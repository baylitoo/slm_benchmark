"""Generic OpenAI-compatible chat + embeddings over the serving stack — the
Playground's "classic queries" / "embed" paths, and platform surfaces in
their own right.

``POST /v1/chat/completions`` routes the request's ``model`` through the same
resolver as everything else (live deployment name, models.yaml profile,
``store:<name>``) and forwards to that upstream — proxying a real
token-by-token SSE stream when the caller asks for one (``stream: true``),
same wire protocol llama-server itself emits. A cold ``store:`` model with
no live placement fires its own load and answers ``202`` with an ETA
instead of a bare error (``_resolve_or_error``, same seam api.py uses for
``/v1/extract/*``). ``POST /v1/embeddings`` does the same for an embedding
deployment (llama-server ``--embedding``), so a RAG/retrieval pipeline
computes document vectors on THIS node — they never leave the infra. ``GET
/v1/models`` advertises the routable ids. Unlike the standalone ``docie
gateway`` (its own process, models.yaml only), this rides the main API:
live deployments work out of the box and the agents' bearer-friendly
tenant guard applies.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, StringConstraints

from docie_bench.agents.api import (
    _client,
    _openai_error,
    agents_tenant_guard,
)
from docie_bench.inngest.serving_api import trigger_deployment_load
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.mojibake import fix_completion_content
from docie_bench.security import TenantContext
from docie_bench.serving import recency
from docie_bench.serving.placement_resolver import (
    STORE_PROFILE_PREFIX,
    PlacementNotFoundError,
    PlacementNotReadyError,
    endpoint_is_loopback,
)
from docie_bench.serving.profile_resolver import (
    ProfileResolutionError,
    resolve_extraction_profile,
)
from docie_bench.settings import get_settings
from docie_bench.studio import usage_store

router = APIRouter(tags=["chat"], dependencies=[Depends(agents_tenant_guard)])

# Bound on the disposable usage-frame scan buffer (see _stream_chat_completions):
# a single SSE line ought to be at most one completion frame's worth of JSON.
# Past this, give up scanning for a usage block rather than growing unboundedly.
_SSE_USAGE_SCAN_LIMIT = 262_144

# Same dependency the router already guards with -- FastAPI caches a repeated
# dependency per request, so declaring it again as a route parameter hands the
# route the SAME authenticated TenantContext without a second quota acquire.
TenantParam = Annotated[TenantContext, Depends(agents_tenant_guard)]

_NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class ChatCompletionRequest(BaseModel):
    """Body of ``POST /v1/chat/completions``.

    Only the fields ``chat_completions`` itself reads and branches on are
    declared -- ``extra="allow"`` is load-bearing, not incidental: the rest
    of the OpenAI-compatible body (``messages``, ``temperature``, ``tools``,
    ...) is forwarded upstream largely as-is (``forward = dict(body)``-style)
    and must never be dropped or rejected by a closed schema.
    """

    model_config = ConfigDict(extra="allow")

    model: _NonEmptyStr
    stream: bool | None = None
    mcp_servers: list[_NonEmptyStr] | None = None
    session_id: str | None = None
    # Human-in-the-loop pause/resume (#383): opt-in, per request -- a
    # programmatic caller of this same endpoint has no human to pause for,
    # so the synthetic ask_user tool and the Pause control's respond channel
    # are never silently injected. Only meaningful together with
    # ``mcp_servers`` + ``stream: true`` (``_stream_chat_with_mcp_tools``);
    # ignored everywhere else, including the non-streaming ``mcp_servers``
    # path and the standalone Agents surface, which never read it at all.
    enable_ask_user: bool = False


def _validation_error_to_openai(exc: RequestValidationError) -> JSONResponse:
    """Reformat FastAPI's ``{"detail": [...]}`` 422 shape into this
    endpoint's existing ``_openai_error(...)`` 400 shape, which clients of
    ``/v1/chat/completions`` already depend on."""
    errors = exc.errors()
    first = errors[0] if errors else None
    if first is None:
        message = "invalid request body"
    else:
        field = ".".join(str(part) for part in first["loc"] if part != "body")
        msg = first.get("msg", "invalid request")
        message = f"{field!r}: {msg}" if field else msg
    return _openai_error(message, status_code=400, error_type="invalid_request_error")


class _ChatCompletionsValidationRoute(APIRoute):
    """Scoped to the single ``/v1/chat/completions`` route it's attached to
    (via ``route_class_override`` on ``add_api_route`` below) -- every other
    route on this router, and every other route in the app, keeps FastAPI's
    default validation-error response untouched."""

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except RequestValidationError as exc:
                return _validation_error_to_openai(exc)

        return custom_route_handler


def _record_usage_outcome(
    profile_name: str,
    surface: str,
    tenant_id: str,
    started: float,
    outcome: Any,
) -> None:
    """One usage-ledger row for a request that RESOLVED to a deployment.

    The single per-surface seam: called once per non-streaming request with
    whatever the surface is about to return -- a completion dict (status
    ``ok``, token counts lifted from its OpenAI ``usage`` block when present)
    or an error ``JSONResponse`` (status ``error``, no tokens). The streaming
    chat path records its own row (``_stream_chat_completions``): tokens land
    there too when the upstream's final SSE frame carries a usage block,
    ``None`` otherwise. ``record_usage`` never raises -- a ledger hiccup
    cannot fail the request it describes.
    """
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    status = "ok"
    if isinstance(outcome, JSONResponse):
        status = "error"
    elif isinstance(outcome, dict):
        usage = outcome.get("usage")
        if isinstance(usage, dict):
            raw_prompt = usage.get("prompt_tokens")
            raw_completion = usage.get("completion_tokens")
            prompt_tokens = raw_prompt if isinstance(raw_prompt, int) else None
            completion_tokens = raw_completion if isinstance(raw_completion, int) else None
    usage_store.record_usage(
        deployment=profile_name,
        surface=surface,
        tenant_id=tenant_id,
        latency_ms=int((time.perf_counter() - started) * 1000),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        status=status,
    )


def _loading_response(triggered: tuple[str, float]) -> JSONResponse:
    name, eta = triggered
    return JSONResponse(
        status_code=202,
        content={
            "status": "loading",
            "deployment": name,
            "eta_seconds": round(eta, 1),
            "message": (
                f"Model {name!r} is starting — this can take a moment "
                f"the first time. Retry in ~{int(eta)}s."
            ),
        },
    )


async def _resolve_or_error(
    model: str, *, session_id: str | None = None
) -> ModelProfile | JSONResponse:
    """Resolve ``model`` the same way every route here needs to, folding in
    load-on-demand for a store model that isn't live yet.

    ``session_id``, when given, is forwarded to ``resolve_extraction_profile``
    so a scaled ``store:`` model pins the replica pick to the conversation
    (see ``placement_resolver.session_affinity_choice``) instead of
    round-robining every turn. Callers that have no session concept
    (embeddings, rerank) simply omit it and get today's round-robin
    behavior, unchanged.

    Mirrors api.py's resolve_profile (same trigger_deployment_load seam): a
    store: model with no live placement fires its own load/deploy and answers
    202 with an ETA instead of a bare 404/409 — the point being a caller of
    /v1/chat/completions, /v1/embeddings or /v1/rerank shouldn't need to know
    in advance whether the model it asked for happens to be warm, any more
    than a caller of /v1/extract/* does. A name that isn't a catalog entry at
    all still 404s; there's nothing to start.

    Also tries the trigger for a BARE (non-``store:``-prefixed) name that
    fails resolution: the Playground's Chat/Vision pickers submit a raw
    deployment name, never ``store:``-prefixed (unlike an explicit API
    caller), so an evicted deployment picked there would otherwise 404
    outright — trigger_deployment_load safely returns None for a name that
    isn't a real catalog entry, so this is a no-op for a genuinely unknown
    model.

    A resolved ``store:`` profile also gets api.py's worker-loopback guard:
    the deploy runtime records a placement's endpoint from the WORKER's
    point of view, so a loopback endpoint is unreachable from this (api)
    process. Unlike api.py's extract path, there is no worker-side chat
    equivalent to redirect to — a chat/embeddings/rerank caller stuck behind
    this guard has to fix the deployment (a non-loopback advertised
    endpoint), not switch routes.
    """
    try:
        profile = resolve_extraction_profile(model_profile=model, session_id=session_id)
    except (PlacementNotFoundError, PlacementNotReadyError) as exc:
        store_name = (
            model[len(STORE_PROFILE_PREFIX) :] if model.startswith(STORE_PROFILE_PREFIX) else None
        )
        triggered = await trigger_deployment_load(store_name) if store_name else None
        if triggered is not None:
            return _loading_response(triggered)
        not_found = isinstance(exc, PlacementNotFoundError)
        error_type = "model_not_found" if not_found else "model_not_ready"
        status_code = 404 if not_found else 409
        return _openai_error(str(exc), status_code=status_code, error_type=error_type)
    except ProfileResolutionError as exc:
        triggered = await trigger_deployment_load(model)
        if triggered is not None:
            return _loading_response(triggered)
        return _openai_error(str(exc), status_code=404, error_type="model_not_found")
    if (
        model.startswith(STORE_PROFILE_PREFIX)
        and endpoint_is_loopback(profile.base_url)
    ):
        # See api.py's resolve_profile: the deploy runtime records a
        # placement's endpoint from the WORKER's point of view, so a
        # loopback endpoint is unreachable from this (api) process. Without
        # this guard the request "resolves" fine and then burns
        # timeout_seconds x retries on a doomed connect before a confusing
        # upstream_unavailable 502.
        return _openai_error(
            f"{model} resolved to {profile.base_url}, which is loopback on "
            "the worker that deployed it and not reachable from the API. "
            "Record a non-loopback advertised endpoint at deploy time, or "
            "extract through the worker (POST /v1/studio/extract) if "
            "extraction — not chat — is what you actually need.",
            status_code=501,
            error_type="deployment_unreachable",
        )
    return profile


@router.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Routable model ids: models.yaml profiles + live deployments."""
    from docie_bench.llm.model_profiles import load_model_profiles
    from docie_bench.serving.profile_resolver import (
        DEFAULT_MODELS_CONFIG,
        _default_live_deployments,
        _is_live,
        build_profile_table,
    )

    path = DEFAULT_MODELS_CONFIG
    yaml_profiles = load_model_profiles(path) if path.exists() else {}
    live = {
        record.spec.name: record
        for record in _default_live_deployments()
        if _is_live(record)
    }
    table = build_profile_table(yaml_profiles, live)
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "docie"}
            for name in sorted(table)
        ],
    }


async def chat_completions(payload: ChatCompletionRequest, tenant: TenantParam) -> Any:
    model = payload.model
    session_id = payload.session_id
    resolved = await _resolve_or_error(model, session_id=session_id)
    if isinstance(resolved, JSONResponse):
        return resolved
    profile = resolved
    if profile.kind != "passthrough":
        return _openai_error(
            f"model {model!r} is a {profile.kind!r} solution profile — use the "
            "gateway or an agent for solution kinds",
            status_code=400,
            error_type="invalid_request_error",
        )

    wants_stream = bool(payload.stream)
    mcp_server_names = payload.mcp_servers
    # exclude_unset: only fields the caller actually sent are forwarded --
    # same as the old dict(body) -- so a field this model declares but the
    # caller omitted (stream, mcp_servers, session_id) doesn't materialize
    # as an explicit null in the upstream request.
    forward = payload.model_dump(exclude_unset=True)
    forward["model"] = profile.model
    url = f"{profile.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {profile.api_key}",
        "Content-Type": "application/json",
    }
    client: httpx.AsyncClient = _client()
    started = time.perf_counter()

    if mcp_server_names:
        names = [str(n) for n in mcp_server_names]
        if wants_stream:
            # Not the OpenAI token-stream format (there is no meaningful
            # token stream for a tool-calling round) -- each executed tool
            # call is relayed as its own SSE event as it finishes, so the
            # caller sees the agentic search happening instead of only the
            # final answer after everything completes silently.
            return await _stream_chat_with_mcp_tools(
                client,
                url,
                headers,
                forward,
                profile,
                names,
                session_id,
                tenant_id=tenant.tenant_id,
                started=started,
                enable_ask_user=payload.enable_ask_user,
            )
        outcome = await _chat_with_mcp_tools(
            client, url, headers, forward, profile, names, session_id
        )
        # run_tool_loop already summed usage across every tool round into the
        # final completion's usage block, so this one row carries the whole
        # exchange's token cost.
        _record_usage_outcome(profile.name, "chat", tenant.tenant_id, started, outcome)
        return outcome

    if wants_stream:
        return await _stream_chat_completions(
            client,
            url,
            headers,
            forward,
            profile.timeout_seconds,
            profile.name,
            tenant_id=tenant.tenant_id,
            started=started,
        )

    completion = await _post_upstream(client, url, headers, forward, profile)
    _record_usage_outcome(profile.name, "chat", tenant.tenant_id, started, completion)
    if isinstance(completion, JSONResponse):
        return completion
    # PR-4 recency: this surface serves traffic too — stamp last_served like
    # api.py's extract path, or a deployment driven only through chat reads
    # as idle forever and becomes the first idle-TTL/LRU eviction victim.
    recency.stamp_served_profile(profile.name)
    # Repair model-emitted UTF-8 mojibake in the answer (accented OCR/description
    # on small vision models — the Playground Vision path lands here).
    if get_settings().fix_mojibake:
        completion = fix_completion_content(completion)
    return completion


# route_class_override, not the @router.post(...) decorator (it doesn't expose
# this kwarg) -- scopes _ChatCompletionsValidationRoute's OpenAI-shaped 400 to
# this one route without touching any other route's default FastAPI 422.
router.add_api_route(
    "/v1/chat/completions",
    chat_completions,
    methods=["POST"],
    route_class_override=_ChatCompletionsValidationRoute,
)


async def _post_upstream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    forward: dict[str, Any],
    profile: ModelProfile,
    *,
    error_hint: str = "",
) -> dict[str, Any] | JSONResponse:
    """One non-streaming upstream completion, errors mapped to OpenAI shape.

    Shared by the plain chat path, every round of the MCP tool loop, and the
    embeddings/rerank forwards — same forwarding, same error taxonomy,
    regardless of which path posts. ``error_hint`` lets a surface append its
    own diagnosis to an upstream 4xx/5xx (e.g. "is this deployment an
    embedding model?") without owning a copy of the whole error ladder.
    """
    body = dict(forward)
    body.pop("stream", None)
    body.pop("mcp_servers", None)
    body.pop("session_id", None)
    body.pop("enable_ask_user", None)
    try:
        upstream = await client.post(
            url, json=body, headers=headers, timeout=profile.timeout_seconds
        )
    except httpx.RequestError as exc:
        return _openai_error(
            f"upstream {profile.base_url} is unreachable: {exc}",
            status_code=502,
            error_type="upstream_unavailable",
        )
    if upstream.status_code >= 400:
        return _openai_error(
            f"upstream returned {upstream.status_code}: {upstream.text[:300]}{error_hint}",
            status_code=upstream.status_code,
            error_type="upstream_error",
        )
    try:
        completion = upstream.json()
    except ValueError:
        return _openai_error(
            "upstream returned a non-JSON response",
            status_code=502,
            error_type="upstream_error",
        )
    if not isinstance(completion, dict):
        return _openai_error(
            "upstream returned a non-object completion",
            status_code=502,
            error_type="upstream_error",
        )
    return completion


async def _resolve_mcp_specs(
    server_names: list[str], session_id: str | None
) -> list[Any] | JSONResponse:
    """Registry lookup + the ``session_id`` docs-search directory override —
    shared setup between the sync and SSE ``mcp_servers`` chat paths so the
    registry-only security check and the session-scoped override (#296)
    have exactly one implementation, not two that can drift.

    Registry-only security: every requested name must exist in
    ``settings.mcp_servers_config`` — the request can never point the server
    at an arbitrary URL or command. The ``mcp`` SDK being absent is a clean
    501 (the extra is optional), not an ImportError.

    ``session_id``: when given and ``docs-search`` is among ``server_names``,
    that server's spec is launched with its documents directory overridden
    to this session's upload directory instead of the operator's shared
    one — a Playground attachment uploaded via
    ``POST /v1/studio/session-documents`` becomes searchable for this
    conversation without ever touching the shared corpus.
    """
    from dataclasses import replace

    from docie_bench import mcp_tools as mcp_mod

    try:
        registry = mcp_mod.load_mcp_registry()
    except mcp_mod.MCPConfigError as exc:
        return _openai_error(str(exc), status_code=500, error_type="mcp_config_error")
    unknown = [name for name in server_names if name not in registry]
    if unknown:
        return _openai_error(
            f"unregistered MCP server(s): {', '.join(unknown)} — register them in "
            f"{get_settings().mcp_servers_config} (see GET /v1/mcp/servers)",
            status_code=400,
            error_type="mcp_server_not_registered",
        )
    try:
        mcp_mod._require_mcp()
    except mcp_mod.MCPUnavailableError as exc:
        return _openai_error(str(exc), status_code=501, error_type="mcp_unavailable")
    specs = [registry[name] for name in server_names]

    if session_id is not None and "docs-search" in server_names:
        from docie_bench.mcp_servers.docs_search import DOCS_DIR_ENV
        from docie_bench.mcp_session_documents import SessionDocumentError, session_documents_dir

        try:
            session_dir = session_documents_dir(session_id)
        except SessionDocumentError as exc:
            return _openai_error(str(exc), status_code=400, error_type="invalid_request_error")
        specs = [
            replace(spec, env={**spec.env, DOCS_DIR_ENV: str(session_dir)})
            if spec.name == "docs-search"
            else spec
            for spec in specs
        ]
    return specs


async def _chat_with_mcp_tools(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    forward: dict[str, Any],
    profile: ModelProfile,
    server_names: list[str],
    session_id: str | None = None,
) -> Any:
    """Serve one chat request with MCP tools: connect, advertise, loop."""
    from contextlib import AsyncExitStack

    from docie_bench import mcp_tools as mcp_mod

    specs = await _resolve_mcp_specs(server_names, session_id)
    if isinstance(specs, JSONResponse):
        return specs

    async def post(body: dict[str, Any]) -> dict[str, Any] | JSONResponse:
        return await _post_upstream(client, url, headers, body, profile)

    tool_call_trace: list[dict[str, Any]] = []
    record_tool_call = mcp_mod.make_trace_recorder(tool_call_trace)

    try:
        async with AsyncExitStack() as stack:
            try:
                sessions = await mcp_mod.open_mcp_sessions(stack, specs)
                tools, mapping = await mcp_mod.collect_openai_tools(sessions)
            except Exception as exc:  # noqa: BLE001 - connect/handshake failure is a gateway error
                return _openai_error(
                    f"could not connect to MCP server(s): {exc}",
                    status_code=502,
                    error_type="mcp_server_unreachable",
                )
            completion = await mcp_mod.run_tool_loop(
                post, forward, sessions, mapping, tools, on_tool_call=record_tool_call
            )
    except Exception as exc:  # noqa: BLE001 - transport teardown (ExitStack unwind) failure
        return _openai_error(
            f"MCP session error: {exc}",
            status_code=502,
            error_type="mcp_server_unreachable",
        )
    if completion is None:
        return _openai_error(
            f"model kept calling tools for {get_settings().mcp_max_tool_iterations} "
            "rounds without a final answer — raise mcp_max_tool_iterations or "
            "simplify the request",
            status_code=502,
            error_type="mcp_tool_loop_exhausted",
        )
    if isinstance(completion, JSONResponse):
        return completion
    recency.stamp_served_profile(profile.name)
    if get_settings().fix_mojibake:
        completion = fix_completion_content(completion)
    if tool_call_trace:
        completion = {**completion, "docie_agent": {"tool_calls": tool_call_trace}}
    return completion


def _context_length_for_profile(profile: ModelProfile) -> int | None:
    """The resolved deployment's own ``context_length`` ceiling for
    ``run_tool_loop``'s context-budget warning (#344), or ``None`` when it
    can't be resolved.

    Same lookup #333's batch fan-out capacity uses for ``n_parallel`` (both
    read the live deployment record straight off ``deployments.json`` and
    pull a field off ``spec.launch``) — here for ``context_length`` instead.
    ``profile.name`` is the deployment's own name for a live-deployment
    profile (``profile_resolver._synthesize_profile``); a profile that isn't
    backed by one (a plain ``models.yaml`` entry, or the env fallback) has
    no matching record, which is exactly the "unknown/unpriceable" case —
    fail open, same convention as every other fit/pricing gate in this
    codebase, rather than guess or block the request.
    """
    from docie_bench.serving.profile_resolver import _default_live_deployments

    for record in _default_live_deployments():
        if record.spec.name == profile.name:
            return record.spec.launch.context_length
    return None


def _tool_calls_supported_for_profile(profile: ModelProfile) -> bool | None:
    """The resolved deployment's persisted ``tool_calls_supported`` verdict
    (or ``None`` when unknown), for the pre-loop warning below.

    Same live-deployment lookup as ``_context_length_for_profile`` — read
    straight off ``deployments.json`` via ``_default_live_deployments()`` and
    matched by ``profile.name`` — here for
    ``DeploymentRecord.tool_calls_supported`` instead of
    ``spec.launch.context_length``. That field is written by the reconciler
    each health cycle from ``LlamaCppRuntime.health()``'s own ``GET /props``
    check (``chat_template_caps.supports_tool_calls`` — see
    ``runtime.llamacpp_tool_calls_mismatch``): ``True``/``False`` is the real
    signal, ``None`` covers every "can't tell" case (non-llamacpp runtime, an
    older llama-server build, an unresolvable profile, or no health cycle has
    observed it yet) — fail open, same convention as
    ``_context_length_for_profile``.
    """
    from docie_bench.serving.profile_resolver import _default_live_deployments

    for record in _default_live_deployments():
        if record.spec.name == profile.name:
            return record.tool_calls_supported
    return None


def _sse_event(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _error_payload(message: str, error_type: str) -> dict[str, Any]:
    return {"message": message, "type": error_type, "code": error_type}


async def _post_upstream_streamed(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    forward: dict[str, Any],
    profile: ModelProfile,
    queue: asyncio.Queue[dict[str, Any] | None],
) -> dict[str, Any] | JSONResponse:
    """Streaming counterpart to ``_post_upstream``, used ONLY by
    ``_stream_chat_with_mcp_tools``'s ``post`` closure — every other caller
    of ``run_tool_loop`` (the non-streaming ``mcp_servers`` chat path, the
    standalone Agents surface) keeps using the blocking ``_post_upstream``,
    completely unchanged. This is why a tool-calling round's reasoning and
    final answer used to both land as one lump the instant llama-server
    finished generating: every round was one non-streaming POST.

    Same forwarding contract as ``_post_upstream`` — the same fields
    (``mcp_servers``, ``session_id``, ``enable_ask_user``) are stripped,
    the same profile/timeout apply — except the upstream call itself is
    made with ``stream: true`` + ``stream_options: {"include_usage": true}``,
    the exact convention ``_stream_chat_completions`` already established,
    instead of one blocking non-streaming POST.

    As each OpenAI-shaped SSE delta frame arrives, ``delta.content`` and
    ``delta.reasoning_content`` fragments are pushed onto ``queue`` as
    ``content_delta``/``reasoning_delta`` events — the SAME queue
    ``_stream_chat_with_mcp_tools.drive()`` already writes its other events
    to — for live rendering, while ``delta.tool_calls`` fragments accumulate
    silently (standard OpenAI streaming tool-call accumulation: fragments
    keyed by ``index``, ``function.name``/``function.arguments`` string
    pieces concatenated as they arrive, ``arguments`` never parsed as JSON
    here — that stays ``execute_tool_call``'s job once the round is whole).

    Once the stream ends (``[DONE]``, or the upstream just closes the
    connection), the accumulated pieces are reassembled into a completion
    dict in EXACTLY the shape ``_post_upstream`` returns for an equivalent
    non-streaming request — ``choices[0].message.{content, reasoning_content,
    tool_calls}``, plus ``usage`` from the trailing frame when present.
    ``run_tool_loop``'s tool-call-vs-final-answer detection, and every other
    caller of ``post()``, are completely unaffected — fed by an accumulated
    stream instead of a single blocking response.

    Connection failures (before or during the stream) and 4xx/5xx upstream
    responses map to the same OpenAI-shaped error dict ``_post_upstream``
    returns for the same conditions — streaming must never regress error
    handling to get token-by-token rendering.
    """
    body = dict(forward)
    body.pop("mcp_servers", None)
    body.pop("session_id", None)
    body.pop("enable_ask_user", None)
    body["stream"] = True
    stream_options = body.get("stream_options")
    if not isinstance(stream_options, dict):
        stream_options = {}
    stream_options.setdefault("include_usage", True)
    body["stream_options"] = stream_options

    stream_ctx = client.stream(
        "POST", url, json=body, headers=headers, timeout=profile.timeout_seconds
    )
    try:
        upstream = await stream_ctx.__aenter__()
    except httpx.RequestError as exc:
        return _openai_error(
            f"upstream {profile.base_url} is unreachable: {exc}",
            status_code=502,
            error_type="upstream_unavailable",
        )
    try:
        if upstream.status_code >= 400:
            error_bytes = await upstream.aread()
            return _openai_error(
                f"upstream returned {upstream.status_code}: "
                f"{error_bytes[:300].decode('utf-8', 'replace')}",
                status_code=upstream.status_code,
                error_type="upstream_error",
            )

        role = "assistant"
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Keyed by the fragment's own "index" (its position among this
        # round's tool calls, NOT a byte offset) -- the OpenAI streaming
        # tool-call contract, mirrored verbatim: the first fragment for a
        # given index carries "id"/"type"/a name piece, every later
        # fragment for that same index carries only more "arguments" text.
        tool_call_fragments: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        completion_id = ""
        model_name = str(body.get("model") or "")

        async for line in upstream.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload in ("", "[DONE]"):
                continue
            try:
                frame = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(frame, dict):
                continue
            frame_id = frame.get("id")
            if isinstance(frame_id, str) and frame_id:
                completion_id = frame_id
            frame_model = frame.get("model")
            if isinstance(frame_model, str) and frame_model:
                model_name = frame_model
            frame_usage = frame.get("usage")
            if isinstance(frame_usage, dict):
                # The OpenAI/llama-server streaming contract sends this on
                # the TRAILING frame (stream_options.include_usage, set
                # above) -- a later frame's usage always wins, but there is
                # normally at most one.
                usage = frame_usage
            choices = frame.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            choice_finish = choice.get("finish_reason")
            if isinstance(choice_finish, str) and choice_finish:
                finish_reason = choice_finish
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            delta_role = delta.get("role")
            if isinstance(delta_role, str) and delta_role:
                role = delta_role
            content_piece = delta.get("content")
            if isinstance(content_piece, str) and content_piece:
                content_parts.append(content_piece)
                queue.put_nowait({"type": "content_delta", "text": content_piece})
            reasoning_piece = delta.get("reasoning_content")
            if isinstance(reasoning_piece, str) and reasoning_piece:
                reasoning_parts.append(reasoning_piece)
                queue.put_nowait({"type": "reasoning_delta", "text": reasoning_piece})
            delta_tool_calls = delta.get("tool_calls")
            if isinstance(delta_tool_calls, list):
                for fragment in delta_tool_calls:
                    if not isinstance(fragment, dict):
                        continue
                    index = fragment.get("index")
                    if not isinstance(index, int):
                        index = 0
                    entry = tool_call_fragments.setdefault(
                        index,
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    fragment_id = fragment.get("id")
                    if isinstance(fragment_id, str) and fragment_id:
                        entry["id"] = fragment_id
                    fragment_type = fragment.get("type")
                    if isinstance(fragment_type, str) and fragment_type:
                        entry["type"] = fragment_type
                    function_fragment = fragment.get("function")
                    if isinstance(function_fragment, dict):
                        name_piece = function_fragment.get("name")
                        if isinstance(name_piece, str):
                            entry["function"]["name"] += name_piece
                        # A raw JSON string streamed in pieces -- concatenate
                        # as they arrive, never parse/validate until the
                        # round is fully done (execute_tool_call's job, once
                        # run_tool_loop has the whole thing).
                        arguments_piece = function_fragment.get("arguments")
                        if isinstance(arguments_piece, str):
                            entry["function"]["arguments"] += arguments_piece
    except httpx.RequestError as exc:
        return _openai_error(
            f"upstream {profile.base_url} is unreachable: {exc}",
            status_code=502,
            error_type="upstream_unavailable",
        )
    finally:
        await stream_ctx.__aexit__(None, None, None)

    message: dict[str, Any] = {
        "role": role,
        "content": "".join(content_parts) if content_parts else None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_call_fragments:
        message["tool_calls"] = [tool_call_fragments[i] for i in sorted(tool_call_fragments)]

    completion: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        completion["usage"] = usage
    return completion


async def _stream_chat_with_mcp_tools(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    forward: dict[str, Any],
    profile: ModelProfile,
    server_names: list[str],
    session_id: str | None,
    *,
    tenant_id: str,
    started: float,
    enable_ask_user: bool = False,
) -> StreamingResponse:
    """SSE variant of ``_chat_with_mcp_tools``: each executed tool call is
    relayed to the client the moment it finishes, instead of the whole
    multi-round exchange completing silently before anything is sent back —
    "Waiting for the model…" with no visibility into the agentic search
    actually running underneath it.

    NOT the OpenAI streaming wire format (there is no meaningful token
    stream for a tool-calling round) — a caller opts in by setting BOTH
    ``stream`` and ``mcp_servers``. Event shapes, one JSON object per
    ``data:`` frame:
      ``{"type": "exchange", "exchange_id": <str>}`` — fired FIRST, before
        any other event, ONLY when the request set ``enable_ask_user``
        (#383). ``exchange_id`` is a fresh id minted for this one request —
        NOT the caller's own ``session_id`` (which spans a whole
        conversation) — and is the id a client uses with the sibling
        ``POST /v1/chat/completions/pause`` and
        ``.../respond`` endpoints below to pause this exchange, or answer an
        ``awaiting_input`` event, while it's still running. Never fires when
        ``enable_ask_user`` is unset — the registry/pause machinery is
        opt-in per request, since a programmatic (non-Playground) caller of
        this same endpoint has no human to pause for.
      ``{"type": "awaiting_input", "question"?, "choices"?}`` — fired every
        time ``run_tool_loop`` pauses waiting for a human answer (#383): a
        model-issued ``ask_user`` tool call carries ``question`` (and
        ``choices`` when the model gave any — render a picker; free text
        otherwise), a user-initiated pause (``POST .../pause``) carries
        neither — render a plain "add context" free-text box. The exchange
        is genuinely suspended at this point (no further rounds, no further
        tool calls) until ``POST /v1/chat/completions/respond`` answers it,
        or ``settings.mcp_ask_user_timeout_seconds`` elapses and the
        exchange ends with an ``error`` event instead. Can fire more than
        once per exchange (a conversation may pause more than once) and can
        interleave with ``tool_call``/``reasoning`` events from rounds
        before and after it. Only ever fires when ``enable_ask_user`` was
        set on the request.
      ``{"type": "system_addendum", "text": <addendum>}`` — fired once per
        request, before the first model round, when ``run_tool_loop`` folds
        ``TOOL_DISCIPLINE_DIRECTIVE`` (and any eager-list context) into the
        request's system message. This is real content injected on top of
        whatever the caller's own system prompt says, but was never
        surfaced anywhere until now — a one-time header, not a per-round
        event. Note a docs-search request's eager-list ``tool_call`` event
        (see ``_eager_list_context``) is emitted BEFORE this one, since that
        listing call happens while the addendum text is still being built.
      ``{"type": "content_delta", "text": <fragment>}`` / ``{"type":
        "reasoning_delta", "text": <fragment>}`` — real token-by-token
        streaming this time: the ``post`` this function builds now streams
        each round's upstream call (see ``_post_upstream_streamed``)
        instead of ``_post_upstream``'s one blocking response, and every
        fragment of that round's ``delta.content``/``delta.reasoning_content``
        is pushed the instant it arrives, interleaved with each other in
        emission order. Every delta for a round lands BEFORE that round's
        own ``usage``, ``reasoning``, and ``tool_call``/``content`` events
        below, which are still built from the round's fully-accumulated
        completion once its stream ends — deltas are purely additive, for
        live rendering only (the Playground appends each fragment to the
        in-progress bubble as it streams), never a replacement for those
        payloads (the final ``content`` event still carries the complete
        ``docie_agent.tool_calls`` trace, etc).
      ``{"type": "tool_call", "tool", "status", "latency_ms", "arguments",
        "result"}`` — same shape as the static ``docie_agent.tool_calls``
        trace (#261/#262), so the frontend's existing ``ToolCallTrace``
        component renders each one as it arrives instead of only after
        the fact.
      ``{"type": "reasoning", "text": <round's reasoning_content>}`` — a
        reasoning-capable model's "why" for that round (calling a tool, or
        the final answer), when the chat template emits one separately
        from ``content``/``tool_calls``. Answers "is there a hidden
        thinking step before the tool call" with the model's own words
        instead of leaving it invisible.
      ``{"type": "usage", "round": {...}, "cumulative": {...}}`` — that
        round's own token usage plus the running cumulative totals through
        this round (``run_tool_loop``'s ``on_usage``), the raw counts with
        no context-window denominator. Lets a client show how much a live
        agentic exchange is consuming before the final completion lands,
        instead of only after the whole exchange finishes.
      ``{"type": "context_budget", "cumulative_tokens", "context_length",
        "threshold_fraction"}`` — fired AT MOST ONCE per exchange (#344),
        the first round whose cumulative usage crosses
        ``settings.mcp_context_budget_warn_fraction`` (default 80%) of the
        resolved deployment's own ``context_length`` (see
        ``_context_length_for_profile``). A WARNING only — a long agentic
        exchange can otherwise run several real rounds before a LATER
        round's cumulative usage exceeds the deployment's context window and
        llama-server hard-400s, losing the whole in-progress exchange with
        no prior warning. Never fires when the ceiling can't be resolved
        (an unknown/unpriceable deployment) — fail-open, same as every
        other fit/pricing gate in this codebase.
      ``{"type": "tool_calls_unsupported", "message"}`` — fired AT MOST ONCE,
        BEFORE the tool loop runs a single round, when the resolved
        deployment's persisted ``tool_calls_supported`` (see
        ``_tool_calls_supported_for_profile``) is explicitly ``False``: the
        model's ACTUAL chat template does not support real tool-calling
        (llama-server's own ``chat_template_caps.supports_tool_calls``),
        regardless of what ``RuntimeFeature.TOOL_CALLS`` advertises. A round
        of ``docs-search__list_files`` can still succeed even here — that
        call is the framework's own eager pre-round listing
        (``mcp_tools._eager_list_context``), not something the model itself
        emitted — but every round after that needs the model to decide to
        call a tool via its own chat template, which this deployment cannot
        do; it will describe using a tool in prose instead of ever emitting a
        real ``tool_calls`` field. Known upfront from the deployment's own
        health state, not learned mid-exchange, unlike ``context_budget``
        above — so it fires before ``run_tool_loop`` even starts. Never fires
        when the verdict is ``True`` or unresolvable (``None``) — fail-open,
        same convention as ``context_budget``.
      ``{"type": "content", "completion": <final OpenAI-shaped completion>}``
      ``{"type": "error", "error": {"message", "type", "code"}}`` — ``type``/
        ``code`` is ``"ask_user_timeout"`` specifically when a pause got no
        answer in time (#383); every other error keeps its existing
        ``error_type`` (``mcp_server_unreachable``, ``mcp_tool_loop_exhausted``, …).
    Always terminated by a literal ``data: [DONE]\\n\\n`` frame, the same
    convention ``_stream_chat_completions`` uses.
    """
    import contextlib
    import uuid
    from contextlib import AsyncExitStack

    from docie_bench import mcp_tools as mcp_mod

    context_length_ceiling = _context_length_for_profile(profile)
    tool_calls_supported = _tool_calls_supported_for_profile(profile)
    exchange_id = uuid.uuid4().hex if enable_ask_user else None
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    class _QueueTraceSink(list[dict[str, Any]]):
        """``make_trace_recorder`` only knows how to ``.append()`` to a
        list — this subclass reuses its exact stringify/truncate formatting
        while also pushing each entry onto the SSE queue the instant it's
        recorded, without needing a second, separately-formatted event."""

        def append(self, item: dict[str, Any]) -> None:
            super().append(item)
            queue.put_nowait({"type": "tool_call", **item})

    async def drive() -> None:
        outcome: Any = None
        try:
            if exchange_id is not None:
                # Fired FIRST, before anything else -- a client needs this id
                # to call POST .../pause at any later point in the exchange,
                # including before the first round has even run.
                mcp_mod.open_pending_input(exchange_id)
                queue.put_nowait({"type": "exchange", "exchange_id": exchange_id})
            if tool_calls_supported is False:
                queue.put_nowait(
                    {
                        "type": "tool_calls_unsupported",
                        "message": (
                            "This deployment's chat template does not support real "
                            "tool-calling (llama-server reports "
                            "chat_template_caps.supports_tool_calls=false) — the model "
                            "may describe using tools instead of actually calling them."
                        ),
                    }
                )
            specs = await _resolve_mcp_specs(server_names, session_id)
            if isinstance(specs, JSONResponse):
                body = json.loads(bytes(specs.body))
                outcome = specs
                queue.put_nowait({"type": "error", "error": body.get("error", body)})
                return

            async def post(body: dict[str, Any]) -> dict[str, Any] | JSONResponse:
                return await _post_upstream_streamed(client, url, headers, body, profile, queue)

            trace = _QueueTraceSink()
            record_tool_call = mcp_mod.make_trace_recorder(trace)
            try:
                async with AsyncExitStack() as stack:
                    try:
                        sessions = await mcp_mod.open_mcp_sessions(stack, specs)
                        tools, mapping = await mcp_mod.collect_openai_tools(sessions)
                    except Exception as exc:  # noqa: BLE001 - connect/handshake failure
                        message = f"could not connect to MCP server(s): {exc}"
                        outcome = _openai_error(
                            message, status_code=502, error_type="mcp_server_unreachable"
                        )
                        error = _error_payload(message, "mcp_server_unreachable")
                        queue.put_nowait({"type": "error", "error": error})
                        return
                    completion = await mcp_mod.run_tool_loop(
                        post,
                        forward,
                        sessions,
                        mapping,
                        tools,
                        on_tool_call=record_tool_call,
                        on_reasoning=lambda text: queue.put_nowait(
                            {"type": "reasoning", "text": text}
                        ),
                        on_system_addendum=lambda text: queue.put_nowait(
                            {"type": "system_addendum", "text": text}
                        ),
                        on_usage=lambda usage: queue.put_nowait({"type": "usage", **usage}),
                        context_length_ceiling=context_length_ceiling,
                        on_context_budget=lambda budget: queue.put_nowait(
                            {"type": "context_budget", **budget}
                        ),
                        exchange_id=exchange_id,
                        on_awaiting_input=(
                            lambda payload: queue.put_nowait(
                                {"type": "awaiting_input", **payload}
                            )
                        )
                        if exchange_id is not None
                        else None,
                    )
            except mcp_mod.AskUserTimeoutError as exc:
                # A paused exchange (ask_user, or a user-initiated pause) got
                # no answer in time -- fail cleanly, never hang the request.
                message = str(exc)
                outcome = _openai_error(message, status_code=504, error_type="ask_user_timeout")
                queue.put_nowait(
                    {"type": "error", "error": _error_payload(message, "ask_user_timeout")}
                )
                return
            except Exception as exc:  # noqa: BLE001 - transport teardown failure
                message = f"MCP session error: {exc}"
                outcome = _openai_error(
                    message, status_code=502, error_type="mcp_server_unreachable"
                )
                queue.put_nowait(
                    {"type": "error", "error": _error_payload(message, "mcp_server_unreachable")}
                )
                return
            if completion is None:
                message = (
                    f"model kept calling tools for {get_settings().mcp_max_tool_iterations} "
                    "rounds without a final answer — raise mcp_max_tool_iterations or "
                    "simplify the request"
                )
                outcome = _openai_error(
                    message, status_code=502, error_type="mcp_tool_loop_exhausted"
                )
                queue.put_nowait(
                    {"type": "error", "error": _error_payload(message, "mcp_tool_loop_exhausted")}
                )
                return
            if isinstance(completion, JSONResponse):
                outcome = completion
                body = json.loads(bytes(completion.body))
                queue.put_nowait({"type": "error", "error": body.get("error", body)})
                return
            recency.stamp_served_profile(profile.name)
            if get_settings().fix_mojibake:
                completion = fix_completion_content(completion)
            if trace:
                completion = {**completion, "docie_agent": {"tool_calls": trace}}
            outcome = completion
            queue.put_nowait({"type": "content", "completion": completion})
        except asyncio.CancelledError:
            # The client disconnected (body_iterator's finally cancels us) --
            # not a failure to report, just stop.
            raise
        except Exception as exc:  # noqa: BLE001 - last-resort net: an unexpected bug here must
            # still reach the client as an error frame, never a silent early
            # [DONE] with no explanation (that's strictly worse than a loud
            # 500 -- it looks like the model just... stopped).
            outcome = _openai_error(
                f"unexpected error in the MCP tool loop: {exc}",
                status_code=500,
                error_type="internal_error",
            )
            queue.put_nowait({"type": "error", "error": _error_payload(str(exc), "internal_error")})
        finally:
            # Always reaped, on every exit path (answered, timed out, a
            # different error, or the client disconnecting mid-CancelledError
            # unwind) -- an abandoned exchange must never leak its registry
            # entry, regardless of how it ended.
            if exchange_id is not None:
                mcp_mod.close_pending_input(exchange_id)
            _record_usage_outcome(profile.name, "chat", tenant_id, started, outcome)
            queue.put_nowait(None)

    async def body_iterator() -> AsyncIterator[bytes]:
        task = asyncio.create_task(drive())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse_event(item)
        finally:
            if not task.done():
                task.cancel()
            # Only a cancellation WE just triggered (client disconnect) is
            # expected here -- drive() itself already turned every other
            # failure into an error frame before its sentinel, so nothing
            # else should reach this await.
            with contextlib.suppress(asyncio.CancelledError):
                await task
        yield b"data: [DONE]\n\n"

    return StreamingResponse(body_iterator(), media_type="text/event-stream")


class _ExchangePauseRequest(BaseModel):
    """Body of ``POST /v1/chat/completions/pause`` (#383)."""

    exchange_id: _NonEmptyStr


class _ExchangeRespondRequest(BaseModel):
    """Body of ``POST /v1/chat/completions/respond`` (#383)."""

    exchange_id: _NonEmptyStr
    text: str = ""


def _unknown_exchange_error(exchange_id: str) -> JSONResponse:
    return _openai_error(
        f"no exchange {exchange_id!r} is awaiting input — it may have already "
        "finished, timed out, or human-in-the-loop was never enabled for it",
        status_code=404,
        error_type="unknown_exchange",
    )


@router.post("/v1/chat/completions/pause")
async def pause_chat_exchange(payload: _ExchangePauseRequest) -> Any:
    """User-initiated pause (#383): flags the exchange named by
    ``exchange_id`` (from that exchange's own ``exchange`` SSE event) to
    suspend at its NEXT round boundary — usable at any point while the
    exchange is running, including mid tool-search, since ``run_tool_loop``
    checks for this at the top of every round. This endpoint only requests
    the pause; the exchange's own SSE stream emits ``awaiting_input`` once
    it actually takes effect, and that is what the Playground waits on
    before showing its free-text prompt as active.

    404s (``unknown_exchange``) when ``exchange_id`` names no exchange that
    ever enabled human-in-the-loop, or one that already finished/timed out —
    the in-memory registry entry is gone either way.
    """
    from docie_bench import mcp_tools as mcp_mod

    if not mcp_mod.request_pause(payload.exchange_id):
        return _unknown_exchange_error(payload.exchange_id)
    return {"paused": True, "exchange_id": payload.exchange_id}


@router.post("/v1/chat/completions/respond")
async def respond_to_chat_exchange(payload: _ExchangeRespondRequest) -> Any:
    """Answers a paused exchange's ``awaiting_input`` event (#383) — a
    model-issued ``ask_user`` question, or a user-initiated pause (``POST
    .../pause``) — with ``text``. Both resolve through this SAME endpoint:
    by the time an answer is being typed, the exchange is in the identical
    awaiting-input state either way.

    Safe to call even before the exchange has actually reached its pause
    checkpoint (e.g. right after ``POST .../pause``, before the current
    round finishes) — the answer is recorded immediately and consumed the
    moment ``run_tool_loop`` gets there, never dropped by the race.

    404s (``unknown_exchange``) under the same conditions as ``.../pause``
    above.
    """
    from docie_bench import mcp_tools as mcp_mod

    if not mcp_mod.submit_pending_answer(payload.exchange_id, payload.text):
        return _unknown_exchange_error(payload.exchange_id)
    return {"accepted": True, "exchange_id": payload.exchange_id}


@router.get("/v1/mcp/servers")
async def list_mcp_servers() -> Any:
    """The registered MCP servers a chat request may name in ``mcp_servers``.

    Secrets never leave: header/env VALUES are redacted to their key names —
    this endpoint exists so a caller (or the Studio) can discover what to
    put in ``mcp_servers``, not to export operator credentials.
    """
    from docie_bench import mcp_tools as mcp_mod

    try:
        registry = mcp_mod.load_mcp_registry()
    except mcp_mod.MCPConfigError as exc:
        return _openai_error(str(exc), status_code=500, error_type="mcp_config_error")
    return {
        "servers": [
            {
                "name": spec.name,
                "transport": spec.transport,
                "url": spec.url,
                "command": list(spec.command) or None,
                "headers": sorted(spec.headers) or None,
                "env": sorted(spec.env) or None,
            }
            for spec in registry.values()
        ]
    }


async def _stream_chat_completions(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    forward: dict[str, Any],
    timeout: float,
    profile_name: str,
    *,
    tenant_id: str,
    started: float,
) -> Any:
    """Real token-by-token proxy: forward raw SSE bytes as the upstream emits
    them, instead of buffering the full completion first. Ported from the
    gateway's ``_forward_stream`` — same shape, chat-only entry point.

    No mojibake repair here (unlike the non-streaming path): a fix-up needs
    the full decoded string, and a multi-byte UTF-8 sequence can straddle a
    chunk boundary. The gateway's proven streaming path skips it too.

    Usage accounting: bytes are relayed to the caller completely unchanged
    (this must never become a de-facto buffering point), but a SECOND,
    disposable line-buffer is fed the same bytes to look for the final
    ``data: {...}`` frame's ``usage`` block — the OpenAI/llama-server
    streaming contract sends one when the request carries
    ``stream_options: {"include_usage": true}`` (set below if the caller
    didn't already ask for it). If no such frame ever arrives (older
    llama-server, a caller that stripped the option back out, a dropped
    connection), the row falls back to ``tokens=None`` exactly as before —
    this is pure upside, never a regression.
    """
    forward["stream"] = True
    stream_options = forward.get("stream_options")
    if not isinstance(stream_options, dict):
        stream_options = {}
    stream_options.setdefault("include_usage", True)
    forward["stream_options"] = stream_options
    stream_ctx = client.stream("POST", url, json=forward, headers=headers, timeout=timeout)
    try:
        upstream = await stream_ctx.__aenter__()
    except httpx.RequestError as exc:
        error = _openai_error(
            f"upstream {url} is unreachable: {exc}",
            status_code=502,
            error_type="upstream_unavailable",
        )
        _record_usage_outcome(profile_name, "chat", tenant_id, started, error)
        return error
    media_type = upstream.headers.get("content-type", "text/event-stream")
    if upstream.status_code >= 400:
        body_bytes = await upstream.aread()
        await stream_ctx.__aexit__(None, None, None)
        detail = body_bytes[:300].decode("utf-8", "replace")
        error = _openai_error(
            f"upstream returned {upstream.status_code}: {detail}",
            status_code=upstream.status_code,
            error_type="upstream_error",
        )
        _record_usage_outcome(profile_name, "chat", tenant_id, started, error)
        return error
    # PR-4 recency, see the non-streaming path's comment above — a stream
    # that gets this far has an accepted upstream connection, which counts
    # as this deployment serving traffic.
    recency.stamp_served_profile(profile_name)

    async def body_iterator() -> AsyncIterator[bytes]:
        usage: dict[str, Any] | None = None
        parse_buffer: bytearray | None = bytearray()
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
                # Byte-level scan of a COPY of what was just relayed -- never
                # holds up delivery, never touches the bytes handed to the
                # caller. Splitting on b"\n" is safe even mid multi-byte UTF-8
                # character: continuation bytes are 0x80-0xBF, never 0x0A.
                if parse_buffer is not None:
                    parse_buffer += chunk
                    if len(parse_buffer) > _SSE_USAGE_SCAN_LIMIT:
                        # Give up silently; any usage already parsed stands.
                        parse_buffer = None
                        continue
                    *lines, parse_buffer = parse_buffer.split(b"\n")
                    parse_buffer = bytearray(parse_buffer)
                    for raw_line in lines:
                        line = bytes(raw_line).strip()
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[len(b"data:") :].strip()
                        if payload in (b"", b"[DONE]"):
                            continue
                        try:
                            frame = json.loads(payload)
                        except ValueError:
                            continue
                        if isinstance(frame, dict) and isinstance(frame.get("usage"), dict):
                            usage = frame["usage"]
        finally:
            await stream_ctx.__aexit__(None, None, None)
            # ``usage`` is populated only when the upstream's final SSE frame
            # actually carried a usage block (stream_options.include_usage,
            # set above) -- otherwise this is None exactly as before the fix.
            outcome = {"usage": usage} if usage is not None else None
            _record_usage_outcome(profile_name, "chat", tenant_id, started, outcome)

    return StreamingResponse(
        body_iterator(), status_code=upstream.status_code, media_type=media_type
    )


@router.post("/v1/embeddings")
async def embeddings(request: Request, tenant: TenantParam) -> Any:
    """OpenAI embeddings over an embedding deployment (llama-server --embedding).

    Body: ``model`` (deployment/profile) + ``input`` (string or list). Forwarded
    verbatim to the upstream's ``/embeddings`` with the served model id. The
    whole point is locality: vectors are computed on this node, so a retrieval
    pipeline never ships document text to a frontier embedding API.
    """
    try:
        body = await request.json()
    except ValueError:
        return _openai_error(
            "request body must be valid JSON",
            status_code=400,
            error_type="invalid_request_error",
        )
    if not isinstance(body, dict):
        return _openai_error(
            "request body must be a JSON object",
            status_code=400,
            error_type="invalid_request_error",
        )
    model = str(body.get("model") or "")
    if not model:
        return _openai_error(
            "missing required 'model' field (an embedding deployment)",
            status_code=400,
            error_type="invalid_request_error",
        )
    if body.get("input") in (None, "", []):
        return _openai_error(
            "missing required 'input' field",
            status_code=400,
            error_type="invalid_request_error",
        )
    resolved = await _resolve_or_error(model)
    if isinstance(resolved, JSONResponse):
        return resolved
    profile = resolved

    forward = dict(body)
    forward["model"] = profile.model
    url = f"{profile.base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {profile.api_key}",
        "Content-Type": "application/json",
    }
    client: httpx.AsyncClient = _client()
    started = time.perf_counter()
    result = await _post_upstream(
        client,
        url,
        headers,
        forward,
        profile,
        error_hint=" (is this deployment an embedding model, served with --embedding?)",
    )
    _record_usage_outcome(profile.name, "embed", tenant.tenant_id, started, result)
    if isinstance(result, JSONResponse):
        return result
    recency.stamp_served_profile(profile.name)  # PR-4 recency, see chat_completions
    return result


@router.post("/v1/rerank")
async def rerank(request: Request, tenant: TenantParam) -> Any:
    """Rerank documents against a query over a reranker deployment.

    Two families answer this surface with one wire contract: ``reranker`` (a
    GGUF, llama-server --reranking + --embedding --pooling rank — llama.cpp
    requires both flags together) and ``multi_vector`` (a safetensors ColBERT
    / PyLate checkpoint, the sentence-transformers MultiVectorEncoder runtime,
    MaxSim scoring). The proxy forwards identically to either.

    Body: ``model`` (deployment/profile), ``query``, ``documents`` (list of
    strings), optional ``top_n``. Forwarded verbatim to the upstream's
    ``/rerank`` with the served model id. Same locality point as embeddings:
    scoring happens on this node, so ranking never ships document text to a
    frontier reranking API.
    """
    try:
        body = await request.json()
    except ValueError:
        return _openai_error(
            "request body must be valid JSON",
            status_code=400,
            error_type="invalid_request_error",
        )
    if not isinstance(body, dict):
        return _openai_error(
            "request body must be a JSON object",
            status_code=400,
            error_type="invalid_request_error",
        )
    model = str(body.get("model") or "")
    if not model:
        return _openai_error(
            "missing required 'model' field (a reranker deployment)",
            status_code=400,
            error_type="invalid_request_error",
        )
    if not body.get("query"):
        return _openai_error(
            "missing required 'query' field",
            status_code=400,
            error_type="invalid_request_error",
        )
    documents = body.get("documents")
    if not isinstance(documents, list) or not documents:
        return _openai_error(
            "missing required 'documents' field (a non-empty list of strings)",
            status_code=400,
            error_type="invalid_request_error",
        )
    resolved = await _resolve_or_error(model)
    if isinstance(resolved, JSONResponse):
        return resolved
    profile = resolved

    forward = dict(body)
    forward["model"] = profile.model
    url = f"{profile.base_url}/rerank"
    headers = {
        "Authorization": f"Bearer {profile.api_key}",
        "Content-Type": "application/json",
    }
    client: httpx.AsyncClient = _client()
    started = time.perf_counter()
    result = await _post_upstream(
        client,
        url,
        headers,
        forward,
        profile,
        error_hint=" (is this deployment a reranker? family 'reranker' = llama-server "
        "--reranking --embedding --pooling rank; family 'multi_vector' = the "
        "sentence-transformers MultiVectorEncoder runtime)",
    )
    _record_usage_outcome(profile.name, "rerank", tenant.tenant_id, started, result)
    if isinstance(result, JSONResponse):
        return result
    recency.stamp_served_profile(profile.name)  # PR-4 recency, see chat_completions
    return result
