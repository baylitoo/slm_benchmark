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

from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from docie_bench.agents.api import (
    _client,
    _openai_error,
    agents_tenant_guard,
)
from docie_bench.inngest.serving_api import trigger_deployment_load
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.mojibake import fix_completion_content
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

router = APIRouter(tags=["chat"], dependencies=[Depends(agents_tenant_guard)])


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


async def _resolve_or_error(model: str) -> ModelProfile | JSONResponse:
    """Resolve ``model`` the same way every route here needs to, folding in
    load-on-demand for a store model that isn't live yet.

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
        profile = resolve_extraction_profile(model_profile=model)
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


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
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
            "missing required 'model' field (a deployment name or profile)",
            status_code=400,
            error_type="invalid_request_error",
        )
    resolved = await _resolve_or_error(model)
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

    wants_stream = bool(body.get("stream"))
    mcp_server_names = body.get("mcp_servers")
    if mcp_server_names is not None and (
        not isinstance(mcp_server_names, list)
        or not all(isinstance(name, str) and name for name in mcp_server_names)
    ):
        return _openai_error(
            "'mcp_servers' must be a list of registered MCP server names",
            status_code=400,
            error_type="invalid_request_error",
        )
    forward = dict(body)
    forward["model"] = profile.model
    url = f"{profile.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {profile.api_key}",
        "Content-Type": "application/json",
    }
    client: httpx.AsyncClient = _client()

    if mcp_server_names:
        # The serving side drives the whole model<->tools exchange, so the
        # response the caller gets is the FINAL answer — a token stream of
        # intermediate tool_calls rounds has no meaningful SSE shape here.
        if wants_stream:
            return _openai_error(
                "'mcp_servers' does not support 'stream': the server runs the "
                "tool exchange and returns the final completion",
                status_code=400,
                error_type="invalid_request_error",
            )
        return await _chat_with_mcp_tools(
            client, url, headers, forward, profile, [str(n) for n in mcp_server_names]
        )

    if wants_stream:
        return await _stream_chat_completions(
            client, url, headers, forward, profile.timeout_seconds, profile.name
        )

    completion = await _post_upstream(client, url, headers, forward, profile)
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


async def _post_upstream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    forward: dict[str, Any],
    profile: ModelProfile,
) -> dict[str, Any] | JSONResponse:
    """One non-streaming upstream completion, errors mapped to OpenAI shape.

    Shared by the plain chat path and every round of the MCP tool loop —
    same forwarding, same error taxonomy, regardless of which path posts.
    """
    body = dict(forward)
    body.pop("stream", None)
    body.pop("mcp_servers", None)
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
            f"upstream returned {upstream.status_code}: {upstream.text[:300]}",
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


async def _chat_with_mcp_tools(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    forward: dict[str, Any],
    profile: ModelProfile,
    server_names: list[str],
) -> Any:
    """Serve one chat request with MCP tools: connect, advertise, loop.

    Registry-only security: every requested name must exist in
    ``settings.mcp_servers_config`` — the request can never point the server
    at an arbitrary URL or command. The ``mcp`` SDK being absent is a clean
    501 (the extra is optional), not an ImportError.
    """
    from contextlib import AsyncExitStack

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

    async def post(body: dict[str, Any]) -> dict[str, Any] | JSONResponse:
        return await _post_upstream(client, url, headers, body, profile)

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
            completion = await mcp_mod.run_tool_loop(post, forward, sessions, mapping, tools)
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
    return completion


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
) -> Any:
    """Real token-by-token proxy: forward raw SSE bytes as the upstream emits
    them, instead of buffering the full completion first. Ported from the
    gateway's ``_forward_stream`` — same shape, chat-only entry point.

    No mojibake repair here (unlike the non-streaming path): a fix-up needs
    the full decoded string, and a multi-byte UTF-8 sequence can straddle a
    chunk boundary. The gateway's proven streaming path skips it too.
    """
    forward["stream"] = True
    stream_ctx = client.stream("POST", url, json=forward, headers=headers, timeout=timeout)
    try:
        upstream = await stream_ctx.__aenter__()
    except httpx.RequestError as exc:
        return _openai_error(
            f"upstream {url} is unreachable: {exc}",
            status_code=502,
            error_type="upstream_unavailable",
        )
    media_type = upstream.headers.get("content-type", "text/event-stream")
    if upstream.status_code >= 400:
        body_bytes = await upstream.aread()
        await stream_ctx.__aexit__(None, None, None)
        detail = body_bytes[:300].decode("utf-8", "replace")
        return _openai_error(
            f"upstream returned {upstream.status_code}: {detail}",
            status_code=upstream.status_code,
            error_type="upstream_error",
        )
    # PR-4 recency, see the non-streaming path's comment above — a stream
    # that gets this far has an accepted upstream connection, which counts
    # as this deployment serving traffic.
    recency.stamp_served_profile(profile_name)

    async def body_iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await stream_ctx.__aexit__(None, None, None)

    return StreamingResponse(
        body_iterator(), status_code=upstream.status_code, media_type=media_type
    )


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> Any:
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
    try:
        upstream = await client.post(
            url, json=forward, headers=headers, timeout=profile.timeout_seconds
        )
    except httpx.RequestError as exc:
        return _openai_error(
            f"upstream {profile.base_url} is unreachable: {exc}",
            status_code=502,
            error_type="upstream_unavailable",
        )
    if upstream.status_code >= 400:
        return _openai_error(
            f"upstream returned {upstream.status_code}: {upstream.text[:300]} "
            "(is this deployment an embedding model, served with --embedding?)",
            status_code=upstream.status_code,
            error_type="upstream_error",
        )
    try:
        result = upstream.json()
    except ValueError:
        return _openai_error(
            "upstream returned a non-JSON response",
            status_code=502,
            error_type="upstream_error",
        )
    recency.stamp_served_profile(profile.name)  # PR-4 recency, see chat_completions
    return result


@router.post("/v1/rerank")
async def rerank(request: Request) -> Any:
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
    try:
        upstream = await client.post(
            url, json=forward, headers=headers, timeout=profile.timeout_seconds
        )
    except httpx.RequestError as exc:
        return _openai_error(
            f"upstream {profile.base_url} is unreachable: {exc}",
            status_code=502,
            error_type="upstream_unavailable",
        )
    if upstream.status_code >= 400:
        return _openai_error(
            f"upstream returned {upstream.status_code}: {upstream.text[:300]} "
            "(is this deployment a reranker? family 'reranker' = llama-server "
            "--reranking --embedding --pooling rank; family 'multi_vector' = the "
            "sentence-transformers MultiVectorEncoder runtime)",
            status_code=upstream.status_code,
            error_type="upstream_error",
        )
    try:
        result = upstream.json()
    except ValueError:
        return _openai_error(
            "upstream returned a non-JSON response",
            status_code=502,
            error_type="upstream_error",
        )
    recency.stamp_served_profile(profile.name)  # PR-4 recency, see chat_completions
    return result
