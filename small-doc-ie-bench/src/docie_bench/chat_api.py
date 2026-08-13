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
from docie_bench.serving.placement_resolver import (
    STORE_PROFILE_PREFIX,
    PlacementNotFoundError,
    PlacementNotReadyError,
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
    """
    try:
        return resolve_extraction_profile(model_profile=model)
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
    forward = dict(body)
    forward["model"] = profile.model
    url = f"{profile.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {profile.api_key}",
        "Content-Type": "application/json",
    }
    client: httpx.AsyncClient = _client()

    if wants_stream:
        return await _stream_chat_completions(
            client, url, headers, forward, profile.timeout_seconds
        )

    forward.pop("stream", None)
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
    # Repair model-emitted UTF-8 mojibake in the answer (accented OCR/description
    # on small vision models — the Playground Vision path lands here).
    if get_settings().fix_mojibake:
        completion = fix_completion_content(completion)
    return completion


async def _stream_chat_completions(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    forward: dict[str, Any],
    timeout: float,
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
        return upstream.json()
    except ValueError:
        return _openai_error(
            "upstream returned a non-JSON response",
            status_code=502,
            error_type="upstream_error",
        )


@router.post("/v1/rerank")
async def rerank(request: Request) -> Any:
    """Rerank documents against a query over a reranker deployment (llama-server
    --reranking + --embedding --pooling rank — llama.cpp requires both flags
    together; --reranking alone leaves the endpoint unable to score).

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
            "(is this deployment a reranker, served with --reranking --embedding --pooling rank?)",
            status_code=upstream.status_code,
            error_type="upstream_error",
        )
    try:
        return upstream.json()
    except ValueError:
        return _openai_error(
            "upstream returned a non-JSON response",
            status_code=502,
            error_type="upstream_error",
        )
