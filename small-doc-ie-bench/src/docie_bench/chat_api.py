"""Generic OpenAI-compatible chat + embeddings over the serving stack — the
Playground's "classic queries" / "embed" paths, and platform surfaces in
their own right.

``POST /v1/chat/completions`` routes the request's ``model`` through the same
resolver as everything else (live deployment name, models.yaml profile,
``store:<name>``) and forwards to that upstream. ``POST /v1/embeddings`` does
the same for an embedding deployment (llama-server ``--embedding``), so a
RAG/retrieval pipeline computes document vectors on THIS node — they never
leave the infra. ``GET /v1/models`` advertises the routable ids. Unlike the
standalone ``docie gateway`` (its own process, models.yaml only), this rides
the main API: live deployments work out of the box and the agents'
bearer-friendly tenant guard applies.

Kept deliberately small: non-streaming forward (a ``stream: true`` request is
answered as a single SSE chunk, same convention as the agents surface). The
full streaming proxy remains the gateway's job.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request

from docie_bench.agents.api import (
    _client,
    _openai_error,
    _single_chunk_sse,
    agents_tenant_guard,
)
from docie_bench.serving.profile_resolver import (
    ProfileResolutionError,
    resolve_extraction_profile,
)

router = APIRouter(tags=["chat"], dependencies=[Depends(agents_tenant_guard)])


@router.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Routable model ids: models.yaml profiles + live deployments."""
    from docie_bench.serving.profile_resolver import (
        DEFAULT_MODELS_CONFIG,
        _default_live_deployments,
        _is_live,
        build_profile_table,
    )
    from docie_bench.llm.model_profiles import load_model_profiles

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
    try:
        profile = resolve_extraction_profile(model_profile=model)
    except ProfileResolutionError as exc:
        return _openai_error(str(exc), status_code=404, error_type="model_not_found")
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
    forward.pop("stream", None)
    url = f"{profile.base_url}/chat/completions"
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
    if wants_stream:
        return _single_chunk_sse(completion)
    return completion


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
    try:
        profile = resolve_extraction_profile(model_profile=model)
    except ProfileResolutionError as exc:
        return _openai_error(str(exc), status_code=404, error_type="model_not_found")

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
