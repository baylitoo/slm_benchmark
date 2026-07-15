"""Agents endpoints: registry CRUD for the Studio tab + the OpenAI surface.

Two audiences, one router:

* The **Studio Agents tab** manages configurations — ``GET/POST /v1/agents``,
  ``GET /v1/agents/templates``, ``PUT/DELETE /v1/agents/{name}``.
* **Agents platforms** consume agents as OpenAI models. Point an OpenAI client
  at ``base_url = <api>/v1/agents`` (``/models`` lists every enabled agent,
  ``/chat/completions`` routes by the ``model`` field) or at the per-agent
  ``base_url = <api>/v1/agents/{name}`` for a single-agent credential scope.

Auth: the same tenant guard as every privileged router, but accepting the key
from ``Authorization: Bearer …`` as well as ``x-api-key`` — OpenAI SDKs send
the former by default, so an external platform needs no header customization.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from docie_bench.agents.registry import (
    AgentConflictError,
    AgentNotFoundError,
    AgentRegistry,
    AgentRegistryError,
)
from docie_bench.agents.runtime import AgentError, complete_agent
from docie_bench.agents.spec import AgentSpec
from docie_bench.agents.templates import AGENT_TEMPLATES, template_by_id
from docie_bench.security import TenantContext, get_quota_manager


async def agents_tenant_guard(
    x_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> AsyncIterator[TenantContext]:
    """tenant_guard, plus ``Authorization: Bearer`` for stock OpenAI clients."""
    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    manager = get_quota_manager()
    context = manager.authenticate(key)
    manager.acquire(context)
    try:
        yield context
    finally:
        manager.release(context)


router = APIRouter(
    prefix="/v1/agents", tags=["agents"], dependencies=[Depends(agents_tenant_guard)]
)

# One pooled client for all upstream forwards (module-level because the router
# outlives any single request). Tests inject a MockTransport via
# `configure_http_transport`.
_http_client: httpx.AsyncClient | None = None
_transport_override: httpx.AsyncBaseTransport | None = None


def _client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(transport=_transport_override)
    return _http_client


def configure_http_transport(transport: httpx.AsyncBaseTransport | None) -> None:
    """Test seam: route upstream forwards through ``transport`` (None resets)."""
    global _http_client, _transport_override
    _transport_override = transport
    _http_client = None


def _registry() -> AgentRegistry:
    # Fresh per request — reads the shared agents.json (see registry docstring).
    return AgentRegistry()


def _spec_view(spec: AgentSpec) -> dict[str, Any]:
    view = spec.model_dump(mode="json")
    # The addressable OpenAI-compatible base path for this agent (the UI
    # prepends the public API origin).
    view["endpoint"] = f"/v1/agents/{spec.name}"
    return view


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, AgentNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, AgentConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, AgentRegistryError):
        return HTTPException(status_code=500, detail=str(exc))
    raise exc


# ---------------------------------------------------------------------------
# Registry CRUD (the Studio tab). Literal paths are declared before /{name}.
# ---------------------------------------------------------------------------


class CreateAgentRequest(BaseModel):
    name: str
    template: str | None = None
    kind: str | None = None
    display_name: str | None = None
    description: str | None = None
    model_profile: str | None = None
    system_prompt: str | None = None
    options: dict[str, Any] | None = None
    enabled: bool = True


class UpdateAgentRequest(BaseModel):
    display_name: str | None = None
    description: str | None = None
    model_profile: str | None = None
    system_prompt: str | None = None
    options: dict[str, Any] | None = None
    enabled: bool | None = None


@router.get("")
async def list_agents() -> list[dict[str, Any]]:
    try:
        return [_spec_view(spec) for spec in _registry().list()]
    except AgentRegistryError as exc:
        raise _http_error(exc) from exc


@router.post("", status_code=201)
async def create_agent(payload: CreateAgentRequest) -> dict[str, Any]:
    template = template_by_id(payload.template) if payload.template else None
    if payload.template and template is None:
        raise HTTPException(status_code=400, detail=f"unknown template {payload.template!r}")

    kind = payload.kind or (template["kind"] if template else None)
    if not kind:
        raise HTTPException(status_code=400, detail="either 'template' or 'kind' is required")

    defaults: dict[str, Any] = dict(template["defaults"]) if template else {}
    options = dict(defaults.get("options") or {})
    if payload.options:
        options.update(payload.options)
    system_prompt = (
        payload.system_prompt
        if payload.system_prompt is not None
        else defaults.get("system_prompt")
    )

    try:
        spec = AgentSpec(
            name=payload.name,
            kind=kind,  # type: ignore[arg-type] — pydantic re-validates the literal
            display_name=payload.display_name
            or (template["display_name"] if template else payload.name),
            description=payload.description or (template["description"] if template else ""),
            model_profile=payload.model_profile,
            system_prompt=system_prompt,
            options=options,
            enabled=payload.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return _spec_view(_registry().create(spec))
    except AgentRegistryError as exc:
        raise _http_error(exc) from exc


@router.get("/templates")
async def list_templates() -> list[dict[str, Any]]:
    return AGENT_TEMPLATES


# ---------------------------------------------------------------------------
# OpenAI surface — platform-wide (base_url = <api>/v1/agents).
# ---------------------------------------------------------------------------


def _openai_error(message: str, *, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": error_type}},
    )


def _openai_model(spec: AgentSpec) -> dict[str, Any]:
    return {"id": spec.name, "object": "model", "created": 0, "owned_by": "docie-agents"}


@router.get("/models")
async def list_agent_models() -> dict[str, Any]:
    try:
        specs = _registry().list()
    except AgentRegistryError as exc:
        raise _http_error(exc) from exc
    return {
        "object": "list",
        "data": [_openai_model(spec) for spec in specs if spec.enabled],
    }


async def _serve_completion(spec: AgentSpec, request: Request) -> Any:
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
    wants_stream = bool(body.get("stream"))
    try:
        completion = await complete_agent(spec, body, http_client=_client())
    except AgentError as exc:
        return _openai_error(exc.message, status_code=exc.status_code, error_type=exc.error_type)
    if wants_stream:
        return _single_chunk_sse(completion)
    return JSONResponse(completion)


@router.post("/chat/completions")
async def agents_chat_completions(request: Request) -> Any:
    # Peek at the model field for routing; _serve_completion re-reads the body
    # (Starlette caches it, so the double read is free).
    try:
        body = await request.json()
    except ValueError:
        body = None
    model = body.get("model", "") if isinstance(body, dict) else ""
    if not model:
        return _openai_error(
            "missing required 'model' field (an agent name)",
            status_code=400,
            error_type="invalid_request_error",
        )
    try:
        spec = _registry().get(str(model))
    except AgentNotFoundError:
        return _openai_error(
            f"model {model!r} is not a configured agent",
            status_code=404,
            error_type="model_not_found",
        )
    except AgentRegistryError as exc:
        raise _http_error(exc) from exc
    return await _serve_completion(spec, request)


# ---------------------------------------------------------------------------
# Per-agent routes (base_url = <api>/v1/agents/{name}) + CRUD by name.
# ---------------------------------------------------------------------------


@router.get("/{name}")
async def get_agent(name: str) -> dict[str, Any]:
    try:
        return _spec_view(_registry().get(name))
    except AgentRegistryError as exc:
        raise _http_error(exc) from exc


@router.put("/{name}")
async def update_agent(name: str, payload: UpdateAgentRequest) -> dict[str, Any]:
    patch = payload.model_dump(exclude_unset=True)
    try:
        return _spec_view(_registry().update(name, patch))
    except AgentRegistryError as exc:
        raise _http_error(exc) from exc


@router.delete("/{name}")
async def delete_agent(name: str) -> dict[str, Any]:
    try:
        _registry().delete(name)
    except AgentRegistryError as exc:
        raise _http_error(exc) from exc
    return {"deleted": name}


@router.get("/{name}/models")
async def agent_models(name: str) -> dict[str, Any]:
    """Single-agent OpenAI model list, so `base_url=<api>/v1/agents/{name}` works."""
    try:
        spec = _registry().get(name)
    except AgentRegistryError as exc:
        raise _http_error(exc) from exc
    return {"object": "list", "data": [_openai_model(spec)] if spec.enabled else []}


@router.post("/{name}/chat/completions")
async def agent_chat_completions(name: str, request: Request) -> Any:
    try:
        spec = _registry().get(name)
    except AgentNotFoundError:
        return _openai_error(
            f"model {name!r} is not a configured agent",
            status_code=404,
            error_type="model_not_found",
        )
    except AgentRegistryError as exc:
        raise _http_error(exc) from exc
    return await _serve_completion(spec, request)


def _single_chunk_sse(completion: dict[str, Any]) -> StreamingResponse:
    """Emit a finished completion as one OpenAI SSE chunk (stream clients work;
    agents buffer anyway because of the post-processing pass)."""
    choices = completion.get("choices") or [{}]
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    chunk = {
        "id": completion.get("id", "chatcmpl-agent"),
        "object": "chat.completion.chunk",
        "created": completion.get("created", 0),
        "model": completion.get("model", ""),
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": message.get("content", "")},
                "finish_reason": first.get("finish_reason", "stop"),
            }
        ],
    }

    async def body_iterator() -> AsyncIterator[bytes]:
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(body_iterator(), media_type="text/event-stream")
