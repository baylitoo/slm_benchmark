"""Unified OpenAI-compatible gateway — one `/v1` endpoint fronting every runtime.

Every runtime already speaks OpenAI (Ollama :11434, llama-server :8088, vLLM
:8000, remote). This gateway puts a single OpenAI-compatible endpoint in front of
all of them so a client points `base_url` at ONE URL and stops caring which
runtime serves which model. `configs/models.yaml` is the routing table.

Routing (`resolve_profile`): the request's `model` is matched first against a
profile *name* (what `/v1/models` advertises), then — so the benchmark, which
sends the upstream id like ``qwen2.5:1.5b``, can repoint `base_url` here without
code changes — against a unique profile `model`. Profiles that share an upstream
`model` on the *same* `base_url` (e.g. ``nuextract3`` / ``nuextract3_think``)
forward identically; a clash across *different* base_urls is rejected as
ambiguous rather than guessed. After resolving, the upstream `model` is always
substituted before forwarding.

Scope is deliberately small: `/v1/chat/completions` (streaming and not),
`/v1/models`, `/healthz`. Passthrough only — no solution adapters, no
circuit-breaker. Those layer on top later.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from docie_bench.llm.model_profiles import ModelProfile, load_model_profiles
from docie_bench.serving.solutions import SolutionError, build_solution

DEFAULT_MODELS_CONFIG = Path("configs/models.yaml")


class GatewayRoutingError(Exception):
    """A requested model could not be routed to exactly one upstream."""

    def __init__(self, message: str, *, status_code: int, error_type: str) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


def resolve_profile(model: str, profiles: dict[str, ModelProfile]) -> ModelProfile:
    """Map a requested model to exactly one profile (name first, then upstream id)."""
    if not model:
        raise GatewayRoutingError(
            "missing required 'model' field", status_code=400, error_type="invalid_request_error"
        )
    if model in profiles:
        return profiles[model]
    matches = [profile for profile in profiles.values() if profile.model == model]
    if not matches:
        raise GatewayRoutingError(
            f"model {model!r} is not a known profile name or upstream model id",
            status_code=404,
            error_type="model_not_found",
        )
    base_urls = {profile.base_url for profile in matches}
    if len(base_urls) > 1:
        names = ", ".join(sorted(profile.name for profile in matches))
        raise GatewayRoutingError(
            f"model {model!r} is ambiguous across profiles with different base_urls "
            f"({names}); request it by profile name instead",
            status_code=409,
            error_type="ambiguous_model",
        )
    return matches[0]


def _openai_error(message: str, *, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": error_type}},
    )


def create_gateway_app(
    models_config_path: str | Path | None = None,
    *,
    profiles: dict[str, ModelProfile] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the gateway app.

    `profiles` (or `models_config_path`) is the routing table. `transport` is an
    injection seam for tests (e.g. ``httpx.MockTransport``) — production passes
    nothing and a default networked client is used.
    """
    if profiles is None:
        profiles = load_model_profiles(models_config_path or DEFAULT_MODELS_CONFIG)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One shared client; absolute upstream URLs are passed per request so a
        # single pool serves every base_url. Auth is per-request (profiles can
        # share a base_url with different keys).
        client = httpx.AsyncClient(transport=transport)
        app.state.client = client
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(
        title="docie gateway",
        summary="Unified OpenAI-compatible endpoint over all serving runtimes.",
        lifespan=lifespan,
    )
    app.state.profiles = profiles

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"status": "ok", "profiles": len(app.state.profiles)}

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        # Advertise profile names as model ids (the unambiguous routing key).
        data = [
            {"id": name, "object": "model", "created": 0, "owned_by": "docie"}
            for name in sorted(app.state.profiles)
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
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

        try:
            profile = resolve_profile(body.get("model", ""), app.state.profiles)
        except GatewayRoutingError as exc:
            return _openai_error(
                exc.message, status_code=exc.status_code, error_type=exc.error_type
            )

        # Non-passthrough profiles are served by a local solution adapter
        # (OCR engine, pipeline, …) rather than proxied to an upstream.
        if profile.kind != "passthrough":
            return await _dispatch_solution(
                profile, body, profiles=app.state.profiles, http_client=app.state.client
            )

        # Always forward the upstream model id, regardless of how it was matched.
        body["model"] = profile.model
        url = f"{profile.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {profile.api_key}",
            "Content-Type": "application/json",
        }
        client: httpx.AsyncClient = app.state.client
        timeout = profile.timeout_seconds

        if body.get("stream"):
            return await _forward_stream(client, url, headers, body, timeout)
        try:
            upstream = await client.post(url, json=body, headers=headers, timeout=timeout)
        except httpx.RequestError as exc:
            return _openai_error(
                f"upstream {profile.base_url} is unreachable: {exc}",
                status_code=502,
                error_type="upstream_unavailable",
            )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    return app


async def _forward_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, object],
    timeout: float,
) -> Response:
    """Proxy a streaming completion, reflecting upstream status and SSE framing.

    The stream context is entered manually so the upstream status can be
    inspected (a pre-stream error becomes a normal Response) and so the context
    is always closed via the generator's ``finally``.
    """
    stream = client.stream("POST", url, json=body, headers=headers, timeout=timeout)
    try:
        upstream = await stream.__aenter__()
    except httpx.RequestError as exc:
        return _openai_error(
            f"upstream is unreachable: {exc}",
            status_code=502,
            error_type="upstream_unavailable",
        )

    media_type = upstream.headers.get("content-type", "text/event-stream")
    if upstream.status_code >= 400:
        payload = await upstream.aread()
        await stream.__aexit__(None, None, None)
        return Response(content=payload, status_code=upstream.status_code, media_type=media_type)

    async def body_iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await stream.__aexit__(None, None, None)

    return StreamingResponse(
        body_iterator(), status_code=upstream.status_code, media_type=media_type
    )


async def _dispatch_solution(
    profile: ModelProfile,
    body: dict[str, object],
    *,
    profiles: Mapping[str, ModelProfile],
    http_client: httpx.AsyncClient,
) -> Response:
    """Serve a non-passthrough profile via its local adapter, OpenAI-shaped."""
    try:
        solution = build_solution(profile, profiles=profiles, http_client=http_client)
        completion = await solution.complete(body)
    except SolutionError as exc:
        return _openai_error(exc.message, status_code=exc.status_code, error_type=exc.error_type)
    if body.get("stream"):
        return _solution_sse(completion)
    return JSONResponse(completion)


def _solution_sse(completion: dict[str, object]) -> StreamingResponse:
    """Emit a single completion as an OpenAI SSE stream so stream clients work."""
    choice = completion["choices"][0]  # type: ignore[index]
    content = choice["message"]["content"]  # type: ignore[index]
    chunk = {
        "id": completion.get("id", "chatcmpl-solution"),
        "object": "chat.completion.chunk",
        "created": completion.get("created", 0),
        "model": completion.get("model", ""),
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }

    async def body_iterator() -> AsyncIterator[bytes]:
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(body_iterator(), media_type="text/event-stream")
