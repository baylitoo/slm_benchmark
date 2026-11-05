"""Small OpenAI-compatible transport seam shared by agents and extraction.

Callers continue to own prompt shaping and response interpretation. This
module owns profile scheduling, capability preflight, HTTP error mapping,
bounded pre-output retries, and stream lifetime.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from docie_bench.llm.model_gateway import (
    ErrorClassification,
    InvalidModelResponseError,
    ModelGateway,
    ModelGatewayError,
    classify_response_error,
)
from docie_bench.llm.model_profiles import ModelProfile


def model_error_type(error: ModelGatewayError) -> str:
    """Stable public error codes shared by agent and extraction adapters."""
    return {
        ErrorClassification.AUTHENTICATION: "upstream_authentication_error",
        ErrorClassification.RATE_LIMITED: "upstream_rate_limited",
        ErrorClassification.CAPABILITY: "unsupported_capability",
        ErrorClassification.QUEUE_FULL: "model_queue_full",
        ErrorClassification.CIRCUIT_OPEN: "model_unavailable",
        ErrorClassification.TRANSIENT: "upstream_unavailable",
        ErrorClassification.TRANSPORT: "upstream_transport_error",
        ErrorClassification.INVALID_RESPONSE: "malformed_upstream_response",
    }.get(error.classification, "upstream_error")


def _required_capabilities(body: dict[str, Any]) -> set[str]:
    required: set[str] = set()
    if body.get("tools"):
        required.add("tools")
    if body.get("tool_choice") is not None:
        required.add("tool_choice")
    if isinstance(body.get("response_format"), dict):
        required.add("structured_output")
    if body.get("stream"):
        required.add("streaming")
    if isinstance(body.get("stream_options"), dict) and body["stream_options"].get(
        "include_usage"
    ):
        required.add("usage")
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list) and any(
                isinstance(part, dict) and part.get("type") == "image_url" for part in content
            ):
                required.add("vision")
                break
    return required


def _headers(profile: ModelProfile) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {profile.api_key}",
        "Content-Type": "application/json",
    }


def _generation_defaults(profile: ModelProfile, payload: dict[str, Any]) -> None:
    payload.setdefault("temperature", profile.temperature)
    payload.setdefault("top_p", profile.top_p)
    payload.setdefault("max_tokens", profile.max_tokens)
    if profile.stop_sequences:
        payload.setdefault("stop", list(profile.stop_sequences))


async def post_chat_completion(
    profile: ModelProfile,
    client: httpx.AsyncClient,
    body: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
    on_call: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one buffered model round with profile scheduling and safe retries."""
    payload = dict(body)
    payload["model"] = profile.model
    payload.pop("stream", None)
    _generation_defaults(profile, payload)
    gateway = ModelGateway(profile, client)
    await gateway.validate_capabilities(_required_capabilities(payload))
    url = f"{profile.base_url}/chat/completions"
    started = time.monotonic()
    outcome = "error"
    result: dict[str, Any] | None = None
    try:
        async def request() -> dict[str, Any]:
            return await gateway.request_json(
                "POST",
                url,
                json=payload,
                headers=_headers(profile),
                timeout=timeout_seconds or profile.timeout_seconds,
            )

        if timeout_seconds is None:
            result = await request()
        else:
            async with asyncio.timeout(timeout_seconds):
                result = await request()
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise InvalidModelResponseError("Model endpoint returned no chat completion choice")
        if not isinstance(choices[0].get("message"), dict):
            raise InvalidModelResponseError("Model endpoint returned a malformed chat completion")
        outcome = "success"
        return result
    except TimeoutError as exc:
        raise ModelGatewayError(
            "Model call exceeded the workflow deadline", status_code=504
        ) from exc
    finally:
        if on_call is not None:
            on_call(
                {
                    "profile": profile.name,
                    "model": profile.model,
                    "execution": "local" if profile.execution_location == "local" else "remote",
                    "configured_location": profile.execution_location,
                    "attempts": gateway.last_attempts,
                    "queue_wait_ms": round(gateway.last_wait_seconds * 1000),
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                    "outcome": outcome,
                    "usage": result.get("usage") if isinstance(result, dict) else None,
                }
            )


async def post_upstream_response(
    profile: ModelProfile,
    client: httpx.AsyncClient,
    body: dict[str, Any],
    *,
    allowed_statuses: frozenset[int] = frozenset(),
    timeout_seconds: float | None = None,
) -> httpx.Response:
    """Return a raw response for specialized adapters, under gateway policy.

    Adapters may allow selected permanent statuses when they implement a
    specific response-format fallback. All other errors use the shared
    classification and retry policy.
    """
    payload = dict(body)
    payload["model"] = profile.model
    _generation_defaults(profile, payload)
    gateway = ModelGateway(profile, client)
    await gateway.validate_capabilities(_required_capabilities(payload))

    async def operation() -> httpx.Response:
        response = await client.post(
            f"{profile.base_url}/chat/completions",
            json=payload,
            headers=_headers(profile),
            timeout=timeout_seconds or profile.timeout_seconds,
        )
        if response.status_code >= 400 and response.status_code not in allowed_statuses:
            raise classify_response_error(response)
        return response

    return await gateway.execute(operation)


async def open_chat_stream(
    profile: ModelProfile,
    client: httpx.AsyncClient,
    body: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
    on_call: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, str, AsyncIterator[bytes]]:
    """Open a scheduled stream, retrying only before any output can escape."""
    payload = dict(body)
    payload["model"] = profile.model
    payload["stream"] = True
    _generation_defaults(profile, payload)
    required = _required_capabilities(payload)
    stream_options = dict(payload.get("stream_options") or {})
    stream_options.setdefault("include_usage", True)
    payload["stream_options"] = stream_options
    gateway = ModelGateway(profile, client)
    await gateway.validate_capabilities(required | {"streaming"})
    lease = gateway.slot()
    lease_entered = False
    stream_context: Any = None
    stream_entered = False
    response: httpx.Response | None = None
    started = time.monotonic()
    trace = {
        "profile": profile.name,
        "model": profile.model,
        "execution": "local" if profile.execution_location == "local" else "remote",
        "configured_location": profile.execution_location,
        "attempts": 0,
        "queue_wait_ms": round(gateway.last_wait_seconds * 1000),
        "elapsed_ms": 0,
        "outcome": "error",
        "usage": None,
    }
    try:
        try:
            if timeout_seconds is None:
                await lease.__aenter__()
            else:
                async with asyncio.timeout(timeout_seconds):
                    await lease.__aenter__()
            lease_entered = True
        except TimeoutError as exc:
            raise ModelGatewayError(
                "Model call exceeded the workflow deadline", status_code=504
            ) from exc
        trace["queue_wait_ms"] = round(gateway.last_wait_seconds * 1000)
        for attempt in range(1, profile.retry_max_attempts + 1):
            trace["attempts"] = attempt
            stream_context = client.stream(
                "POST",
                f"{profile.base_url}/chat/completions",
                json=payload,
                headers=_headers(profile),
                timeout=timeout_seconds or profile.timeout_seconds,
            )
            try:
                response = await stream_context.__aenter__()
                stream_entered = True
            except httpx.RequestError as exc:
                error = gateway._classify_exception(exc)
            else:
                if response.status_code < 400:
                    break
                await response.aread()
                error = classify_response_error(response)
                await stream_context.__aexit__(None, None, None)
                stream_context = None
                stream_entered = False
            gateway._record_failure(error)
            if not error.retryable or attempt == profile.retry_max_attempts:
                raise error
            await gateway._sleep(gateway._retry_delay(error, attempt))
        else:  # pragma: no cover - loop always breaks or raises
            raise AssertionError("stream retry loop exited unexpectedly")
    except BaseException:
        if stream_context is not None and stream_entered:
            await stream_context.__aexit__(None, None, None)
        if lease_entered:
            await lease.__aexit__(None, None, None)
        trace["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        if on_call is not None:
            on_call(trace)
        raise

    assert response is not None
    assert lease_entered
    media_type = response.headers.get("content-type", "text/event-stream")

    async def iterator() -> AsyncIterator[bytes]:
        usage_buffer = bytearray()

        def scan_usage(chunk: bytes) -> None:
            usage_buffer.extend(chunk)
            while b"\n" in usage_buffer:
                line, _, remainder = usage_buffer.partition(b"\n")
                usage_buffer[:] = remainder
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == b"[DONE]":
                    continue
                try:
                    frame = json.loads(payload)
                except (TypeError, ValueError):
                    continue
                if isinstance(frame, dict) and isinstance(frame.get("usage"), dict):
                    trace["usage"] = frame["usage"]
            if len(usage_buffer) > 64 * 1024:
                usage_buffer.clear()

        try:
            if timeout_seconds is None:
                async for chunk in response.aiter_raw():
                    scan_usage(chunk)
                    yield chunk
            else:
                async with asyncio.timeout(timeout_seconds):
                    async for chunk in response.aiter_raw():
                        scan_usage(chunk)
                        yield chunk
            gateway._record_success()
            trace["outcome"] = "success"
        except TimeoutError:
            trace["outcome"] = "deadline_exceeded"
            raise
        except asyncio.CancelledError:
            trace["outcome"] = "cancelled"
            raise
        except Exception:
            trace["outcome"] = "stream_error"
            raise
        finally:
            trace["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            if on_call is not None:
                on_call(trace)
            await stream_context.__aexit__(None, None, None)
            await lease.__aexit__(None, None, None)

    return response.status_code, media_type, iterator()
