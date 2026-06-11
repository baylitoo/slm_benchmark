from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from docie_bench.llm.model_gateway import (
    CircuitOpenError,
    ModelCapabilityError,
    ModelGatewayError,
    ModelQueueFullError,
    reset_gateway_state,
)
from docie_bench.llm.model_profiles import ModelProfile, load_model_profiles
from docie_bench.llm.openai_client import OpenAICompatibleClient


@pytest.fixture(autouse=True)
def _reset_gateway() -> None:
    reset_gateway_state()


def _profile(**overrides: Any) -> ModelProfile:
    values: dict[str, Any] = {
        "name": "test",
        "model": "test-model",
        "base_url": "http://model.test/v1",
        "api_key": "test",
        "response_format_style": "json_object",
        "retry_max_attempts": 1,
        "retry_backoff_base_seconds": 0,
        "retry_backoff_max_seconds": 0,
        "circuit_breaker_failure_threshold": 5,
        "queue_timeout_seconds": 1,
    }
    values.update(overrides)
    return ModelProfile(**values)


def _completion(content: str = "{}") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


async def _client(
    profile: ModelProfile,
    handler: Callable[[httpx.Request], httpx.Response | Any],
) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(profile)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=profile.base_url,
        transport=httpx.MockTransport(handler),
    )
    return client


async def _chat(client: OpenAICompatibleClient) -> dict[str, Any]:
    result, _usage, _raw = await client.chat_json(
        system_prompt="system",
        user_prompt="user",
        schema_name="test",
        schema={"type": "object"},
    )
    return result


@pytest.mark.asyncio
async def test_capability_discovery_validates_profile_and_is_cached() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        assert request.url.path == "/v1/models"
        calls += 1
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "test-model",
                        "capabilities": {
                            "vision": True,
                            "response_format_styles": ["json_object"],
                        },
                    }
                ]
            },
        )

    client = await _client(_profile(capability_discovery="required", vision=True), handler)
    try:
        first = await client.discover_capabilities()
        second = await client.discover_capabilities()
    finally:
        await client.aclose()

    assert first == second
    assert first.vision is True
    assert first.response_format_styles == frozenset({"json_object"})
    assert calls == 1


@pytest.mark.asyncio
async def test_capability_discovery_rejects_missing_model_and_unsupported_format() -> None:
    responses = [
        {"data": [{"id": "other-model"}]},
        {
            "data": [
                {
                    "id": "test-model",
                    "response_format_styles": ["openai_json_schema"],
                }
            ]
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0))

    missing = await _client(_profile(base_url="http://missing.test/v1"), handler)
    unsupported = await _client(_profile(base_url="http://unsupported.test/v1"), handler)
    try:
        with pytest.raises(ModelCapabilityError, match="was not returned"):
            await missing.discover_capabilities()
        with pytest.raises(ModelCapabilityError, match="does not report support"):
            await unsupported.discover_capabilities()
    finally:
        await missing.aclose()
        await unsupported.aclose()


@pytest.mark.asyncio
async def test_optional_discovery_falls_back_but_required_discovery_fails() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(404, text="not implemented")
        return _completion()

    optional = await _client(_profile(capability_discovery="optional"), handler)
    required = await _client(
        _profile(
            base_url="http://required.test/v1",
            capability_discovery="required",
        ),
        handler,
    )
    try:
        assert await _chat(optional) == {}
        with pytest.raises(ModelGatewayError, match="404"):
            await _chat(required)
    finally:
        await optional.aclose()
        await required.aclose()

    assert paths == ["/v1/models", "/v1/chat/completions", "/v1/models"]


@pytest.mark.asyncio
async def test_transient_and_rate_limited_responses_retry_but_permanent_error_does_not() -> None:
    statuses = [429, 503, 200]
    calls = 0

    def retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        status = statuses.pop(0)
        if status == 200:
            return _completion('{"ok": true}')
        return httpx.Response(status, text="try later", headers={"Retry-After": "0"})

    retrying = await _client(_profile(retry_max_attempts=3), retry_handler)
    try:
        assert await _chat(retrying) == {"ok": True}
    finally:
        await retrying.aclose()
    assert calls == 3

    permanent_calls = 0

    def permanent_handler(request: httpx.Request) -> httpx.Response:
        nonlocal permanent_calls
        permanent_calls += 1
        return httpx.Response(400, text="bad request")

    permanent = await _client(
        _profile(base_url="http://permanent.test/v1", retry_max_attempts=3),
        permanent_handler,
    )
    try:
        with pytest.raises(ModelGatewayError, match="400"):
            await _chat(permanent)
    finally:
        await permanent.aclose()
    assert permanent_calls == 1


@pytest.mark.asyncio
async def test_invalid_model_response_is_retried() -> None:
    responses = [_completion("not-json"), _completion('{"ok": true}')]
    client = await _client(
        _profile(retry_max_attempts=2),
        lambda request: responses.pop(0),
    )
    try:
        assert await _chat(client) == {"ok": True}
    finally:
        await client.aclose()
    assert responses == []


@pytest.mark.asyncio
async def test_circuit_breaker_rejects_then_allows_recovery_probe() -> None:
    now = 0.0
    calls = 0
    failing = True

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="down") if failing else _completion('{"ok": true}')

    client = await _client(
        _profile(circuit_breaker_failure_threshold=2, circuit_breaker_reset_seconds=10),
        handler,
    )
    client._gateway._monotonic = lambda: now
    try:
        for _ in range(2):
            with pytest.raises(ModelGatewayError, match="503"):
                await _chat(client)
        with pytest.raises(CircuitOpenError):
            await _chat(client)
        assert calls == 2

        now = 11
        failing = False
        assert await _chat(client) == {"ok": True}
    finally:
        await client.aclose()
    assert calls == 3


@pytest.mark.asyncio
async def test_shared_per_model_concurrency_and_queue_limit() -> None:
    active = 0
    max_active = 0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        entered.set()
        await release.wait()
        active -= 1
        return _completion()

    profile = _profile(max_concurrency=1, queue_limit=1)
    first = await _client(profile, handler)
    second = await _client(profile, handler)
    third = await _client(profile, handler)
    first_task = asyncio.create_task(_chat(first))
    try:
        await entered.wait()
        second_task = asyncio.create_task(_chat(second))
        await asyncio.sleep(0)
        with pytest.raises(ModelQueueFullError):
            await _chat(third)
        release.set()
        await first_task
        await second_task
    finally:
        release.set()
        await first.aclose()
        await second.aclose()
        await third.aclose()
    assert max_active == 1


def test_model_profile_loads_gateway_controls(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
profiles:
  limited:
    model: test-model
    base_url: http://model.test/v1
    api_key: test
    capability_discovery: required
    retry_max_attempts: 4
    circuit_breaker_failure_threshold: 2
    max_concurrency: 1
    queue_limit: 3
""",
        encoding="utf-8",
    )

    profile = load_model_profiles(config)["limited"]

    assert profile.capability_discovery == "required"
    assert profile.retry_max_attempts == 4
    assert profile.circuit_breaker_failure_threshold == 2
    assert profile.max_concurrency == 1
    assert profile.queue_limit == 3
