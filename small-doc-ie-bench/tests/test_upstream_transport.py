from __future__ import annotations

import httpx
import pytest

from docie_bench.llm.model_gateway import ModelCapabilityError, reset_gateway_state
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.upstream_transport import open_chat_stream, post_chat_completion


def _profile(**overrides: object) -> ModelProfile:
    values: dict[str, object] = {
        "name": "transport-test",
        "model": "test-model",
        "base_url": "http://upstream.test/v1",
        "api_key": "server-secret",
        "retry_max_attempts": 2,
        "retry_backoff_base_seconds": 0,
        "retry_backoff_max_seconds": 0,
    }
    values.update(overrides)
    return ModelProfile(**values)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _clear_gateway() -> None:
    reset_gateway_state()


@pytest.mark.asyncio
async def test_profile_generation_settings_auth_and_retry_after_are_shared() -> None:
    seen: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await post_chat_completion(
            _profile(temperature=0.2, top_p=0.8, max_tokens=77, stop_sequences=("END",)),
            client,
            {"messages": [{"role": "user", "content": "hello"}]},
        )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert len(seen) == 2
    assert seen[1].headers["Authorization"] == "Bearer server-secret"
    import json

    payload = json.loads(seen[1].content)
    assert (payload["temperature"], payload["top_p"], payload["max_tokens"]) == (0.2, 0.8, 77)
    assert payload["stop"] == ["END"]


@pytest.mark.asyncio
async def test_known_unsupported_tools_fail_before_the_upstream_call() -> None:
    calls = 0

    async def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ModelCapabilityError):
            await post_chat_completion(
                _profile(capabilities={"tools": False}),
                client,
                {"messages": [], "tools": [{"type": "function", "function": {"name": "x"}}]},
            )
    assert calls == 0


@pytest.mark.asyncio
async def test_stream_failure_after_first_chunk_is_never_replayed() -> None:
    calls = 0

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: first\n\n"
            raise httpx.ReadError("connection lost")

    async def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=BrokenStream(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        _, _, chunks = await open_chat_stream(
            _profile(), client, {"messages": [{"role": "user", "content": "hello"}]}
        )
        assert await anext(chunks) == b"data: first\n\n"
        with pytest.raises(httpx.ReadError):
            await anext(chunks)
    assert calls == 1
