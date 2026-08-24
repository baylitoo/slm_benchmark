"""Generic OpenAI chat surface (POST /v1/chat/completions on the main API)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docie_bench.agents.api import configure_http_transport
from docie_bench.chat_api import router as chat_router
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.profile_resolver import ProfileResolutionError

UPSTREAM = ModelProfile(
    name="lfm2.5-350m", model="lfm2.5-350m-served", base_url="http://upstream/v1", api_key="k"
)


@pytest.fixture()
def api(monkeypatch) -> tuple[TestClient, list[httpx.Request]]:
    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "lfm2.5-350m":
            return UPSTREAM
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)

    captured: list[httpx.Request] = []

    async def _sse_chunks() -> AsyncIterator[bytes]:
        for piece in ("Hel", "lo", "!"):
            yield f'data: {{"choices":[{{"delta":{{"content":"{piece}"}}}}]}}\n\n'.encode()
        yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse_chunks()
            )
        if body.get("tools"):
            return httpx.Response(
                200,
                json={
                    "id": "c2",
                    "object": "chat.completion",
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_invoice_total",
                                            "arguments": '{"invoice_id": "INV-7"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
        last = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"echo: {last}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    yield client, captured
    configure_http_transport(None)


def test_chat_forwards_with_served_model_id(api) -> None:
    client, captured = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "echo: hello"
    sent = json.loads(captured[-1].content)
    # The upstream sees ITS served id, never the deployment name.
    assert sent["model"] == "lfm2.5-350m-served"
    assert sent["messages"][0]["role"] == "system"
    assert captured[-1].url.host == "upstream"


def test_chat_stamps_recency_for_the_served_deployment(api, monkeypatch) -> None:
    # PR-4 recency must be stamped by every surface that serves traffic (see
    # recency.py's docstring) -- api.py's extract path already does this; this
    # generic chat surface didn't, so a deployment driven only through chat
    # read as idle forever and became the first idle-TTL/LRU eviction victim.
    client, _ = api
    calls: list[str | None] = []
    monkeypatch.setattr(
        "docie_bench.chat_api.recency.stamp_served_profile", lambda name: calls.append(name)
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, response.text
    assert calls == ["lfm2.5-350m"]


def test_chat_stream_stamps_recency_for_the_served_deployment(api, monkeypatch) -> None:
    # Streaming skips completion parsing entirely, so recency must be stamped
    # once the upstream connection is accepted -- not deferred to a
    # non-streaming-only code path that a streaming request never reaches.
    client, _ = api
    calls: list[str | None] = []
    monkeypatch.setattr(
        "docie_bench.chat_api.recency.stamp_served_profile", lambda name: calls.append(name)
    )
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())  # drain the stream
    assert calls == ["lfm2.5-350m"]


def test_chat_unknown_model_is_openai_404(api) -> None:
    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


def test_chat_requires_model(api) -> None:
    client, _ = api
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 400


def test_chat_stream_proxies_real_upstream_chunks(api) -> None:
    """A stream: true request gets the upstream's actual token-by-token SSE
    frames relayed as they arrive — not one buffered chunk built after the
    full completion finishes. The mock upstream emits 4 separate SSE events;
    all 4 must show up verbatim, in order, and the forwarded request must
    still carry stream: true (previously stripped before reaching upstream)."""
    client, captured = api
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert events == [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"delta":{"content":"!"}}]}',
        "data: [DONE]",
    ]
    sent = json.loads(captured[-1].content)
    assert sent["stream"] is True


# ── store: model that isn't live yet: load-on-demand, not a bare error ──────


@pytest.fixture
def api_store(monkeypatch) -> TestClient:
    from docie_bench.serving.placement_resolver import (
        PlacementNotFoundError,
        PlacementNotReadyError,
    )

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "store:never-deployed":
            raise PlacementNotFoundError("store model 'never-deployed' is not in the catalog")
        if model_profile == "store:evicted":
            raise PlacementNotReadyError("store model 'evicted' placement is 'stopped'")
        if model_profile == "store:truly-unseeded":
            raise PlacementNotFoundError("store model 'truly-unseeded' is not in the catalog")
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)

    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def test_chat_store_model_not_live_triggers_load_and_returns_202(
    api_store, monkeypatch
) -> None:
    async def fake_trigger(name: str) -> tuple[str, float] | None:
        assert name in ("never-deployed", "evicted")
        return name, 30.0

    monkeypatch.setattr("docie_bench.chat_api.trigger_deployment_load", fake_trigger)

    for model in ("store:never-deployed", "store:evicted"):
        response = api_store.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["status"] == "loading"
        assert body["eta_seconds"] == 30.0
        assert "30s" in body["message"]


def test_chat_bare_evicted_deployment_name_also_triggers_load(
    api_store, monkeypatch
) -> None:
    """The Playground's Chat/Vision pickers submit a raw deployment name,
    never store:-prefixed (unlike an explicit store:<name> API call) — an
    evicted deployment picked there must still cold-start, not 404, or the
    picker offering evicted deployments at all would be dead UI."""

    async def fake_trigger(name: str) -> tuple[str, float] | None:
        assert name == "gemma-2-2b-it"
        return name, 45.0

    monkeypatch.setattr("docie_bench.chat_api.trigger_deployment_load", fake_trigger)

    response = api_store.post(
        "/v1/chat/completions",
        json={"model": "gemma-2-2b-it", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "loading"
    assert body["deployment"] == "gemma-2-2b-it"


def test_chat_store_model_worker_loopback_endpoint_is_rejected(monkeypatch) -> None:
    """Mirrors api.py's resolve_profile guard: the deploy runtime records a
    placement's endpoint from the WORKER's point of view, so a loopback
    endpoint is unreachable from the api container this route runs in.
    Without the guard, this "resolves" fine and then burns
    timeout_seconds x retries on a doomed connect before a confusing
    upstream_unavailable 502 -- fail fast with a 501 instead."""
    loopback_profile = ModelProfile(
        name="store:qwen2.5-1.5b",
        model="qwen2.5-1.5b",
        base_url="http://127.0.0.1:8088/v1",
        api_key="local-not-used",
    )

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "store:qwen2.5-1.5b":
            return loopback_profile
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "store:qwen2.5-1.5b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 501, response.text
    body = response.json()
    assert "127.0.0.1:8088" in body["error"]["message"]
    assert "/v1/studio/extract" in body["error"]["message"]


def test_chat_deployment_selector_with_loopback_endpoint_is_not_guarded(monkeypatch) -> None:
    """The guard is scoped to store: refs only, mirroring api.py's guard
    exactly. This is a KNOWN, pre-existing gap, not a covered case: a bare
    deployment name can ALSO carry a loopback endpoint (control_plane.py's
    _guard_deterministic_advertise fail-opens on loopback for both serve()
    and serve_store_model() alike), and load-on-demand only helps a COLD
    model -- a warm-but-loopback deployment hits the same doomed-connect
    failure this guard exists to prevent, unguarded. Widening the check
    naively would risk false-positive 501s on a legitimate same-host local
    `docie up` deployment (control_plane.py treats that as reachable on
    purpose), so this needs its own deliberate fix, not a blind broaden --
    tracked as a follow-up, not silently claimed as covered."""
    loopback_profile = ModelProfile(
        name="local-dep", model="local-dep", base_url="http://127.0.0.1:8088/v1", api_key="k"
    )

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "local-dep":
            return loopback_profile
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "local-dep", "messages": [{"role": "user", "content": "hi"}]},
    )
    # Not rejected by the loopback guard; it proceeds to the actual (failing,
    # unreachable-in-tests) upstream call instead of a 501.
    assert response.status_code != 501, response.text


def test_chat_store_model_genuinely_unseeded_still_404s(api_store, monkeypatch) -> None:
    async def fake_trigger(name: str) -> tuple[str, float] | None:
        return None  # not a catalog entry at all -- nothing to trigger

    monkeypatch.setattr("docie_bench.chat_api.trigger_deployment_load", fake_trigger)

    response = api_store.post(
        "/v1/chat/completions",
        json={"model": "store:truly-unseeded", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


def test_chat_forwards_tools_and_preserves_tool_calls(api) -> None:
    """OpenAI tool calling end to end through the proxy: `tools`/`tool_choice`
    reach the upstream verbatim, and the upstream's `tool_calls` response --
    content null, finish_reason "tool_calls" -- comes back unmangled. Pins the
    two places that COULD have broken it: the forward body (built via
    dict(body), so unknown keys must survive) and fix_completion_content
    (which must skip a null content and never touch tool_calls)."""
    client, captured = api
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_invoice_total",
                "description": "Total for an invoice id",
                "parameters": {
                    "type": "object",
                    "properties": {"invoice_id": {"type": "string"}},
                    "required": ["invoice_id"],
                },
            },
        }
    ]
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "total of INV-7?"}],
            "tools": tools,
            "tool_choice": "auto",
        },
    )
    assert resp.status_code == 200, resp.text
    sent = json.loads(captured[-1].content)
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"

    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    (call,) = choice["message"]["tool_calls"]
    assert call["function"]["name"] == "get_invoice_total"
    assert json.loads(call["function"]["arguments"]) == {"invoice_id": "INV-7"}


def test_chat_forwards_tool_role_messages(api) -> None:
    """Round 2 of a tool exchange: the assistant tool_calls message and the
    `role: "tool"` result message forward verbatim -- the proxy must not
    validate or reshape the message list."""
    client, captured = api
    messages = [
        {"role": "user", "content": "total of INV-7?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_invoice_total", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "5400.00 EUR"},
    ]
    resp = client.post(
        "/v1/chat/completions", json={"model": "lfm2.5-350m", "messages": messages}
    )
    assert resp.status_code == 200, resp.text
    assert json.loads(captured[-1].content)["messages"] == messages
