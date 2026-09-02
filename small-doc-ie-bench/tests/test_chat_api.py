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


@pytest.fixture
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
        if last == "explode":
            return httpx.Response(500, text="upstream kaboom")
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
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
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


def test_chat_malformed_body_is_openai_shaped_400(api) -> None:
    """A body that isn't even valid JSON must still 400 in the SAME
    OpenAI error shape as every hand-rolled ``_openai_error(...)`` in this
    file, not FastAPI's default ``{"detail": [...]}`` validation shape."""
    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert set(error) == {"message", "type", "code"}
    assert error["type"] == "invalid_request_error"


def test_chat_forwards_unrecognized_openai_fields_untouched(api) -> None:
    """Fields this handler never inspects (temperature, top_p, ...) must
    still reach the upstream verbatim -- the point of ``extra="allow"``."""
    client, captured = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "top_p": 0.9,
        },
    )
    assert response.status_code == 200, response.text
    sent = json.loads(captured[-1].content)
    assert sent["temperature"] == 0.2
    assert sent["top_p"] == 0.9


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


def test_chat_mcp_stream_relays_each_tool_call_as_its_own_sse_event(api, monkeypatch) -> None:
    # Each executed tool call arrives as its own SSE frame the moment it
    # finishes, instead of the whole exchange completing silently before
    # anything reaches the client ("Waiting for the model…" with zero
    # visibility into the agentic search actually running).
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        assert on_tool_call is not None
        on_tool_call("calc__add", True, 12, {"a": 1, "b": 2}, "3")
        return {"choices": [{"message": {"role": "assistant", "content": "3"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what is 1+2?"}],
            "mcp_servers": ["calc"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    tool_events = [e for e in events if e["type"] == "tool_call"]
    content_events = [e for e in events if e["type"] == "content"]
    assert tool_events == [
        {
            "type": "tool_call",
            "tool": "calc__add",
            "status": "ok",
            "latency_ms": 12,
            "arguments": '{"a": 1, "b": 2}',
            "result": "3",
        }
    ]
    (content_event,) = content_events
    assert content_event["completion"]["choices"][0]["message"]["content"] == "3"
    assert response.text.strip().endswith("data: [DONE]")


def test_chat_mcp_stream_relays_reasoning_content_as_its_own_sse_event(api, monkeypatch) -> None:
    # A reasoning-capable model's "why" for calling a tool arrives as its
    # own event too -- answers "is there a hidden thinking step" instead of
    # silently discarding message.reasoning_content.
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        assert on_reasoning is not None
        on_reasoning("the user asked for 1+2, so I should call calc.add")
        return {"choices": [{"message": {"role": "assistant", "content": "3"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what is 1+2?"}],
            "mcp_servers": ["calc"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    (reasoning_event,) = [e for e in events if e["type"] == "reasoning"]
    assert reasoning_event["text"] == "the user asked for 1+2, so I should call calc.add"


def test_chat_mcp_stream_relays_system_addendum_as_its_own_sse_event_once(api, monkeypatch) -> None:
    # run_tool_loop's TOOL_DISCIPLINE_DIRECTIVE (folded into the request's
    # system message before the first round) is real, load-bearing content
    # that was never surfaced anywhere -- it now arrives as its own one-time
    # SSE event, with the exact addendum text, not a placeholder.
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import TOOL_DISCIPLINE_DIRECTIVE, MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        assert on_system_addendum is not None
        on_system_addendum(TOOL_DISCIPLINE_DIRECTIVE)
        return {"choices": [{"message": {"role": "assistant", "content": "3"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what is 1+2?"}],
            "mcp_servers": ["calc"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    addendum_events = [e for e in events if e["type"] == "system_addendum"]
    assert addendum_events == [{"type": "system_addendum", "text": TOOL_DISCIPLINE_DIRECTIVE}]


def test_chat_mcp_stream_relays_usage_as_its_own_sse_event(api, monkeypatch) -> None:
    # run_tool_loop's on_usage (per-round + cumulative token counts) arrives
    # as its own event too -- lets a client show context consumption before
    # the final completion lands, not only after the whole exchange finishes.
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        assert on_usage is not None
        on_usage(
            {
                "round": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "cumulative": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
        return {"choices": [{"message": {"role": "assistant", "content": "3"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what is 1+2?"}],
            "mcp_servers": ["calc"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    (usage_event,) = [e for e in events if e["type"] == "usage"]
    assert usage_event == {
        "type": "usage",
        "round": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "cumulative": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_chat_mcp_stream_relays_context_budget_as_its_own_sse_event(api, monkeypatch) -> None:
    # run_tool_loop's on_context_budget (#344) arrives as its own event too --
    # a client warning that cumulative usage crossed the resolved
    # deployment's context-window threshold, before the exchange runs out of
    # room entirely.
    from docie_bench import chat_api, mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)
    monkeypatch.setattr(chat_api, "_context_length_for_profile", lambda profile: 4096)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        assert context_length_ceiling == 4096
        assert on_context_budget is not None
        on_context_budget(
            {
                "cumulative_tokens": 3300,
                "context_length": 4096,
                "threshold_fraction": 0.8,
            }
        )
        return {"choices": [{"message": {"role": "assistant", "content": "3"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what is 1+2?"}],
            "mcp_servers": ["calc"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    (budget_event,) = [e for e in events if e["type"] == "context_budget"]
    assert budget_event == {
        "type": "context_budget",
        "cumulative_tokens": 3300,
        "context_length": 4096,
        "threshold_fraction": 0.8,
    }


def test_chat_mcp_stream_unresolvable_ceiling_skips_context_budget_check(
    api, monkeypatch, tmp_path
) -> None:
    # A profile with no matching live deployment record (a plain models.yaml
    # profile, or -- as here -- the test's stub resolver) can't be priced
    # against a context window at all; run_tool_loop must get None and never
    # be asked to warn, rather than guess a ceiling or block the request.
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)
    # No monkeypatch of _context_length_for_profile itself: the real lookup
    # runs, against an empty DOCIE_SERVING_HOME (an isolated tmp_path, not
    # this machine's real deployments.json -- whatever it happens to have
    # deployed right now is irrelevant), so it deterministically finds no
    # live deployment record named "lfm2.5-350m" and resolves to None -- the
    # "can't be priced" path this test covers.
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    seen_ceiling: list[int | None] = []

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        seen_ceiling.append(context_length_ceiling)
        assert on_context_budget is not None  # still wired -- just never called
        return {"choices": [{"message": {"role": "assistant", "content": "3"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what is 1+2?"}],
            "mcp_servers": ["calc"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert seen_ceiling == [None]
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert [e for e in events if e["type"] == "context_budget"] == []


def test_chat_mcp_stream_emits_tool_calls_unsupported_before_any_round(api, monkeypatch) -> None:
    # #353: a deployment whose chat template is known NOT to support real
    # tool-calling must warn the caller ONCE, BEFORE run_tool_loop runs a
    # single round -- known upfront from the deployment's own health state.
    from docie_bench import chat_api, mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)
    monkeypatch.setattr(chat_api, "_tool_calls_supported_for_profile", lambda profile: False)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "read the pdf"}],
            "mcp_servers": ["calc"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    unsupported_events = [e for e in events if e["type"] == "tool_calls_unsupported"]
    assert len(unsupported_events) == 1
    assert "chat_template_caps.supports_tool_calls=false" in unsupported_events[0]["message"]
    # Fired before any other event (content included) -- known upfront, not
    # learned mid-exchange.
    assert events[0]["type"] == "tool_calls_unsupported"


@pytest.mark.parametrize("verdict", [True, None])
def test_chat_mcp_stream_skips_tool_calls_unsupported_when_true_or_unknown(
    api, monkeypatch, verdict: bool | None
) -> None:
    from docie_bench import chat_api, mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)
    monkeypatch.setattr(chat_api, "_tool_calls_supported_for_profile", lambda profile: verdict)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "read the pdf"}],
            "mcp_servers": ["calc"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert [e for e in events if e["type"] == "tool_calls_unsupported"] == []


def test_chat_mcp_stream_unregistered_server_sends_an_error_event(api) -> None:
    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": ["ghost"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    (error_event,) = events
    assert error_event["type"] == "error"
    assert "unregistered MCP server" in error_event["error"]["message"]


# ── human-in-the-loop pause/resume (#383) ────────────────────────────────


def test_chat_mcp_stream_mints_no_exchange_without_ask_user_opt_in(api, monkeypatch) -> None:
    # enable_ask_user defaults False -- a caller that never opts in gets
    # exactly today's behavior: no exchange id minted, no ask_user tool.
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    seen_exchange_ids: list[str | None] = []

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        seen_exchange_ids.append(exchange_id)
        assert on_awaiting_input is None
        return {"choices": [{"message": {"role": "assistant", "content": "3"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what is 1+2?"}],
            "mcp_servers": ["calc"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert [e for e in events if e["type"] == "exchange"] == []
    assert seen_exchange_ids == [None]


def test_chat_mcp_stream_emits_exchange_event_first_when_ask_user_is_enabled(
    api, monkeypatch
) -> None:
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    seen_exchange_ids: list[str | None] = []

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        seen_exchange_ids.append(exchange_id)
        assert exchange_id is not None
        assert on_awaiting_input is not None
        return {"choices": [{"message": {"role": "assistant", "content": "3"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what is 1+2?"}],
            "mcp_servers": ["calc"],
            "stream": True,
            "enable_ask_user": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    # exchange is the FIRST event, before anything else.
    assert events[0]["type"] == "exchange"
    assert events[0]["exchange_id"] == seen_exchange_ids[0]
    # And the registry entry is gone once the exchange has finished.
    assert not mcp_tools.has_pending_input(events[0]["exchange_id"])


def test_chat_mcp_stream_relays_awaiting_input_as_its_own_sse_event(api, monkeypatch) -> None:
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(
        post,
        body,
        sessions,
        mapping,
        tools,
        on_tool_call=None,
        on_reasoning=None,
        on_system_addendum=None,
        on_usage=None,
        context_length_ceiling=None,
        on_context_budget=None,
        exchange_id=None,
        on_awaiting_input=None,
    ):
        assert on_awaiting_input is not None
        on_awaiting_input({"question": "which invoice?", "choices": ["A", "B"]})
        return {"choices": [{"message": {"role": "assistant", "content": "picked A"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "which invoice?"}],
            "mcp_servers": ["calc"],
            "stream": True,
            "enable_ask_user": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    (awaiting_event,) = [e for e in events if e["type"] == "awaiting_input"]
    assert awaiting_event == {
        "type": "awaiting_input",
        "question": "which invoice?",
        "choices": ["A", "B"],
    }


def test_chat_mcp_stream_ask_user_timeout_ends_the_exchange_with_an_error_event(
    api, monkeypatch
) -> None:
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(*args, **kwargs):
        raise mcp_tools.AskUserTimeoutError("no answer arrived within 0s")

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": ["calc"],
            "stream": True,
            "enable_ask_user": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    (error_event,) = [e for e in events if e["type"] == "error"]
    assert error_event["error"]["type"] == "ask_user_timeout"
    assert error_event["error"]["code"] == "ask_user_timeout"
    # Never leaked -- cleaned up even though the exchange ended via an error.
    exchange_event = next(e for e in events if e["type"] == "exchange")
    assert not mcp_tools.has_pending_input(exchange_event["exchange_id"])


async def test_chat_mcp_stream_cancellation_during_a_pause_clears_the_registry() -> None:
    # A closed tab / abandoned request: the client disconnects while an
    # exchange is genuinely paused, awaiting an answer nobody will ever send.
    # The registry entry must not survive that.
    import time

    from docie_bench import mcp_tools
    from docie_bench.chat_api import _stream_chat_with_mcp_tools
    from docie_bench.llm.model_profiles import ModelProfile

    profile = ModelProfile(
        name="lfm2.5-350m", model="lfm2.5-350m-served", base_url="http://upstream/v1", api_key="k"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        # The MCP tool loop's post() now streams every round (see
        # _post_upstream_streamed) -- the mock upstream answers with the
        # SSE-delta shape, not one blocking JSON completion.
        async def _chunks() -> AsyncIterator[bytes]:
            frame = {
                "id": "r1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "ask_user",
                                        "arguments": '{"question": "which invoice?"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
            yield f"data: {json.dumps(frame)}\n\n".encode()
            yield b'data: {"choices": [], "usage": {}}\n\n'
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_chunks()
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await _stream_chat_with_mcp_tools(
            client,
            "http://upstream/v1/chat/completions",
            {},
            {"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "hi"}]},
            profile,
            [],  # no real MCP servers needed for this
            None,
            tenant_id="t",
            started=time.perf_counter(),
            enable_ask_user=True,
        )
        gen = resp.body_iterator
        exchange_id: str | None = None
        async for chunk in gen:
            line = bytes(chunk).decode()
            if not line.startswith("data: ") or line.startswith("data: [DONE]"):
                continue
            event = json.loads(line[len("data: ") :].strip())
            if event["type"] == "exchange":
                exchange_id = event["exchange_id"]
            elif event["type"] == "awaiting_input":
                assert exchange_id is not None
                assert mcp_tools.has_pending_input(exchange_id)
                break  # simulate the client giving up mid-pause
        assert exchange_id is not None
        await gen.aclose()  # drives body_iterator's finally: cancel + await drive()
        assert not mcp_tools.has_pending_input(exchange_id)


def test_pause_endpoint_flags_a_registered_exchange_and_404s_for_unknown_ones(api) -> None:
    from docie_bench import mcp_tools

    mcp_tools.open_pending_input("exch-pause-route")
    try:
        client, _ = api
        resp = client.post(
            "/v1/chat/completions/pause", json={"exchange_id": "exch-pause-route"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"paused": True, "exchange_id": "exch-pause-route"}
        assert mcp_tools._pending_inputs["exch-pause-route"].pause_requested is True
    finally:
        mcp_tools.close_pending_input("exch-pause-route")

    resp = client.post("/v1/chat/completions/pause", json={"exchange_id": "no-such-exchange"})
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "unknown_exchange"


def test_respond_endpoint_resolves_a_registered_exchange_and_404s_for_unknown_ones(api) -> None:
    from docie_bench import mcp_tools

    mcp_tools.open_pending_input("exch-respond-route")
    try:
        client, _ = api
        resp = client.post(
            "/v1/chat/completions/respond",
            json={"exchange_id": "exch-respond-route", "text": "use invoice A"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"accepted": True, "exchange_id": "exch-respond-route"}
        pending = mcp_tools._pending_inputs["exch-respond-route"]
        assert pending.answer == "use invoice A"
        assert pending.event.is_set()
    finally:
        mcp_tools.close_pending_input("exch-respond-route")

    resp = client.post(
        "/v1/chat/completions/respond",
        json={"exchange_id": "no-such-exchange", "text": "x"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "unknown_exchange"


def test_chat_mcp_field_must_be_a_string_list(api) -> None:
    client, _ = api
    for bad in ("calc", [1, 2], [""], {"name": "calc"}):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "lfm2.5-350m",
                "messages": [{"role": "user", "content": "hi"}],
                "mcp_servers": bad,
            },
        )
        assert resp.status_code == 400, bad


def test_chat_mcp_unregistered_server_is_400(api, monkeypatch) -> None:
    # Registry-only security: a request picks servers BY NAME from the
    # operator's config -- it can never introduce its own server.
    from docie_bench import mcp_tools

    monkeypatch.setattr(mcp_tools, "load_mcp_registry", lambda path=None: {})
    client, _ = api
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": ["calc"],
        },
    )
    assert resp.status_code == 400
    assert "unregistered MCP server" in resp.json()["error"]["message"]


def test_chat_mcp_sdk_missing_is_501(api, monkeypatch) -> None:
    # The mcp package is an optional extra: absent SDK answers a clear 501
    # with the install command, never an ImportError 500.
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec, MCPUnavailableError

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )

    def raiser() -> None:
        raise MCPUnavailableError("MCP tool support needs the optional 'mcp' package")

    monkeypatch.setattr(mcp_tools, "_require_mcp", raiser)
    client, _ = api
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": ["calc"],
        },
    )
    assert resp.status_code == 501
    assert "'mcp' package" in resp.json()["error"]["message"]


def test_chat_mcp_response_carries_the_tool_call_trace(api, monkeypatch) -> None:
    # The Playground's Chat mode (and any other caller of the generic
    # `mcp_servers` surface) needs the same "Try it" trace shape the Agents
    # surface already returns -- `run_tool_loop`'s `on_tool_call` seam is
    # shared via `mcp_tools.make_trace_recorder`.
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(
        post, body, sessions, mapping, tools, on_tool_call=None, on_reasoning=None
    ):
        assert on_tool_call is not None
        on_tool_call("calc__add", True, 12, {"a": 1, "b": 2}, "3")
        return {
            "id": "chatcmpl-1",
            "choices": [{"message": {"role": "assistant", "content": "3"}}],
        }

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what is 1+2?"}],
            "mcp_servers": ["calc"],
        },
    )
    assert resp.status_code == 200
    trace = resp.json()["docie_agent"]["tool_calls"]
    assert trace == [
        {
            "tool": "calc__add",
            "status": "ok",
            "latency_ms": 12,
            "arguments": '{"a": 1, "b": 2}',
            "result": "3",
        }
    ]


def test_chat_mcp_session_id_points_docs_search_at_the_session_directory(
    api, monkeypatch, tmp_path
) -> None:
    # A Playground attachment uploaded via POST /v1/studio/session-documents
    # (#296) is only searchable if docs-search's spec is launched with its
    # documents directory overridden to that session's upload directory.
    from docie_bench import mcp_session_documents, mcp_tools
    from docie_bench.mcp_servers.docs_search import DOCS_DIR_ENV
    from docie_bench.mcp_tools import MCPServerSpec
    from docie_bench.settings import get_settings

    monkeypatch.setenv("MCP_SESSION_DOCUMENTS_ROOT", str(tmp_path))
    get_settings.cache_clear()

    session_id, _ = mcp_session_documents.save_document(None, "invoice.pdf", b"%PDF-fake")

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "docs-search": MCPServerSpec(
                name="docs-search", transport="streamable-http", url="http://x"
            )
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    captured_specs: list[list] = []

    async def fake_open_sessions(stack, specs):
        captured_specs.append(specs)
        return {"docs-search": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    async def fake_run_tool_loop(
        post, body, sessions, mapping, tools, on_tool_call=None, on_reasoning=None
    ):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    monkeypatch.setattr(mcp_tools, "run_tool_loop", fake_run_tool_loop)

    client, _ = api
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what's in the attached file?"}],
            "mcp_servers": ["docs-search"],
            "session_id": session_id,
        },
    )
    assert resp.status_code == 200
    (specs,) = captured_specs
    (spec,) = specs
    assert spec.env[DOCS_DIR_ENV] == str(tmp_path / session_id)
    get_settings.cache_clear()


def test_chat_mcp_unknown_session_id_is_400(api, monkeypatch) -> None:
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "docs-search": MCPServerSpec(
                name="docs-search", transport="streamable-http", url="http://x"
            )
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)
    client, _ = api
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": ["docs-search"],
            "session_id": "not-a-real-session",
        },
    )
    assert resp.status_code == 400
    assert "invalid session id" in resp.json()["error"]["message"]


def test_chat_mcp_session_id_must_be_a_string(api) -> None:
    client, _ = api
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": ["calc"],
            "session_id": 12345,
        },
    )
    assert resp.status_code == 400
    assert "session_id" in resp.json()["error"]["message"]


# ── usage ledger: every resolved request writes one durable row ─────────────


@pytest.fixture
def usage_db(tmp_path):
    from docie_bench.storage.db import dispose_engine, init_engine

    init_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    yield
    dispose_engine()


def _usage_rows():
    from sqlalchemy import select

    from docie_bench.storage.db import session_scope
    from docie_bench.studio.models import UsageRecord

    with session_scope() as session:
        assert session is not None
        return session.scalars(select(UsageRecord).order_by(UsageRecord.id)).all()


def test_chat_success_writes_usage_row_with_tokens(api, usage_db) -> None:
    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, response.text
    (row,) = _usage_rows()
    assert row.deployment == "lfm2.5-350m"
    assert row.surface == "chat"
    assert row.status == "ok"
    # Lifted from the upstream completion's usage block, in hand at the seam.
    assert row.prompt_tokens == 7
    assert row.completion_tokens == 3
    assert row.latency_ms >= 0
    # The authenticated principal's id (an unauthenticated caller gets its
    # per-client anon bucket, e.g. "anon:testclient" -- never empty).
    assert row.tenant_id


def test_chat_upstream_error_writes_error_usage_row(api, usage_db) -> None:
    # An upstream 5xx is still a request against the deployment: it must be
    # counted (status=error, no tokens), not silently missing from the ledger.
    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "explode"}]},
    )
    assert response.status_code == 500
    (row,) = _usage_rows()
    assert row.status == "error"
    assert row.surface == "chat"
    assert row.prompt_tokens is None
    assert row.completion_tokens is None


def test_chat_stream_writes_usage_row_without_tokens(api, usage_db) -> None:
    # This mock upstream's SSE frames never carry a usage block (unlike the
    # dedicated fixtures below) -- the row must still land, tokens None,
    # exactly the pre-fix fallback behavior.
    client, _ = api
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
    (row,) = _usage_rows()
    assert row.status == "ok"
    assert row.surface == "chat"
    assert row.prompt_tokens is None
    assert row.completion_tokens is None


def test_chat_stream_requests_include_usage_by_default(api) -> None:
    # llama-server/OpenAI only emit a final usage frame when asked -- the
    # relay must ask, without a caller having to know that itself.
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
        list(response.iter_lines())
    sent = json.loads(captured[-1].content)
    assert sent["stream_options"] == {"include_usage": True}


def test_chat_stream_preserves_callers_stream_options(api) -> None:
    client, captured = api
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "stream": True,
            "stream_options": {"include_usage": False, "custom_flag": "x"},
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        list(response.iter_lines())
    sent = json.loads(captured[-1].content)
    # A caller that explicitly opted OUT (or set other options) is honored --
    # this only fills in the default, never overrides an explicit choice.
    assert sent["stream_options"] == {"include_usage": False, "custom_flag": "x"}


@pytest.fixture
def api_stream_with_usage(monkeypatch):
    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "lfm2.5-350m":
            return UPSTREAM
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)

    async def _sse_chunks() -> AsyncIterator[bytes]:
        # The final usage frame split ACROSS two raw chunks -- proves the
        # byte-buffer scan reassembles a frame straddling a chunk boundary,
        # not just one that happens to land whole in a single yield.
        yield b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: {"choices":['
        yield (
            b'],"usage":{"prompt_tokens":11,"completion_tokens":4,'
            b'"total_tokens":15}}\n\ndata: [DONE]\n\n'
        )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_chunks()
        )

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    yield client
    configure_http_transport(None)


def test_chat_stream_captures_usage_split_across_chunks(api_stream_with_usage, usage_db) -> None:
    with api_stream_with_usage.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        assert response.status_code == 200
        events = [line for line in response.iter_lines() if line.startswith("data: ")]

    # Relayed bytes are UNCHANGED -- the usage-scanning is a side channel.
    assert events[-1] == "data: [DONE]"
    (row,) = _usage_rows()
    assert row.status == "ok"
    assert row.prompt_tokens == 11
    assert row.completion_tokens == 4


def test_chat_usage_ledger_down_never_fails_the_chat(api, monkeypatch) -> None:
    # The never-raise contract: a database that errors on the insert (blip,
    # pool exhaustion, gone entirely) must not turn a served completion into
    # a 500 -- the ledger degrades, the chat answers.
    from sqlalchemy.exc import OperationalError

    from docie_bench.studio import usage_store

    def broken_session_scope():
        raise OperationalError("insert", None, Exception("database is down"))

    monkeypatch.setattr(usage_store, "session_scope", broken_session_scope)
    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "echo: hi"


def test_chat_without_database_still_answers(api) -> None:
    # No DATABASE_URL at all: recording degrades to a no-op, the surface is
    # unaffected.
    from docie_bench.storage.db import dispose_engine
    from docie_bench.studio.usage_store import record_usage

    dispose_engine()  # ensure no engine leaks in from an earlier test

    assert (
        record_usage(
            deployment="lfm2.5-350m", surface="chat", tenant_id="anonymous", latency_ms=1
        )
        is False
    )
    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, response.text


def test_mcp_servers_listing_redacts_secret_values(api, monkeypatch) -> None:
    # Discovery endpoint: names + transports out, header/env VALUES never.
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "docs": MCPServerSpec(
                name="docs",
                transport="streamable-http",
                url="http://mcp.internal/mcp",
                headers={"Authorization": "Bearer topsecret"},
            )
        },
    )
    client, _ = api
    resp = client.get("/v1/mcp/servers")
    assert resp.status_code == 200
    (entry,) = resp.json()["servers"]
    assert entry["name"] == "docs"
    assert entry["headers"] == ["Authorization"]
    assert "topsecret" not in resp.text


# ── _post_upstream_streamed: token-by-token MCP tool loop rounds (#389) ─────
#
# Real token streaming for the MCP tool loop's own upstream calls -- every
# round used to be one blocking _post_upstream() POST, so a reasoning model's
# thinking and its final answer both landed as one lump the instant
# llama-server finished generating. These tests exercise the new streaming
# `post` closure (_post_upstream_streamed) directly against a fake
# multi-chunk SSE upstream -- the highest-risk part is tool-call fragment
# accumulation (a wrong reconstruction silently breaks every agentic tool
# call), so that gets a byte-identical comparison against the non-streaming
# path's own output for the same logical content, not just an eyeballed
# happy path.


def _sse_body(*frames: dict) -> bytes:
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    return (body + "data: [DONE]\n\n").encode()


@pytest.fixture
def streamed_profile() -> ModelProfile:
    return ModelProfile(
        name="lfm2.5-350m", model="lfm2.5-350m-served", base_url="http://upstream/v1", api_key="k"
    )


async def test_post_upstream_streamed_accumulates_a_tool_call_split_across_many_chunks(
    streamed_profile,
) -> None:
    # The highest-risk part (#389): function.name AND function.arguments
    # split across several chunks -- realistic streaming, not one chunk per
    # tool call -- reassembled and compared BYTE-IDENTICAL to what
    # _post_upstream (the non-streaming path) returns for the same logical
    # completion.
    import asyncio

    from docie_bench.chat_api import _post_upstream, _post_upstream_streamed

    def streaming_handler(request: httpx.Request) -> httpx.Response:
        frames = [
            {
                "id": "c1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_", "arguments": ""},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "invoice_total",
                                        "arguments": '{"invoi',
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'ce_id": '}}
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"INV-7"}'}}
                            ]
                        },
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body(*frames)
        )

    def non_streaming_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "model": "lfm2.5-350m-served",
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

    forward = {"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "total?"}]}
    queue: asyncio.Queue = asyncio.Queue()

    async with httpx.AsyncClient(transport=httpx.MockTransport(streaming_handler)) as client:
        streamed = await _post_upstream_streamed(
            client, "http://upstream/v1/chat/completions", {}, forward, streamed_profile, queue
        )
    async with httpx.AsyncClient(transport=httpx.MockTransport(non_streaming_handler)) as client:
        non_streamed = await _post_upstream(
            client, "http://upstream/v1/chat/completions", {}, forward, streamed_profile
        )

    assert isinstance(streamed, dict)
    assert isinstance(non_streamed, dict)
    assert streamed["choices"][0]["message"] == non_streamed["choices"][0]["message"]
    assert streamed["choices"][0]["finish_reason"] == non_streamed["choices"][0]["finish_reason"]
    assert streamed["choices"][0]["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_invoice_total", "arguments": '{"invoice_id": "INV-7"}'},
        }
    ]
    # A tool-call-only round never streams content/reasoning fragments.
    assert queue.empty()


async def test_post_upstream_streamed_accumulates_two_tool_calls_by_index(
    streamed_profile,
) -> None:
    # Two tool calls in the same round, each streamed as its own fragment
    # sequence -- the "index" keying, not arrival order, must keep them apart.
    import asyncio

    from docie_bench.chat_api import _post_upstream_streamed

    def handler(request: httpx.Request) -> httpx.Response:
        frames = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "calc__add", "arguments": ""},
                                },
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "calc__sub", "arguments": ""},
                                },
                            ],
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 1, "function": {"arguments": '{"a": 5, '}},
                                {"index": 0, "function": {"arguments": '{"a": 1, '}},
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"b": 2}'}},
                                {"index": 1, "function": {"arguments": '"b": 3}'}},
                            ]
                        },
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body(*frames)
        )

    forward = {"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "1+2 and 5-3?"}]}
    queue: asyncio.Queue = asyncio.Queue()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _post_upstream_streamed(
            client, "http://upstream/v1/chat/completions", {}, forward, streamed_profile, queue
        )
    assert isinstance(result, dict)
    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "calc__add", "arguments": '{"a": 1, "b": 2}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "calc__sub", "arguments": '{"a": 5, "b": 3}'},
        },
    ]


async def test_post_upstream_streamed_accumulates_reasoning_only_deltas(streamed_profile) -> None:
    import asyncio

    from docie_bench.chat_api import _post_upstream_streamed

    def handler(request: httpx.Request) -> httpx.Response:
        frames = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "reasoning_content": "Let "}}
                ]
            },
            {"choices": [{"index": 0, "delta": {"reasoning_content": "me "}}]},
            {"choices": [{"index": 0, "delta": {"reasoning_content": "think."}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body(*frames)
        )

    forward = {"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "hi"}]}
    queue: asyncio.Queue = asyncio.Queue()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _post_upstream_streamed(
            client, "http://upstream/v1/chat/completions", {}, forward, streamed_profile, queue
        )
    assert isinstance(result, dict)
    message = result["choices"][0]["message"]
    assert message["reasoning_content"] == "Let me think."
    assert message["content"] is None
    assert "tool_calls" not in message

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [e["type"] for e in events] == ["reasoning_delta", "reasoning_delta", "reasoning_delta"]
    assert [e["text"] for e in events] == ["Let ", "me ", "think."]


async def test_post_upstream_streamed_accumulates_content_only_deltas(streamed_profile) -> None:
    import asyncio

    from docie_bench.chat_api import _post_upstream_streamed

    def handler(request: httpx.Request) -> httpx.Response:
        frames = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
            {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
            {"choices": [{"index": 0, "delta": {"content": "!"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body(*frames)
        )

    forward = {"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "hi"}]}
    queue: asyncio.Queue = asyncio.Queue()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _post_upstream_streamed(
            client, "http://upstream/v1/chat/completions", {}, forward, streamed_profile, queue
        )
    assert isinstance(result, dict)
    message = result["choices"][0]["message"]
    assert message["content"] == "Hello!"
    assert "reasoning_content" not in message
    assert "tool_calls" not in message

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [e["type"] for e in events] == ["content_delta", "content_delta", "content_delta"]
    assert [e["text"] for e in events] == ["Hel", "lo", "!"]


async def test_post_upstream_streamed_captures_usage_from_the_trailing_frame(
    streamed_profile,
) -> None:
    import asyncio

    from docie_bench.chat_api import _post_upstream_streamed

    def handler(request: httpx.Request) -> httpx.Response:
        frames = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body(*frames)
        )

    forward = {"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "hi"}]}
    queue: asyncio.Queue = asyncio.Queue()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _post_upstream_streamed(
            client, "http://upstream/v1/chat/completions", {}, forward, streamed_profile, queue
        )
    assert isinstance(result, dict)
    assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


async def test_post_upstream_streamed_upstream_error_status_maps_like_post_upstream(
    streamed_profile,
) -> None:
    import asyncio

    from fastapi.responses import JSONResponse

    from docie_bench.chat_api import _post_upstream_streamed

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream kaboom")

    forward = {"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "hi"}]}
    queue: asyncio.Queue = asyncio.Queue()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _post_upstream_streamed(
            client, "http://upstream/v1/chat/completions", {}, forward, streamed_profile, queue
        )
    assert isinstance(result, JSONResponse)
    assert result.status_code == 500
    body = json.loads(bytes(result.body))
    assert body["error"]["type"] == "upstream_error"
    assert "upstream kaboom" in body["error"]["message"]


async def test_post_upstream_streamed_connection_error_maps_to_upstream_unavailable(
    streamed_profile,
) -> None:
    import asyncio

    from fastapi.responses import JSONResponse

    from docie_bench.chat_api import _post_upstream_streamed

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    forward = {"model": "lfm2.5-350m", "messages": [{"role": "user", "content": "hi"}]}
    queue: asyncio.Queue = asyncio.Queue()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _post_upstream_streamed(
            client, "http://upstream/v1/chat/completions", {}, forward, streamed_profile, queue
        )
    assert isinstance(result, JSONResponse)
    assert result.status_code == 502
    body = json.loads(bytes(result.body))
    assert body["error"]["type"] == "upstream_unavailable"


def test_chat_mcp_stream_content_and_reasoning_deltas_precede_the_rounds_one_shot_events(
    api, monkeypatch
) -> None:
    # #389, end to end through the real (unmocked) run_tool_loop: every
    # content_delta/reasoning_delta for a round must land BEFORE that
    # round's own usage/reasoning/content events, which are only built once
    # the round's stream has fully ended.
    from docie_bench import mcp_tools
    from docie_bench.mcp_tools import MCPServerSpec

    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://x")
        },
    )
    monkeypatch.setattr(mcp_tools, "_require_mcp", lambda: None)

    async def fake_open_sessions(stack, specs):
        return {"calc": object()}

    async def fake_collect_tools(sessions):
        return [], {}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open_sessions)
    monkeypatch.setattr(mcp_tools, "collect_openai_tools", fake_collect_tools)
    # run_tool_loop is left REAL here -- the point is exercising the actual
    # streaming post() closure end to end, not a stand-in for it.

    def handler(request: httpx.Request) -> httpx.Response:
        frames = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "reasoning_content": "thinking "}}
                ]
            },
            {"choices": [{"index": 0, "delta": {"reasoning_content": "it over"}}]},
            {"choices": [{"index": 0, "delta": {"content": "the "}}]},
            {"choices": [{"index": 0, "delta": {"content": "answer"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body(*frames)
        )

    configure_http_transport(httpx.MockTransport(handler))
    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [{"role": "user", "content": "what is 1+2?"}],
            "mcp_servers": ["calc"],
            "stream": True,
        },
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert [e["type"] for e in events] == [
        "reasoning_delta",
        "reasoning_delta",
        "content_delta",
        "content_delta",
        "usage",
        "reasoning",
        "content",
    ]
    reasoning_deltas = "".join(e["text"] for e in events if e["type"] == "reasoning_delta")
    content_deltas = "".join(e["text"] for e in events if e["type"] == "content_delta")
    assert reasoning_deltas == "thinking it over"
    assert content_deltas == "the answer"
    (reasoning_event,) = [e for e in events if e["type"] == "reasoning"]
    assert reasoning_event["text"] == "thinking it over"
    (content_event,) = [e for e in events if e["type"] == "content"]
    assert content_event["completion"]["choices"][0]["message"]["content"] == "the answer"
