"""custom-kind agents with options.mcp_servers: same tool loop the generic
chat surface uses (mcp_tools.run_tool_loop), reused rather than duplicated.

The MCP side is REAL — an in-process MCPServer + ClientSession over
mcp.shared.memory streams, same protocol handshake/framing a remote server
would speak. Only the LLM upstream and the registry are scripted.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.client.session import ClientSession
from mcp.server.mcpserver import MCPServer
from mcp.shared.memory import create_client_server_memory_streams

from docie_bench import mcp_tools
from docie_bench.agents.api import configure_http_transport
from docie_bench.agents.api import router as agents_router
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.mcp_tools import MCPServerSpec

UPSTREAM = ModelProfile(name="alpha", model="up-alpha", base_url="http://upstream/v1", api_key="k")


def _calc_server() -> MCPServer:
    server = MCPServer("calc")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def subtract(a: int, b: int) -> int:
        """Subtract two integers."""
        return a - b

    return server


@asynccontextmanager
async def _memory_session(server: MCPServer) -> AsyncIterator[ClientSession]:
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        low = server._lowlevel_server
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: low.run(server_read, server_write, low.create_initialization_options())
            )
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                yield session
            tg.cancel_scope.cancel()


def _tool_calls_completion(name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": "r1",
        "object": "chat.completion",
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
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _final_completion(content: str) -> dict[str, Any]:
    return {
        "id": "r2",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
    }


@pytest.fixture
def api(tmp_path, monkeypatch) -> tuple[TestClient, list[httpx.Request]]:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile in (None, "alpha"):
            return UPSTREAM
        return replace(UPSTREAM, name=str(model_profile), model=str(model_profile))

    monkeypatch.setattr("docie_bench.agents.runtime.resolve_extraction_profile", fake_resolver)

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        has_tool_result = any(m.get("role") == "tool" for m in body["messages"])
        if not has_tool_result:
            return httpx.Response(200, json=_tool_calls_completion("calc__add", '{"a": 2, "b": 3}'))
        result = next(m["content"] for m in body["messages"] if m.get("role") == "tool")
        return httpx.Response(200, json=_final_completion(f"the sum is {result}"))

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(agents_router)
    client = TestClient(app)
    yield client, captured
    configure_http_transport(None)


def _create_agent(client: TestClient, **overrides: object) -> None:
    payload: dict[str, object] = {
        "name": "tool-helper",
        "template": "custom",
        "model_profile": "alpha",
    }
    payload.update(overrides)
    response = client.post("/v1/agents", json=payload)
    assert response.status_code in (200, 201), response.text


def test_custom_agent_without_mcp_servers_is_unaffected(api) -> None:
    # No options.mcp_servers -> the original bare-forward behavior, unchanged.
    client, captured = api
    _create_agent(client)
    response = client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, response.text
    sent = json.loads(captured[-1].content)
    assert "tools" not in sent


def test_custom_agent_runs_the_tool_loop_when_mcp_servers_set(api, monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://unused")
        },
    )
    server = _calc_server()

    async def fake_open(stack, specs):
        assert [s.name for s in specs] == ["calc"]
        session = await stack.enter_async_context(_memory_session(server))
        return {"calc": session}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open)

    client, captured = api
    _create_agent(client, options={"mcp_servers": ["calc"]})
    response = client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "what is 2+3?"}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "the sum is 5"
    assert body["docie_agent"]["agent"] == "tool-helper"
    assert body["docie_agent"]["kind"] == "custom"
    # Per-call tool trace (#261) — the "Try it" view's per-request detail.
    (call,) = body["docie_agent"]["tool_calls"]
    assert call["tool"] == "calc__add"
    assert call["status"] == "ok"
    assert isinstance(call["latency_ms"], int)
    # Full arguments/result ride the live trace (#262) for the "Try it" view.
    assert call["arguments"] == '{"a": 2, "b": 3}'
    assert call["result"] == "5"
    # Usage summed across both rounds, same contract as the chat surface.
    assert body["usage"] == {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}
    # Round 2 carries the executed exchange.
    round2 = json.loads(captured[-1].content)["messages"]
    assert round2[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "5"}


def test_custom_agent_injects_system_prompt_before_the_tool_loop(api, monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://unused")
        },
    )
    server = _calc_server()

    async def fake_open(stack, specs):
        session = await stack.enter_async_context(_memory_session(server))
        return {"calc": session}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open)

    client, captured = api
    _create_agent(
        client,
        options={"mcp_servers": ["calc"]},
        system_prompt="You are a calculator assistant.",
    )
    client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "what is 2+3?"}]},
    )
    first_round = json.loads(captured[0].content)["messages"]
    assert first_round[0] == {"role": "system", "content": "You are a calculator assistant."}


def test_custom_agent_unregistered_mcp_server_is_400(api) -> None:
    client, _ = api
    _create_agent(client, options={"mcp_servers": ["ghost"]})
    response = client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert "unregistered MCP server" in response.json()["error"]["message"]


def test_custom_agent_bad_mcp_servers_option_is_a_clear_error(api) -> None:
    client, _ = api
    _create_agent(client, options={"mcp_servers": "calc"})  # must be a list, not a bare string
    response = client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "invalid_agent_config"


# ── per-server tool allowlisting (options.mcp_tools) ─────────────────────────


@pytest.fixture
def api_with_calc(api, monkeypatch) -> tuple[TestClient, list[httpx.Request]]:
    """The calc server (add + subtract) wired to a fresh open_mcp_sessions,
    for every allowlist test below."""
    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://unused")
        },
    )
    server = _calc_server()

    async def fake_open(stack, specs):
        session = await stack.enter_async_context(_memory_session(server))
        return {"calc": session}

    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open)
    return api


def test_mcp_tools_allowlist_filters_advertised_tools(api_with_calc) -> None:
    client, captured = api_with_calc
    _create_agent(client, options={"mcp_servers": ["calc"], "mcp_tools": {"calc": ["add"]}})
    client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "what is 2+3?"}]},
    )
    sent_tools = json.loads(captured[0].content)["tools"]
    names = {t["function"]["name"] for t in sent_tools}
    assert names == {"calc__add"}  # subtract is NOT advertised


def test_mcp_tools_allowlist_naming_unknown_tool_is_500(api_with_calc) -> None:
    client, _ = api_with_calc
    _create_agent(
        client, options={"mcp_servers": ["calc"], "mcp_tools": {"calc": ["multiply"]}}
    )
    response = client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "invalid_agent_config"
    assert "multiply" in response.json()["error"]["message"]


def test_mcp_tools_allowlist_naming_unselected_server_is_500(api_with_calc) -> None:
    client, _ = api_with_calc
    _create_agent(
        client, options={"mcp_servers": ["calc"], "mcp_tools": {"other": ["add"]}}
    )
    response = client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500
    assert "not in options.mcp_servers" in response.json()["error"]["message"]


def test_mcp_tools_allowlist_malformed_is_500(api_with_calc) -> None:
    client, _ = api_with_calc
    _create_agent(
        client, options={"mcp_servers": ["calc"], "mcp_tools": ["calc"]}  # must be an object
    )
    response = client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "invalid_agent_config"


def test_no_mcp_tools_option_advertises_every_tool(api_with_calc) -> None:
    client, captured = api_with_calc
    _create_agent(client, options={"mcp_servers": ["calc"]})  # no allowlist -> unrestricted
    client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    names = {t["function"]["name"] for t in json.loads(captured[0].content)["tools"]}
    assert names == {"calc__add", "calc__subtract"}


# ── usage-ledger recording (#261: tool-call observability) ──────────────────


def test_agent_completion_records_a_usage_row_with_the_tool_trace(
    api_with_calc, monkeypatch
) -> None:
    from docie_bench.studio import usage_store

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(usage_store, "record_usage", lambda **kw: calls.append(kw) or True)
    client, _ = api_with_calc
    _create_agent(client, options={"mcp_servers": ["calc"]})
    response = client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "what is 2+3?"}]},
    )
    assert response.status_code == 200, response.text
    (row,) = calls
    assert row["deployment"] == "tool-helper"
    assert row["surface"] == "agent"
    assert row["status"] == "ok"
    assert row["prompt_tokens"] == 30
    assert row["completion_tokens"] == 12
    (call,) = row["tool_calls"]
    assert call["tool"] == "calc__add"
    assert call["status"] == "ok"
    # The ledger is aggregate stats, not a store for tool payloads (#262):
    # arguments/result are stripped before persisting.
    assert set(call) == {"tool", "status", "latency_ms"}


def test_agent_completion_without_tools_records_no_tool_call_trace(api, monkeypatch) -> None:
    from docie_bench.studio import usage_store

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(usage_store, "record_usage", lambda **kw: calls.append(kw) or True)
    client, _ = api
    _create_agent(client)  # no options.mcp_servers -> bare forward
    client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    (row,) = calls
    assert row["surface"] == "agent"
    assert row["status"] == "ok"
    assert row["tool_calls"] is None


def test_agent_error_records_a_usage_row_with_error_status(api, monkeypatch) -> None:
    from docie_bench.studio import usage_store

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(usage_store, "record_usage", lambda **kw: calls.append(kw) or True)
    client, _ = api
    _create_agent(client, options={"mcp_servers": ["ghost"]})  # unregistered -> 400
    response = client.post(
        "/v1/agents/tool-helper/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    (row,) = calls
    assert row["status"] == "error"
    assert row["prompt_tokens"] is None
    assert row["tool_calls"] is None
