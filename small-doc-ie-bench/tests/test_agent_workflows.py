"""workflow-kind agents (#265): a fixed, ORDERED sequence of steps, each its
own backing model/prompt, run server-side in one request -- "prompt
chaining". Step 1 sees the caller's own messages; each later step sees ONLY
the previous step's answer, not the accumulated history.
"""

from __future__ import annotations

import json
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


def _completion(content: str, *, usage: dict[str, int]) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


@pytest.fixture
def api(tmp_path, monkeypatch) -> tuple[TestClient, list[httpx.Request]]:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        return replace(UPSTREAM, name=str(model_profile), model=str(model_profile))

    monkeypatch.setattr("docie_bench.agents.runtime.resolve_extraction_profile", fake_resolver)

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json=_completion(
                f"echo: {last}",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
        )

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(agents_router)
    client = TestClient(app)
    yield client, captured
    configure_http_transport(None)


def _create_workflow(client: TestClient, steps: list[dict[str, Any]], **overrides: object) -> None:
    payload: dict[str, object] = {
        "name": "workflow-test",
        "template": "workflow-agent",
        "options": {"steps": steps},
    }
    payload.update(overrides)
    response = client.post("/v1/agents", json=payload)
    assert response.status_code == 201, response.text


def test_workflow_chains_steps_each_seeing_only_the_previous_answer(api) -> None:
    client, captured = api
    _create_workflow(
        client,
        [
            {"model_profile": "alpha", "system_prompt": "step one"},
            {"model_profile": "beta", "system_prompt": "step two"},
        ],
    )
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "echo: echo: hello"
    # Usage summed across both steps, same contract as the MCP tool loop.
    assert body["usage"] == {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}

    step1_req, step2_req = (json.loads(r.content) for r in captured)
    assert step1_req["model"] == "alpha"
    assert step1_req["messages"] == [
        {"role": "system", "content": "step one"},
        {"role": "user", "content": "hello"},
    ]
    # Step 2 sees ONLY step 1's answer -- not the original user message, not
    # both messages accumulated.
    assert step2_req["model"] == "beta"
    assert step2_req["messages"] == [
        {"role": "system", "content": "step two"},
        {"role": "user", "content": "echo: hello"},
    ]

    assert body["docie_agent"]["steps"] == [
        {"step": 0, "model_profile": "alpha", "content": "echo: hello"},
        {"step": 1, "model_profile": "beta", "content": "echo: echo: hello"},
    ]


def test_workflow_requires_a_non_empty_steps_list(api) -> None:
    client, _ = api
    _create_workflow(client, [])
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "invalid_agent_config"


def test_workflow_rejects_a_step_without_a_model_profile(api) -> None:
    client, _ = api
    _create_workflow(client, [{"system_prompt": "no model"}])
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500
    assert "model_profile" in response.json()["error"]["message"]


def test_workflow_rejects_malformed_steps_option(api) -> None:
    client, _ = api
    _create_workflow(client, "not-a-list")  # type: ignore[arg-type]
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "invalid_agent_config"


# ── a step may use its own MCP tools (#259), same as a custom agent ─────────


def _calc_server() -> MCPServer:
    server = MCPServer("calc")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    return server


@asynccontextmanager
async def _memory_session(server: MCPServer):
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        low = server._lowlevel_server
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: low.run(*server_streams, low.create_initialization_options())
            )
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                yield session
            tg.cancel_scope.cancel()


def test_workflow_step_can_use_its_own_mcp_tools(api, monkeypatch) -> None:
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

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        has_tool_result = any(m.get("role") == "tool" for m in body["messages"])
        if not has_tool_result:
            return httpx.Response(
                200,
                json={
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
                                        "function": {
                                            "name": "calc__add",
                                            "arguments": '{"a": 2, "b": 3}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
        return httpx.Response(
            200,
            json=_completion(
                "the sum is 5",
                usage={"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
            ),
        )

    configure_http_transport(httpx.MockTransport(handler))

    _create_workflow(client, [{"model_profile": "alpha", "mcp_servers": ["calc"]}])
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "what is 2+3?"}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "the sum is 5"
    (call,) = body["docie_agent"]["tool_calls"]
    assert call["tool"] == "calc__add"
