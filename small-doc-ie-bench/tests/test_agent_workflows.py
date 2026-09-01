"""workflow-kind agents (#265): a fixed, ORDERED sequence of steps, each its
own backing model/prompt, run server-side in one request -- "prompt
chaining". Step 1 sees the caller's own messages; each later step sees ONLY
the previous step's answer, not the accumulated history.

A step may also carry a `route` (#266): a cheap classifier whose own answer
picks a NAMED next step instead of the next one in the list.
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
        {
            "step": 0,
            "name": "0",
            "model_profile": "alpha",
            "content": "echo: hello",
            "routed_to": None,
        },
        {
            "step": 1,
            "name": "1",
            "model_profile": "beta",
            "content": "echo: echo: hello",
            "routed_to": None,
        },
    ]


def test_workflow_stream_relays_each_step_completion_as_it_finishes(api) -> None:
    # #346: each step's completion is relayed as its own SSE event the
    # moment that step finishes -- not just the final step after the whole
    # multi-step chain completes silently.
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
        json={"stream": True, "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert response.text.strip().endswith("data: [DONE]")

    step_events = [e for e in events if e["type"] == "step"]
    assert step_events == [
        {
            "type": "step",
            "step": 0,
            "name": "0",
            "model_profile": "alpha",
            "content": "echo: hello",
            "routed_to": None,
        },
        {
            "type": "step",
            "step": 1,
            "name": "1",
            "model_profile": "beta",
            "content": "echo: echo: hello",
            "routed_to": None,
        },
    ]
    # Step 0's event arrives before step 1 even starts (only 1 upstream
    # request has been made by then) -- a real incremental relay, not the
    # whole chain collapsed into one frame at the end.
    (content_event,) = [e for e in events if e["type"] == "content"]
    completion = content_event["completion"]
    assert completion["choices"][0]["message"]["content"] == "echo: echo: hello"
    assert completion["usage"] == {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    assert completion["docie_agent"]["steps"] == [
        {k: v for k, v in e.items() if k != "type"} for e in step_events
    ]
    assert [e["type"] for e in events] == ["step", "step", "content"]


# ── conditional routing (#266): a step's own answer picks a NAMED next
# step instead of always falling through to the next one in the list ───────


def test_workflow_route_step_matches_a_messy_answer_and_skips_chaining_its_label(api) -> None:
    """The classifier's answer ("Label: billing", not the bare label
    "billing") still matches by substring -- and the step it routes to
    sees the ORIGINAL request text, never the classifier's own label."""
    client, captured = api

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        if body["model"] == "classifier":
            return httpx.Response(
                200,
                json=_completion(
                    "Label: billing",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                ),
            )
        last = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json=_completion(
                f"handled: {last}",
                usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            ),
        )

    configure_http_transport(httpx.MockTransport(handler))
    _create_workflow(
        client,
        [
            {
                "model_profile": "classifier",
                "route": {"routes": {"billing": "handle-billing"}},
            },
            {"name": "handle-billing", "model_profile": "billing-handler"},
        ],
    )
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "why was I charged twice?"}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "handled: why was I charged twice?"

    classify_req, handle_req = (json.loads(r.content) for r in captured)
    assert classify_req["messages"][-1]["content"] == "why was I charged twice?"
    # The target step sees the original text, NOT the classifier's own
    # "Label: billing" answer.
    assert handle_req["messages"][-1]["content"] == "why was I charged twice?"

    assert body["docie_agent"]["steps"] == [
        {
            "step": 0,
            "name": "0",
            "model_profile": "classifier",
            "content": "Label: billing",
            "routed_to": "handle-billing",
        },
        {
            "step": 1,
            "name": "handle-billing",
            "model_profile": "billing-handler",
            "content": "handled: why was I charged twice?",
            "routed_to": None,
        },
    ]


def test_workflow_route_falls_back_to_default_when_no_label_matches(api) -> None:
    client, captured = api

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        content = "no idea what this is" if body["model"] == "classifier" else "handled generically"
        return httpx.Response(
            200,
            json=_completion(
                content, usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            ),
        )

    configure_http_transport(httpx.MockTransport(handler))
    _create_workflow(
        client,
        [
            {
                "model_profile": "classifier",
                "route": {"routes": {"billing": "handle-billing"}, "default": "handle-other"},
            },
            {"name": "handle-billing", "model_profile": "billing-handler"},
            {"name": "handle-other", "model_profile": "generic-handler"},
        ],
    )
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "what time is it?"}]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "handled generically"


def test_workflow_route_step_with_no_match_and_no_default_is_unroutable(api) -> None:
    client, _ = api
    configure_http_transport(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=_completion(
                    "not a known label",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                ),
            )
        )
    )
    _create_workflow(
        client,
        [
            {"model_profile": "classifier", "route": {"routes": {"billing": "handle-billing"}}},
            {"name": "handle-billing", "model_profile": "billing-handler"},
        ],
    )
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "workflow_unroutable"


def test_workflow_route_target_must_be_a_real_step_name(api) -> None:
    client, _ = api
    _create_workflow(
        client,
        [{"model_profile": "classifier", "route": {"routes": {"billing": "does-not-exist"}}}],
    )
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "billing please"}]},
    )
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "invalid_agent_config"


def test_workflow_routing_loop_hits_the_step_budget(api) -> None:
    client, _ = api
    configure_http_transport(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=_completion(
                    "ping", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                ),
            )
        )
    )
    _create_workflow(
        client,
        [
            {"name": "a", "model_profile": "alpha", "route": {"routes": {"ping": "b"}}},
            {"name": "b", "model_profile": "beta", "route": {"routes": {"ping": "a"}}},
        ],
    )
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "workflow_budget_exhausted"


def test_workflow_rejects_a_duplicate_step_name(api) -> None:
    client, _ = api
    _create_workflow(
        client,
        [
            {"name": "dupe", "model_profile": "alpha"},
            {"name": "dupe", "model_profile": "beta"},
        ],
    )
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "invalid_agent_config"


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
    assert call["step"] == 0
    assert call["step_name"] == "0"


def test_workflow_tool_calls_are_tagged_with_the_step_that_made_them(api, monkeypatch) -> None:
    """A multi-step workflow where more than one step calls tools -- the
    trace must attribute each call to its OWN step, not flatten them into
    one unattributed list (#282-adjacent UI-visibility fix)."""
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

    def tool_call_response(args: str) -> httpx.Response:
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
                                    "function": {"name": "calc__add", "arguments": args},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        if any(m.get("role") == "tool" for m in body["messages"]):
            return httpx.Response(
                200,
                json=_completion(
                    "5", usage={"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27}
                ),
            )
        args = '{"a": 2, "b": 3}' if body["model"] == "alpha" else '{"a": 10, "b": 20}'
        return tool_call_response(args)

    configure_http_transport(httpx.MockTransport(handler))

    _create_workflow(
        client,
        [
            {"name": "first", "model_profile": "alpha", "mcp_servers": ["calc"]},
            {"name": "second", "model_profile": "beta", "mcp_servers": ["calc"]},
        ],
    )
    response = client.post(
        "/v1/agents/workflow-test/chat/completions",
        json={"messages": [{"role": "user", "content": "add some numbers"}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    call_1, call_2 = body["docie_agent"]["tool_calls"]
    assert call_1["step"] == 0
    assert call_1["step_name"] == "first"
    assert call_2["step"] == 1
    assert call_2["step_name"] == "second"
