"""MCP tool sources for served chat models (docie_bench.mcp_tools).

The MCP side of every test is REAL: an in-process ``MCPServer`` wired to a
``ClientSession`` over ``mcp.shared.memory`` streams — the same protocol
handshake, tool listing, and call/result framing a remote server would
speak. Only the LLM upstream is scripted (a fake ``post`` for the loop
unit tests, ``httpx.MockTransport`` for the route-level test), because the
model's half of the exchange is exactly what each test needs to control.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.client.session import ClientSession
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from mcp.shared.memory import create_client_server_memory_streams

from docie_bench import mcp_tools
from docie_bench.mcp_tools import (
    TOOL_DISCIPLINE_DIRECTIVE,
    TRACE_TEXT_LIMIT,
    MCPConfigError,
    MCPServerSpec,
    collect_openai_tools,
    execute_tool_call,
    load_mcp_registry,
    make_trace_recorder,
    run_tool_loop,
)


def _calc_server() -> MCPServer:
    server = MCPServer("calc")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def crash(reason: str) -> str:
        """Always fails."""
        raise ValueError(f"boom: {reason}")

    return server


@asynccontextmanager
async def _memory_session(server: MCPServer):
    """A real ClientSession talking to ``server`` over in-memory streams."""
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


# ---------------------------------------------------------------- registry


def test_registry_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_mcp_registry(tmp_path / "nope.json") == {}


def test_registry_parses_both_transports(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "docs": {
                        "transport": "streamable-http",
                        "url": "http://mcp.internal/mcp",
                        "headers": {"Authorization": "Bearer x"},
                    },
                    "fs": {"transport": "stdio", "command": ["npx", "server-fs", "/data"]},
                }
            }
        ),
        encoding="utf-8",
    )
    registry = load_mcp_registry(path)
    assert registry["docs"].transport == "streamable-http"
    assert registry["docs"].url == "http://mcp.internal/mcp"
    assert registry["docs"].headers == {"Authorization": "Bearer x"}
    assert registry["fs"].command == ("npx", "server-fs", "/data")


@pytest.mark.parametrize(
    "payload",
    [
        "[]",  # not an object
        '{"servers": {"x": {"transport": "websocket"}}}',  # unknown transport
        '{"servers": {"x": {"transport": "streamable-http"}}}',  # http without url
        '{"servers": {"x": {"transport": "stdio", "command": []}}}',  # empty command
        "not json at all",
    ],
)
def test_registry_rejects_malformed_config(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(MCPConfigError):
        load_mcp_registry(path)


# ------------------------------------------------- tool listing + execution


async def test_collect_openai_tools_qualifies_names_and_passes_schema() -> None:
    async with _memory_session(_calc_server()) as session:
        tools, mapping = await collect_openai_tools({"calc": session})
    names = {t["function"]["name"] for t in tools}
    assert names == {"calc__add", "calc__crash"}
    assert mapping["calc__add"] == ("calc", "add")
    add_schema = next(t for t in tools if t["function"]["name"] == "calc__add")["function"][
        "parameters"
    ]
    # MCP input_schema is JSON Schema and must reach OpenAI 'parameters' verbatim.
    assert add_schema["type"] == "object"
    assert set(add_schema["required"]) == {"a", "b"}
    assert add_schema["properties"]["a"]["type"] == "integer"


async def test_execute_tool_call_success_error_and_bad_args() -> None:
    mapping = {"calc__add": ("calc", "add"), "calc__crash": ("calc", "crash")}
    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        assert await execute_tool_call(sessions, mapping, "calc__add", '{"a": 2, "b": 3}') == "5"
        # A tool failure comes back as TEXT for the model to reason about,
        # never as an exception that kills the chat request.
        failed = await execute_tool_call(sessions, mapping, "calc__crash", '{"reason": "x"}')
        assert failed.startswith("error:")
        bad = await execute_tool_call(sessions, mapping, "calc__add", "{not json")
        assert bad.startswith("error: tool arguments were not valid JSON")


# ------------------------------------------------------------------ the loop


def _tool_calls_completion(name: str, arguments: str, usage: dict[str, int]) -> dict[str, Any]:
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
        "usage": usage,
    }


def _final_completion(content: str, usage: dict[str, int]) -> dict[str, Any]:
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
        "usage": usage,
    }


async def test_run_tool_loop_executes_and_sums_usage() -> None:
    posted: list[dict[str, Any]] = []
    responses = [
        _tool_calls_completion(
            "calc__add",
            '{"a": 2, "b": 3}',
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
        _final_completion(
            "sum is 5", {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27}
        ),
    ]

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        posted.append(json.loads(json.dumps(body)))
        return responses[len(posted) - 1]

    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {"model": "m", "messages": [{"role": "user", "content": "2+3?"}]}
        completion = await run_tool_loop(post, body, sessions, mapping, tools, max_iterations=4)

    assert completion["choices"][0]["message"]["content"] == "sum is 5"
    # Usage is the WHOLE exchange, not the last round.
    assert completion["usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "total_tokens": 42,
    }
    # Round 2 upstream request carries the executed exchange: the assistant's
    # tool_calls message followed by the role:"tool" result.
    round2 = posted[1]["messages"]
    assert round2[-2]["tool_calls"][0]["function"]["name"] == "calc__add"
    assert round2[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "5"}
    # Both rounds advertised the MCP tools.
    assert {t["function"]["name"] for t in posted[0]["tools"]} == {"calc__add", "calc__crash"}


def test_trace_recorder_clips_at_the_raised_limit_not_the_old_4000() -> None:
    # #284: a docs-search read_document/search_text result on a real PDF
    # routinely exceeded the old 4000-char cap and got silently clipped.
    trace: list[dict] = []
    record = make_trace_recorder(trace)
    long_result = "x" * (TRACE_TEXT_LIMIT + 500)

    record("docs-search__read_document", True, 1, {}, long_result)

    assert TRACE_TEXT_LIMIT > 4000
    assert len(trace[0]["result"]) < len(long_result)
    assert trace[0]["result"].startswith("x" * TRACE_TEXT_LIMIT)


async def test_run_tool_loop_inserts_the_directive_when_no_system_message_exists() -> None:
    posted: list[dict[str, Any]] = []

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        posted.append(json.loads(json.dumps(body)))
        return _final_completion("hi", {})

    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        await run_tool_loop(post, body, sessions, mapping, tools, max_iterations=4)

    messages = posted[0]["messages"]
    assert messages[0] == {"role": "system", "content": TOOL_DISCIPLINE_DIRECTIVE}
    assert messages[1] == {"role": "user", "content": "hi"}


async def test_run_tool_loop_appends_the_directive_to_an_existing_system_message() -> None:
    posted: list[dict[str, Any]] = []

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        posted.append(json.loads(json.dumps(body)))
        return _final_completion("hi", {})

    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
        }
        await run_tool_loop(post, body, sessions, mapping, tools, max_iterations=4)

    messages = posted[0]["messages"]
    assert sum(1 for m in messages if m["role"] == "system") == 1
    assert messages[0] == {
        "role": "system",
        "content": f"You are helpful.\n\n{TOOL_DISCIPLINE_DIRECTIVE}",
    }


async def test_run_tool_loop_eagerly_lists_docs_search_files_before_the_first_round(
    tmp_path: Path, monkeypatch
) -> None:
    # docs-search's catalog entry declares `eager_list_tool="list_files"` --
    # registered under that exact name so the CATALOG lookup in
    # `_eager_list_context` matches.
    from docie_bench.mcp_servers import docs_search

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello")
    monkeypatch.setenv(docs_search.DOCS_DIR_ENV, str(docs))

    posted: list[dict[str, Any]] = []
    traced: list[tuple[str, bool, int, Any, str]] = []

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        posted.append(json.loads(json.dumps(body)))
        return _final_completion("hi", {})

    async with _memory_session(docs_search.build_server()) as session:
        sessions = {"docs-search": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        await run_tool_loop(
            post,
            body,
            sessions,
            mapping,
            tools,
            max_iterations=4,
            on_tool_call=lambda name, ok, latency_ms, args, result: traced.append(
                (name, ok, latency_ms, args, result)
            ),
        )

    system_content = posted[0]["messages"][0]["content"]
    assert "a.txt" in system_content
    # Reported through on_tool_call exactly like a model-issued call, so the
    # trace shows it happened rather than it running invisibly.
    (call,) = traced
    assert call[0] == "docs-search__list_files"
    assert call[1] is True
    assert "a.txt" in call[4]


async def test_run_tool_loop_skips_eager_listing_when_no_catalog_entry_declares_one() -> None:
    posted: list[dict[str, Any]] = []

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        posted.append(json.loads(json.dumps(body)))
        return _final_completion("hi", {})

    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        await run_tool_loop(post, body, sessions, mapping, tools, max_iterations=4)

    # Only the tool-discipline directive, no eager-listing text appended --
    # `calc` has no `eager_list_tool` in the catalog.
    assert posted[0]["messages"][0]["content"] == TOOL_DISCIPLINE_DIRECTIVE


async def test_run_tool_loop_reports_each_call_through_on_tool_call() -> None:
    responses = [
        _tool_calls_completion("calc__add", '{"a": 2, "b": 3}', {}),
        _final_completion("sum is 5", {}),
    ]

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        return responses.pop(0)

    traced: list[tuple[str, bool, int, Any, str]] = []

    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {"model": "m", "messages": [{"role": "user", "content": "2+3?"}]}
        await run_tool_loop(
            post,
            body,
            sessions,
            mapping,
            tools,
            max_iterations=4,
            on_tool_call=lambda name, ok, latency_ms, args, result: traced.append(
                (name, ok, latency_ms, args, result)
            ),
        )

    (call,) = traced
    name, ok, latency_ms, arguments, result = call
    assert name == "calc__add"
    assert ok is True
    assert isinstance(latency_ms, int)
    assert latency_ms >= 0
    assert arguments == '{"a": 2, "b": 3}'
    assert result == "5"


async def test_run_tool_loop_reports_reasoning_content_for_every_round() -> None:
    # LFM2.5's bundled chat template opens <think> unconditionally --
    # llama-server (with --jinja) surfaces it as reasoning_content SEPARATE
    # from content/tool_calls. Fires for the tool-calling round AND the
    # final round, since "is there a hidden thinking step" applies to both.
    tool_round = _tool_calls_completion("calc__add", '{"a": 2, "b": 3}', {})
    tool_round["choices"][0]["message"]["reasoning_content"] = "the user wants 2+3, call calc.add"
    final_round = _final_completion("sum is 5", {})
    final_round["choices"][0]["message"]["reasoning_content"] = "I have the answer, respond"
    responses = [tool_round, final_round]

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        return responses.pop(0)

    reasoning: list[str] = []

    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {"model": "m", "messages": [{"role": "user", "content": "2+3?"}]}
        await run_tool_loop(
            post, body, sessions, mapping, tools, max_iterations=4, on_reasoning=reasoning.append
        )

    assert reasoning == ["the user wants 2+3, call calc.add", "I have the answer, respond"]


async def test_run_tool_loop_reports_system_addendum_exactly_once() -> None:
    # Fired once, right after _with_system_addendum runs, before the first
    # round -- not once per round like on_reasoning/on_tool_call.
    responses = [
        _tool_calls_completion("calc__add", '{"a": 2, "b": 3}', {}),
        _final_completion("sum is 5", {}),
    ]

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        return responses.pop(0)

    addenda: list[str] = []

    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {"model": "m", "messages": [{"role": "user", "content": "2+3?"}]}
        await run_tool_loop(
            post,
            body,
            sessions,
            mapping,
            tools,
            max_iterations=4,
            on_system_addendum=addenda.append,
        )

    assert addenda == [TOOL_DISCIPLINE_DIRECTIVE]


async def test_run_tool_loop_system_addendum_matches_the_folded_system_message(
    tmp_path: Path, monkeypatch
) -> None:
    # The reported addendum must be the exact text folded into the request
    # (directive + eager-list context), not a placeholder -- assert it
    # against what the upstream `post` actually received.
    from docie_bench.mcp_servers import docs_search

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello")
    monkeypatch.setenv(docs_search.DOCS_DIR_ENV, str(docs))

    posted: list[dict[str, Any]] = []
    addenda: list[str] = []

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        posted.append(json.loads(json.dumps(body)))
        return _final_completion("hi", {})

    async with _memory_session(docs_search.build_server()) as session:
        sessions = {"docs-search": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        await run_tool_loop(
            post,
            body,
            sessions,
            mapping,
            tools,
            max_iterations=4,
            on_system_addendum=addenda.append,
        )

    (addendum,) = addenda
    assert posted[0]["messages"][0]["content"] == addendum
    assert "a.txt" in addendum


# ---------------------------------------------------------- docs-search resources (#332)


async def test_docs_search_list_resources_has_one_resource_per_file(
    tmp_path: Path, monkeypatch
) -> None:
    from docie_bench.mcp_servers import docs_search

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello")
    (docs / "b.txt").write_text("world")
    # Nested dir with a space, same as list_files (rglob) already handles --
    # e.g. a mcp_session_documents (#296) upload named by a user.
    (docs / "sub dir").mkdir()
    (docs / "sub dir" / "My Invoice.txt").write_text("nested")
    monkeypatch.setenv(docs_search.DOCS_DIR_ENV, str(docs))

    async with _memory_session(docs_search.build_server()) as session:
        result = await session.list_resources()

    by_uri = {str(r.uri): r for r in result.resources}
    assert set(by_uri) == {
        "docs://a.txt",
        "docs://b.txt",
        "docs://sub dir/My Invoice.txt",
    }
    assert by_uri["docs://a.txt"].name == "a.txt"
    assert by_uri["docs://a.txt"].mime_type == "text/plain"
    assert by_uri["docs://sub dir/My Invoice.txt"].name == "sub dir/My Invoice.txt"


async def test_docs_search_read_resource_returns_full_document_text(
    tmp_path: Path, monkeypatch
) -> None:
    from docie_bench.mcp_servers import docs_search

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("total due: 42 EUR")
    monkeypatch.setenv(docs_search.DOCS_DIR_ENV, str(docs))

    async with _memory_session(docs_search.build_server()) as session:
        result = await session.read_resource("docs://a.txt")

    (content,) = result.contents
    assert content.text == "total due: 42 EUR"
    assert content.mime_type == "text/plain"


async def test_docs_search_read_resource_unknown_uri_raises_mcp_error(
    tmp_path: Path, monkeypatch
) -> None:
    from docie_bench.mcp_servers import docs_search

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("hello")
    monkeypatch.setenv(docs_search.DOCS_DIR_ENV, str(docs))

    async with _memory_session(docs_search.build_server()) as session:
        with pytest.raises(MCPError, match="Unknown resource"):
            await session.read_resource("docs://nope.txt")


async def test_run_tool_loop_reports_per_round_and_cumulative_usage() -> None:
    # on_usage fires once per round with that round's OWN usage plus the
    # RUNNING cumulative totals through that round -- round 2's cumulative
    # must include round 1, not just repeat round 1's own numbers.
    responses = [
        _tool_calls_completion(
            "calc__add",
            '{"a": 2, "b": 3}',
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
        _final_completion(
            "sum is 5", {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27}
        ),
    ]

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        return responses.pop(0)

    usage_events: list[dict[str, Any]] = []

    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {"model": "m", "messages": [{"role": "user", "content": "2+3?"}]}
        await run_tool_loop(
            post, body, sessions, mapping, tools, max_iterations=4, on_usage=usage_events.append
        )

    assert usage_events == [
        {
            "round": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "cumulative": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        {
            "round": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
            "cumulative": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        },
    ]


async def test_run_tool_loop_returns_caller_owned_calls_untouched() -> None:
    # The model calling a tool the CALLER advertised (not an MCP one) ends the
    # server-side loop: executing the caller's function is the caller's job.
    scripted = _tool_calls_completion(
        "callers_own_lookup",
        '{"q": "x"}',
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    )

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        return scripted

    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "callers_own_lookup", "parameters": {"type": "object"}},
                }
            ],
        }
        completion = await run_tool_loop(post, body, sessions, mapping, tools, max_iterations=4)

    calls = completion["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "callers_own_lookup"


async def test_run_tool_loop_exhaustion_returns_none() -> None:
    count = 0

    async def post(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal count
        count += 1
        return _tool_calls_completion(
            "calc__add",
            '{"a": 1, "b": 1}',
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async with _memory_session(_calc_server()) as session:
        sessions = {"calc": session}
        tools, mapping = await collect_openai_tools(sessions)
        body = {"model": "m", "messages": [{"role": "user", "content": "loop"}]}
        result = await run_tool_loop(post, body, sessions, mapping, tools, max_iterations=3)

    assert result is None
    assert count == 3  # the bound is exact, not off-by-one


# ------------------------------------------------------------ route (end-to-end)


def test_chat_route_runs_the_full_mcp_exchange(monkeypatch) -> None:
    """POST /v1/chat/completions with mcp_servers: real MCP server in memory,
    scripted llama-style upstream via MockTransport, final answer + summed
    usage out — the whole serving path, not just the loop function."""
    from docie_bench.agents.api import configure_http_transport
    from docie_bench.chat_api import router as chat_router
    from docie_bench.llm.model_profiles import ModelProfile

    upstream_profile = ModelProfile(
        name="lfm2.5-350m", model="lfm2.5-350m-served", base_url="http://upstream/v1", api_key="k"
    )
    monkeypatch.setattr(
        "docie_bench.chat_api.resolve_extraction_profile",
        lambda **_: upstream_profile,
    )
    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_registry",
        lambda path=None: {
            "calc": MCPServerSpec(name="calc", transport="streamable-http", url="http://unused")
        },
    )

    server = _calc_server()

    async def fake_open(stack, specs):
        assert [spec.name for spec in specs] == ["calc"]
        session = await stack.enter_async_context(_memory_session(server))
        return {"calc": session}

    # The transport seam is the ONLY faked MCP piece: the registry points at a
    # URL, but the session comes from the in-memory pair — everything after
    # connect (initialize, list_tools, call_tool) is the real protocol.
    monkeypatch.setattr(mcp_tools, "open_mcp_sessions", fake_open)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "mcp_servers" not in body  # never leaks upstream
        has_tool_result = any(m.get("role") == "tool" for m in body["messages"])
        if not has_tool_result:
            return httpx.Response(
                200,
                json=_tool_calls_completion(
                    "calc__add",
                    '{"a": 19, "b": 23}',
                    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                ),
            )
        result = next(m["content"] for m in body["messages"] if m.get("role") == "tool")
        return httpx.Response(
            200,
            json=_final_completion(
                f"the answer is {result}",
                {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26},
            ),
        )

    configure_http_transport(httpx.MockTransport(handler))
    try:
        app = FastAPI()
        app.include_router(chat_router)
        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "lfm2.5-350m",
                "messages": [{"role": "user", "content": "what is 19+23?"}],
                "mcp_servers": ["calc"],
            },
        )
    finally:
        configure_http_transport(None)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "the answer is 42"
    assert payload["usage"] == {"prompt_tokens": 30, "completion_tokens": 11, "total_tokens": 41}
