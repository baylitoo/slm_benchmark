"""call_llm MCP server: dispatch a scoped sub-task to another model (#285)."""

from __future__ import annotations

import httpx
import pytest

from docie_bench.mcp_servers import call_llm


async def test_dispatch_posts_task_and_context_and_returns_the_answer(monkeypatch) -> None:
    monkeypatch.setenv(call_llm.API_KEY_ENV, "k")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "3 pages"}}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )

    answer = await call_llm.dispatch(
        "how many pages?", "page 1\npage 2\npage 3", model_profile="lfm2.5-350m"
    )

    assert answer == "3 pages"
    body = captured[0].content
    assert b'"model": "lfm2.5-350m"' in body or b'"model":"lfm2.5-350m"' in body
    assert captured[0].headers["Authorization"] == "Bearer k"


async def test_dispatch_requires_a_model_profile_when_no_default_is_set() -> None:
    with pytest.raises(ValueError, match="no model_profile given"):
        await call_llm.dispatch("task", "context")
