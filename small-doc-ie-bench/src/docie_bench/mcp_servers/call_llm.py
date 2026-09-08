"""call_llm MCP server (#285): dispatch a scoped sub-task to another model.

Run: ``python -m docie_bench.mcp_servers.call_llm``.

Recommendation (a) from #285: a single upstream chat completion, no tools,
no recursion. An agent's tool loop hands a raw dump (a search_text/
read_document result, say) to a separate, typically cheap model whose only
job is answering ONE question about it -- instead of stuffing that dump
into the orchestrating model's own context.

This process is spawned as its own subprocess (same as every other catalog
server) and has no in-process access to the api's resolver/auth -- it loops
back over HTTP to docie's OWN ``/v1/chat/completions`` instead of talking to
a runtime endpoint directly. The target is picked by ``model_profile`` and
resolved through the exact same path any other caller uses, so it gets the
same cold-start-retry, usage-ledger, and mojibake-fix behavior for free
rather than reimplementing any of it here.
"""

from __future__ import annotations

import os
from typing import Any

API_BASE_ENV = "DOCIE_MCP_CALL_LLM_API_BASE"
API_KEY_ENV = "DOCIE_MCP_CALL_LLM_API_KEY"
DEFAULT_PROFILE_ENV = "DOCIE_MCP_CALL_LLM_DEFAULT_PROFILE"
# Correct when this server runs in the same container as the api process
# (the common case: it's spawned BY that process) -- see settings.api_port.
_DEFAULT_API_BASE = "http://127.0.0.1:8080"
_TIMEOUT_SECONDS = 60.0
_ERROR_BODY_CHARS = 500


def _resolve_profile(model_profile: str | None) -> str:
    profile = model_profile or os.environ.get(DEFAULT_PROFILE_ENV)
    if not profile:
        raise ValueError(
            f"no model_profile given and {DEFAULT_PROFILE_ENV} is not set -- the "
            "operator must configure a default, or every call must pass "
            "model_profile explicitly"
        )
    return profile


async def dispatch(task: str, context: str, model_profile: str | None = None) -> str:
    import httpx

    if not task.strip():
        raise ValueError("task must not be empty")
    profile = _resolve_profile(model_profile)
    base = os.environ.get(API_BASE_ENV, _DEFAULT_API_BASE).rstrip("/")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(API_KEY_ENV)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    content = f"{task}\n\n---\n{context}" if context.strip() else task
    body = {"model": profile, "messages": [{"role": "user", "content": content}]}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{base}/v1/chat/completions", json=body, headers=headers
            )
    except httpx.HTTPError as exc:
        raise ValueError(f"call_llm request failed: {exc}") from exc
    if response.status_code >= 400:
        raise ValueError(
            f"call_llm sub-request failed: HTTP {response.status_code}: "
            f"{response.text[:_ERROR_BODY_CHARS]}"
        )
    data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices or not isinstance(choices, list):
        raise ValueError(f"call_llm sub-request returned no choices: {data!r}"[:_ERROR_BODY_CHARS])
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def build_server() -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("docie-call-llm")

    @server.tool()
    async def call_llm(task: str, context: str = "", model_profile: str | None = None) -> str:
        """Dispatch a scoped sub-task to another model instead of reasoning
        about a large context dump yourself. `task` is the question to
        answer; `context` is the raw material (e.g. a search_text/
        read_document result) to answer it from. Returns the sub-model's
        plain-text answer. One completion -- no tools, no recursion."""
        return await dispatch(task, context, model_profile)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
