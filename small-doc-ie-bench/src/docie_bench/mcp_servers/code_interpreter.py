"""Code-interpreter MCP server (#264): sandboxed Python execution via a
self-hosted Judge0 instance (docker-compose.yml's judge0-server/judge0-worker,
opt-in via `docker compose --profile sandbox up`). See
``docs/adr-agent-sandboxing.md`` for why Judge0 was picked over hand-rolling
a sandbox: `isolate`-based isolation, an actual persistent worker pool
(rather than a cold spawn per call), and a fixable, understood security
history rather than an unaudited one-off. Run:
``python -m docie_bench.mcp_servers.code_interpreter``.

Wired the same way every other MCP tool is (#259) — an agent opts in via
``options.mcp_servers: ["code-interpreter"]``, no new plumbing.
"""

from __future__ import annotations

import os
from typing import Any

URL_ENV = "DOCIE_MCP_CODE_INTERPRETER_URL"
TOKEN_ENV = "DOCIE_MCP_CODE_INTERPRETER_TOKEN"  # noqa: S105 - an env VAR NAME, not a credential
_DEFAULT_URL = "http://judge0-server:2358"
# "Python (3.8.1)" on the judge0/judge0 image — stable across judge0
# releases (a language's id doesn't change once assigned; verified against
# judge0's own /api/v2/languages docs, not assumed).
_PYTHON_LANGUAGE_ID = 71
_CPU_TIME_LIMIT_SECONDS = 5.0
_WALL_TIME_LIMIT_SECONDS = 10.0
_MEMORY_LIMIT_KB = 256_000
# Judge0 status id for "Time Limit Exceeded" (verified against judge0's own
# status list: 1 In Queue, 2 Processing, 3 Accepted, 4 Wrong Answer, 5 Time
# Limit Exceeded, 6 Compilation Error, 7-12 Runtime Error variants, 13
# Internal Error, 14 Exec Format Error).
_STATUS_TIME_LIMIT_EXCEEDED = 5


class CodeInterpreterUnavailableError(RuntimeError):
    """No auth token configured — see docs/adr-agent-sandboxing.md. Fails
    CLOSED: never submits code to an unauthenticated Judge0 instance, the
    same shape as ``mcp_tools.MCPUnavailableError`` for the optional ``mcp``
    SDK dependency."""


def _require_token() -> str:
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise CodeInterpreterUnavailableError(
            f"code_interpreter needs {TOKEN_ENV} set — register it with the "
            "same token value as JUDGE0_AUTH_TOKEN in .env when enabling "
            "this server. Refusing to submit code to an unauthenticated "
            "Judge0 instance."
        )
    return token


def submit_code(code: str, *, url: str | None = None) -> dict[str, Any]:
    """Submit ``code`` to Judge0 for sandboxed execution and return
    ``{stdout, stderr, exit_code, timed_out, truncated}``. Never raises for
    the SNIPPET's own failures — a traceback in stderr or a nonzero exit
    code is a normal, informative outcome — only for a missing token
    (:class:`CodeInterpreterUnavailableError`) or a Judge0 request itself
    failing (connection error, non-2xx response).
    """
    import httpx

    token = _require_token()
    base_url = url if url is not None else os.environ.get(URL_ENV, _DEFAULT_URL)
    response = httpx.post(
        f"{base_url}/submissions",
        params={"wait": "true", "base64_encoded": "false"},
        headers={"X-Auth-Token": token},
        json={
            "source_code": code,
            "language_id": _PYTHON_LANGUAGE_ID,
            "cpu_time_limit": _CPU_TIME_LIMIT_SECONDS,
            "wall_time_limit": _WALL_TIME_LIMIT_SECONDS,
            "memory_limit": _MEMORY_LIMIT_KB,
        },
        timeout=_WALL_TIME_LIMIT_SECONDS + 5.0,
    )
    response.raise_for_status()
    result = response.json()
    status = result.get("status") or {}
    return {
        "stdout": result.get("stdout") or "",
        "stderr": result.get("stderr") or result.get("compile_output") or "",
        "exit_code": result.get("exit_code"),
        "timed_out": status.get("id") == _STATUS_TIME_LIMIT_EXCEEDED,
        # Judge0 enforces its own response size limits server-side; nothing
        # extra to truncate on this side.
        "truncated": False,
    }


def build_server() -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("docie-code-interpreter")

    @server.tool()
    def run_python(code: str) -> dict[str, Any]:
        """Run a short Python snippet in an isolated sandbox (Judge0: no
        network, no persistent filesystem, resource- and time-limited). No
        state persists across calls and no packages can be installed.
        Returns {stdout, stderr, exit_code, timed_out, truncated}. Use
        print() to produce output — a bare expression's value is not
        captured."""
        return submit_code(code)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
