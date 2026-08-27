"""Static catalog of first-party MCP servers; enabling an entry materializes
a stdio registry entry in configs/mcp-servers.json (see mcp_tools).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CatalogParam:
    name: str
    env_var: str
    description: str
    required: bool = False


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    title: str
    description: str
    module: str
    tools: tuple[str, ...]
    params: tuple[CatalogParam, ...] = field(default_factory=tuple)


CATALOG: dict[str, CatalogEntry] = {
    entry.name: entry
    for entry in (
        CatalogEntry(
            name="calculator",
            title="Calculator",
            description=(
                "Exact arithmetic: evaluate expressions and verify sums. Aimed "
                "squarely at the totals small models miscompute — the model "
                "reads the line items, the tool does the math."
            ),
            module="docie_bench.mcp_servers.calculator",
            tools=("calc", "sum_check"),
        ),
        CatalogEntry(
            name="dates",
            title="Dates",
            description=(
                "Parse dates in any common format to ISO 8601, compute exact "
                "day differences, and tell the model today's date."
            ),
            module="docie_bench.mcp_servers.dates",
            tools=("parse_date", "date_diff", "today"),
        ),
        CatalogEntry(
            name="web-fetch",
            title="Web fetch",
            description=(
                "HTTP GET restricted to an operator-set host allowlist (deny "
                "by default, redirects never followed). Lets the model ground "
                "an answer in a page you explicitly allow."
            ),
            module="docie_bench.mcp_servers.web_fetch",
            tools=("fetch",),
            params=(
                CatalogParam(
                    name="allowed_hosts",
                    env_var="DOCIE_MCP_FETCH_ALLOWED_HOSTS",
                    description=(
                        "Comma-separated hostnames the fetch tool may reach "
                        "('*' allows every host). Empty = every fetch refused."
                    ),
                ),
            ),
        ),
        CatalogEntry(
            name="docs-search",
            title="Document Search",
            description=(
                "Agentic RAG demo: list, read, and search a shared directory "
                "of PDFs/text files (parsed via liteparse, the same PDF "
                "backend the rest of the platform uses). Built to prove out "
                "small models — search first, then answer from what was "
                "actually found."
            ),
            module="docie_bench.mcp_servers.docs_search",
            tools=("list_files", "read_document", "search_text"),
            params=(
                CatalogParam(
                    name="docs_dir",
                    env_var="DOCIE_MCP_DOCS_SEARCH_DIR",
                    description=(
                        "Directory of documents this server may read (relative "
                        "paths resolve against the server process's cwd). "
                        "Defaults to 'data/agent-docs' if left blank — drop "
                        "PDFs/text files there for the agent to search."
                    ),
                ),
                CatalogParam(
                    name="backend",
                    env_var="DOCIE_MCP_DOCS_SEARCH_BACKEND",
                    description=(
                        "Retrieval strategy (see docs_search.SearchBackend). "
                        "Only 'substring' is implemented today; defaults to "
                        "it if left blank. An unrecognized value is a config "
                        "error the first time search_text is called."
                    ),
                ),
            ),
        ),
        CatalogEntry(
            name="code-interpreter",
            title="Code Interpreter",
            description=(
                "Sandboxed Python execution via a self-hosted Judge0 instance "
                "(docker-compose.yml's judge0-server, start with "
                "`docker compose --profile sandbox up`) — lets an agent "
                "actually run a snippet instead of reasoning about it. See "
                "docs/adr-agent-sandboxing.md. Refuses to run without a "
                "matching auth token; never falls back to running code "
                "unsandboxed."
            ),
            module="docie_bench.mcp_servers.code_interpreter",
            tools=("run_python",),
            params=(
                CatalogParam(
                    name="url",
                    env_var="DOCIE_MCP_CODE_INTERPRETER_URL",
                    description=(
                        "Judge0 base URL. Defaults to "
                        "'http://judge0-server:2358' (the compose-network "
                        "address of the judge0-server service) if left blank."
                    ),
                ),
                CatalogParam(
                    name="token",
                    env_var="DOCIE_MCP_CODE_INTERPRETER_TOKEN",
                    description=(
                        "Auth token — must match JUDGE0_AUTH_TOKEN in .env. "
                        "Required: the tool refuses to run without it rather "
                        "than submitting code to an unauthenticated Judge0."
                    ),
                    required=True,
                ),
            ),
        ),
    )
}


def registry_entry_for(entry: CatalogEntry, params: dict[str, str]) -> dict[str, object]:
    env = {
        param.env_var: params[param.name] for param in entry.params if params.get(param.name)
    }
    record: dict[str, object] = {
        "transport": "stdio",
        "command": ["python", "-m", entry.module],
        "catalog": entry.name,
    }
    if env:
        record["env"] = env
    return record
