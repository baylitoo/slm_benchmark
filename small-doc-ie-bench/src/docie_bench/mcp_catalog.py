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
    # A zero-argument tool name (one of ``tools``) that lists this server's
    # addressable identifiers (docs-search's `list_files`). When set,
    # mcp_tools.run_tool_loop calls it once up front and folds the real
    # listing into context -- the model starts every request already
    # knowing valid identifiers instead of discovering them (or inventing
    # them) via a tool call it may skip.
    eager_list_tool: str | None = None


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
            eager_list_tool="list_files",
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
                CatalogParam(
                    name="snippet_window",
                    env_var="DOCIE_MCP_DOCS_SEARCH_SNIPPET_WINDOW",
                    description=(
                        "Characters of context kept on each side of a "
                        "search_text match. Defaults to 400 if left blank -- "
                        "lower this if long-document searches are filling "
                        "the model's context across several rounds."
                    ),
                ),
                CatalogParam(
                    name="snippet_max_chars",
                    env_var="DOCIE_MCP_DOCS_SEARCH_SNIPPET_MAX_CHARS",
                    description=(
                        "Hard ceiling on one match's snippet, in case many "
                        "scattered hits on the same page merge into a large "
                        "span. Defaults to 4000 if left blank."
                    ),
                ),
                CatalogParam(
                    name="peek_char_budget",
                    env_var="DOCIE_MCP_DOCS_SEARCH_PEEK_CHAR_BUDGET",
                    description=(
                        "Characters returned by read_document's default "
                        "'peek' (no page range) before it stops and points "
                        "the model at search_text instead. Defaults to 4000 "
                        "if left blank."
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
        CatalogEntry(
            name="call-llm",
            title="Call LLM (sub-agent dispatch)",
            description=(
                "Dispatch a scoped sub-task -- e.g. 'extract the answer from "
                "this search result' -- to a separate, typically cheap model "
                "instead of reasoning about a large context dump in the "
                "orchestrating model's own context. One completion, no "
                "tools, no recursion (#285)."
            ),
            module="docie_bench.mcp_servers.call_llm",
            tools=("call_llm",),
            params=(
                CatalogParam(
                    name="default_model_profile",
                    env_var="DOCIE_MCP_CALL_LLM_DEFAULT_PROFILE",
                    description=(
                        "model_profile every call_llm call dispatches to "
                        "unless the caller overrides it per-call. Required "
                        "unless every caller always passes model_profile "
                        "explicitly."
                    ),
                ),
                CatalogParam(
                    name="api_base",
                    env_var="DOCIE_MCP_CALL_LLM_API_BASE",
                    description=(
                        "Base URL of docie's own OpenAI-compatible API -- "
                        "the sub-request loops back through it. Defaults to "
                        "'http://127.0.0.1:8080' if left blank, correct when "
                        "this server runs in the same container as the api "
                        "process (the normal case)."
                    ),
                ),
                CatalogParam(
                    name="api_key",
                    env_var="DOCIE_MCP_CALL_LLM_API_KEY",
                    description=(
                        "API key for the loopback request when AUTH_REQUIRED "
                        "is on. Must be one of the operator's API_KEYS."
                    ),
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
