"""Static catalog of first-party MCP servers; enabling an entry materializes
a stdio registry entry in configs/mcp-servers.json (see mcp_tools).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class CatalogParam:
    name: str
    env_var: str
    description: str
    required: bool = False
    # Display-only: masks the value as a password field in the enable form.
    # The value still persists as plaintext env in the server registry file
    # (mcp_tools.save_registry_entry) -- same trust model as every other
    # catalog param, not a secrets-manager integration.
    secret: bool = False
    # Display hint for the enable form's param widget:
    # "text" (default) -- genuinely open-ended (paths, URLs, hostnames,
    #   credentials), rendered as a free-typed TextInput.
    # "number" -- rendered as TextInput type="number".
    # "enum" -- a fixed value space (see `choices`), rendered as a <Select>.
    # "model_profile" -- rendered as a <Select> of live chat-capable
    #   deployments instead of free text, so picking the wrong profile name
    #   is a config-time error, not a silent runtime failure.
    kind: Literal["text", "number", "enum", "model_profile"] = "text"
    # Only meaningful when kind="enum" -- the fixed set of valid values.
    choices: tuple[str, ...] = ()


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
    # A multi-step USAGE PROTOCOL a tool's own JSON-schema description
    # already documents but a small model reliably fails to extract --
    # llama.cpp's --jinja rendering of `tools` is one dense inline JSON dump
    # per server (verified against a real running LFM2.5 chat_template: tool
    # descriptions get string-escaped and crammed into one "List of tools:
    # [...]" system-prompt line), not readable prose a small model attends to
    # well. `mcp_tools.run_tool_loop` folds this into the SAME clean,
    # dedicated system-message channel `TOOL_DISCIPLINE_DIRECTIVE` already
    # uses -- a model attends to that far more reliably than a schema
    # description buried in a JSON blob. `None` (default): no promotion,
    # the tool's own description is the only source, unchanged behavior.
    usage_notes: str | None = None


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
            tools=("list_files", "read_document", "search_text", "write_note", "read_notes"),
            eager_list_tool="list_files",
            usage_notes=(
                "docs-search reading protocol: call read_document ONCE with no "
                "start_page/end_page first -- its total_pages tells you how long "
                "the document really is, and a short document is answered "
                "directly from that first call. On a longer document, call "
                "search_text with a specific query to find which page(s) actually "
                "answer the question, THEN call read_document again with "
                "start_page/end_page set to those pages. Never guess a page range "
                "on the first call -- you don't know it yet.\n"
                "Notes protocol: call read_notes on a document before "
                "re-analyzing it from scratch -- an earlier pass may already "
                "have found (or flagged) something worth reusing. write_note is "
                "for a short, page-anchored observation or discrepancy, never a "
                "restatement of the page's own text -- if you're about to quote "
                "more than a sentence or two into a note, that content belongs "
                "in your answer, not the note."
            ),
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
                        "Retrieval strategy (see docs_search.SearchBackend): "
                        "'substring' (default, exact case-insensitive match) "
                        "or 'hybrid' (substring pre-filter, then semantic "
                        "rerank via a multi_vector_server deployment -- "
                        "needs reranker_url set). An unrecognized value is a "
                        "config error the first time search_text is called."
                    ),
                    kind="enum",
                    choices=("substring", "hybrid"),
                ),
                CatalogParam(
                    name="reranker_url",
                    env_var="DOCIE_MCP_DOCS_SEARCH_RERANKER_URL",
                    description=(
                        "Base URL of a running multi_vector_server deployment "
                        "-- only consulted when backend='hybrid'. No default: "
                        "a multi_vector deployment's URL is assigned by the "
                        "serving control plane, not a fixed compose-network "
                        "address. Required for hybrid; search_text fails "
                        "clearly if it's unset or unreachable rather than "
                        "silently falling back to substring."
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
                    kind="number",
                ),
                CatalogParam(
                    name="snippet_max_chars",
                    env_var="DOCIE_MCP_DOCS_SEARCH_SNIPPET_MAX_CHARS",
                    description=(
                        "Hard ceiling on one match's snippet, in case many "
                        "scattered hits on the same page merge into a large "
                        "span. Defaults to 4000 if left blank."
                    ),
                    kind="number",
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
                    kind="number",
                ),
                CatalogParam(
                    name="note_max_chars",
                    env_var="DOCIE_MCP_DOCS_SEARCH_NOTE_MAX_CHARS",
                    description=(
                        "Max characters for one write_note call -- a note is "
                        "a short observation, not another full-text dump. "
                        "Defaults to 2000 if left blank; a note over the cap "
                        "is rejected, never truncated."
                    ),
                    kind="number",
                ),
                CatalogParam(
                    name="max_notes_per_doc",
                    env_var="DOCIE_MCP_DOCS_SEARCH_MAX_NOTES_PER_DOC",
                    description=(
                        "Max notes write_note will accumulate per document "
                        "-- notes are append-only and never evicted, so this "
                        "is a hard ceiling, not a rolling window. Defaults "
                        "to 100 if left blank; write_note past the cap is "
                        "rejected."
                    ),
                    kind="number",
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
                    secret=True,
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
                    kind="model_profile",
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
        CatalogEntry(
            name="sql-agent",
            title="SQL Agent (PostgreSQL)",
            description=(
                "Agentic querying over a live PostgreSQL database (e.g. an "
                "ERP) -- discover tables, inspect a table's schema, then run "
                "a read-only query. Every connection is opened with "
                "Postgres's own default_transaction_read_only session "
                "setting on, so a write is rejected by the server itself "
                "even inside a CTE; the operator's DB user should still be "
                "granted SELECT only, as defense in depth."
            ),
            module="docie_bench.mcp_servers.sql_agent",
            tools=("list_tables", "describe_table", "run_query"),
            eager_list_tool="list_tables",
            usage_notes=(
                "sql-agent querying protocol: ALWAYS call describe_table on a "
                "table before writing a query against it -- never guess a "
                "column name from its table name alone. Every connection is "
                "read-only at the Postgres session level, so don't bother "
                "attempting a write query -- it will be rejected by the "
                "database itself, not silently ignored. If a question needs "
                "data from more than one table, describe_table "
                "each one first, then write a single query joining them -- "
                "don't run several separate single-table queries and combine "
                "the results yourself."
            ),
            params=(
                CatalogParam(
                    name="host",
                    env_var="DOCIE_MCP_SQL_AGENT_HOST",
                    description="PostgreSQL host.",
                    required=True,
                ),
                CatalogParam(
                    name="port",
                    env_var="DOCIE_MCP_SQL_AGENT_PORT",
                    description="PostgreSQL port. Defaults to 5432 if left blank.",
                    kind="number",
                ),
                CatalogParam(
                    name="user",
                    env_var="DOCIE_MCP_SQL_AGENT_USER",
                    description="DB user -- should be granted SELECT only.",
                    required=True,
                ),
                CatalogParam(
                    name="password",
                    env_var="DOCIE_MCP_SQL_AGENT_PASSWORD",
                    description="DB password for the above user.",
                    required=True,
                    secret=True,
                ),
                CatalogParam(
                    name="dbname",
                    env_var="DOCIE_MCP_SQL_AGENT_DBNAME",
                    description="Database name to connect to.",
                    required=True,
                ),
                CatalogParam(
                    name="row_limit",
                    env_var="DOCIE_MCP_SQL_AGENT_ROW_LIMIT",
                    description=(
                        "Max rows run_query returns before capping with a "
                        "notice. Defaults to 200 if left blank."
                    ),
                    kind="number",
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
