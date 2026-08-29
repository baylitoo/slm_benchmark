"""SQL agent MCP server (#310): agentic querying over a live PostgreSQL
database -- discover tables, inspect one table's schema, then run a
read-only query. Run: ``python -m docie_bench.mcp_servers.sql_agent``.

Same discover -> inspect -> read shape as docs_search's
list_files/read_document/search_text, but NOT a docs_search SearchBackend --
that ABC's ``search(query, targets) -> [{path, page, snippet}]`` return
type is document-shaped, and forcing SQL result rows through it would
distort the interface rather than reuse it. Each new data source (this one,
later a vector-DB or knowledge-graph server) is its own catalog entry with
its own tool shape, unified only by registration in ``mcp_catalog.CATALOG``
and selection through ``mcp_tools.run_tool_loop`` -- the same way
web-fetch/calculator/code-interpreter/call-llm already differ from
docs-search in tool shape while sharing that same orchestration layer.

Read-only is enforced in two layers. The real boundary is Postgres's own
``default_transaction_read_only`` session setting, turned on for every
connection this server opens (see ``_connect``) -- the server itself
rejects any write, including one hidden inside a CTE (e.g.
``WITH x AS (INSERT INTO ... RETURNING *) SELECT * FROM x``), which a
client-side statement-prefix check alone cannot catch. ``_reject_if_not_select``
is a fast-fail convenience on top of that, not a substitute for it. The
operator's DB user should additionally be granted SELECT only -- defense in
depth against a Postgres GUC misconfiguration, not this server's job to
verify.

``run_query`` caps its result at ``ROW_LIMIT_ENV`` rows (default 200) with a
truncation notice -- an unbounded row set is the same problem
``read_document`` had before pagination (#306) and ``search_text`` had
before windowing (#307-#309), just for rows instead of characters.
"""

from __future__ import annotations

import os
import re
from typing import Any

HOST_ENV = "DOCIE_MCP_SQL_AGENT_HOST"
PORT_ENV = "DOCIE_MCP_SQL_AGENT_PORT"
USER_ENV = "DOCIE_MCP_SQL_AGENT_USER"
PASSWORD_ENV = "DOCIE_MCP_SQL_AGENT_PASSWORD"  # noqa: S105 - an env VAR NAME, not a credential
DBNAME_ENV = "DOCIE_MCP_SQL_AGENT_DBNAME"
ROW_LIMIT_ENV = "DOCIE_MCP_SQL_AGENT_ROW_LIMIT"
_DEFAULT_PORT = "5432"
_DEFAULT_ROW_LIMIT = 200
# Defensive-only ceiling on list_tables -- a real schema (even a large ERP)
# is expected to stay well under this; if it doesn't, the eager-injection
# wording in mcp_tools._eager_list_context ("the ONLY valid identifiers ...
# is: {result}") would otherwise assert a listing is complete when it isn't.
_TABLE_LIST_SAFETY_CEILING = 500

_DISALLOWED_LEADING_KEYWORDS = frozenset(
    {
        "insert", "update", "delete", "drop", "alter", "truncate", "create",
        "grant", "revoke", "call", "merge", "copy", "vacuum", "reindex", "refresh",
    }
)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _connect() -> Any:
    import psycopg
    from psycopg.rows import dict_row

    host = os.environ.get(HOST_ENV)
    user = os.environ.get(USER_ENV)
    password = os.environ.get(PASSWORD_ENV)
    dbname = os.environ.get(DBNAME_ENV)
    fields = (("host", host), ("user", user), ("password", password), ("dbname", dbname))
    missing = [name for name, value in fields if not value]
    if missing:
        raise ValueError(
            f"sql-agent is not configured: missing {', '.join(missing)} -- set them as "
            "catalog params when enabling this server"
        )
    port = os.environ.get(PORT_ENV) or _DEFAULT_PORT
    return psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        connect_timeout=10,
        row_factory=dict_row,
        # The real read-only boundary (see module docstring) -- Postgres
        # itself rejects a write for the rest of this session, including one
        # hidden inside a CTE that _reject_if_not_select cannot see.
        options="-c default_transaction_read_only=on",
    )


def _json_safe(value: Any) -> Any:
    """Postgres types with no direct JSON mapping (Decimal, date/datetime,
    UUID, ...) stringified -- a tool result that fails to serialize would
    surface as an opaque MCP transport error instead of the actual row."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def list_tables_impl(schema: str | None = None) -> dict[str, Any]:
    """Every base table visible to this connection, as ``schema.table``
    names. Pass ``schema`` to narrow to one Postgres schema."""
    query = (
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_type = 'BASE TABLE' "
        "AND table_schema NOT IN ('pg_catalog', 'information_schema')"
    )
    params: tuple[Any, ...] = ()
    if schema:
        query += " AND table_schema = %s"
        params = (schema,)
    query += " ORDER BY table_schema, table_name"
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    names = [f"{r['table_schema']}.{r['table_name']}" for r in rows]
    result: dict[str, Any] = {"tables": names[:_TABLE_LIST_SAFETY_CEILING]}
    if len(names) > _TABLE_LIST_SAFETY_CEILING:
        result["notice"] = (
            f"{len(names)} tables matched; showing the first "
            f"{_TABLE_LIST_SAFETY_CEILING}. Pass schema= to narrow the search."
        )
    return result


def describe_table_impl(table: str) -> dict[str, Any]:
    """One table's columns, primary key, and foreign keys. ``table`` MUST be
    one of the exact ``schema.table`` strings returned by list_tables."""
    schema, sep, name = table.partition(".")
    if not sep:
        schema, name = "public", schema
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (schema, name),
        )
        columns = cur.fetchall()
        if not columns:
            raise ValueError(
                f"no such table: {table!r} -- call list_tables and use one of its exact "
                "schema-qualified names, don't invent one"
            )
        cur.execute(
            "SELECT kcu.column_name FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema "
            "WHERE tc.constraint_type = 'PRIMARY KEY' "
            "AND tc.table_schema = %s AND tc.table_name = %s",
            (schema, name),
        )
        primary_key = [r["column_name"] for r in cur.fetchall()]
        cur.execute(
            "SELECT kcu.column_name, ccu.table_schema AS foreign_schema, "
            "ccu.table_name AS foreign_table, ccu.column_name AS foreign_column "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema "
            "JOIN information_schema.constraint_column_usage ccu "
            "ON tc.constraint_name = ccu.constraint_name "
            "WHERE tc.constraint_type = 'FOREIGN KEY' "
            "AND tc.table_schema = %s AND tc.table_name = %s",
            (schema, name),
        )
        foreign_keys = [
            {
                "column": r["column_name"],
                "references": f"{r['foreign_schema']}.{r['foreign_table']}.{r['foreign_column']}",
            }
            for r in cur.fetchall()
        ]
    return {
        "table": f"{schema}.{name}",
        "columns": [
            {
                "name": c["column_name"],
                "type": c["data_type"],
                "nullable": c["is_nullable"] == "YES",
                "default": c["column_default"],
            }
            for c in columns
        ],
        "primary_key": primary_key,
        "foreign_keys": foreign_keys,
    }


def _reject_if_not_select(sql: str) -> None:
    """Fast-fail convenience only -- see the module docstring for why the
    Postgres session's own read-only setting, not this check, is the real
    enforcement (a CTE can wrap a write behind a leading SELECT/WITH)."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise ValueError("query must not be empty")
    if ";" in stripped:
        raise ValueError(
            "only a single statement is allowed -- remove the ';' and anything after it"
        )
    first_word = re.split(r"[\s(]", stripped, maxsplit=1)[0].lower()
    if first_word not in ("select", "with"):
        raise ValueError(
            f"only read-only queries are allowed (must start with SELECT or WITH), got: "
            f"{first_word!r}"
        )
    found_words = {w.lower() for w in re.findall(r"[A-Za-z_]+", stripped)}
    leading_keywords = found_words & _DISALLOWED_LEADING_KEYWORDS
    if leading_keywords:
        raise ValueError(
            f"query contains a write keyword ({', '.join(sorted(leading_keywords))}) -- "
            "only read-only queries are allowed"
        )


def run_query_impl(sql: str) -> dict[str, Any]:
    """Execute one read-only SQL statement and return its rows (capped at
    ``ROW_LIMIT_ENV`` rows, default 200)."""
    import psycopg

    _reject_if_not_select(sql)
    limit = _int_env(ROW_LIMIT_ENV, _DEFAULT_ROW_LIMIT)
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(sql)
            fetched = cur.fetchmany(limit + 1)
            columns = [d.name for d in cur.description] if cur.description else []
    except psycopg.Error as exc:
        raise ValueError(f"query failed: {exc}") from exc
    truncated = len(fetched) > limit
    rows = [{k: _json_safe(v) for k, v in row.items()} for row in fetched[:limit]]
    result: dict[str, Any] = {"columns": columns, "rows": rows, "row_count": len(rows)}
    if truncated:
        result["notice"] = (
            f"results capped at {limit} rows -- narrow the query (add a WHERE clause, "
            "LIMIT, or an aggregation) rather than assuming this is the full result set."
        )
    return result


def build_server() -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("docie-sql-agent")

    @server.tool()
    def list_tables(schema: str | None = None) -> dict[str, Any]:
        """List every base table visible to this connection, as
        `schema.table` names. ALWAYS call this first -- pass `schema` to
        narrow to one Postgres schema if you already know it."""
        return list_tables_impl(schema)

    @server.tool()
    def describe_table(table: str) -> dict[str, Any]:
        """Describe one table's columns (name, type, nullable, default),
        primary key, and foreign keys. `table` MUST be one of the exact
        `schema.table` strings returned by list_tables -- never invent,
        guess, or construct one from a name seen elsewhere."""
        return describe_table_impl(table)

    @server.tool()
    def run_query(sql: str) -> dict[str, Any]:
        """Execute one read-only SQL query (must start with SELECT or WITH,
        a single statement) and return its rows, capped at a row limit with
        a notice if truncated. Inspect a table's schema with describe_table
        before querying it -- don't guess column names or types."""
        return run_query_impl(sql)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
