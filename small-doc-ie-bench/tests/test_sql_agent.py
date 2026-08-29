"""sql_agent MCP server (#310) -- the logic that doesn't need a live
Postgres: read-only statement validation, JSON-safety coercion, and the
missing-config refusal. Query execution itself needs a real connection and
is out of scope for these tests."""

from __future__ import annotations

import datetime
from decimal import Decimal

import pytest

from docie_bench.mcp_servers import sql_agent


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM invoices",
        "  select id from invoices  ",
        "WITH t AS (SELECT 1) SELECT * FROM t",
        "SELECT * FROM invoices;",
    ],
)
def test_reject_if_not_select_allows_read_only_queries(sql: str) -> None:
    sql_agent._reject_if_not_select(sql)  # must not raise


@pytest.mark.parametrize(
    ("sql", "match"),
    [
        ("", "must not be empty"),
        ("   ", "must not be empty"),
        ("DROP TABLE invoices", "must start with SELECT or WITH"),
        ("SELECT * FROM a; DROP TABLE a", "single statement"),
        ("SELECT * FROM invoices WHERE id = 1 OR DELETE", "write keyword"),
        ("UPDATE invoices SET total = 0", "must start with SELECT or WITH"),
    ],
)
def test_reject_if_not_select_refuses_writes(sql: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        sql_agent._reject_if_not_select(sql)


def test_json_safe_stringifies_non_primitive_postgres_types() -> None:
    assert sql_agent._json_safe(Decimal("12.50")) == "12.50"
    assert sql_agent._json_safe(datetime.date(2026, 1, 1)) == "2026-01-01"
    assert sql_agent._json_safe(None) is None
    assert sql_agent._json_safe("thales") == "thales"
    assert sql_agent._json_safe(42) == 42
    assert sql_agent._json_safe(True) is True


def test_connect_refuses_without_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for env in (
        sql_agent.HOST_ENV,
        sql_agent.USER_ENV,
        sql_agent.PASSWORD_ENV,
        sql_agent.DBNAME_ENV,
    ):
        monkeypatch.delenv(env, raising=False)
    with pytest.raises(ValueError, match="not configured"):
        sql_agent._connect()
