"""Durable per-deployment usage ledger: one row per serving request.

Write side (``record_usage``) is called from the serving surfaces themselves
(chat/embed/rerank in chat_api.py, extract in api.py) and mirrors
``seed_store``'s degrade-on-any-database-trouble contract: no DATABASE_URL and
a CONFIGURED-but-erroring database are both normal modes that must never fail
the request being tracked -- a chat completion that already succeeded upstream
cannot be turned into a 500 by a ledger insert.

Read side (``usage_summary``) aggregates the raw rows per deployment over a
bounded window at query time -- requests, errors, token totals, avg/p95
latency, last-used. Aggregation is done in Python over a column-projected
window scan rather than in SQL: a portable per-group p95 needs
``percentile_cont`` (PostgreSQL) with no sqlite equivalent, and the windows
are bounded (<= 30 days of dashboard traffic), so the honest cross-dialect
version is also the simple one. ``aggregate_usage``/``percentile`` are pure
and unit-tested without a database.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any

from sqlalchemy import Table, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from docie_bench.storage.db import session_scope
from docie_bench.studio.models import UsageRecord, utcnow

logger = logging.getLogger("docie_bench.studio.usage_store")

# Arbitrary-but-stable advisory-lock key ("docie usage records v1"), distinct
# from every other migration's key -- see seed_store.py/batch_store.py's
# identical pattern for the sibling new-table migrations.
_USAGE_RECORD_LOCK_KEY = 0x0D0C1E0B

SURFACES = ("chat", "extract", "embed", "rerank", "agent")

# window query param -> hours of history it covers.
WINDOW_HOURS = {"24h": 24, "7d": 7 * 24, "30d": 30 * 24}


def ensure_usage_record_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``usage_records`` if missing.

    Same shape as ``ensure_seed_run_table``: ``CREATE TABLE IF NOT EXISTS``
    under ``pg_advisory_xact_lock`` on PostgreSQL, plus an explicit
    ``CreateIndex(..., if_not_exists=True)`` loop so every process's
    concurrent ``init_engine`` (api, worker, N replicas) is safe.
    """
    from sqlalchemy.schema import CreateIndex, CreateTable

    usage_table = UsageRecord.__table__
    assert isinstance(usage_table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"), {"key": _USAGE_RECORD_LOCK_KEY}
            )
        existed = sa_inspect(connection).has_table("usage_records")
        connection.execute(CreateTable(usage_table, if_not_exists=True))
        for index in usage_table.indexes:
            connection.execute(CreateIndex(index, if_not_exists=True))
        # Forward migration for tables created before tool_calls_json existed
        # (#261) -- same ADD-COLUMN-if-missing shape as ensure_batch_tables.
        # Nullable, no rewrite.
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text(
                    "ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS tool_calls_json JSON"
                )
            )
        else:
            present = {col["name"] for col in sa_inspect(connection).get_columns("usage_records")}
            if "tool_calls_json" not in present:
                connection.execute(
                    sa_text("ALTER TABLE usage_records ADD COLUMN tool_calls_json JSON")
                )
    return not existed


def record_usage(
    *,
    deployment: str,
    surface: str,
    tenant_id: str,
    latency_ms: int,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    status: str = "ok",
    tool_calls: list[dict[str, Any]] | None = None,
) -> bool:
    """Insert one usage row. Best-effort by contract: NEVER raises.

    Returns whether a row was written -- ``False`` covers both "no
    DATABASE_URL" (persistence disabled, a normal mode) and "database errored
    on this insert" (connection blip, pool exhaustion). Either way the serving
    request this row describes already has its answer; the ledger must never
    take it down. Callers therefore never check the return value -- it exists
    for tests.
    """
    try:
        with session_scope() as session:
            if session is None:
                return False
            session.add(
                UsageRecord(
                    deployment=deployment[:200],
                    surface=surface,
                    tenant_id=(tenant_id or "anonymous")[:128],
                    latency_ms=max(0, int(latency_ms)),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    status=status if status in ("ok", "error") else "error",
                    tool_calls_json=tool_calls or None,
                )
            )
        return True
    except SQLAlchemyError:
        logger.warning(
            "usage ledger unavailable (record, deployment=%s surface=%s): request served untracked",
            deployment,
            surface,
            exc_info=True,
        )
        return False


def percentile(sorted_values: list[int], fraction: float) -> float | None:
    """Nearest-rank percentile over an ascending-sorted list; None when empty."""
    if not sorted_values:
        return None
    rank = math.ceil(fraction * len(sorted_values))
    index = min(len(sorted_values) - 1, max(0, rank - 1))
    return float(sorted_values[index])


def _fold_tool_calls(
    per_tool: dict[str, dict[str, Any]], tool_calls: list[dict[str, Any]]
) -> None:
    for call in tool_calls:
        name = call.get("tool")
        if not isinstance(name, str) or not name:
            continue
        stats = per_tool.setdefault(name, {"calls": 0, "errors": 0, "latencies": []})
        stats["calls"] += 1
        if call.get("status") == "error":
            stats["errors"] += 1
        latency = call.get("latency_ms")
        if isinstance(latency, int):
            stats["latencies"].append(latency)


UsageRow = tuple[
    str, str, int | None, int | None, int, dt.datetime | None, list[dict[str, Any]] | None
]


def aggregate_usage(rows: list[UsageRow]) -> list[dict[str, Any]]:
    """Fold raw ``(deployment, status, prompt_tokens, completion_tokens,
    latency_ms, created_at, tool_calls_json)`` rows into one summary entry per
    deployment.

    Pure (no database) so the p95/averaging math is unit-testable. Output is
    sorted by request count descending -- busiest deployment first, which is
    the order the Usage table renders in. ``tool_calls`` folds every row's MCP
    tool-call trace (agent surface only, see ``UsageRecord.tool_calls_json``)
    into per-tool call/error counts and average latency -- the Observability
    Usage view's expandable per-agent detail.
    """
    grouped: dict[str, dict[str, Any]] = {}
    latencies: dict[str, list[int]] = {}
    last_used: dict[str, dt.datetime] = {}
    tool_stats: dict[str, dict[str, dict[str, Any]]] = {}
    for (
        deployment,
        status,
        prompt_tokens,
        completion_tokens,
        latency_ms,
        created_at,
        tool_calls,
    ) in rows:
        entry = grouped.setdefault(
            deployment,
            {
                "deployment": deployment,
                "requests": 0,
                "errors": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "avg_latency_ms": None,
                "p95_latency_ms": None,
                "last_used_at": None,
                "tool_calls": [],
            },
        )
        entry["requests"] += 1
        if status == "error":
            entry["errors"] += 1
        entry["prompt_tokens"] += prompt_tokens or 0
        entry["completion_tokens"] += completion_tokens or 0
        latencies.setdefault(deployment, []).append(int(latency_ms))
        if created_at is not None:
            # Normalize before comparing: sqlite hands back naive datetimes,
            # PostgreSQL aware ones, and the two don't compare.
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=dt.UTC)
            if deployment not in last_used or created_at > last_used[deployment]:
                last_used[deployment] = created_at
        if tool_calls:
            _fold_tool_calls(tool_stats.setdefault(deployment, {}), tool_calls)
    for deployment, values in latencies.items():
        values.sort()
        grouped[deployment]["avg_latency_ms"] = round(sum(values) / len(values), 1)
        grouped[deployment]["p95_latency_ms"] = percentile(values, 0.95)
    for deployment, stamp in last_used.items():
        grouped[deployment]["last_used_at"] = stamp.isoformat()
    for deployment, per_tool in tool_stats.items():
        grouped[deployment]["tool_calls"] = [
            {
                "tool": name,
                "calls": stats["calls"],
                "errors": stats["errors"],
                "avg_latency_ms": (
                    round(sum(stats["latencies"]) / len(stats["latencies"]), 1)
                    if stats["latencies"]
                    else None
                ),
            }
            for name, stats in sorted(per_tool.items(), key=lambda kv: -kv[1]["calls"])
        ]
    return sorted(grouped.values(), key=lambda entry: (-entry["requests"], entry["deployment"]))


def usage_summary(*, tenant_id: str, window_hours: int) -> list[dict[str, Any]]:
    """Per-deployment aggregates over the last ``window_hours`` for one tenant.

    Missing DATABASE_URL degrades to an empty list, same contract as every
    other Studio listing route (seeds/batches/runs).
    """
    since = utcnow() - dt.timedelta(hours=window_hours)
    with session_scope() as session:
        if session is None:
            return []
        rows = session.execute(
            select(
                UsageRecord.deployment,
                UsageRecord.status,
                UsageRecord.prompt_tokens,
                UsageRecord.completion_tokens,
                UsageRecord.latency_ms,
                UsageRecord.created_at,
                UsageRecord.tool_calls_json,
            )
            .where(UsageRecord.tenant_id == tenant_id)
            .where(UsageRecord.created_at >= since)
        ).all()
        return aggregate_usage([tuple(row) for row in rows])


__all__ = [
    "SURFACES",
    "WINDOW_HOURS",
    "aggregate_usage",
    "ensure_usage_record_table",
    "percentile",
    "record_usage",
    "usage_summary",
]
