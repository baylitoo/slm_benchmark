"""Durable store for extraction-run outcomes.

See ``ExtractionRunResult`` (studio/models.py) for why this exists: the
plain-HTTP polling contract documented in docs/server-integration.md needs
the extraction's actual output, and proxying Inngest's own run-status API
does not reliably carry it on this project's self-hosted Inngest server.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from sqlalchemy import Table
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateIndex, CreateTable

from docie_bench.storage.db import session_scope
from docie_bench.studio.models import ExtractionRunResult

logger = logging.getLogger("docie_bench.studio.extraction_results")

# Arbitrary-but-stable advisory-lock key ("docie extraction results v1"),
# distinct from every other migration's key -- see dynamic_schemas.py's
# identical pattern for the sibling new-table migrations.
_EXTRACTION_RESULT_LOCK_KEY = 0x0D0C1E09


def ensure_extraction_result_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``extraction_run_results`` if missing.

    Same shape as ``ensure_seed_run_table``/``ensure_dynamic_schema_table``:
    ``CREATE TABLE IF NOT EXISTS`` under ``pg_advisory_xact_lock`` on
    PostgreSQL, safe against every process's concurrent ``init_engine()`` at
    stack-up.
    """
    table = ExtractionRunResult.__table__
    assert isinstance(table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _EXTRACTION_RESULT_LOCK_KEY},
            )
        existed = sa_inspect(connection).has_table("extraction_run_results")
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in table.indexes:
            connection.execute(CreateIndex(index, if_not_exists=True))
    return not existed


def _to_dict(row: ExtractionRunResult) -> dict[str, Any]:
    return {
        "event_id": row.event_id,
        "tenant_id": row.tenant_id,
        "status": row.status,
        "output": row.output_json,
        "error": row.error_text,
        "created_at": _isoformat(row.created_at),
    }


def _isoformat(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.isoformat()


def record_extraction_result(
    *,
    event_id: str,
    tenant_id: str,
    status: str,
    output: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Record an extraction's final outcome (success or failure).

    Idempotent upsert by ``event_id`` -- Inngest is at-least-once, so a
    redelivered ``doc/extract.requested`` re-running ``extract_document`` must
    not raise on a duplicate row, it should just overwrite with the latest
    outcome. Never allowed to fail the extraction it's tracking: the same
    "never fail the work over the tracking" contract as
    ``seed_store.complete_seed_run`` -- an unavailable or errored database is
    logged and swallowed, never raised. The realtime ``result``/``error``
    topics already delivered the outcome to any Inngest-aware subscriber
    before this is even called; a tracking failure here only degrades the
    plain-HTTP polling fallback, it never loses the actual result.
    """
    try:
        with session_scope() as session:
            if session is None:
                return
            row = session.get(ExtractionRunResult, event_id)
            if row is None:
                row = ExtractionRunResult(event_id=event_id, tenant_id=tenant_id or "anonymous")
                session.add(row)
            row.status = status
            row.output_json = output
            row.error_text = error[:4000] if error else None
            session.flush()
    except SQLAlchemyError:
        logger.warning(
            "extraction-result tracking unavailable (event_id=%s): the realtime "
            "channel already delivered this outcome, only the HTTP polling "
            "fallback is degraded",
            event_id,
            exc_info=True,
        )


def get_extraction_result(event_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    """Tenant-scoped read. ``None`` for both "not found" and "not yours" --
    never confirm another tenant's event id, mirroring ``RunStore.get_run``."""
    with session_scope() as session:
        if session is None:
            return None
        row = session.get(ExtractionRunResult, event_id)
        if row is None or row.tenant_id != tenant_id:
            return None
        return _to_dict(row)


__all__ = [
    "ensure_extraction_result_table",
    "get_extraction_result",
    "record_extraction_result",
]
