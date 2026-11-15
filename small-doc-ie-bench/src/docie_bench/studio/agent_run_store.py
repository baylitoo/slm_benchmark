"""Agent runs kept for evaluation (``AgentRunRecord``), with bounded retention."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, cast

from sqlalchemy import Table, delete, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import CursorResult, Engine
from sqlalchemy.schema import CreateIndex, CreateTable

from docie_bench.storage.db import session_scope
from docie_bench.studio.models import AgentRunRecord, isoformat, utcnow

# Arbitrary-but-stable advisory-lock key ("docie agent runs v1").
_AGENT_RUN_LOCK_KEY = 0x0D0C1E0F
RETENTION = dt.timedelta(days=30)


def ensure_agent_run_table(engine: Engine) -> bool:
    """Race-safe forward migration, same shape as ``ensure_evaluator_table``."""
    table = AgentRunRecord.__table__
    assert isinstance(table, Table)
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"), {"key": _AGENT_RUN_LOCK_KEY}
            )
        existed = sa_inspect(connection).has_table("agent_runs")
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in table.indexes:
            connection.execute(CreateIndex(index, if_not_exists=True))
    return not existed


def save_agent_run(
    *,
    agent: str,
    tenant_id: str,
    api_key_id: str | None,
    messages: list[dict[str, Any]],
    output_text: str,
    tool_calls: list[dict[str, Any]],
    evaluators: list[str],
    now: dt.datetime | None = None,
) -> str | None:
    """Store one run and return its id, or None without a database."""
    created = now or utcnow()
    with session_scope() as session:
        if session is None:
            return None
        run_id = str(uuid.uuid4())
        session.add(
            AgentRunRecord(
                id=run_id,
                agent=agent,
                tenant_id=tenant_id,
                api_key_id=api_key_id,
                messages_json=messages,
                output_text=output_text,
                tool_calls_json=tool_calls,
                evaluators_json=evaluators,
                created_at=created,
                expires_at=created + RETENTION,
            )
        )
        return run_id


def _view(row: AgentRunRecord) -> dict[str, Any]:
    return {
        "id": row.id,
        "agent": row.agent,
        "tenant_id": row.tenant_id,
        "messages": row.messages_json,
        "output": row.output_text,
        "tool_calls": row.tool_calls_json,
        "evaluators": row.evaluators_json,
        "created_at": isoformat(row.created_at),
        "expires_at": isoformat(row.expires_at),
    }


def get_agent_run(run_id: str) -> dict[str, Any] | None:
    with session_scope() as session:
        if session is None:
            return None
        row = session.execute(
            select(AgentRunRecord).where(AgentRunRecord.id == run_id)
        ).scalar_one_or_none()
        return _view(row) if row is not None else None


def purge_expired_runs(now: dt.datetime | None = None) -> int:
    """Delete runs past their retention; returns how many were removed."""
    with session_scope() as session:
        if session is None:
            return 0
        result = cast(
            CursorResult[Any],
            session.execute(
                delete(AgentRunRecord).where(AgentRunRecord.expires_at < (now or utcnow()))
            ),
        )
        return int(result.rowcount or 0)
