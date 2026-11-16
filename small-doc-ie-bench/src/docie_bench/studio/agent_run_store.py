"""Agent runs kept for evaluation (``AgentRunRecord``).

Retention: a run lives until it is scored and, when an evaluator flags it,
until its review is decided. ``delete_agent_run`` is called at both points.
``UNEVALUATED_TTL`` is only the backstop for a run that was never scored
(queue down, event lost); a run with an open review is never purged.
"""

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
from docie_bench.studio.models import (
    AgentRunRecord,
    EvaluationResultRecord,
    isoformat,
    utcnow,
)

# Arbitrary-but-stable advisory-lock key ("docie agent runs v1").
_AGENT_RUN_LOCK_KEY = 0x0D0C1E0F
# Evaluation results: verdicts only (an LLM judge's reason is cleared when
# its run is deleted), kept for the Evaluators page history.
RETENTION = dt.timedelta(days=30)
UNEVALUATED_TTL = dt.timedelta(days=7)


def ensure_agent_run_table(engine: Engine) -> bool:
    """Race-safe forward migration for ``agent_runs`` and ``evaluation_results``,
    same shape as ``ensure_evaluator_table``."""
    created = False
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"), {"key": _AGENT_RUN_LOCK_KEY}
            )
        for model in (AgentRunRecord, EvaluationResultRecord):
            table = model.__table__
            assert isinstance(table, Table)
            created |= not sa_inspect(connection).has_table(table.name)
            connection.execute(CreateTable(table, if_not_exists=True))
            for index in table.indexes:
                connection.execute(CreateIndex(index, if_not_exists=True))
    return created


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
                expires_at=created + UNEVALUATED_TTL,
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


def save_evaluation(
    *,
    run_id: str,
    evaluator: str,
    agent: str,
    tenant_id: str,
    verdicts: list[dict[str, Any]],
    review_reasons: list[str],
    now: dt.datetime | None = None,
) -> None:
    """One row per (run, evaluator): a retried step replaces, never duplicates."""
    created = now or utcnow()
    with session_scope() as session:
        if session is None:
            return
        session.execute(
            delete(EvaluationResultRecord).where(
                EvaluationResultRecord.run_id == run_id,
                EvaluationResultRecord.evaluator == evaluator,
            )
        )
        session.add(
            EvaluationResultRecord(
                run_id=run_id,
                evaluator=evaluator,
                agent=agent,
                tenant_id=tenant_id,
                verdicts_json=verdicts,
                review_reasons_json=review_reasons,
                created_at=created,
                expires_at=created + RETENTION,
            )
        )


def list_evaluations(
    evaluator: str, *, tenant_id: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    """An evaluator's most recent results, newest first; one tenant's when given."""
    with session_scope() as session:
        if session is None:
            return []
        statement = select(EvaluationResultRecord).where(
            EvaluationResultRecord.evaluator == evaluator
        )
        if tenant_id is not None:
            statement = statement.where(EvaluationResultRecord.tenant_id == tenant_id)
        rows = session.execute(
            statement.order_by(
                EvaluationResultRecord.created_at.desc(), EvaluationResultRecord.id.desc()
            ).limit(limit)
        ).scalars()
        return [
            {
                "run_id": row.run_id,
                "agent": row.agent,
                "verdicts": row.verdicts_json,
                "review_reasons": row.review_reasons_json,
                "created_at": isoformat(row.created_at),
            }
            for row in rows
        ]


def delete_agent_run(run_id: str) -> bool:
    """Delete a run's text, and the LLM judges' reasons that may quote it.

    The verdicts themselves stay for the Evaluators page history.
    """
    with session_scope() as session:
        if session is None:
            return False
        results = session.execute(
            select(EvaluationResultRecord).where(EvaluationResultRecord.run_id == run_id)
        ).scalars()
        for row in results:
            row.verdicts_json = [{**v, "reason": None} for v in row.verdicts_json or []]
        result = cast(
            CursorResult[Any],
            session.execute(delete(AgentRunRecord).where(AgentRunRecord.id == run_id)),
        )
        return bool(result.rowcount)


def purge_expired_runs(now: dt.datetime | None = None) -> int:
    """Delete runs and evaluation results past their retention; returns how
    many runs were removed."""
    cutoff = now or utcnow()
    with session_scope() as session:
        if session is None:
            return 0
        session.execute(
            delete(EvaluationResultRecord).where(EvaluationResultRecord.expires_at < cutoff)
        )
        from docie_bench.storage.db import ReviewTask

        open_review = select(ReviewTask.source_request_id).where(
            ReviewTask.kind == "agent_run",
            ReviewTask.status.in_(("pending", "claimed")),
        )
        result = cast(
            CursorResult[Any],
            session.execute(
                delete(AgentRunRecord).where(
                    AgentRunRecord.expires_at < cutoff,
                    AgentRunRecord.id.not_in(open_review),
                )
            ),
        )
        return int(result.rowcount or 0)
