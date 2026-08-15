"""Durable index for Studio seed-download jobs (Ollama / Hugging Face).

Mirrors ``studio.store.RunStore``'s claim/complete/fail lifecycle, scaled down
for what a seed job actually needs: no artifacts (a seed's output is a store
entry + catalog row, not a downloadable blob this API serves), no
idempotency-key rotation (the trigger routes never computed one for seeds --
same as today, unchanged here). See ``SeedRun`` (studio/models.py) for why
this exists: seed outcomes had no durable record before this module, only the
ephemeral realtime topic and a pollable-but-cleared-on-settle sidecar file.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import Table, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from docie_bench.storage.db import session_scope
from docie_bench.studio.models import SeedRun

# Arbitrary-but-stable advisory-lock key ("docie seed runs v1"), distinct from
# every other migration's key -- see dynamic_schemas.py/routing_policies.py's
# identical pattern for the sibling new-table migrations.
_SEED_RUN_LOCK_KEY = 0x0D0C1E08


def ensure_seed_run_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``seed_runs`` if missing.

    Same shape and reasoning as ``ensure_routing_policy_table``: ``CREATE
    TABLE IF NOT EXISTS`` under ``pg_advisory_xact_lock`` on PostgreSQL, plus
    an explicit ``CreateIndex(..., if_not_exists=True)`` loop for
    ``channel``'s ``unique=True`` index, which ``Base.metadata.create_all()``
    would otherwise silently skip on any process that isn't the one racing
    through table creation.
    """
    from sqlalchemy.schema import CreateIndex, CreateTable

    seed_table = SeedRun.__table__
    assert isinstance(seed_table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"), {"key": _SEED_RUN_LOCK_KEY}
            )
        existed = sa_inspect(connection).has_table("seed_runs")
        connection.execute(CreateTable(seed_table, if_not_exists=True))
        for index in seed_table.indexes:
            connection.execute(CreateIndex(index, if_not_exists=True))
    return not existed


def _to_dict(row: SeedRun) -> dict[str, Any]:
    return {
        "event_id": row.event_id,
        "channel": row.channel,
        "tenant_id": row.tenant_id,
        "kind": row.kind,
        "reference": row.reference,
        "name": row.name,
        "status": row.status,
        "error": row.error_text,
        "result": row.result_json,
        "created_at": _isoformat(row.created_at),
        "updated_at": _isoformat(row.updated_at),
    }


def _isoformat(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.isoformat()


def claim_seed_run(
    *,
    event_id: str,
    channel: str,
    tenant_id: str,
    kind: str,
    name: str,
    reference: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Reserve a seed-run row *before* doing any download work.

    Same redelivery contract as ``RunStore.claim``: same ``event_id`` already
    ``completed`` -> ``("exists", record)`` (a redelivery of a finished seed,
    do not re-download); same ``event_id`` still ``running``/``failed`` ->
    ``("claimed", record)`` (Inngest is at-least-once -- a failed attempt must
    be allowed to retry).

    No DATABASE_URL is a normal degraded mode for a worker job (unlike the
    write routes in studio_api, which 503 -- there is no HTTP caller here to
    hand an error to), not an exception: returns ``("unavailable", {})`` and
    the caller proceeds without persistence, same as ``RunStore.enabled``
    gating every write call site in ``_run_benchmark_job``.
    """
    with session_scope() as session:
        if session is None:
            return "unavailable", {}
        existing = session.get(SeedRun, event_id)
        if existing is not None:
            if existing.status == "completed":
                return "exists", _to_dict(existing)
            existing.status = "running"
            existing.error_text = None
            session.flush()
            return "claimed", _to_dict(existing)
        row = SeedRun(
            event_id=event_id,
            channel=channel,
            tenant_id=tenant_id or "anonymous",
            kind=kind,
            reference=reference,
            name=name,
            status="running",
        )
        session.add(row)
        try:
            session.flush()
        except IntegrityError:
            # A duplicate trigger already owns this channel under a different
            # event id -- do not double-run; answer with the existing row.
            session.rollback()
            found = session.scalars(select(SeedRun).where(SeedRun.channel == channel)).first()
            if found is None:  # pragma: no cover - defensive; row must exist
                raise
            return "exists", _to_dict(found)
        return "claimed", _to_dict(row)


def complete_seed_run(*, event_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
    """Mark a claimed run completed. ``result["name"]`` (if present) overwrites
    the row's ``name`` -- an HF seed's store name may be empty/None at claim
    time (derived from the repo mid-job), so this is where the real,
    server-resolved name lands."""
    with session_scope() as session:
        if session is None:
            return None
        row = session.get(SeedRun, event_id)
        if row is None:
            return None
        row.status = "completed"
        row.result_json = result
        row.error_text = None
        resolved_name = result.get("name")
        if isinstance(resolved_name, str) and resolved_name:
            row.name = resolved_name
        session.flush()
        return _to_dict(row)


def fail_seed_run(*, event_id: str, error: str) -> dict[str, Any] | None:
    with session_scope() as session:
        if session is None:
            return None
        row = session.get(SeedRun, event_id)
        if row is None:
            return None
        row.status = "failed"
        row.error_text = error[:4000]
        session.flush()
        return _to_dict(row)


def list_seed_runs(*, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
    with session_scope() as session:
        if session is None:
            return []
        rows = session.scalars(
            select(SeedRun)
            .where(SeedRun.tenant_id == tenant_id)
            .order_by(SeedRun.created_at.desc())
            .limit(limit)
        ).all()
        return [_to_dict(row) for row in rows]


__all__ = [
    "claim_seed_run",
    "complete_seed_run",
    "ensure_seed_run_table",
    "fail_seed_run",
    "list_seed_runs",
]
