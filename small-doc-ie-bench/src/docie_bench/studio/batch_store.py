"""Durable index for batch-extraction jobs (``BatchRun`` + per-document
``BatchItem``).

Mirrors ``studio.seed_store``'s claim/settle lifecycle and race-safe table
migration, with one deliberate difference: a batch REQUIRES the database.
Seed tracking can degrade silently (the download is the product; the row is
a convenience). A batch's item results ARE the product -- there is nothing
to hand the caller without them -- so the trigger route answers 503 when
persistence is unavailable rather than running an unrecoverable job. Inside
the worker the item writes still catch ``SQLAlchemyError`` and log, so a
transient blip mid-batch loses one item's persisted state, not the run.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from sqlalchemy import Table, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import selectinload

from docie_bench.storage.db import session_scope
from docie_bench.studio.models import BatchItem, BatchRun

logger = logging.getLogger("docie_bench.studio.batch_store")

# Distinct advisory-lock key from every other new-table migration
# (dynamic_schemas / routing_policies / seed_runs / review_evidence).
_BATCH_LOCK_KEY = 0x0D0C1E0A


class BatchStoreUnavailableError(RuntimeError):
    """No DATABASE_URL (or it errored): batches cannot run without persistence."""


def ensure_batch_tables(engine: Engine) -> bool:
    """Race-safe forward migration: create ``batch_runs`` + ``batch_items`` if
    missing. Same shape as ``ensure_seed_run_table``: CREATE TABLE IF NOT
    EXISTS under ``pg_advisory_xact_lock`` on PostgreSQL, explicit
    ``if_not_exists`` index creation -- every process's concurrent
    ``init_engine`` (api, worker, N replicas) would otherwise race
    ``create_all`` into a duplicate-table abort. Order matters: the child
    table's FK needs the parent to exist first."""
    from sqlalchemy.schema import CreateIndex, CreateTable

    runs_table = BatchRun.__table__
    items_table = BatchItem.__table__
    assert isinstance(runs_table, Table)  # narrow FromClause for the compiler
    assert isinstance(items_table, Table)
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"), {"key": _BATCH_LOCK_KEY}
            )
        existed = sa_inspect(connection).has_table("batch_runs")
        for table in (runs_table, items_table):
            connection.execute(CreateTable(table, if_not_exists=True))
            for index in table.indexes:
                connection.execute(CreateIndex(index, if_not_exists=True))
    return not existed


def _isoformat(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.isoformat()


def _item_to_dict(item: BatchItem) -> dict[str, Any]:
    return {
        "position": item.position,
        "filename": item.filename,
        "status": item.status,
        "result": item.result_json,
        "error": item.error_text,
        "latency_ms": item.latency_ms,
        "updated_at": _isoformat(item.updated_at),
    }


def _run_to_dict(run: BatchRun, *, include_items: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "event_id": run.event_id,
        "channel": run.channel,
        "tenant_id": run.tenant_id,
        "name": run.name,
        "schema_name": run.schema_name,
        "model_selector": run.model_selector,
        "status": run.status,
        "total_items": run.total_items,
        "done_items": run.done_items,
        "failed_items": run.failed_items,
        "error": run.error_text,
        "artifacts": run.artifacts_json or [],
        "created_at": _isoformat(run.created_at),
        "updated_at": _isoformat(run.updated_at),
    }
    if include_items:
        out["items"] = [_item_to_dict(item) for item in run.items]
    return out


def claim_batch_run(
    *,
    event_id: str,
    channel: str,
    tenant_id: str,
    name: str,
    schema_name: str,
    model_selector: str | None,
    filenames: list[str],
) -> tuple[str, dict[str, Any]]:
    """Reserve the run row + one PENDING item per document, BEFORE any
    extraction. Same redelivery contract as the other stores: same
    ``event_id`` already ``completed`` -> ``("exists", record)`` (do not
    re-run); still ``running``/``failed`` -> ``("claimed", record)`` with the
    existing items left as they are (Inngest is at-least-once, and a
    resumed run must keep the per-item progress it already made -- that is
    the whole point of per-item state). Raises BatchStoreUnavailableError
    when persistence is unavailable."""
    try:
        with session_scope() as session:
            if session is None:
                raise BatchStoreUnavailableError(
                    "batch extraction requires a configured DATABASE_URL (item results "
                    "are persisted per document)"
                )
            existing = session.get(BatchRun, event_id, options=[selectinload(BatchRun.items)])
            if existing is not None:
                if existing.status == "completed":
                    return "exists", _run_to_dict(existing, include_items=True)
                existing.status = "running"
                existing.error_text = None
                session.flush()
                return "claimed", _run_to_dict(existing, include_items=True)
            row = BatchRun(
                event_id=event_id,
                channel=channel,
                tenant_id=tenant_id or "anonymous",
                name=name,
                schema_name=schema_name,
                model_selector=model_selector,
                status="running",
                total_items=len(filenames),
            )
            row.items = [
                BatchItem(position=i, filename=fn, status="pending")
                for i, fn in enumerate(filenames)
            ]
            session.add(row)
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                found = session.scalars(
                    select(BatchRun)
                    .where(BatchRun.channel == channel)
                    .options(selectinload(BatchRun.items))
                ).first()
                if found is None:  # pragma: no cover - defensive
                    raise
                return "exists", _run_to_dict(found, include_items=True)
            return "claimed", _run_to_dict(row, include_items=True)
    except BatchStoreUnavailableError:
        raise
    except SQLAlchemyError as exc:
        raise BatchStoreUnavailableError(f"batch persistence unavailable: {exc}") from exc


def record_batch_item(
    *,
    event_id: str,
    position: int,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    latency_ms: int | None = None,
) -> None:
    """Persist one document's outcome and bump the run's denormalized
    counters. Best-effort: a transient DB blip here loses this item's
    persisted state, never the running batch (logged, not raised)."""
    try:
        with session_scope() as session:
            if session is None:
                return
            item = session.scalars(
                select(BatchItem).where(
                    BatchItem.run_event_id == event_id, BatchItem.position == position
                )
            ).first()
            if item is None:
                return
            was_terminal = item.status in ("done", "failed")
            item.status = status
            item.result_json = result
            item.error_text = error
            item.latency_ms = latency_ms
            run = session.get(BatchRun, event_id)
            if run is not None and not was_terminal:
                if status == "done":
                    run.done_items += 1
                elif status == "failed":
                    run.failed_items += 1
            session.flush()
    except SQLAlchemyError:
        logger.warning(
            "batch item persistence failed (event_id=%s, position=%s); batch continues",
            event_id,
            position,
            exc_info=True,
        )


def settle_batch_run(
    *,
    event_id: str,
    status: str,
    artifacts: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any] | None:
    """Mark the run terminal (``completed`` -- possibly with some failed
    items -- or ``failed`` for a whole-run error) and attach artifact refs."""
    try:
        with session_scope() as session:
            if session is None:
                return None
            run = session.get(BatchRun, event_id, options=[selectinload(BatchRun.items)])
            if run is None:
                return None
            run.status = status
            run.error_text = error
            if artifacts is not None:
                run.artifacts_json = artifacts
            session.flush()
            return _run_to_dict(run, include_items=True)
    except SQLAlchemyError:
        logger.warning(
            "batch settle persistence failed (event_id=%s)", event_id, exc_info=True
        )
        return None


def list_batch_runs(*, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """This tenant's batches, newest first, WITHOUT items (the list view)."""
    with session_scope() as session:
        if session is None:
            raise BatchStoreUnavailableError("batch persistence requires a configured DATABASE_URL")
        rows = session.scalars(
            select(BatchRun)
            .where(BatchRun.tenant_id == tenant_id)
            .order_by(BatchRun.created_at.desc())
            .limit(limit)
        ).all()
        return [_run_to_dict(row, include_items=False) for row in rows]


def get_batch_run(event_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    """One batch WITH its items, tenant-scoped (a foreign tenant's id reads as
    not-found, never a 403 that confirms existence)."""
    with session_scope() as session:
        if session is None:
            raise BatchStoreUnavailableError("batch persistence requires a configured DATABASE_URL")
        run = session.get(BatchRun, event_id, options=[selectinload(BatchRun.items)])
        if run is None or run.tenant_id != tenant_id:
            return None
        return _run_to_dict(run, include_items=True)
