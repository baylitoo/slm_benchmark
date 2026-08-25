"""Durable store for recurring batch-extraction schedules (``BatchSchedule``).

Mirrors ``studio.batch_store``'s split contract between the two sides that
touch it:

* The CRUD used by the API routes REQUIRES the database (schedule rows are
  the product; there is nothing to save without them) -- those functions
  raise :class:`ScheduleStoreUnavailableError`, which the routes answer as
  503, exactly like the batch routes.
* The cron-tick side (``due_schedules`` / ``mark_fired``) degrades
  gracefully on ANY database trouble, same as ``seed_store``: the tick runs
  every minute forever, and a transient connection blip must log-and-skip
  one tick, never crash-loop the worker's cron function.

Firing material also lives here (``scheduled_batch_event_data``): both the
cron tick and the ``run-now`` route need "turn this schedule into a
``doc/batch.requested`` event by re-reading the source batch's stored
documents", and keeping it beside the store keeps the two paths identical.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Any

from sqlalchemy import Table, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from docie_bench.storage.db import session_scope
from docie_bench.studio import store as studio_store
from docie_bench.studio.models import BatchSchedule, utcnow

logger = logging.getLogger("docie_bench.studio.schedule_store")

# Distinct advisory-lock key from every other new-table migration (batch runs
# hold 0x0D0C1E0A; see batch_store.py's identical pattern).
_SCHEDULE_LOCK_KEY = 0x0D0C1E0B

# The safe interval set. "every_n_minutes" reads the minute count from the
# row's ``every_n_minutes`` column, floor-limited so a schedule can never tick
# faster than the cron that drives it can keep up with.
INTERVALS = ("hourly", "daily", "weekly", "every_n_minutes")
MIN_EVERY_N_MINUTES = 15
MAX_EVERY_N_MINUTES = 7 * 24 * 60  # one week, same ceiling as "weekly"

_FIXED_DELTAS = {
    "hourly": dt.timedelta(hours=1),
    "daily": dt.timedelta(days=1),
    "weekly": dt.timedelta(weeks=1),
}


class ScheduleStoreUnavailableError(RuntimeError):
    """No DATABASE_URL (or it errored): schedules cannot exist without rows."""


class ScheduleValidationError(ValueError):
    """The interval / every_n_minutes combination is not a valid schedule."""


def interval_delta(interval: str, every_n_minutes: int | None) -> dt.timedelta:
    """The timedelta between firings, or :class:`ScheduleValidationError`."""
    if interval in _FIXED_DELTAS:
        return _FIXED_DELTAS[interval]
    if interval == "every_n_minutes":
        if every_n_minutes is None:
            raise ScheduleValidationError(
                "interval 'every_n_minutes' requires 'every_n_minutes'"
            )
        if not MIN_EVERY_N_MINUTES <= every_n_minutes <= MAX_EVERY_N_MINUTES:
            raise ScheduleValidationError(
                f"'every_n_minutes' must be between {MIN_EVERY_N_MINUTES} and "
                f"{MAX_EVERY_N_MINUTES} (got {every_n_minutes})"
            )
        return dt.timedelta(minutes=every_n_minutes)
    raise ScheduleValidationError(
        f"unknown interval {interval!r}: expected one of {', '.join(INTERVALS)}"
    )


def ensure_batch_schedule_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``batch_schedules`` if missing.

    Same shape as ``ensure_batch_tables``: CREATE TABLE IF NOT EXISTS under
    ``pg_advisory_xact_lock`` on PostgreSQL plus explicit ``if_not_exists``
    index creation, because every process's concurrent ``init_engine`` (api,
    serving, N workers) would otherwise race ``create_all`` into a
    duplicate-table abort."""
    from sqlalchemy.schema import CreateIndex, CreateTable

    table = BatchSchedule.__table__
    assert isinstance(table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"), {"key": _SCHEDULE_LOCK_KEY}
            )
        existed = sa_inspect(connection).has_table("batch_schedules")
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


def _to_dict(row: BatchSchedule) -> dict[str, Any]:
    return {
        "id": row.id,
        "tenant_id": row.tenant_id,
        "name": row.name,
        "source_event_id": row.source_event_id,
        "schema_name": row.schema_name,
        "selectors": row.selectors_json or {},
        "interval": row.interval,
        "every_n_minutes": row.every_n_minutes,
        "enabled": row.enabled,
        "next_run_at": _isoformat(row.next_run_at),
        "last_run_at": _isoformat(row.last_run_at),
        "last_event_id": row.last_event_id,
        "last_error": row.last_error,
        "created_at": _isoformat(row.created_at),
        "updated_at": _isoformat(row.updated_at),
    }


# -- route-facing CRUD (database REQUIRED, like batch_store) -----------------


def create_schedule(
    *,
    tenant_id: str,
    name: str,
    source_event_id: str,
    schema_name: str,
    selectors: dict[str, Any] | None,
    interval: str,
    every_n_minutes: int | None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Persist a new schedule; ``next_run_at`` starts one interval from now
    (a schedule is "run me every X", not "run me right now" -- run-now is its
    own explicit route). Raises ScheduleValidationError on a bad interval."""
    delta = interval_delta(interval, every_n_minutes)
    try:
        with session_scope() as session:
            if session is None:
                raise ScheduleStoreUnavailableError(
                    "batch schedules require a configured DATABASE_URL"
                )
            row = BatchSchedule(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id or "anonymous",
                name=name,
                source_event_id=source_event_id,
                schema_name=schema_name,
                selectors_json=selectors,
                interval=interval,
                every_n_minutes=every_n_minutes if interval == "every_n_minutes" else None,
                enabled=enabled,
                next_run_at=utcnow() + delta,
            )
            session.add(row)
            session.flush()
            return _to_dict(row)
    except ScheduleStoreUnavailableError:
        raise
    except SQLAlchemyError as exc:
        raise ScheduleStoreUnavailableError(
            f"schedule persistence unavailable: {exc}"
        ) from exc


def list_schedules(*, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """This tenant's schedules, newest first."""
    try:
        with session_scope() as session:
            if session is None:
                raise ScheduleStoreUnavailableError(
                    "batch schedules require a configured DATABASE_URL"
                )
            rows = session.scalars(
                select(BatchSchedule)
                .where(BatchSchedule.tenant_id == tenant_id)
                .order_by(BatchSchedule.created_at.desc())
                .limit(limit)
            ).all()
            return [_to_dict(row) for row in rows]
    except ScheduleStoreUnavailableError:
        raise
    except SQLAlchemyError as exc:
        raise ScheduleStoreUnavailableError(
            f"schedule persistence unavailable: {exc}"
        ) from exc


def get_schedule(schedule_id: str, *, tenant_id: str) -> dict[str, Any] | None:
    """One schedule, tenant-scoped (a foreign tenant's id reads as not-found,
    never a 403 that confirms existence)."""
    try:
        with session_scope() as session:
            if session is None:
                raise ScheduleStoreUnavailableError(
                    "batch schedules require a configured DATABASE_URL"
                )
            row = session.get(BatchSchedule, schedule_id)
            if row is None or row.tenant_id != tenant_id:
                return None
            return _to_dict(row)
    except ScheduleStoreUnavailableError:
        raise
    except SQLAlchemyError as exc:
        raise ScheduleStoreUnavailableError(
            f"schedule persistence unavailable: {exc}"
        ) from exc


_UNSET: Any = object()


def update_schedule(
    schedule_id: str,
    *,
    tenant_id: str,
    name: str | None = None,
    enabled: bool | None = None,
    interval: str | None = None,
    every_n_minutes: int | None | Any = _UNSET,
) -> dict[str, Any] | None:
    """Patch a schedule. Changing the interval -- or re-enabling a disabled
    schedule -- recomputes ``next_run_at`` from NOW (+ the new interval), so a
    schedule disabled for a month never fires a "catch-up" the second it is
    switched back on. Returns None for an unknown/foreign id."""
    try:
        with session_scope() as session:
            if session is None:
                raise ScheduleStoreUnavailableError(
                    "batch schedules require a configured DATABASE_URL"
                )
            row = session.get(BatchSchedule, schedule_id)
            if row is None or row.tenant_id != tenant_id:
                return None
            reschedule = False
            if interval is not None:
                n = every_n_minutes if every_n_minutes is not _UNSET else row.every_n_minutes
                interval_delta(interval, n)  # validate before writing anything
                row.interval = interval
                row.every_n_minutes = n if interval == "every_n_minutes" else None
                reschedule = True
            elif every_n_minutes is not _UNSET:
                interval_delta(row.interval, every_n_minutes)
                row.every_n_minutes = every_n_minutes
                reschedule = True
            if name is not None:
                row.name = name
            if enabled is not None:
                if enabled and not row.enabled:
                    reschedule = True
                row.enabled = enabled
            if reschedule:
                row.next_run_at = utcnow() + interval_delta(
                    row.interval, row.every_n_minutes
                )
            session.flush()
            return _to_dict(row)
    except (ScheduleStoreUnavailableError, ScheduleValidationError):
        raise
    except SQLAlchemyError as exc:
        raise ScheduleStoreUnavailableError(
            f"schedule persistence unavailable: {exc}"
        ) from exc


def delete_schedule(schedule_id: str, *, tenant_id: str) -> bool:
    """True when the row existed (tenant-scoped) and is now gone."""
    try:
        with session_scope() as session:
            if session is None:
                raise ScheduleStoreUnavailableError(
                    "batch schedules require a configured DATABASE_URL"
                )
            row = session.get(BatchSchedule, schedule_id)
            if row is None or row.tenant_id != tenant_id:
                return False
            session.delete(row)
            session.flush()
            return True
    except ScheduleStoreUnavailableError:
        raise
    except SQLAlchemyError as exc:
        raise ScheduleStoreUnavailableError(
            f"schedule persistence unavailable: {exc}"
        ) from exc


# -- cron-facing scan/advance (degrades gracefully, like seed_store) ---------


def due_schedules(
    *, now: dt.datetime | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Enabled schedules with ``next_run_at <= now``, oldest-due first --
    across ALL tenants (the cron ticks for everyone; each fired event carries
    its schedule's own ``tenant_id``). Degrades to ``[]`` on any database
    trouble: the tick runs every minute, a blip skips one tick, never
    crash-loops the cron."""
    try:
        with session_scope() as session:
            if session is None:
                return []
            rows = session.scalars(
                select(BatchSchedule)
                .where(
                    BatchSchedule.enabled.is_(True),
                    BatchSchedule.next_run_at <= (now or utcnow()),
                )
                .order_by(BatchSchedule.next_run_at.asc())
                .limit(limit)
            ).all()
            return [_to_dict(row) for row in rows]
    except SQLAlchemyError:
        logger.warning("batch-schedule scan unavailable; skipping this tick", exc_info=True)
        return []


def mark_fired(
    schedule_id: str,
    *,
    event_id: str | None = None,
    error: str | None = None,
    advance: bool = True,
) -> dict[str, Any] | None:
    """Record a firing (or a firing that could not happen) and, with
    ``advance``, push ``next_run_at`` one interval past NOW. The error path
    also advances so a broken schedule surfaces ``last_error`` once per
    interval instead of retrying every minute. ``advance=False`` is the
    run-now route: it stamps last_run/last_event without touching the
    cadence. Degrades to None (logged) -- the fired event is already sent;
    tracking must never fail it."""
    try:
        with session_scope() as session:
            if session is None:
                return None
            row = session.get(BatchSchedule, schedule_id)
            if row is None:
                return None
            now = utcnow()
            if event_id is not None:
                row.last_run_at = now
                row.last_event_id = event_id
            row.last_error = error
            if advance:
                row.next_run_at = now + interval_delta(row.interval, row.every_n_minutes)
            session.flush()
            return _to_dict(row)
    except SQLAlchemyError:
        logger.warning(
            "batch-schedule advance unavailable (id=%s)", schedule_id, exc_info=True
        )
        return None


# -- firing material ---------------------------------------------------------


def scheduled_batch_event_data(
    schedule: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """``(event_data, None)`` ready to fire as ``doc/batch.requested``, or
    ``(None, reason)`` when the schedule cannot fire.

    Re-materializes the source batch's documents the same way the
    retry-failed route does: every item's ``input_relkey`` must still exist
    in the shared blob store -- a schedule fires ALL of the source's
    documents or none (a silently partial recurring run would look like data
    loss downstream). The blob store is resolved via
    ``studio_store.default_blob_store`` attribute access so one monkeypatch
    covers this module too (see ``studio_api._shared``'s docstring for the
    pattern)."""
    from docie_bench.studio.batch_store import BatchStoreUnavailableError, get_batch_run

    source_event_id = str(schedule["source_event_id"])
    try:
        run = get_batch_run(source_event_id, tenant_id=str(schedule["tenant_id"]))
    except BatchStoreUnavailableError as exc:
        return None, str(exc)
    if run is None:
        return None, f"source batch {source_event_id!r} no longer exists"
    items = list(run.get("items") or [])
    if not items:
        return None, f"source batch {source_event_id!r} has no documents"
    missing = [item["filename"] for item in items if not item.get("input_relkey")]
    if missing:
        return None, (
            "these source documents predate durable input storage and cannot be "
            f"re-run: {', '.join(missing[:5])}"
        )
    blobs = studio_store.default_blob_store()
    gone = [item["filename"] for item in items if not blobs.exists(str(item["input_relkey"]))]
    if gone:
        return None, f"input documents no longer in the store: {', '.join(gone[:5])}"

    data: dict[str, Any] = {
        "channel": f"batch:{uuid.uuid4().hex}",
        "tenant_id": schedule["tenant_id"],
        # Capped at BatchRun.name's String(200): the fired batch's claim
        # writes this verbatim, and Postgres raises on over-length.
        "name": f"scheduled: {schedule['name']}"[:200],
        "schema_name": schedule["schema_name"],
        "scheduled_by": schedule["id"],
        "scheduled_from": source_event_id,
        "inputs": [
            {"filename": item["filename"], "relkey": item["input_relkey"]} for item in items
        ],
    }
    for key, value in (schedule.get("selectors") or {}).items():
        if value:
            data[key] = value
    return data, None


__all__ = [
    "INTERVALS",
    "MAX_EVERY_N_MINUTES",
    "MIN_EVERY_N_MINUTES",
    "ScheduleStoreUnavailableError",
    "ScheduleValidationError",
    "create_schedule",
    "delete_schedule",
    "due_schedules",
    "ensure_batch_schedule_table",
    "get_schedule",
    "interval_delta",
    "list_schedules",
    "mark_fired",
    "scheduled_batch_event_data",
    "update_schedule",
]
