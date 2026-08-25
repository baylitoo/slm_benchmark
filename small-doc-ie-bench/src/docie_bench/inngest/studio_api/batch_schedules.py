"""Batch schedules: recurring batch extraction, saved as a config and driven
by the worker's once-a-minute cron (``functions.batch_schedule_tick_job``).

A schedule references a previously-run batch (``source_event_id``) and re-runs
that batch's durably stored documents (``BatchItem.input_relkey`` in the
shared blob store -- the same no-re-upload seam ``retry-failed`` uses) on an
interval. Every firing is an ordinary ``doc/batch.requested`` event, so a
scheduled run is a normal ``BatchRun`` with per-item state, results and
retry-failed.

Route glue only -- persistence and the firing material live in
``docie_bench.studio.schedule_store``. Same DATABASE_URL contract as the
batch routes: schedule rows are the product, so persistence trouble is a 503
here, while the cron side degrades gracefully.
"""

from __future__ import annotations

import uuid
from typing import Any

import inngest
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docie_bench.inngest.client import inngest_client, send_or_503
from docie_bench.security import TenantDependency
from docie_bench.studio.batch_store import BatchStoreUnavailableError, get_batch_run
from docie_bench.studio.schedule_store import (
    ScheduleStoreUnavailableError,
    ScheduleValidationError,
    create_schedule,
    delete_schedule,
    get_schedule,
    interval_delta,
    list_schedules,
    mark_fired,
    scheduled_batch_event_data,
    update_schedule,
)

from . import _shared
from .batch import BATCH_EVENT

router = APIRouter()


class BatchScheduleCreateRequest(BaseModel):
    """A recurring re-run of ``source_event_id``'s documents. Selectors
    default to the source batch's own (``selectors_json``); an explicit
    model override here replaces the model selector, mirroring
    ``retry-failed`` -- e.g. schedule the nightly re-run on the stronger
    model."""

    name: str | None = Field(default=None, max_length=200)
    source_event_id: str = Field(min_length=1, max_length=128)
    # "hourly" | "daily" | "weekly" | "every_n_minutes" (+ the minute count).
    interval: str = Field(max_length=32)
    every_n_minutes: int | None = None
    enabled: bool = True
    deployment: str | None = None
    model_profile: str | None = None
    routing_policy: str | None = None


class BatchSchedulePatchRequest(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    enabled: bool | None = None
    interval: str | None = Field(default=None, max_length=32)
    every_n_minutes: int | None = None


@router.post("/batch-schedules", status_code=201)
async def create_batch_schedule(
    payload: BatchScheduleCreateRequest, tenant: TenantDependency
) -> dict[str, Any]:
    override = [payload.deployment, payload.model_profile, payload.routing_policy]
    if sum(1 for sel in override if sel) > 1:
        raise HTTPException(
            status_code=400,
            detail="'deployment', 'model_profile' and 'routing_policy' are mutually exclusive",
        )
    try:
        interval_delta(payload.interval, payload.every_n_minutes)
    except ScheduleValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        run = get_batch_run(payload.source_event_id, tenant_id=tenant.tenant_id)
    except BatchStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"batch {payload.source_event_id!r} not found",
            headers=_shared._DOMAIN_404,
        )
    items = run.get("items") or []
    missing = [item["filename"] for item in items if not item.get("input_relkey")]
    if not items or missing:
        raise HTTPException(
            status_code=409,
            detail=(
                "this batch's documents predate durable input storage and cannot be "
                "scheduled without a re-upload"
                + (f": {', '.join(missing[:5])}" if missing else "")
            ),
        )

    selectors = dict(run.get("selectors") or {})
    if any(override):
        for key in ("deployment", "model_profile", "routing_policy"):
            selectors.pop(key, None)
        for key, value in (
            ("deployment", payload.deployment),
            ("model_profile", payload.model_profile),
            ("routing_policy", payload.routing_policy),
        ):
            if value:
                selectors[key] = value

    try:
        return create_schedule(
            tenant_id=tenant.tenant_id,
            # Hard cap: the derived default can exceed String(200) when the
            # source batch's own name is near the limit (Postgres raises on
            # over-length, unlike sqlite -- see the routing_label precedent).
            name=(payload.name or f"re-run: {run['name']}")[:200],
            source_event_id=payload.source_event_id,
            schema_name=str(run["schema_name"]),
            selectors=selectors or None,
            interval=payload.interval,
            every_n_minutes=payload.every_n_minutes,
            enabled=payload.enabled,
        )
    except ScheduleStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/batch-schedules")
async def list_batch_schedules(tenant: TenantDependency) -> list[dict[str, Any]]:
    """This tenant's schedules, newest first, with next/last run state."""
    try:
        return list_schedules(tenant_id=tenant.tenant_id)
    except ScheduleStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/batch-schedules/{schedule_id}")
async def patch_batch_schedule(
    schedule_id: str, payload: BatchSchedulePatchRequest, tenant: TenantDependency
) -> dict[str, Any]:
    """Enable/disable, rename, or change the interval. An interval change --
    or a re-enable -- reschedules ``next_run_at`` from now, never a
    catch-up firing."""
    fields = payload.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=422, detail="empty patch: nothing to change")
    kwargs: dict[str, Any] = {
        key: fields[key] for key in ("name", "enabled", "interval") if key in fields
    }
    if "every_n_minutes" in fields:
        kwargs["every_n_minutes"] = fields["every_n_minutes"]
    try:
        updated = update_schedule(schedule_id, tenant_id=tenant.tenant_id, **kwargs)
    except ScheduleValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ScheduleStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(
            status_code=404,
            detail=f"schedule {schedule_id!r} not found",
            headers=_shared._DOMAIN_404,
        )
    return updated


@router.delete("/batch-schedules/{schedule_id}")
async def delete_batch_schedule(schedule_id: str, tenant: TenantDependency) -> dict[str, Any]:
    try:
        deleted = delete_schedule(schedule_id, tenant_id=tenant.tenant_id)
    except ScheduleStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"schedule {schedule_id!r} not found",
            headers=_shared._DOMAIN_404,
        )
    return {"deleted": schedule_id}


@router.post("/batch-schedules/{schedule_id}/run-now", response_model=_shared.TriggerResponse)
async def run_batch_schedule_now(
    schedule_id: str, tenant: TenantDependency
) -> _shared.TriggerResponse:
    """Fire the schedule immediately, without touching its cadence:
    ``last_run_at``/``last_event_id`` are stamped but ``next_run_at`` stays
    where the interval put it. Exactly the event the cron would send."""
    try:
        schedule = get_schedule(schedule_id, tenant_id=tenant.tenant_id)
    except ScheduleStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if schedule is None:
        raise HTTPException(
            status_code=404,
            detail=f"schedule {schedule_id!r} not found",
            headers=_shared._DOMAIN_404,
        )
    data, error = scheduled_batch_event_data(schedule)
    if data is None:
        # The source batch (or its stored documents) is gone -- same honest
        # "the referenced bytes no longer exist" contract as retry-failed.
        raise HTTPException(status_code=410, detail=str(error))
    # A run-now gets its own channel (fresh uuid) so two clicks never share a
    # realtime stream with each other or with the cron's own firings.
    data["channel"] = f"batch:{uuid.uuid4().hex}"
    ids = await send_or_503(inngest_client, inngest.Event(name=BATCH_EVENT, data=data))
    event_ids = [str(i) for i in ids]
    _shared._record_event_owners(event_ids, tenant.tenant_id)
    mark_fired(schedule_id, event_id=event_ids[0] if event_ids else None, advance=False)
    return _shared.TriggerResponse(
        event_ids=event_ids, channel=str(data["channel"]), topics=_shared.DEFAULT_TOPICS
    )


__all__ = [
    "BatchScheduleCreateRequest",
    "BatchSchedulePatchRequest",
    "create_batch_schedule",
    "delete_batch_schedule",
    "list_batch_schedules",
    "patch_batch_schedule",
    "router",
    "run_batch_schedule_now",
]
