"""Realtime subscription, run status/listing, comparison, and artifact routes."""

from __future__ import annotations

import os
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from docie_bench.inngest.realtime import subscription_token
from docie_bench.security import TenantDependency
from docie_bench.studio.store import RunStoreUnavailableError

from . import _shared

router = APIRouter()


@router.get("/realtime-token")
async def realtime_token(
    channel: Annotated[str, Query(min_length=1)],
    tenant: TenantDependency,
    topics: Annotated[list[str] | None, Query()] = None,
) -> Any:
    """Mint a realtime subscription token (authenticated).

    Every sibling trigger route requires a key; this route previously handed
    out subscription JWTs for an arbitrary channel string unauthenticated.
    Channels are unguessable uuid4 hexes, so it was exposure-by-guessing
    rather than a leak — but there is no reason for the one credential-minting
    route to be the only open one.
    """
    del tenant  # authenticated principal required; channels are per-run uuids
    try:
        return await subscription_token(channel, topics or _shared.DEFAULT_TOPICS)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.get("/runs/{event_id}")
async def event_runs(event_id: str, tenant: TenantDependency) -> Any:
    """Run status for an event.

    Benchmark runs have a durable index row (metrics + addressable artifact URIs)
    resolved here, tenant-scoped. Extraction runs have no durable row, so this
    falls back to proxying the Inngest server's run status.
    """
    store = _shared.default_run_store()
    if store.enabled:
        try:
            owner = store.run_owner(event_id)
        except RunStoreUnavailableError:
            owner = None
        if owner is not None:
            # Ownership is recorded (benchmark run row or extraction event owner):
            # serve it only to its owner. A cross-tenant id is 404, never proxied.
            if owner != tenant.tenant_id:
                raise HTTPException(
                    status_code=404, detail="Run not found", headers=_shared._DOMAIN_404
                )
            record = store.get_run(event_id, tenant_id=tenant.tenant_id)
            if record is not None:
                # Durable benchmark run: answer from the index.
                return record
            # Owned extraction run: fall through to the proxy below (scoped by the
            # ownership check we just passed).
        else:
            # Store is enabled but no ownership is recorded for this id: we cannot
            # prove the caller owns it, so refuse rather than leak another
            # principal's run status/output through the tenant-agnostic proxy.
            raise HTTPException(
                status_code=404, detail="Run not found", headers=_shared._DOMAIN_404
            )

    base = os.getenv("INNGEST_BASE_URL", "http://localhost:8288").rstrip("/")
    headers = {}
    signing_key = os.getenv("INNGEST_SIGNING_KEY")
    if signing_key:
        headers["Authorization"] = f"Bearer {signing_key}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{base}/v1/events/{event_id}/runs", headers=headers)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@router.get("/runs")
async def list_runs(
    tenant: TenantDependency,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> Any:
    """List this tenant's durable benchmark runs (metrics + artifact URIs)."""
    store = _shared.default_run_store()
    if not store.enabled:
        return []
    try:
        return store.list_runs(tenant_id=tenant.tenant_id, limit=limit)
    except RunStoreUnavailableError:
        return []


class ComparisonRequest(BaseModel):
    baseline_event_id: str
    candidate_event_id: str


@router.post("/comparisons")
async def create_comparison(payload: ComparisonRequest, tenant: TenantDependency) -> dict[str, Any]:
    """Compare two of this tenant's completed runs -- the CLI's ``compare_runs``
    logic (deltas, confidence intervals, sign tests, root causes), reachable
    without leaving the Studio.

    Stateless by design: each run's ``metrics_json`` already lives in Postgres
    (``StudioRun``), so recomputing the comparison on every call is cheap pure
    computation, not a second I/O round-trip. No budgets/promote/persistence
    yet -- this is the comparison-viewing slice; gating and a revisitable
    comparison id are deliberately deferred until there's a real caller for them.
    """
    from docie_bench.benchmark.comparison import build_comparison_payload

    store = _shared.default_run_store()
    if not store.enabled:
        raise HTTPException(
            status_code=503, detail="Studio run index requires a configured DATABASE_URL"
        )
    baseline = store.get_run(payload.baseline_event_id, tenant_id=tenant.tenant_id)
    candidate = store.get_run(payload.candidate_event_id, tenant_id=tenant.tenant_id)
    if baseline is None or candidate is None:
        raise HTTPException(status_code=404, detail="Run not found", headers=_shared._DOMAIN_404)
    for label, run in (("baseline", baseline), ("candidate", candidate)):
        if not run.get("metrics"):
            if run["status"] == "failed":
                # Permanently unusable -- distinct from "running" so a client
                # doesn't poll/retry a comparison that can never succeed.
                raise HTTPException(
                    status_code=422,
                    detail=f"{label} run {run['event_id']!r} failed and has no metrics "
                    f"to compare: {run.get('error') or 'no error detail recorded'}",
                )
            raise HTTPException(
                status_code=409,
                detail=f"{label} run {run['event_id']!r} has no metrics yet "
                f"(status={run['status']!r}) -- wait for it to complete",
            )
    try:
        return build_comparison_payload(
            baseline["metrics"],
            candidate["metrics"],
            baseline_meta={
                "event_id": baseline["event_id"],
                "dataset": baseline["dataset"],
                "model_profile": baseline["model_profile"],
                "created_at": baseline["created_at"],
            },
            candidate_meta={
                "event_id": candidate["event_id"],
                "dataset": candidate["dataset"],
                "model_profile": candidate["model_profile"],
                "created_at": candidate["created_at"],
            },
        )
    except ValueError as exc:
        # A malformed metrics_json row (shape drift, hand-edited DB row) has no
        # file-read boundary to reject it at the way the CLI's Path-based
        # compare_runs does -- surface it as a client-facing 422, not a 500.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/artifacts/{artifact_id}")
async def download_artifact(artifact_id: str, tenant: TenantDependency) -> Response:
    """Stream a run artifact (``report.html`` / ``predictions.jsonl`` / ``metrics.json``).

    Resolved purely by ``artifact_id -> DB row -> shared blob store`` (never a
    worker-local path), so it is reachable from any non-worker replica. A
    cross-tenant id returns 404 (not 403) so run existence is never confirmed.
    """
    store = _shared.default_run_store()
    if not store.enabled:
        raise HTTPException(
            status_code=404, detail="Artifact not found", headers=_shared._DOMAIN_404
        )
    try:
        resolved = store.open_artifact(artifact_id, tenant_id=tenant.tenant_id)
    except RunStoreUnavailableError:
        raise HTTPException(
            status_code=404, detail="Artifact not found", headers=_shared._DOMAIN_404
        ) from None
    if resolved is None:
        raise HTTPException(
            status_code=404, detail="Artifact not found", headers=_shared._DOMAIN_404
        )
    meta, content = resolved
    return Response(
        content=content,
        media_type=meta["media_type"],
        headers={"Content-Disposition": f'attachment; filename="{meta["name"]}"'},
    )
