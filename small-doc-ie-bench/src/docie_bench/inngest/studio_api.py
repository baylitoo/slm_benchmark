"""FastAPI glue for DocIE Studio: trigger jobs + consume results.

Mounted on the main API (``app.include_router(studio_router)``). Three routes
give the frontend everything it needs:

  - ``POST /v1/studio/extract``        -- fire ``doc/extract.requested``; returns
                                          the event id(s) and the realtime channel.
  - ``GET  /v1/studio/realtime-token`` -- mint a subscription token (the "hook").
  - ``GET  /v1/studio/runs/{id}``      -- polling fallback: proxy the Inngest
                                          server's run status for an event.

Realtime is best-effort; if the experimental API is unavailable the token route
returns 501 and the frontend falls back to polling ``/runs``.
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated, Any

import httpx
import inngest
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from docie_bench.inngest.client import inngest_client
from docie_bench.inngest.functions import benchmark_idempotency_key
from docie_bench.inngest.realtime import (
    TOPIC_ERROR,
    TOPIC_PROGRESS,
    TOPIC_RESULT,
    TOPIC_STATUS,
    subscription_token,
)
from docie_bench.security import TenantDependency
from docie_bench.serving.resources import DEFAULT_DEPLOY_CONTEXT_LENGTH
from docie_bench.studio.store import RunStoreUnavailableError, default_run_store

router = APIRouter(prefix="/v1/studio", tags=["studio"])

DEFAULT_TOPICS = [TOPIC_STATUS, TOPIC_PROGRESS, TOPIC_RESULT, TOPIC_ERROR]
EXTRACT_EVENT = "doc/extract.requested"
BENCHMARK_EVENT = "benchmark/run.requested"
DEPLOY_EVENT = "serving/deploy.requested"
SEED_EVENT = "serving/seed.requested"


class ExtractRequest(BaseModel):
    text: str | None = None
    content_b64: str | None = None
    filename: str | None = None
    schema_name: str = "invoice"
    # Explicit live-deployment selector (a DeploymentRecord ``spec.name``). Wins
    # over ``model_profile``; forwarded verbatim into the event data (auto-included
    # by ``model_dump(exclude_none=True)`` below) for the worker's resolver.
    deployment: str | None = None
    model_profile: str | None = None
    ocr_backend: str | None = None
    language: str | None = None


class TriggerResponse(BaseModel):
    event_ids: list[str]
    channel: str
    topics: list[str]


def _record_event_owners(event_ids: list[str], tenant_id: str) -> None:
    """Bind each triggered event id to its principal so ``/runs`` can be scoped.

    Best-effort: if the run index is unconfigured the run-status proxy stays
    unscoped in that degraded mode (see ``event_runs``).
    """
    store = default_run_store()
    if not store.enabled:
        return
    try:
        for event_id in event_ids:
            store.record_event_owner(event_id=event_id, tenant_id=tenant_id)
    except RunStoreUnavailableError:
        pass


@router.post("/extract", response_model=TriggerResponse)
async def trigger_extract(payload: ExtractRequest, tenant: TenantDependency) -> TriggerResponse:
    if not payload.text and not payload.content_b64:
        raise HTTPException(status_code=422, detail="Provide either 'text' or 'content_b64'")
    channel = f"extract:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    # Bind provenance to the authenticated principal (mirrors trigger_benchmark) so
    # the worker's audit row is tenant-scoped rather than anonymous.
    data["tenant_id"] = tenant.tenant_id
    ids = await inngest_client.send(inngest.Event(name=EXTRACT_EVENT, data=data))
    # Record ownership so the run-status proxy is tenant-scoped: an extraction run
    # has no durable StudioRun row, so this is its only ownership signal.
    _record_event_owners(list(ids), tenant.tenant_id)
    return TriggerResponse(event_ids=list(ids), channel=channel, topics=DEFAULT_TOPICS)


class BenchmarkRequest(BaseModel):
    dataset: str
    split: str | None = None
    model_profile: str | None = None
    schema_name: str = "invoice"
    concurrency: int = 1
    repeat: int = 1
    language: str | None = None
    # Optional override; pass a nonce to force a fresh run of an identical request.
    idempotency_key: str | None = None


@router.post("/benchmark", response_model=TriggerResponse)
async def trigger_benchmark(payload: BenchmarkRequest, tenant: TenantDependency) -> TriggerResponse:
    channel = f"benchmark:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    # Bind provenance to the authenticated principal (never a client body field)
    # so downloads/listing can be tenant-scoped and a forged event can only
    # mis-file the attacker's own run, not read a victim's (B2, extended).
    data["tenant_id"] = tenant.tenant_id
    # Materialize the idempotency key here so both the platform-level Inngest dedup
    # (event.data.idempotency_key) and the worker's durable claim use the same key.
    # Namespace it by the authenticated principal: without this, two tenants firing
    # an identical request would collide on one global key — denying one tenant's
    # run and leaking the other's record through the dedup short-circuit. Prefixing
    # covers both the derived and the client-supplied key branches.
    base_key = f"{tenant.tenant_id}:{benchmark_idempotency_key(data)}"
    # Rotate the key once the prior run for it has terminally failed, so a
    # legitimate re-request is not deduped away for the 24h window (a genuine
    # duplicate of an in-flight/succeeded run still resolves to the same key).
    store = default_run_store()
    effective_key = base_key
    if store.enabled:
        try:
            effective_key = store.effective_idempotency_key(base_key)
        except RunStoreUnavailableError:
            effective_key = base_key
    data["idempotency_key"] = effective_key
    ids = await inngest_client.send(inngest.Event(name=BENCHMARK_EVENT, data=data))
    _record_event_owners(list(ids), tenant.tenant_id)
    return TriggerResponse(event_ids=list(ids), channel=channel, topics=DEFAULT_TOPICS)


class DeployRequest(BaseModel):
    model: str
    name: str | None = None
    runtime: str | None = None
    # None => the control plane auto-allocates a free port at deploy time; the
    # UI sends no port unless the operator explicitly overrides it. model_dump(
    # exclude_none=True) at trigger time drops a None so the worker sees no port.
    port: int | None = None
    # The shared deploy-default context (resources.DEFAULT_DEPLOY_CONTEXT_LENGTH)
    # — the SAME constant the sizing engine prices uncalibrated fits at, so the
    # fit table and a default deploy consume the same KV budget.
    context_length: int = DEFAULT_DEPLOY_CONTEXT_LENGTH
    replicas: int = 1


@router.post("/deploy", response_model=TriggerResponse)
async def trigger_deploy(payload: DeployRequest, tenant: TenantDependency) -> TriggerResponse:
    channel = f"deploy:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    ids = await inngest_client.send(inngest.Event(name=DEPLOY_EVENT, data=data))
    # Deploy has no durable StudioRun row; record ownership so the triggering
    # principal can poll its status via /runs and a cross-tenant id is 404 (never
    # proxied). Every event-producing trigger records an owner for parity — an
    # unregistered one would 404 its own status polling (the Deploy.tsx fallback).
    _record_event_owners(list(ids), tenant.tenant_id)
    return TriggerResponse(event_ids=list(ids), channel=channel, topics=DEFAULT_TOPICS)


class SeedOllamaRequest(BaseModel):
    reference: str  # e.g. "qwen2.5:1.5b" or "hf.co/numind/NuExtract3-GGUF:Q4_K_M"
    name: str  # store entry name
    family: str = "openai_chat"
    # Optional on-disk vision projector (GGUF) for needs_mmproj families whose
    # pulled model ships no projector layer (e.g. a separately-downloaded
    # NuExtract3 mmproj). Path must be reachable inside the serving container.
    mmproj: str | None = None


@router.post("/seed-ollama", response_model=TriggerResponse)
async def trigger_seed_ollama(
    payload: SeedOllamaRequest, tenant: TenantDependency
) -> TriggerResponse:
    channel = f"seed:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    ids = await inngest_client.send(inngest.Event(name=SEED_EVENT, data=data))
    # Seed has no durable StudioRun row; record ownership so the triggering
    # principal can poll its status via /runs (no 404 regression) while a
    # cross-tenant id stays 404 rather than leaking through the Inngest proxy.
    _record_event_owners(list(ids), tenant.tenant_id)
    return TriggerResponse(event_ids=list(ids), channel=channel, topics=DEFAULT_TOPICS)


@router.get("/realtime-token")
async def realtime_token(
    channel: Annotated[str, Query(min_length=1)],
    topics: Annotated[list[str] | None, Query()] = None,
) -> Any:
    try:
        return await subscription_token(channel, topics or DEFAULT_TOPICS)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.get("/runs/{event_id}")
async def event_runs(event_id: str, tenant: TenantDependency) -> Any:
    """Run status for an event.

    Benchmark runs have a durable index row (metrics + addressable artifact URIs)
    resolved here, tenant-scoped. Extraction runs have no durable row, so this
    falls back to proxying the Inngest server's run status (the array shape the
    Playground polling client expects).
    """
    store = default_run_store()
    if store.enabled:
        try:
            owner = store.run_owner(event_id)
        except RunStoreUnavailableError:
            owner = None
        if owner is not None:
            # Ownership is recorded (benchmark run row or extraction event owner):
            # serve it only to its owner. A cross-tenant id is 404, never proxied.
            if owner != tenant.tenant_id:
                raise HTTPException(status_code=404, detail="Run not found")
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
            raise HTTPException(status_code=404, detail="Run not found")

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
    store = default_run_store()
    if not store.enabled:
        return []
    try:
        return store.list_runs(tenant_id=tenant.tenant_id, limit=limit)
    except RunStoreUnavailableError:
        return []


@router.get("/artifacts/{artifact_id}")
async def download_artifact(artifact_id: str, tenant: TenantDependency) -> Response:
    """Stream a run artifact (``report.html`` / ``predictions.jsonl`` / ``metrics.json``).

    Resolved purely by ``artifact_id -> DB row -> shared blob store`` (never a
    worker-local path), so it is reachable from any non-worker replica. A
    cross-tenant id returns 404 (not 403) so run existence is never confirmed.
    """
    store = default_run_store()
    if not store.enabled:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        resolved = store.open_artifact(artifact_id, tenant_id=tenant.tenant_id)
    except RunStoreUnavailableError:
        raise HTTPException(status_code=404, detail="Artifact not found") from None
    if resolved is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    meta, content = resolved
    return Response(
        content=content,
        media_type=meta["media_type"],
        headers={"Content-Disposition": f'attachment; filename="{meta["name"]}"'},
    )


__all__ = ["router"]
