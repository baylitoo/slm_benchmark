"""Benchmark trigger route."""

from __future__ import annotations

import uuid
from typing import Any

import inngest
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from docie_bench.inngest.client import inngest_client, send_or_503
from docie_bench.inngest.functions import benchmark_idempotency_key
from docie_bench.security import TenantDependency
from docie_bench.studio.store import RunStoreUnavailableError

from . import _shared

router = APIRouter()


class BenchmarkRequest(BaseModel):
    dataset: str
    split: str | None = None
    model_profile: str | None = None
    # Server-side path to a routing-policy YAML (benchmark.routing_config.
    # load_routing_policy's own format -- see configs/routing-policy.example.yaml).
    # Mutually exclusive with model_profile and routing_policy_name below, same
    # rule the CLI's --routing-policy/--model-profile pair already enforces
    # (cli.py) -- a policy multi-stage-routes a document through several
    # profiles; picking one profile up front makes no sense alongside that.
    routing_policy: str | None = None
    # Name of a RoutingPolicy saved via POST /routing-policies -- the
    # discoverable alternative to routing_policy's raw filesystem path.
    # Resolved to the saved policy at worker time (inngest.functions._run_
    # benchmark_job), not here, since resolution needs a DB session.
    routing_policy_name: str | None = None
    schema_name: str = "invoice"
    concurrency: int = 1
    repeat: int = 1
    language: str | None = None
    # Optional override; pass a nonce to force a fresh run of an identical request.
    idempotency_key: str | None = None


@router.post("/benchmark", response_model=_shared.TriggerResponse)
async def trigger_benchmark(
    payload: BenchmarkRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    routing_fields = [payload.routing_policy, payload.routing_policy_name, payload.model_profile]
    if sum(1 for f in routing_fields if f) > 1:
        raise HTTPException(
            status_code=422,
            detail="routing_policy, routing_policy_name, and model_profile are mutually "
            "exclusive -- a routing policy already selects which profile(s) a document "
            "runs through",
        )
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
    store = _shared.default_run_store()
    effective_key = base_key
    if store.enabled:
        try:
            effective_key = store.effective_idempotency_key(base_key)
        except RunStoreUnavailableError:
            effective_key = base_key
    data["idempotency_key"] = effective_key
    ids = await send_or_503(
        inngest_client, inngest.Event(name=_shared.BENCHMARK_EVENT, data=data)
    )
    _shared._record_event_owners(list(ids), tenant.tenant_id)
    return _shared.TriggerResponse(
        event_ids=list(ids), channel=channel, topics=_shared.DEFAULT_TOPICS
    )
