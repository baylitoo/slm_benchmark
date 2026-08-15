"""Durable seed-download listing (Ollama / Hugging Face) -- the Downloads tab."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from docie_bench.security import TenantDependency

router = APIRouter()


@router.get("/seeds")
async def list_seeds(tenant: TenantDependency) -> list[dict[str, Any]]:
    """This tenant's recent seed-download jobs (running, completed, failed).

    The realtime ``progress`` topic and ``GET /v1/serving/seed-progress``
    cover a download's LIVE percentage while it's in flight; this is the
    durable counterpart -- what happened to a seed, including its error text,
    survives closing the panel or navigating away. Missing DATABASE_URL
    degrades to an empty list, same contract as every other Studio listing
    route (``GET /datasets``/``GET /model-profiles``/etc).
    """
    from docie_bench.studio.seed_store import list_seed_runs

    return list_seed_runs(tenant_id=tenant.tenant_id)
