"""Per-deployment usage aggregates (the Observability tab's Usage section)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from docie_bench.security import TenantDependency

router = APIRouter()


@router.get("/usage")
async def usage_summary_route(
    tenant: TenantDependency,
    window: str = Query(default="24h"),
) -> dict[str, Any]:
    """This tenant's serving usage over a bounded window, aggregated per
    deployment/profile at read time from the raw ``usage_records`` ledger:
    requests, errors, token totals (prompt/completion), avg + p95 latency,
    and last-used. Rows are written by the serving surfaces themselves
    (chat/embed/rerank in chat_api.py, extract in api.py). Missing
    DATABASE_URL degrades to an empty listing, same contract as every other
    Studio listing route (``GET /seeds``/``GET /batches``/etc).
    """
    from docie_bench.studio.usage_store import WINDOW_HOURS, usage_summary

    hours = WINDOW_HOURS.get(window)
    if hours is None:
        raise HTTPException(
            status_code=400,
            detail=f"window must be one of {', '.join(sorted(WINDOW_HOURS))} (got {window!r})",
        )
    return {
        "window": window,
        "deployments": usage_summary(tenant_id=tenant.tenant_id, window_hours=hours),
    }
