"""Named ``RoutingPolicy`` CRUD routes.

Route glue only -- persistence lives in ``docie_bench.studio.routing_policies``
(a different module; this one is the FastAPI-facing counterpart, distinct
package path so there's no actual name collision).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docie_bench.extract.routing import RoutingPolicy

from . import _shared

router = APIRouter()


@router.get("/routing-policies")
async def list_routing_policies_route() -> list[dict[str, Any]]:
    """Named ``RoutingPolicy``s saved via the Studio -- what the Benchmark
    tab's "Routing policy" picker can reference beyond a raw server-side
    filesystem path. Missing DATABASE_URL degrades to an empty list, same
    contract as ``GET /schemas/dynamic``/``GET /model-profiles``.
    """
    from docie_bench.studio.routing_policies import list_routing_policies

    return list_routing_policies()


@router.get("/routing-policies/{name}")
async def get_routing_policy_route(name: str) -> dict[str, Any]:
    from docie_bench.studio.routing_policies import get_routing_policy

    policy = get_routing_policy(name)
    if policy is None:
        raise HTTPException(
            status_code=404,
            detail=f"routing policy {name!r} not found",
            headers=_shared._DOMAIN_404,
        )
    return policy


class RoutingPolicyCreateRequest(BaseModel):
    # max_length matches RoutingPolicyRecord.name's String(64) column -- reject
    # an over-length name here with a 422 instead of a DB-layer DataError (the
    # same overflow shape #194's review caught for the routing_label column).
    name: str = Field(min_length=1, max_length=64)
    policy: RoutingPolicy


@router.post("/routing-policies", status_code=201)
async def create_routing_policy_route(payload: RoutingPolicyCreateRequest) -> dict[str, Any]:
    """Save a ``RoutingPolicy`` under a registry name so it can be referenced
    from a benchmark trigger afterward, instead of every run needing a
    server-side YAML file path.

    ``RoutingPolicy`` (extract.routing) is already fully validated -- FastAPI
    rejects a malformed policy (duplicate stage names, an unknown decision
    literal, etc.) before this handler body even runs. This route is the
    missing "define once, reuse by name" persistence layer, not new routing
    logic. Create-only, matching the dynamic-schema/pipeline-profile routes
    above: 409 on an existing name, no in-place update.
    """
    from docie_bench.studio.routing_policies import (
        RoutingPolicyConflictError,
        RoutingPolicyUnavailableError,
        create_routing_policy,
    )

    try:
        return create_routing_policy(payload.name, payload.policy)
    except RoutingPolicyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RoutingPolicyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/routing-policies/{name}")
async def delete_routing_policy_route(name: str) -> dict[str, Any]:
    from docie_bench.studio.routing_policies import (
        RoutingPolicyNotFoundError,
        RoutingPolicyUnavailableError,
        delete_routing_policy,
    )

    try:
        delete_routing_policy(name)
    except RoutingPolicyNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=str(exc), headers=_shared._DOMAIN_404
        ) from exc
    except RoutingPolicyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"deleted": name}
