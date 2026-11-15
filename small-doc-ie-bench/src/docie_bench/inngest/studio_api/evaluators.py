"""Evaluator CRUD routes. Persistence lives in ``docie_bench.studio.evaluator_store``."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from docie_bench.evaluators.spec import EvaluatorSpec

from . import _shared

router = APIRouter()


def _check_agents(spec: EvaluatorSpec) -> None:
    """Every target must be an existing agent whose runs may be stored.

    A ``proxy_security`` agent exists to keep raw input from leaving the
    request; storing that input for evaluation would undo it.
    """
    from docie_bench.agents.registry import AgentNotFoundError, AgentRegistry

    registry = AgentRegistry()
    for name in spec.agents:
        try:
            agent = registry.get(name)
        except AgentNotFoundError as exc:
            raise HTTPException(status_code=422, detail=f"unknown agent {name!r}") from exc
        if agent.kind == "proxy_security":
            raise HTTPException(
                status_code=422,
                detail=f"agent {name!r} is a proxy_security agent; its runs are never stored",
            )


@router.get("/evaluators")
async def list_evaluators_route() -> list[dict[str, Any]]:
    from docie_bench.studio.evaluator_store import list_evaluators

    return list_evaluators()


@router.get("/evaluators/templates")
async def evaluator_templates_route() -> dict[str, Any]:
    from docie_bench.evaluators.templates import CATEGORIES, TEMPLATES

    return {"categories": CATEGORIES, "templates": TEMPLATES}


@router.get("/evaluators/{name}/results")
async def evaluator_results_route(name: str, limit: int = 100) -> list[dict[str, Any]]:
    from docie_bench.studio.agent_run_store import list_evaluations

    return list_evaluations(name, limit=max(1, min(limit, 500)))


@router.get("/agent-runs/{run_id}")
async def agent_run_route(run_id: str) -> dict[str, Any]:
    """A stored run, for the review view. 404 once it has expired."""
    from docie_bench.studio.agent_run_store import get_agent_run

    run = get_agent_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"agent run {run_id!r} not found or expired",
            headers=_shared._DOMAIN_404,
        )
    return run


@router.get("/evaluators/{name}")
async def get_evaluator_route(name: str) -> dict[str, Any]:
    from docie_bench.studio.evaluator_store import get_evaluator

    spec = get_evaluator(name)
    if spec is None:
        raise HTTPException(
            status_code=404, detail=f"evaluator {name!r} not found", headers=_shared._DOMAIN_404
        )
    return spec.model_dump(mode="json")


@router.post("/evaluators", status_code=201)
async def create_evaluator_route(spec: EvaluatorSpec) -> dict[str, Any]:
    from docie_bench.studio.evaluator_store import (
        EvaluatorConflictError,
        EvaluatorUnavailableError,
        create_evaluator,
    )

    _check_agents(spec)
    try:
        return create_evaluator(spec)
    except EvaluatorConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EvaluatorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.put("/evaluators/{name}")
async def update_evaluator_route(name: str, spec: EvaluatorSpec) -> dict[str, Any]:
    from docie_bench.studio.evaluator_store import (
        EvaluatorNotFoundError,
        EvaluatorStoreError,
        EvaluatorUnavailableError,
        update_evaluator,
    )

    _check_agents(spec)
    try:
        return update_evaluator(name, spec)
    except EvaluatorNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc), headers=_shared._DOMAIN_404) from exc
    except EvaluatorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EvaluatorStoreError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/evaluators/{name}")
async def delete_evaluator_route(name: str) -> dict[str, Any]:
    from docie_bench.studio.evaluator_store import (
        EvaluatorNotFoundError,
        EvaluatorUnavailableError,
        delete_evaluator,
    )

    try:
        delete_evaluator(name)
    except EvaluatorNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc), headers=_shared._DOMAIN_404) from exc
    except EvaluatorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"deleted": name}
