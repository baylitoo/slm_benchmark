from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from docie_bench.orchestrator.schemas import (
    ClaimRequest,
    CompleteRequest,
    FailRequest,
    LeaseRequest,
    RunAction,
    RunCreate,
)
from docie_bench.orchestrator.service import (
    LeaseConflictError,
    NotFoundError,
    OrchestratorError,
    OrchestratorService,
)
from docie_bench.security import TenantDependency

router = APIRouter(prefix="/v1")
_service: OrchestratorService | None = None


def configure_orchestrator(service: OrchestratorService | None) -> None:
    global _service
    _service = service


def service() -> OrchestratorService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Orchestrator database is not configured")
    return _service


def translate_error(exc: OrchestratorError) -> HTTPException:
    if isinstance(exc, NotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, LeaseConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


@router.post("/experiments", status_code=201)
def create_experiment(payload: RunCreate) -> dict[str, Any]:
    try:
        return service().create_run(payload)
    except OrchestratorError as exc:
        raise translate_error(exc) from exc


@router.get("/experiments")
def query_experiments(
    owner: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    q: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return {"runs": service().query_runs(owner=owner, status=status, tag=tag, query=q, limit=limit)}


@router.get("/experiments/{run_id}")
def get_experiment(run_id: str) -> dict[str, Any]:
    try:
        return service().get_run(run_id)
    except OrchestratorError as exc:
        raise translate_error(exc) from exc


@router.post("/experiments/{run_id}/cancel")
def cancel_experiment(run_id: str, payload: RunAction) -> dict[str, Any]:
    try:
        return service().cancel_run(run_id, payload.reason)
    except OrchestratorError as exc:
        raise translate_error(exc) from exc


@router.post("/experiments/{run_id}/resume")
def resume_experiment(run_id: str, payload: RunAction) -> dict[str, Any]:
    try:
        return service().resume_run(run_id, payload.reason)
    except OrchestratorError as exc:
        raise translate_error(exc) from exc


@router.post("/workers/tasks/claim")
def claim_task(payload: ClaimRequest, tenant: TenantDependency) -> dict[str, Any] | None:
    # B2: lease owner is the authenticated principal, not the forgeable payload.
    return service().claim_task(
        worker_id=tenant.tenant_id,
        lease_seconds=payload.lease_seconds,
        run_id=payload.run_id,
    )


@router.post("/workers/tasks/{task_id}/heartbeat")
def heartbeat_task(task_id: str, payload: LeaseRequest, tenant: TenantDependency) -> dict[str, Any]:
    data = {**payload.model_dump(), "worker_id": tenant.tenant_id}  # B2
    try:
        return service().heartbeat(task_id=task_id, **data)
    except OrchestratorError as exc:
        raise translate_error(exc) from exc


@router.post("/workers/tasks/{task_id}/complete")
def complete_task(
    task_id: str, payload: CompleteRequest, tenant: TenantDependency
) -> dict[str, Any]:
    data = {**payload.model_dump(), "worker_id": tenant.tenant_id}  # B2
    try:
        return service().complete_task(task_id=task_id, **data)
    except OrchestratorError as exc:
        raise translate_error(exc) from exc


@router.post("/workers/tasks/{task_id}/fail")
def fail_task(task_id: str, payload: FailRequest, tenant: TenantDependency) -> dict[str, Any]:
    data = {**payload.model_dump(), "worker_id": tenant.tenant_id}  # B2
    try:
        return service().fail_task(task_id=task_id, **data)
    except OrchestratorError as exc:
        raise translate_error(exc) from exc


@router.post("/workers/recover")
def recover_tasks() -> dict[str, int]:
    return {"recovered": service().recover_expired()}
