from __future__ import annotations

import asyncio
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docie_bench.api import app
from docie_bench.orchestrator.api import configure_orchestrator
from docie_bench.orchestrator.artifacts import LocalArtifactStore
from docie_bench.orchestrator.schemas import RunCreate, TaskSpec
from docie_bench.orchestrator.service import LeaseConflictError, OrchestratorService
from docie_bench.orchestrator.worker import ArtifactOutput, BenchmarkWorker, TaskOutput
from docie_bench.storage.db import Base


@pytest.fixture
def service(tmp_path: Path) -> OrchestratorService:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'orchestrator.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return OrchestratorService(sessionmaker(engine, expire_on_commit=False))


def create_run(service: OrchestratorService, count: int = 2, max_attempts: int = 3) -> dict:
    return service.create_run(
        RunCreate(
            name="nightly invoice baseline",
            owner="bench-team",
            tags=["nightly", "invoice"],
            notes="regression candidate",
            metadata={"commit": "abc123"},
            manifest={"dataset": "invoices-v2", "model": "tiny"},
            tasks=[
                TaskSpec(
                    key=f"tiny:doc-{index}", payload={"doc_id": index}, max_attempts=max_attempts
                )
                for index in range(count)
            ],
        )
    )


def test_run_survives_service_restart_and_is_queryable(service: OrchestratorService) -> None:
    created = create_run(service)
    restarted = OrchestratorService(service.sessions)

    found = restarted.query_runs(owner="bench-team", tag="nightly", query="invoice")
    run = restarted.get_run(created["id"])

    assert found[0]["id"] == created["id"]
    assert run["manifest"]["dataset"] == "invoices-v2"
    assert run["progress"] == {"total": 2, "by_status": {"queued": 2}}
    assert run["events"][0]["type"] == "run.created"


def test_concurrent_workers_claim_distinct_tasks_and_stale_completion_is_rejected(
    service: OrchestratorService,
) -> None:
    run = create_run(service)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(
            pool.map(
                lambda worker: service.claim_task(worker_id=worker, lease_seconds=30),
                ["worker-a", "worker-b"],
            )
        )

    assert all(claims)
    assert len({claim["id"] for claim in claims if claim}) == 2
    first = claims[0]
    assert first is not None
    service.complete_task(
        task_id=first["id"],
        worker_id=first["worker_id"],
        lease_token=first["lease_token"],
        result={"score": 1.0},
    )
    with pytest.raises(LeaseConflictError):
        service.complete_task(
            task_id=first["id"],
            worker_id=first["worker_id"],
            lease_token=first["lease_token"],
            result={"score": 0.0},
        )
    assert service.get_run(run["id"])["progress"]["by_status"] == {"completed": 1, "leased": 1}


def test_worker_death_lease_expiry_recovers_and_honors_retry_limit(
    service: OrchestratorService,
) -> None:
    run = create_run(service, count=1, max_attempts=2)
    first = service.claim_task(worker_id="dead-worker", lease_seconds=30)
    assert first is not None

    recovered = service.recover_expired(now=first["lease_expires_at"] + dt.timedelta(seconds=1))
    second = service.claim_task(worker_id="replacement", lease_seconds=30)
    assert recovered == 1
    assert second is not None
    assert second["id"] == first["id"]
    assert second["attempt"] == 2

    service.recover_expired(now=second["lease_expires_at"] + dt.timedelta(seconds=1))
    final = service.get_run(run["id"])
    assert final["status"] == "failed"
    assert final["tasks"][0]["status"] == "failed"
    assert [event["type"] for event in final["events"]].count("task.recovered") == 1


def test_heartbeat_extends_lease_and_failure_retries(service: OrchestratorService) -> None:
    create_run(service, count=1, max_attempts=2)
    claim = service.claim_task(worker_id="worker", lease_seconds=10)
    assert claim is not None

    heartbeat = service.heartbeat(
        task_id=claim["id"],
        worker_id="worker",
        lease_token=claim["lease_token"],
        lease_seconds=60,
    )
    failed = service.fail_task(
        task_id=claim["id"],
        worker_id="worker",
        lease_token=claim["lease_token"],
        error="endpoint unavailable",
    )

    assert heartbeat["lease_expires_at"] > claim["lease_expires_at"]
    assert failed["status"] == "queued"
    assert failed["attempt"] == 1
    assert failed["error"] == "endpoint unavailable"


def test_cancel_invalidates_lease_and_resume_requeues(service: OrchestratorService) -> None:
    run = create_run(service, count=1)
    claim = service.claim_task(worker_id="worker", lease_seconds=30)
    assert claim is not None

    cancelled = service.cancel_run(run["id"], "operator request")
    with pytest.raises(LeaseConflictError):
        service.complete_task(
            task_id=claim["id"],
            worker_id="worker",
            lease_token=claim["lease_token"],
            result={},
        )
    resumed = service.resume_run(run["id"], "capacity restored")

    assert cancelled["status"] == "cancelled"
    assert resumed["tasks"][0]["status"] == "queued"
    assert resumed["events"][-1]["type"] == "run.resumed"


def test_worker_records_artifact_and_retries_after_executor_failure(
    service: OrchestratorService, tmp_path: Path
) -> None:
    run = create_run(service, count=1, max_attempts=2)
    calls = 0

    async def flaky_executor(payload: dict) -> TaskOutput:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("worker process failed")
        return TaskOutput(
            result={"doc_id": payload["doc_id"], "score": 0.75},
            artifacts=[ArtifactOutput("metrics.json", b'{"score":0.75}', "application/json")],
        )

    worker = BenchmarkWorker(
        worker_id="local-1",
        service=service,
        executor=flaky_executor,
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        lease_seconds=30,
    )

    assert asyncio.run(worker.run_once())
    assert asyncio.run(worker.run_once())
    completed = service.get_run(run["id"])
    assert completed["status"] == "completed", completed["tasks"][0]["error"]
    assert completed["tasks"][0]["attempt"] == 2
    assert completed["artifacts"][0]["name"] == "metrics.json"
    assert Path(completed["artifacts"][0]["uri"].removeprefix("file:///")).exists()


def test_competing_attempt_artifacts_do_not_overwrite_each_other(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")

    first = store.put(run_id="run", task_id="task", name="result.json", content=b"first")
    second = store.put(run_id="run", task_id="task", name="result.json", content=b"second")

    assert first.uri != second.uri
    assert Path(first.uri.removeprefix("file:///")).read_bytes() == b"first"
    assert Path(second.uri.removeprefix("file:///")).read_bytes() == b"second"


def test_worker_routes_require_auth_and_bind_identity_to_principal(
    service: OrchestratorService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docie_bench import security
    from docie_bench.security import TenantQuotaManager

    manager = TenantQuotaManager(
        api_keys={"secret": "tenant-x"},
        auth_required=True,
        requests_per_window=100,
        window_seconds=60,
        max_concurrent=10,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)
    configure_orchestrator(service)
    client = TestClient(app)
    headers = {"X-API-Key": "secret"}

    # B1: the worker lease surface is closed without a valid key.
    assert client.post("/v1/workers/tasks/claim", json={"worker_id": "spoof"}).status_code == 401

    client.post(
        "/v1/experiments",
        json={
            "name": "auth-run",
            "owner": "api-user",
            "tasks": [{"key": "model:doc", "payload": {"doc": "1"}}],
        },
        headers=headers,
    )
    # B2: the forgeable payload worker_id is ignored; the lease owner is the
    # authenticated principal (tenant-x), not "spoof".
    claim = client.post(
        "/v1/workers/tasks/claim",
        json={"worker_id": "spoof", "lease_seconds": 30},
        headers=headers,
    ).json()
    assert claim["worker_id"] == "tenant-x"

    configure_orchestrator(None)


def test_experiment_and_worker_http_api(service: OrchestratorService) -> None:
    configure_orchestrator(service)
    client = TestClient(app)
    response = client.post(
        "/v1/experiments",
        json={
            "name": "api-run",
            "owner": "api-user",
            "tags": ["smoke"],
            "tasks": [{"key": "model:doc", "payload": {"doc": "1"}}],
        },
    )
    assert response.status_code == 201
    run_id = response.json()["id"]

    claim = client.post(
        "/v1/workers/tasks/claim",
        json={"worker_id": "remote-worker", "lease_seconds": 30, "run_id": run_id},
    ).json()
    completed = client.post(
        f"/v1/workers/tasks/{claim['id']}/complete",
        json={
            "worker_id": "remote-worker",
            "lease_token": claim["lease_token"],
            "result": {"metric": 1},
        },
    )
    queried = client.get("/v1/experiments", params={"owner": "api-user", "tag": "smoke"})

    assert completed.status_code == 200
    assert queried.json()["runs"][0]["status"] == "completed"
    configure_orchestrator(None)
