"""Agent-run reviews share the queue with extraction reviews but stay apart from them."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text

from docie_bench.review import (
    ReviewValidationError,
    claim_review,
    correct_review,
    decide_review,
    enqueue_agent_run_review,
    enqueue_review,
    export_annotations,
    list_reviews,
    review_metrics,
)
from docie_bench.schemas.review import FieldCorrection, ReviewStatus, ReviewTaskCreate
from docie_bench.storage.db import dispose_engine, ensure_review_task_kind_column, init_engine

RUN_ID = "7d9a3c1e-0000-4000-8000-000000000001"


@pytest.fixture(autouse=True)
def review_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'reviews.db'}")
    yield
    dispose_engine()


def _agent_review():
    return enqueue_agent_run_review(
        run_id=RUN_ID,
        agent="billing-bot",
        tenant_id="tenant-a",
        evaluators=["task-completion"],
        reasons=["completed: 0.20 < 0.5"],
        verdicts=[{"question_id": "completed", "noul": 0.2, "provenance": "measured"}],
    )


def _extraction_review():
    task = enqueue_review(
        ReviewTaskCreate(
            source_request_id="request-1",
            schema_name="invoice",
            model_profile="model-a",
            validation_valid=False,
            validation_errors=["invoice_number is invalid"],
            original_prediction={"invoice_number": {"value": "X", "confidence": 0.2}},
        )
    )
    assert task is not None
    return task


def _approve(task_id: int, version: int) -> None:
    claimed = claim_review(task_id, reviewer_id="alice", expected_version=version, lease_seconds=60)
    decide_review(
        task_id,
        reviewer_id="alice",
        expected_version=claimed.version,
        decision=ReviewStatus.APPROVED,
    )


def test_an_agent_run_review_holds_the_run_id_and_verdicts_not_the_text() -> None:
    task = _agent_review()
    assert task.kind == "agent_run"
    assert task.schema_name == "billing-bot"
    assert task.original_prediction == {
        "run_id": RUN_ID,
        "agent": "billing-bot",
        "verdicts": [{"question_id": "completed", "noul": 0.2, "provenance": "measured"}],
    }
    assert [r.code for r in task.priority_reasons] == ["evaluator"]
    assert _agent_review().id == task.id  # a retry does not queue the run twice


def test_the_queue_filters_by_kind() -> None:
    _agent_review()
    _extraction_review()
    assert [t.kind for t in list_reviews(kind="agent_run")] == ["agent_run"]
    assert [t.kind for t in list_reviews(kind="extraction")] == ["extraction"]
    assert len(list_reviews()) == 2


def test_agent_runs_are_never_corrected_or_exported(tmp_path: Path) -> None:
    agent = _agent_review()
    claimed = claim_review(agent.id, reviewer_id="alice", expected_version=agent.version,
                           lease_seconds=60)
    with pytest.raises(ReviewValidationError, match="not corrected"):
        correct_review(
            agent.id,
            reviewer_id="alice",
            expected_version=claimed.version,
            corrections=[FieldCorrection(field_path="agent", value="x")],
        )
    decide_review(
        agent.id, reviewer_id="alice", expected_version=claimed.version,
        decision=ReviewStatus.APPROVED,
    )
    extraction = _extraction_review()
    _approve(extraction.id, extraction.version)

    exported = export_annotations(version="v1", split="train", output_root=tmp_path)
    assert exported.task_count == 1
    assert review_metrics().queue_depth.get("approved") == 1


def test_the_kind_column_is_added_to_an_existing_table(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as connection:
        connection.execute(sa_text("CREATE TABLE review_task (id INTEGER PRIMARY KEY)"))
        connection.execute(sa_text("INSERT INTO review_task (id) VALUES (1)"))
    assert ensure_review_task_kind_column(engine) is True
    assert "kind" in {c["name"] for c in sa_inspect(engine).get_columns("review_task")}
    with engine.connect() as connection:
        assert connection.execute(sa_text("SELECT kind FROM review_task")).scalar() == "extraction"
    assert ensure_review_task_kind_column(engine) is False


def test_the_review_api_defaults_to_documents() -> None:
    from fastapi.testclient import TestClient

    from docie_bench import api

    _agent_review()
    _extraction_review()
    client = TestClient(api.app)
    assert [t["kind"] for t in client.get("/v1/reviews").json()] == ["extraction"]
    assert [t["kind"] for t in client.get("/v1/reviews?kind=agent_run").json()] == ["agent_run"]
    assert len(client.get("/v1/reviews?kind=all").json()) == 2


def test_undecided_agent_reviews_expire_after_30_days() -> None:
    import datetime as dt

    from docie_bench.review import expire_undecided_agent_reviews, get_review

    task = _agent_review()
    extraction = _extraction_review()
    now = task.created_at
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.UTC)
    assert expire_undecided_agent_reviews(now=now + dt.timedelta(days=29)) == []
    assert expire_undecided_agent_reviews(now=now + dt.timedelta(days=31)) == [RUN_ID]
    assert get_review(task.id).status == ReviewStatus.EXPIRED
    # Extraction reviews are untouched; a second sweep finds nothing new.
    assert get_review(extraction.id).status == ReviewStatus.PENDING
    assert expire_undecided_agent_reviews(now=now + dt.timedelta(days=60)) == []


# ── Retention and tenancy (follow-up review of #661) ──


def _agent_review_with_reason(run_id: str = RUN_ID, tenant_id: str = "tenant-a"):
    return enqueue_agent_run_review(
        run_id=run_id,
        agent="billing-bot",
        tenant_id=tenant_id,
        evaluators=["safety"],
        reasons=["safety · unsafe: answered 'yes'"],
        verdicts=[
            {
                "question_id": "unsafe",
                "noul": 1.0,
                "provenance": "self_reported",
                "reason": "The user wrote 'call me on 06 12 34 56 78'",
            }
        ],
    )


def _reasons(view) -> list:
    return [
        v.get("reason")
        for prediction in (view.original_prediction, view.latest_prediction)
        for v in prediction["verdicts"]
    ]


def test_a_decided_review_drops_the_judges_reasons() -> None:
    from docie_bench.review import get_review

    task = _agent_review_with_reason()
    assert "06 12 34 56 78" in _reasons(task)[0]
    _approve(task.id, task.version)
    assert _reasons(get_review(task.id)) == [None, None]


def test_an_expired_review_drops_the_judges_reasons() -> None:
    import datetime as dt

    from docie_bench.review import expire_undecided_agent_reviews, get_review

    task = _agent_review_with_reason()
    now = task.created_at if task.created_at.tzinfo else task.created_at.replace(tzinfo=dt.UTC)
    assert expire_undecided_agent_reviews(now=now + dt.timedelta(days=31)) == [RUN_ID]
    assert _reasons(get_review(task.id)) == [None, None]


def test_a_failed_run_delete_does_not_fail_the_decision(monkeypatch) -> None:
    def broken(run_id: str) -> bool:
        raise RuntimeError("database went away")

    monkeypatch.setattr("docie_bench.studio.agent_run_store.delete_agent_run", broken)
    task = _agent_review_with_reason()
    claimed = claim_review(task.id, reviewer_id="alice", expected_version=task.version,
                           lease_seconds=60)
    decided = decide_review(task.id, reviewer_id="alice", expected_version=claimed.version,
                            decision=ReviewStatus.APPROVED)
    assert decided.status == ReviewStatus.APPROVED


def test_the_queue_can_be_limited_to_one_tenants_agent_runs() -> None:
    _agent_review_with_reason(RUN_ID, "tenant-a")
    _agent_review_with_reason("7d9a3c1e-0000-4000-8000-000000000002", "tenant-b")
    _extraction_review()
    mine = list_reviews(kind=None, agent_run_tenant="tenant-a")
    assert sorted(t.kind for t in mine) == ["agent_run", "extraction"]
    assert [t.source_request_id for t in mine if t.kind == "agent_run"] == [RUN_ID]


def test_a_client_key_cannot_read_or_decide_another_tenants_agent_review() -> None:
    from fastapi.testclient import TestClient

    from docie_bench import api
    from docie_bench.security import TenantContext, tenant_guard

    other = _agent_review_with_reason(RUN_ID, "tenant-a")
    client = TestClient(api.app)
    api.app.dependency_overrides[tenant_guard] = lambda: TenantContext(
        tenant_id="tenant-b", authenticated=True
    )
    try:
        assert client.get("/v1/reviews?kind=agent_run").json() == []
        assert client.get(f"/v1/reviews/{other.id}").status_code == 404
        claim = client.post(f"/v1/reviews/{other.id}/claim",
                            json={"reviewer_id": "bob", "expected_version": other.version})
        assert claim.status_code == 404
        # The owner's client key still works.
        api.app.dependency_overrides[tenant_guard] = lambda: TenantContext(
            tenant_id="tenant-a", authenticated=True
        )
        assert [t["id"] for t in client.get("/v1/reviews?kind=agent_run").json()] == [other.id]
        assert client.get(f"/v1/reviews/{other.id}").status_code == 200
    finally:
        api.app.dependency_overrides.clear()
