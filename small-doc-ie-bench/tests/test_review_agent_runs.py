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
