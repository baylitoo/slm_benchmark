from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docie_bench import api
from docie_bench.review import (
    ReviewConflictError,
    ReviewValidationError,
    claim_review,
    correct_review,
    decide_review,
    enqueue_review,
    export_annotations,
    get_review,
    release_review,
    review_metrics,
    score_review_candidate,
)
from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation
from docie_bench.schemas.review import FieldCorrection, ReviewStatus, ReviewTaskCreate
from docie_bench.storage.audit import save_extraction_audit
from docie_bench.storage.db import (
    ReviewCorrection,
    ReviewTask,
    dispose_engine,
    init_engine,
    session_scope,
)


@pytest.fixture(autouse=True)
def review_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'reviews.db'}")
    yield
    dispose_engine()


def _payload(request_id: str = "request-1") -> ReviewTaskCreate:
    return ReviewTaskCreate(
        source_request_id=request_id,
        schema_name="invoice",
        model_profile="model-a",
        document_hash="sha256:abc",
        validation_valid=False,
        validation_errors=["invoice_number is invalid"],
        disagreement_score=0.6,
        expected_learning_value=0.8,
        metadata={"doc_id": "invoice-1", "source": "batch-a"},
        original_prediction={
            "document_type": "invoice",
            "invoice_number": {
                "value": "WRONG",
                "confidence": 0.2,
                "evidence_ids": [],
            },
            "vendor_name": {
                "value": "Acme",
                "confidence": 0.9,
                "evidence_ids": ["b1"],
            },
        },
    )


def _claim(task_id: int, version: int, reviewer: str = "alice"):
    return claim_review(
        task_id,
        reviewer_id=reviewer,
        expected_version=version,
        lease_seconds=300,
    )


def test_priority_is_explainable_and_clean_extraction_is_not_queued() -> None:
    priority, reasons = score_review_candidate(_payload())

    assert priority > 0
    assert {reason.code for reason in reasons} == {
        "invalid",
        "low_confidence",
        "weak_evidence",
        "model_disagreement",
        "learning_value",
    }

    clean = _payload("clean")
    clean.validation_valid = True
    clean.validation_errors = []
    clean.disagreement_score = None
    clean.expected_learning_value = None
    clean.original_prediction = {
        "invoice_number": {"value": "INV-1", "confidence": 0.99, "evidence_ids": ["b1"]}
    }
    assert enqueue_review(clean) is None
    forced = enqueue_review(clean, force=True)
    assert forced is not None
    assert forced.priority_reasons[0].code == "manual"


def test_full_correction_approval_lifecycle_is_immutable_and_conflict_safe() -> None:
    task = enqueue_review(_payload())
    assert task is not None
    claimed = _claim(task.id, task.version)

    with pytest.raises(ReviewConflictError, match="expected version"):
        _claim(task.id, task.version, "bob")
    with pytest.raises(ReviewConflictError, match="not claimed"):
        correct_review(
            task.id,
            reviewer_id="bob",
            expected_version=claimed.version,
            corrections=[FieldCorrection(field_path="invoice_number.value", value="INV-1")],
        )
    with pytest.raises(ReviewValidationError, match="unknown schema fields"):
        correct_review(
            task.id,
            reviewer_id="alice",
            expected_version=claimed.version,
            corrections=[FieldCorrection(field_path="invented.value", value="bad")],
        )
    with pytest.raises(ReviewValidationError, match="not writable schema fields"):
        correct_review(
            task.id,
            reviewer_id="alice",
            expected_version=claimed.version,
            corrections=[FieldCorrection(field_path="invoice_number.value.extra", value="bad")],
        )

    corrected = correct_review(
        task.id,
        reviewer_id="alice",
        expected_version=claimed.version,
        corrections=[
            FieldCorrection(
                field_path="invoice_number.value",
                value="INV-1",
                comment="Read from the document",
            )
        ],
        comment="Corrected identifier",
    )
    approved = decide_review(
        task.id,
        reviewer_id="alice",
        expected_version=corrected.version,
        decision=ReviewStatus.APPROVED,
        comment="Verified",
    )

    assert approved.status == ReviewStatus.APPROVED
    assert approved.latest_prediction["invoice_number"]["value"] == "INV-1"
    assert approved.original_prediction["invoice_number"]["value"] == "WRONG"
    assert approved.corrections[0].corrections[0]["field_path"] == "invoice_number.value"
    assert [event.event_type for event in approved.events] == [
        "enqueued",
        "claimed",
        "corrected",
        "approved",
    ]

    with session_scope() as session:
        assert session is not None
        stored = session.get(ReviewCorrection, approved.corrections[0].id)
        assert stored is not None
        assert stored.corrected_prediction_json["invoice_number"]["value"] == "INV-1"


def test_release_reject_and_expired_claim_return_task_to_queue() -> None:
    task = enqueue_review(_payload())
    assert task is not None
    claimed = _claim(task.id, task.version)
    released = release_review(
        task.id,
        reviewer_id="alice",
        expected_version=claimed.version,
        comment="Taking another task",
    )
    assert released.status == ReviewStatus.PENDING

    claimed = _claim(task.id, released.version, "bob")
    with session_scope() as session:
        assert session is not None
        stored = session.get(ReviewTask, task.id)
        assert stored is not None
        stored.claim_expires_at = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1)

    expired = get_review(task.id)
    assert expired.status == ReviewStatus.PENDING
    assert expired.claimed_by is None
    assert expired.events[-1].event_type == "claim_expired"

    with pytest.raises(ReviewConflictError, match="expected version"):
        decide_review(
            task.id,
            reviewer_id="bob",
            expected_version=claimed.version,
            decision=ReviewStatus.REJECTED,
        )

    reclaimed = _claim(task.id, expired.version, "carol")
    rejected = decide_review(
        task.id,
        reviewer_id="carol",
        expected_version=reclaimed.version,
        decision=ReviewStatus.REJECTED,
        comment="Unreadable source",
    )
    assert rejected.status == ReviewStatus.REJECTED


def test_extraction_audit_automatically_admits_review_candidate() -> None:
    save_extraction_audit(
        ExtractionResponse(
            request_id="extraction-request",
            schema_name="invoice",
            model_profile="model-a",
            document_hash="sha256:auto",
            result={
                "invoice_number": {
                    "value": "uncertain",
                    "confidence": 0.1,
                    "evidence_ids": [],
                }
            },
            validation=ExtractionValidation(valid=True),
            latency_ms=10,
        )
    )

    with session_scope() as session:
        assert session is not None
        task = session.query(ReviewTask).filter_by(source_request_id="extraction-request").one()
        assert task.status == ReviewStatus.PENDING
        assert {reason["code"] for reason in task.priority_reasons_json} == {
            "low_confidence",
            "weak_evidence",
        }


def test_reviewer_agreement_compares_latest_independent_reviewer_snapshots() -> None:
    task = enqueue_review(_payload("agreement"))
    assert task is not None
    alice_claim = _claim(task.id, task.version)
    alice_correction = correct_review(
        task.id,
        reviewer_id="alice",
        expected_version=alice_claim.version,
        corrections=[FieldCorrection(field_path="invoice_number.value", value="INV-1")],
    )
    released = release_review(
        task.id,
        reviewer_id="alice",
        expected_version=alice_correction.version,
    )
    bob_claim = _claim(task.id, released.version, "bob")
    bob_correction = correct_review(
        task.id,
        reviewer_id="bob",
        expected_version=bob_claim.version,
        corrections=[FieldCorrection(field_path="invoice_number.value", value="INV-1")],
    )
    decide_review(
        task.id,
        reviewer_id="bob",
        expected_version=bob_correction.version,
        decision=ReviewStatus.APPROVED,
    )

    assert review_metrics().reviewer_agreement == 1.0


def test_dynamic_schema_corrections_accept_declared_nested_fields() -> None:
    payload = _payload("dynamic")
    payload.schema_name = "purchase_order"
    payload.dynamic_schema = {
        "document_type": "purchase_order",
        "fields": [{"name": "order_number", "type": "string"}],
    }
    payload.original_prediction = {
        "document_type": "purchase_order",
        "order_number": {"value": "PO-?", "confidence": 0.1, "evidence_ids": []},
    }
    task = enqueue_review(payload)
    assert task is not None
    claimed = _claim(task.id, task.version)

    corrected = correct_review(
        task.id,
        reviewer_id="alice",
        expected_version=claimed.version,
        corrections=[FieldCorrection(field_path="order_number.value", value="PO-42")],
    )

    assert corrected.latest_prediction["order_number"]["value"] == "PO-42"


def test_approved_export_is_versioned_train_only_and_metrics_quantify_work(tmp_path: Path) -> None:
    first = enqueue_review(_payload("approved"))
    assert first is not None
    claimed = _claim(first.id, first.version)
    corrected = correct_review(
        first.id,
        reviewer_id="alice",
        expected_version=claimed.version,
        corrections=[FieldCorrection(field_path="invoice_number.value", value="INV-9")],
    )
    decide_review(
        first.id,
        reviewer_id="alice",
        expected_version=corrected.version,
        decision=ReviewStatus.APPROVED,
    )

    second = enqueue_review(_payload("rejected"))
    assert second is not None
    claimed_second = _claim(second.id, second.version, "bob")
    decide_review(
        second.id,
        reviewer_id="bob",
        expected_version=claimed_second.version,
        decision=ReviewStatus.REJECTED,
    )

    exported = export_annotations(version="v1", split="train", output_root=tmp_path)
    rows = [
        json.loads(line)
        for line in Path(exported.annotations_path).read_text(encoding="utf-8").splitlines()
    ]
    assert exported.task_count == 1
    assert rows[0]["ground_truth"]["invoice_number"]["value"] == "INV-9"
    assert rows[0]["metadata"]["annotation_version"] == "v1"
    assert "original_prediction" not in rows[0]

    with pytest.raises(ReviewConflictError, match="already exists"):
        export_annotations(version="v1", split="train", output_root=tmp_path)
    with pytest.raises(ReviewValidationError, match="train split"):
        export_annotations(version="eval-v1", split="test", output_root=tmp_path)

    metrics = review_metrics()
    assert metrics.correction_rate == 0.5
    assert metrics.queue_depth == {"approved": 1, "rejected": 1}
    assert metrics.workload_by_reviewer["alice"] == {
        "claims": 1,
        "corrections": 1,
        "approvals": 1,
        "rejections": 0,
    }
    assert metrics.average_queue_latency_seconds is not None


def test_review_api_exposes_lifecycle_conflicts_metrics_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api.settings, "annotation_export_dir", tmp_path / "exports")
    with TestClient(api.app) as client:
        created = client.post("/v1/reviews", json=_payload("api-request").model_dump(mode="json"))
        assert created.status_code == 200
        task = created.json()
        duplicate = client.post("/v1/reviews", json=_payload("api-request").model_dump(mode="json"))
        assert duplicate.status_code == 409

        claimed = client.post(
            f"/v1/reviews/{task['id']}/claim",
            json={"reviewer_id": "alice", "expected_version": task["version"]},
        )
        assert claimed.status_code == 200
        claimed_task = claimed.json()

        conflict = client.post(
            f"/v1/reviews/{task['id']}/claim",
            json={"reviewer_id": "bob", "expected_version": task["version"]},
        )
        assert conflict.status_code == 409

        corrected = client.post(
            f"/v1/reviews/{task['id']}/correct",
            json={
                "reviewer_id": "alice",
                "expected_version": claimed_task["version"],
                "corrections": [{"field_path": "invoice_number.value", "value": "INV-API"}],
            },
        ).json()
        approved = client.post(
            f"/v1/reviews/{task['id']}/approve",
            json={"reviewer_id": "alice", "expected_version": corrected["version"]},
        )
        assert approved.status_code == 200

        assert client.get("/v1/reviews/metrics").json()["correction_rate"] == 1.0
        exported = client.post("/v1/reviews/exports", json={"version": "api-v1"})
        assert exported.status_code == 200
        assert exported.json()["task_count"] == 1
