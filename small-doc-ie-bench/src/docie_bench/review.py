from __future__ import annotations

import copy
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.orm.exc import StaleDataError

from docie_bench.schemas.extraction import schema_json
from docie_bench.schemas.review import (
    AnnotationExportView,
    FieldCorrection,
    ReviewCorrectionView,
    ReviewEventView,
    ReviewMetricsView,
    ReviewReason,
    ReviewStatus,
    ReviewTaskCreate,
    ReviewTaskView,
)
from docie_bench.settings import get_settings
from docie_bench.storage.db import ReviewCorrection, ReviewEvent, ReviewTask, session_scope


class ReviewError(RuntimeError):
    pass


class ReviewNotFoundError(ReviewError):
    pass


class ReviewConflictError(ReviewError):
    pass


class ReviewValidationError(ReviewError):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _as_utc(value: dt.datetime) -> dt.datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)


def _confidence_values(obj: Any) -> list[float]:
    if isinstance(obj, list):
        return [value for item in obj for value in _confidence_values(item)]
    if not isinstance(obj, dict):
        return []
    values: list[float] = []
    confidence = obj.get("confidence")
    if isinstance(confidence, (int, float)):
        values.append(float(confidence))
    for value in obj.values():
        values.extend(_confidence_values(value))
    return values


def _evidence_counts(obj: Any) -> tuple[int, int]:
    if isinstance(obj, list):
        counts = [_evidence_counts(item) for item in obj]
        return sum(item[0] for item in counts), sum(item[1] for item in counts)
    if not isinstance(obj, dict):
        return 0, 0
    if obj.get("value") is not None or obj.get("amount") is not None:
        return 1, int(bool(obj.get("evidence_ids")))
    counts = [_evidence_counts(value) for value in obj.values()]
    return sum(item[0] for item in counts), sum(item[1] for item in counts)


def score_review_candidate(
    payload: ReviewTaskCreate,
    *,
    confidence_threshold: float = 0.7,
    evidence_threshold: float = 0.8,
) -> tuple[float, list[ReviewReason]]:
    reasons: list[ReviewReason] = []
    if not payload.validation_valid:
        reasons.append(
            ReviewReason(code="invalid", score=1.0, detail="Extraction failed validation")
        )

    confidences = _confidence_values(payload.original_prediction)
    if confidences:
        mean_confidence = sum(confidences) / len(confidences)
        if mean_confidence < confidence_threshold:
            uncertainty = 1.0 - mean_confidence
            reasons.append(
                ReviewReason(
                    code="low_confidence",
                    score=uncertainty,
                    detail=(
                        f"Mean field confidence {mean_confidence:.3f} "
                        f"is below {confidence_threshold:.3f}"
                    ),
                )
            )

    evidence_total, evidence_grounded = _evidence_counts(payload.original_prediction)
    if evidence_total:
        coverage = evidence_grounded / evidence_total
        if coverage < evidence_threshold:
            reasons.append(
                ReviewReason(
                    code="weak_evidence",
                    score=1.0 - coverage,
                    detail=f"Evidence coverage {coverage:.3f} is below {evidence_threshold:.3f}",
                )
            )

    if payload.disagreement_score:
        reasons.append(
            ReviewReason(
                code="model_disagreement",
                score=payload.disagreement_score,
                detail=f"Model disagreement score is {payload.disagreement_score:.3f}",
            )
        )
    if payload.expected_learning_value:
        reasons.append(
            ReviewReason(
                code="learning_value",
                score=payload.expected_learning_value,
                detail=f"Expected learning value is {payload.expected_learning_value:.3f}",
            )
        )

    # Additive scoring makes multi-signal tasks rise while retaining a readable breakdown.
    weights = {
        "invalid": 40.0,
        "low_confidence": 25.0,
        "weak_evidence": 20.0,
        "model_disagreement": 30.0,
        "learning_value": 15.0,
    }
    priority = sum(weights[reason.code] * reason.score for reason in reasons)
    return round(priority, 4), reasons


def _require_session(session: Session | None) -> Session:
    if session is None:
        raise ReviewValidationError("Review workflow requires DATABASE_URL")
    return session


def _flush(session: Session) -> None:
    try:
        session.flush()
    except (IntegrityError, StaleDataError) as exc:
        raise ReviewConflictError("Review task changed concurrently; reload it and retry") from exc


def _event(
    task: ReviewTask,
    event_type: str,
    *,
    actor_id: str | None = None,
    from_status: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    task.events.append(
        ReviewEvent(
            event_type=event_type,
            actor_id=actor_id,
            from_status=from_status,
            to_status=task.status,
            task_version=task.version,
            details_json=details or {},
        )
    )


def _query_task(session: Session, task_id: int) -> ReviewTask:
    task = session.scalar(
        select(ReviewTask)
        .where(ReviewTask.id == task_id)
        .options(selectinload(ReviewTask.corrections), selectinload(ReviewTask.events))
    )
    if task is None:
        raise ReviewNotFoundError(f"Review task {task_id} not found")
    return task


def _expire_task_claim(task: ReviewTask, now: dt.datetime) -> bool:
    if (
        task.status == ReviewStatus.CLAIMED
        and task.claim_expires_at is not None
        and _as_utc(task.claim_expires_at) <= now
    ):
        previous_reviewer = task.claimed_by
        task.status = ReviewStatus.PENDING
        task.claimed_by = None
        task.claim_expires_at = None
        task.version += 1
        task.updated_at = now
        _event(
            task,
            "claim_expired",
            from_status=ReviewStatus.CLAIMED,
            details={"previous_reviewer": previous_reviewer},
        )
        return True
    return False


def _assert_version(task: ReviewTask, expected_version: int) -> None:
    if task.version != expected_version:
        raise ReviewConflictError(
            f"Review task {task.id} changed: expected version {expected_version}, "
            f"current version {task.version}"
        )


def _assert_claim_owner(task: ReviewTask, reviewer_id: str) -> None:
    if task.status != ReviewStatus.CLAIMED or task.claimed_by != reviewer_id:
        raise ReviewConflictError(f"Review task {task.id} is not claimed by {reviewer_id!r}")


def _task_view(task: ReviewTask, *, include_history: bool = True) -> ReviewTaskView:
    return ReviewTaskView(
        id=task.id,
        source_request_id=task.source_request_id,
        schema_name=task.schema_name,
        model_profile=task.model_profile,
        document_hash=task.document_hash,
        status=ReviewStatus(task.status),
        priority=task.priority,
        priority_reasons=[ReviewReason.model_validate(item) for item in task.priority_reasons_json],
        original_prediction=task.original_prediction_json,
        latest_prediction=task.latest_prediction_json,
        validation_errors=task.validation_errors_json,
        dynamic_schema=task.dynamic_schema_json,
        metadata=task.metadata_json,
        claimed_by=task.claimed_by,
        claim_expires_at=task.claim_expires_at,
        version=task.version,
        created_at=task.created_at,
        updated_at=task.updated_at,
        decided_at=task.decided_at,
        decided_by=task.decided_by,
        decision_comment=task.decision_comment,
        corrections=[
            ReviewCorrectionView(
                id=correction.id,
                revision=correction.revision,
                reviewer_id=correction.reviewer_id,
                created_at=correction.created_at,
                corrections=correction.corrections_json,
                corrected_prediction=correction.corrected_prediction_json,
                comment=correction.comment,
            )
            for correction in task.corrections
        ]
        if include_history
        else [],
        events=[
            ReviewEventView(
                id=event.id,
                event_type=event.event_type,
                actor_id=event.actor_id,
                from_status=event.from_status,
                to_status=event.to_status,
                task_version=event.task_version,
                created_at=event.created_at,
                details=event.details_json,
            )
            for event in task.events
        ]
        if include_history
        else [],
    )


def enqueue_review(payload: ReviewTaskCreate, *, force: bool = False) -> ReviewTaskView | None:
    settings = get_settings()
    priority, reasons = score_review_candidate(
        payload,
        confidence_threshold=settings.review_confidence_threshold,
        evidence_threshold=settings.review_evidence_threshold,
    )
    if not reasons and not force:
        return None
    if not reasons:
        reasons = [
            ReviewReason(code="manual", score=0.0, detail="Task was manually forced into the queue")
        ]
    with session_scope() as maybe_session:
        session = _require_session(maybe_session)
        task = ReviewTask(
            source_request_id=payload.source_request_id,
            schema_name=payload.schema_name,
            model_profile=payload.model_profile,
            document_hash=payload.document_hash,
            priority=priority,
            priority_reasons_json=[reason.model_dump(mode="json") for reason in reasons],
            original_prediction_json=payload.original_prediction,
            latest_prediction_json=payload.original_prediction,
            validation_errors_json=payload.validation_errors,
            dynamic_schema_json=payload.dynamic_schema,
            metadata_json=payload.metadata,
            status=ReviewStatus.PENDING,
            version=1,
        )
        _event(task, "enqueued", details={"force": force})
        session.add(task)
        _flush(session)
        return _task_view(task)


def list_reviews(
    *,
    status: ReviewStatus | None = None,
    reviewer_id: str | None = None,
    limit: int = 100,
) -> list[ReviewTaskView]:
    now = _utcnow()
    with session_scope() as maybe_session:
        session = _require_session(maybe_session)
        expired_tasks = list(
            session.scalars(
                select(ReviewTask)
                .where(
                    ReviewTask.status == ReviewStatus.CLAIMED,
                    ReviewTask.claim_expires_at <= now,
                )
                .options(selectinload(ReviewTask.events))
            )
        )
        for task in expired_tasks:
            _expire_task_claim(task, now)
        _flush(session)

        statement = select(ReviewTask).order_by(ReviewTask.priority.desc(), ReviewTask.created_at)
        if status is not None:
            statement = statement.where(ReviewTask.status == status)
        if reviewer_id is not None:
            statement = statement.where(ReviewTask.claimed_by == reviewer_id)
        tasks = list(session.scalars(statement.limit(limit)))
        return [_task_view(task, include_history=False) for task in tasks]


def get_review(task_id: int) -> ReviewTaskView:
    with session_scope() as maybe_session:
        session = _require_session(maybe_session)
        task = _query_task(session, task_id)
        _expire_task_claim(task, _utcnow())
        _flush(session)
        return _task_view(task)


def claim_review(
    task_id: int, *, reviewer_id: str, expected_version: int, lease_seconds: int
) -> ReviewTaskView:
    now = _utcnow()
    with session_scope() as maybe_session:
        session = _require_session(maybe_session)
        task = _query_task(session, task_id)
        expired = _expire_task_claim(task, now)
        if expired:
            _flush(session)
        _assert_version(task, expected_version)
        if task.status != ReviewStatus.PENDING:
            raise ReviewConflictError(f"Review task {task.id} is not pending")
        task.status = ReviewStatus.CLAIMED
        task.claimed_by = reviewer_id
        task.claim_expires_at = now + dt.timedelta(seconds=lease_seconds)
        task.version += 1
        task.updated_at = now
        _event(task, "claimed", actor_id=reviewer_id, from_status=ReviewStatus.PENDING)
        _flush(session)
        return _task_view(task)


def release_review(
    task_id: int, *, reviewer_id: str, expected_version: int, comment: str | None = None
) -> ReviewTaskView:
    with session_scope() as maybe_session:
        session = _require_session(maybe_session)
        task = _query_task(session, task_id)
        _expire_task_claim(task, _utcnow())
        _assert_version(task, expected_version)
        _assert_claim_owner(task, reviewer_id)
        task.status = ReviewStatus.PENDING
        task.claimed_by = None
        task.claim_expires_at = None
        task.version += 1
        task.updated_at = _utcnow()
        _event(
            task,
            "released",
            actor_id=reviewer_id,
            from_status=ReviewStatus.CLAIMED,
            details={"comment": comment},
        )
        _flush(session)
        return _task_view(task)


def _set_path(payload: dict[str, Any], field_path: str, value: Any) -> None:
    parts = field_path.split(".")
    current: Any = payload
    for part in parts[:-1]:
        if isinstance(current, dict):
            if part not in current or current[part] is None:
                current[part] = {}
            elif not isinstance(current[part], (dict, list)):
                raise ReviewValidationError(f"Cannot traverse scalar in field_path={field_path!r}")
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ReviewValidationError(f"Cannot apply correction to field_path={field_path!r}")
    leaf = parts[-1]
    if isinstance(current, dict):
        current[leaf] = value
    elif isinstance(current, list) and leaf.isdigit() and int(leaf) < len(current):
        current[int(leaf)] = value
    else:
        raise ReviewValidationError(f"Cannot apply correction to field_path={field_path!r}")


def _validate_correction_paths(task: ReviewTask, corrections: list[FieldCorrection]) -> None:
    if task.dynamic_schema_json:
        allowed = {
            field["name"]
            for field in task.dynamic_schema_json.get("fields", [])
            if isinstance(field, dict) and isinstance(field.get("name"), str)
        }
        allowed.add("document_type")
    else:
        try:
            allowed = set(schema_json(task.schema_name).get("properties", {}))
        except ValueError as exc:
            raise ReviewValidationError(str(exc)) from exc
    unknown = sorted(
        {correction.field_path.split(".", maxsplit=1)[0] for correction in corrections} - allowed
    )
    if unknown:
        raise ReviewValidationError(f"Correction paths contain unknown schema fields: {unknown}")
    allowed_nested_fields = {"value", "amount", "currency", "evidence_ids", "confidence"}
    invalid_nested = sorted(
        correction.field_path
        for correction in corrections
        if len(correction.field_path.split(".")) > 2
        or (
            "." in correction.field_path
            and correction.field_path.split(".", maxsplit=1)[1] not in allowed_nested_fields
        )
    )
    if invalid_nested:
        raise ReviewValidationError(
            f"Correction paths are not writable schema fields: {invalid_nested}"
        )


def correct_review(
    task_id: int,
    *,
    reviewer_id: str,
    expected_version: int,
    corrections: list[FieldCorrection],
    comment: str | None = None,
) -> ReviewTaskView:
    with session_scope() as maybe_session:
        session = _require_session(maybe_session)
        task = _query_task(session, task_id)
        _expire_task_claim(task, _utcnow())
        _assert_version(task, expected_version)
        _assert_claim_owner(task, reviewer_id)
        _validate_correction_paths(task, corrections)
        corrected = copy.deepcopy(task.latest_prediction_json)
        for correction in corrections:
            _set_path(corrected, correction.field_path, correction.value)
        revision = len(task.corrections) + 1
        record = ReviewCorrection(
            revision=revision,
            reviewer_id=reviewer_id,
            corrections_json=[item.model_dump(mode="json") for item in corrections],
            corrected_prediction_json=corrected,
            comment=comment,
        )
        task.corrections.append(record)
        task.latest_prediction_json = corrected
        task.version += 1
        task.updated_at = _utcnow()
        _event(
            task,
            "corrected",
            actor_id=reviewer_id,
            details={
                "revision": revision,
                "field_paths": [item.field_path for item in corrections],
            },
        )
        _flush(session)
        return _task_view(task)


def decide_review(
    task_id: int,
    *,
    reviewer_id: str,
    expected_version: int,
    decision: ReviewStatus,
    comment: str | None = None,
) -> ReviewTaskView:
    if decision not in {ReviewStatus.APPROVED, ReviewStatus.REJECTED}:
        raise ReviewValidationError("decision must be approved or rejected")
    with session_scope() as maybe_session:
        session = _require_session(maybe_session)
        task = _query_task(session, task_id)
        _expire_task_claim(task, _utcnow())
        _assert_version(task, expected_version)
        _assert_claim_owner(task, reviewer_id)
        task.status = decision
        task.claimed_by = None
        task.claim_expires_at = None
        task.decided_at = _utcnow()
        task.decided_by = reviewer_id
        task.decision_comment = comment
        task.version += 1
        task.updated_at = task.decided_at
        _event(task, decision.value, actor_id=reviewer_id, from_status=ReviewStatus.CLAIMED)
        _flush(session)
        return _task_view(task)


def export_annotations(
    *,
    version: str,
    split: str,
    output_root: Path,
    task_ids: list[int] | None = None,
) -> AnnotationExportView:
    if split != "train":
        raise ReviewValidationError(
            "Approved human corrections may only be exported to the train split "
            "to prevent evaluation leakage"
        )
    export_dir = output_root / version
    if export_dir.exists():
        raise ReviewConflictError(f"Annotation version {version!r} already exists")

    with session_scope() as maybe_session:
        session = _require_session(maybe_session)
        statement = select(ReviewTask).where(ReviewTask.status == ReviewStatus.APPROVED)
        if task_ids is not None:
            statement = statement.where(ReviewTask.id.in_(task_ids))
        tasks = list(session.scalars(statement.order_by(ReviewTask.id)))
        if task_ids is not None and len(tasks) != len(set(task_ids)):
            raise ReviewValidationError("All exported task_ids must exist and be approved")
        rows = [
            {
                "doc_id": task.metadata_json.get("doc_id") or task.source_request_id,
                "schema_name": task.schema_name,
                "schema_mode": "dynamic" if task.dynamic_schema_json else "static",
                "dynamic_schema": task.dynamic_schema_json,
                "ground_truth": task.latest_prediction_json,
                "metadata": {
                    **task.metadata_json,
                    "review_task_id": str(task.id),
                    "annotation_version": version,
                    "split": split,
                },
            }
            for task in tasks
        ]
        created_at = _utcnow()
        manifest = {
            "version": version,
            "split": split,
            "created_at": created_at.isoformat(),
            "task_count": len(rows),
            "source_task_ids": [task.id for task in tasks],
        }

    try:
        export_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ReviewConflictError(f"Annotation version {version!r} already exists") from exc
    annotations_path = export_dir / "annotations.jsonl"
    manifest_path = export_dir / "manifest.json"
    annotations_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return AnnotationExportView(
        version=version,
        split=split,
        created_at=created_at,
        task_count=len(rows),
        annotations_path=str(annotations_path),
        manifest_path=str(manifest_path),
    )


def review_metrics() -> ReviewMetricsView:
    with session_scope() as maybe_session:
        session = _require_session(maybe_session)
        tasks = list(
            session.scalars(
                select(ReviewTask).options(
                    selectinload(ReviewTask.corrections), selectinload(ReviewTask.events)
                )
            )
        )
        queue_depth = Counter(task.status for task in tasks)
        decided = [
            task for task in tasks if task.status in {ReviewStatus.APPROVED, ReviewStatus.REJECTED}
        ]
        corrected = [task for task in decided if task.corrections]
        correction_rate = len(corrected) / len(decided) if decided else None

        agreement_pairs: list[bool] = []
        for task in tasks:
            reviewer_snapshots: dict[str, dict[str, Any]] = {}
            for correction in task.corrections:
                reviewer_snapshots[correction.reviewer_id] = correction.corrected_prediction_json
            snapshots = list(reviewer_snapshots.values())
            for index, left in enumerate(snapshots):
                agreement_pairs.extend(left == right for right in snapshots[index + 1 :])
        reviewer_agreement = (
            sum(agreement_pairs) / len(agreement_pairs) if agreement_pairs else None
        )

        latencies = []
        for task in tasks:
            first_claim = next(
                (event for event in task.events if event.event_type == "claimed"),
                None,
            )
            if first_claim is not None:
                latencies.append(
                    (_as_utc(first_claim.created_at) - _as_utc(task.created_at)).total_seconds()
                )
        workload: dict[str, dict[str, int]] = defaultdict(
            lambda: {"claims": 0, "corrections": 0, "approvals": 0, "rejections": 0}
        )
        for task in tasks:
            for event in task.events:
                if event.actor_id and event.event_type in {"claimed", "approved", "rejected"}:
                    key = {
                        "claimed": "claims",
                        "approved": "approvals",
                        "rejected": "rejections",
                    }[event.event_type]
                    workload[event.actor_id][key] += 1
            for correction in task.corrections:
                workload[correction.reviewer_id]["corrections"] += 1
        return ReviewMetricsView(
            queue_depth=dict(queue_depth),
            correction_rate=correction_rate,
            reviewer_agreement=reviewer_agreement,
            average_queue_latency_seconds=sum(latencies) / len(latencies) if latencies else None,
            workload_by_reviewer=dict(workload),
        )
