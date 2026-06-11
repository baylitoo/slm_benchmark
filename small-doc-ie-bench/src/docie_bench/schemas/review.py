from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReviewStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewReason(BaseModel):
    code: Literal[
        "invalid",
        "low_confidence",
        "weak_evidence",
        "model_disagreement",
        "learning_value",
        "manual",
    ]
    score: float = Field(ge=0.0)
    detail: str


class FieldCorrection(BaseModel):
    field_path: str = Field(min_length=1, max_length=512)
    value: Any
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("field_path")
    @classmethod
    def validate_field_path(cls, value: str) -> str:
        parts = value.split(".")
        if any(not part or part.startswith("_") for part in parts):
            raise ValueError("field_path must contain non-empty public path segments")
        return value


class ReviewTaskCreate(BaseModel):
    source_request_id: str = Field(min_length=1, max_length=64)
    schema_name: str = Field(min_length=1, max_length=64)
    model_profile: str = Field(min_length=1, max_length=128)
    document_hash: str | None = Field(default=None, max_length=128)
    original_prediction: dict[str, Any]
    validation_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    dynamic_schema: dict[str, Any] | None = None
    disagreement_score: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_learning_value: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)


class ClaimRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=128)
    expected_version: int = Field(ge=1)
    lease_seconds: int | None = Field(default=None, ge=30, le=86400)


class ReleaseRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=128)
    expected_version: int = Field(ge=1)
    comment: str | None = Field(default=None, max_length=2000)


class CorrectionRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=128)
    expected_version: int = Field(ge=1)
    corrections: list[FieldCorrection] = Field(min_length=1)
    comment: str | None = Field(default=None, max_length=2000)


class DecisionRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=128)
    expected_version: int = Field(ge=1)
    comment: str | None = Field(default=None, max_length=2000)


class ReviewCorrectionView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    revision: int
    reviewer_id: str
    created_at: dt.datetime
    corrections: list[dict[str, Any]]
    corrected_prediction: dict[str, Any]
    comment: str | None


class ReviewEventView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_type: str
    actor_id: str | None
    from_status: str | None
    to_status: str | None
    task_version: int
    created_at: dt.datetime
    details: dict[str, Any]


class ReviewTaskView(BaseModel):
    id: int
    source_request_id: str
    schema_name: str
    model_profile: str
    document_hash: str | None
    status: ReviewStatus
    priority: float
    priority_reasons: list[ReviewReason]
    original_prediction: dict[str, Any]
    latest_prediction: dict[str, Any]
    validation_errors: list[str]
    dynamic_schema: dict[str, Any] | None
    metadata: dict[str, str]
    claimed_by: str | None
    claim_expires_at: dt.datetime | None
    version: int
    created_at: dt.datetime
    updated_at: dt.datetime
    decided_at: dt.datetime | None
    decided_by: str | None
    decision_comment: str | None
    corrections: list[ReviewCorrectionView] = Field(default_factory=list)
    events: list[ReviewEventView] = Field(default_factory=list)


class ReviewMetricsView(BaseModel):
    queue_depth: dict[str, int]
    correction_rate: float | None
    reviewer_agreement: float | None
    average_queue_latency_seconds: float | None
    workload_by_reviewer: dict[str, dict[str, int]]


class AnnotationExportRequest(BaseModel):
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    split: Literal["train", "validation", "test"] = "train"
    task_ids: list[int] | None = None


class AnnotationExportView(BaseModel):
    version: str
    split: str
    created_at: dt.datetime
    task_count: int
    annotations_path: str
    manifest_path: str
