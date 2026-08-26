"""Database records for durable, tenant-scoped ASR jobs.

The synchronous OpenAI-compatible transcription route intentionally remains
stateless.  These tables are the durable counterpart for long recordings and
batches: one job, one row per recording, and addressable output artifacts.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from docie_bench.storage.db import Base


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class ASRJob(Base):
    __tablename__ = "asr_jobs"
    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_asr_job_tenant_key"),
    )

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), index=True)
    channel: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    deployment: Mapped[str] = mapped_column(String(200), index=True)
    model: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), index=True, default="queued")
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    completed_items: Mapped[int] = mapped_column(Integer, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, default=0)
    options_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_retention: Mapped[str] = mapped_column(String(32), default="delete_after_completion")
    raw_expires_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    cancel_requested_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    items: Mapped[list[ASRJobItem]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="ASRJobItem.position"
    )
    artifacts: Mapped[list[ASRJobArtifact]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="ASRJobArtifact.name"
    )


class ASRJobItem(Base):
    __tablename__ = "asr_job_items"
    __table_args__ = (
        UniqueConstraint("job_event_id", "position", name="uq_asr_job_item_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_event_id: Mapped[str] = mapped_column(
        ForeignKey("asr_jobs.event_id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str] = mapped_column(String(300))
    input_relkey: Mapped[str | None] = mapped_column(String(500), nullable=True, index=True)
    input_sha256: Mapped[str] = mapped_column(String(64), index=True)
    input_size_bytes: Mapped[int] = mapped_column(BigInteger)
    mime_type: Mapped[str] = mapped_column(String(150))
    reference_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="pending")
    detected_language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    job: Mapped[ASRJob] = relationship(back_populates="items")


class ASRJobArtifact(Base):
    __tablename__ = "asr_job_artifacts"
    __table_args__ = (
        UniqueConstraint("job_event_id", "name", name="uq_asr_job_artifact_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_event_id: Mapped[str] = mapped_column(
        ForeignKey("asr_jobs.event_id", ondelete="CASCADE"), index=True
    )
    item_position: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(300))
    kind: Mapped[str] = mapped_column(String(32), index=True)
    relkey: Mapped[str] = mapped_column(String(500), index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    media_type: Mapped[str] = mapped_column(String(150))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[ASRJob] = relationship(back_populates="artifacts")


__all__ = ["ASRJob", "ASRJobArtifact", "ASRJobItem", "utcnow"]
