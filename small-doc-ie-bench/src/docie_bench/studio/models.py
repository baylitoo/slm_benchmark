"""SQLAlchemy models for the durable Studio run index.

One ``StudioRun`` row per benchmark job (keyed by the Inngest ``event_id``) holds
the parsed metrics summary + status; ``StudioRunArtifact`` rows point at the
content-addressed blobs (``report.html``, ``predictions.jsonl``, ``metrics.json``).

``idempotency_key`` is unique: a re-fired benchmark with the same key resolves to
the existing run instead of starting a second one (see ``RunStore.claim``).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
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


class StudioRun(Base):
    __tablename__ = "studio_runs"

    # The Inngest event id that triggered the job — the address the UI polls.
    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    # Deterministic dedup key; a double-fire with the same key does not double-run.
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    # Bound to the authenticated principal at trigger time, never a client body
    # field — download/list are filtered by this so tenants can't read each other.
    tenant_id: Mapped[str] = mapped_column(String(128), index=True, default="anonymous")
    status: Mapped[str] = mapped_column(String(32), index=True, default="running")
    dataset: Mapped[str | None] = mapped_column(String(300), nullable=True)
    model_profile: Mapped[str | None] = mapped_column(String(128), nullable=True)
    schema_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Small summary only; the large predictions.jsonl lives in the blob store.
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    artifacts: Mapped[list[StudioRunArtifact]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="StudioRunArtifact.name"
    )


class StudioRunArtifact(Base):
    __tablename__ = "studio_run_artifacts"
    __table_args__ = (UniqueConstraint("run_event_id", "name", name="uq_studio_artifact_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_event_id: Mapped[str] = mapped_column(
        ForeignKey("studio_runs.event_id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    # Store-relative, content-addressed key (never an absolute worker path).
    relkey: Mapped[str] = mapped_column(String(400))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    media_type: Mapped[str] = mapped_column(String(150), default="application/octet-stream")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[StudioRun] = relationship(back_populates="artifacts")


class StudioEventOwner(Base):
    """Lightweight event id -> triggering principal binding.

    Recorded for every triggered job (benchmark *and* extraction), so the
    run-status route can reject a cross-tenant event id instead of proxying it
    from the tenant-agnostic Inngest server. Extraction runs have no
    ``StudioRun`` row, so this is the only ownership signal available for them.
    """

    __tablename__ = "studio_event_owners"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True, default="anonymous")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


__all__ = ["StudioEventOwner", "StudioRun", "StudioRunArtifact", "utcnow"]
