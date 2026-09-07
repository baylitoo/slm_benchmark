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
    BigInteger,
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
    # BigInteger: an artifact can exceed Postgres INTEGER's ~2.147 GB cap.
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    media_type: Mapped[str] = mapped_column(String(150), default="application/octet-stream")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[StudioRun] = relationship(back_populates="artifacts")


class DynamicSchema(Base):
    """A named, reusable ``DynamicSchemaSpec`` (schemas.dynamic) saved from the
    Studio's schema builder.

    Shared operator config like ``models.yaml``/``data/datasets.yaml`` -- no
    ``tenant_id``, unlike ``StudioRun``'s per-tenant scoping. Create-only for now
    (409 on a duplicate ``name``): no version/lifecycle field, matching the "save
    and reuse by name" scope of this first slice -- not the full draft/published
    versioning the original roadmap brief described. ``name`` doubles as
    ``DynamicSchemaSpec.document_type`` (both already share the same snake_case
    pattern), so a saved schema's own name IS the identifier extraction requests
    reference.
    """

    __tablename__ = "dynamic_schemas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class RoutingPolicyRecord(Base):
    """A named, reusable ``RoutingPolicy`` (extract.routing) saved from the
    Studio.

    Same "save and reuse by name" scope as ``DynamicSchema`` above: shared
    operator config, no ``tenant_id``, create-only (409 on a duplicate
    ``name``). Unlike ``DynamicSchema``, ``name`` is a separate registry key
    from the policy's own ``version`` field -- ``RoutingPolicy.version`` is a
    free-form label the policy author already controls (e.g. to track
    edits to a cascade's thresholds) and reusing it as the unique lookup key
    would conflate "which policy" with "which revision of that policy",
    forcing every version bump to also be a rename. ``spec_json`` stores the
    validated ``RoutingPolicy.model_dump()``, not the caller's raw input, so a
    stored row always round-trips through ``RoutingPolicy.model_validate``
    cleanly at benchmark-trigger time.
    """

    __tablename__ = "routing_policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class SeedRun(Base):
    """A durable record of one seed-download job (Ollama or Hugging Face).

    Seed progress itself is already tracked live via the realtime ``progress``
    topic and a pollable sidecar file (``serving.seed_progress``) -- both are
    ephemeral, cleared once the job settles, and were the ONLY record of a
    download's outcome: closing the Studio's seed panel (or navigating away)
    lost the in-flight percentage, and a failed download's error only ever
    reached the worker's own logs, never the Studio. This table is the
    missing durable counterpart -- one row per triggering event id, same
    claim/complete/fail lifecycle as ``StudioRun``, so a "Downloads" tab can
    list recent seeds (running, completed, failed) and show a failure's error
    text after the fact, not just while a subscriber happened to be watching.

    ``channel`` is the correlation key to the LIVE progress sidecar/realtime
    topic while a seed is still running -- the two are separate concerns
    (this row survives the job; the sidecar is deleted once it settles).
    """

    __tablename__ = "seed_runs"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    channel: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True, default="anonymous")
    # "ollama" | "hf" -- which seed route triggered this.
    kind: Mapped[str] = mapped_column(String(16), index=True)
    # Ollama reference (e.g. "qwen2.5:1.5b") or HF repo id, whichever applies.
    reference: Mapped[str | None] = mapped_column(String(300), nullable=True)
    # Store entry name this seed writes/wrote.
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), index=True, default="running")
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The job's own return value on success (family, size_bytes, catalog_registered, ...).
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class UsageRecord(Base):
    """One serving request against one deployment/profile -- the durable usage
    ledger behind ``GET /v1/studio/usage`` (the Observability tab's Usage
    section).

    Raw rows on purpose: one insert per request, aggregation happens at read
    time (``usage_store.usage_summary``). Pre-aggregating on write would need
    an upsert per request (a lock hotspot on the busiest deployment) to save a
    GROUP BY over a bounded window; the read path is a polling dashboard, not
    a hot loop. Prometheus counters (telemetry.py) stay the real-time signal;
    this table is the queryable, per-tenant, restart-surviving record those
    counters can't be.

    ``deployment`` is the RESOLVED profile name (deployment name, models.yaml
    profile, or ``store:<name>``) -- same identifier ``recency.stamp_served_
    profile`` receives, so the Usage table lines up with the rest of the
    serving views. ``prompt_tokens``/``completion_tokens`` are nullable: a
    streamed chat proxies raw SSE bytes and never parses a usage block, and
    an errored request has none -- the request still counts.
    """

    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    deployment: Mapped[str] = mapped_column(String(200), index=True)
    # chat | extract | embed | rerank | agent -- which serving surface answered.
    surface: Mapped[str] = mapped_column(String(16), index=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer)
    # ok | error -- whether the surface answered the caller successfully.
    status: Mapped[str] = mapped_column(String(16), default="ok")
    tenant_id: Mapped[str] = mapped_column(String(128), index=True, default="anonymous")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


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


class ExtractionRunResult(Base):
    """Durable outcome of one extraction run, keyed by its Inngest event id.

    ``GET /v1/studio/runs/{event_id}`` is documented as pollable over plain
    HTTP with only an API key, no Inngest client required -- but that
    previously depended entirely on proxying Inngest's own
    ``GET /v1/events/{id}/runs``, whose ``output`` field is NOT reliably
    populated by this project's self-hosted Inngest server (confirmed by a
    real external integration test, not speculation: Inngest's Cloud REST API
    docs show `output` present on that endpoint, but self-hosted `inngest
    start` does not guarantee the same parity). ``extract_document`` writes
    its own outcome here on completion; the run-status route reads this
    FIRST, before ever touching the Inngest proxy, making the "no Inngest
    client needed" guarantee true regardless of the self-hosted server's REST
    completeness. The realtime ``result``/``error`` topics remain the primary,
    lower-latency channel for an Inngest-aware subscriber; this is the
    durable counterpart for a plain polling caller.
    """

    __tablename__ = "extraction_run_results"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True, default="anonymous")
    status: Mapped[str] = mapped_column(String(32))  # "completed" | "failed"
    output_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class BatchRun(Base):
    """A durable record of one batch-extraction job: N documents through one
    schema + model, each with its own per-item state (see ``BatchItem``).

    Deliberately NOT a ``StudioRun``: that table is benchmark-shaped -- one
    ``metrics_json`` blob and one artifact set written at ``complete()``. A
    batch's whole point is item-level state (running/done/failed EACH), so
    a document that fails is recorded and the batch continues, a crash
    resumes at the document it died on (each item is its own Inngest step),
    and "retry only the failed ones" has something to key on. The two
    counters here are denormalized from ``BatchItem`` so the list view never
    needs an aggregate query per row.

    Item results (JSONL + CSV) go into the SAME ``ArtifactBlobStore`` the
    benchmark uses, served through the same authenticated
    ``/v1/studio/artifacts/{id}`` route -- reused, not rebuilt.
    """

    __tablename__ = "batch_runs"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    channel: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True, default="anonymous")
    name: Mapped[str] = mapped_column(String(200))
    schema_name: Mapped[str] = mapped_column(String(64))
    # The model selector as submitted (deployment name, model_profile, or the
    # resolved default) -- what the operator picked, for the list view.
    model_selector: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="running")
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    done_items: Mapped[int] = mapped_column(Integer, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, default=0)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Artifact refs (JSONL + CSV of every item's result) once the batch settles;
    # ``[{"name","artifact_id","uri","media_type","size_bytes"}, ...]``.
    artifacts_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    # Optional completion webhook: POSTed the settled run summary once the
    # batch reaches a terminal state (see functions._deliver_batch_webhook).
    callback_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    # The ORIGINAL model/schema selectors as submitted (deployment /
    # model_profile / routing_policy / ocr_backend / language /
    # dynamic_schema_name) -- what "retry failed only" re-fires with, since
    # ``model_selector`` above is a display string that lost which KIND of
    # selector it was.
    selectors_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    items: Mapped[list[BatchItem]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="BatchItem.position"
    )


class BatchItem(Base):
    """One document of a batch: its own status, its extraction result (the
    full ExtractionResponse as JSON) or its error. Keyed by position so the
    results line up with the submitted order and a retry can name exactly the
    failed positions."""

    __tablename__ = "batch_items"
    __table_args__ = (UniqueConstraint("run_event_id", "position", name="uq_batch_item_pos"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_event_id: Mapped[str] = mapped_column(
        ForeignKey("batch_runs.event_id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str] = mapped_column(String(300))
    # The document's key in the shared ArtifactBlobStore (written at claim).
    # Load-bearing twice: "retry failed only" re-reads the SAME durable bytes
    # without a re-upload, and RunStore.gc's orphan sweep consults it so batch
    # inputs are never reclaimed as orphans (they are referenced by no
    # StudioRunArtifact row). Nullable only for rows that predate the column.
    input_relkey: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="pending")
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    run: Mapped[BatchRun] = relationship(back_populates="items")


__all__ = [
    "BatchItem",
    "BatchRun",
    "DynamicSchema",
    "RoutingPolicyRecord",
    "SeedRun",
    "StudioEventOwner",
    "StudioRun",
    "StudioRunArtifact",
    "UsageRecord",
    "utcnow",
]
