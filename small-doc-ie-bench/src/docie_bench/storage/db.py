from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy import (
    inspect as sa_inspect,
)
from sqlalchemy import (
    text as sa_text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from docie_bench.settings import get_settings

logger = logging.getLogger(__name__)

metadata = MetaData()


class Base(DeclarativeBase):
    metadata = metadata


class ExtractionAudit(Base):
    __tablename__ = "extraction_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC)
    )
    tenant_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    schema_name: Mapped[str] = mapped_column(String(64), index=True)
    model_profile: Mapped[str] = mapped_column(String(128), index=True)
    document_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    valid: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[int] = mapped_column(Integer)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    warnings_json: Mapped[list[str]] = mapped_column(JSON)
    errors_text: Mapped[str] = mapped_column(Text, default="")


class ReviewTask(Base):
    __tablename__ = "review_task"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    schema_name: Mapped[str] = mapped_column(String(64), index=True)
    model_profile: Mapped[str] = mapped_column(String(128), index=True)
    document_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="pending")
    priority: Mapped[float] = mapped_column(Float, index=True)
    priority_reasons_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    original_prediction_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    latest_prediction_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    validation_errors_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    validation_warnings_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    dynamic_schema_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    claimed_by: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    claim_expires_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.UTC),
        onupdate=lambda: dt.datetime.now(dt.UTC),
    )
    decided_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision_comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    __mapper_args__ = {"version_id_col": version, "version_id_generator": False}

    corrections: Mapped[list[ReviewCorrection]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="ReviewCorrection.revision"
    )
    events: Mapped[list[ReviewEvent]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="ReviewEvent.id"
    )
    evidence: Mapped[ReviewEvidence | None] = relationship(
        back_populates="task", cascade="all, delete-orphan", uselist=False
    )


class ReviewCorrection(Base):
    __tablename__ = "review_correction"
    __table_args__ = (
        UniqueConstraint("task_id", "revision", name="uq_review_correction_revision"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("review_task.id"), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    reviewer_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC)
    )
    corrections_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    corrected_prediction_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    task: Mapped[ReviewTask] = relationship(back_populates="corrections")


class ReviewEvent(Base):
    __tablename__ = "review_event"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("review_task.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    actor_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    task_version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC)
    )
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    task: Mapped[ReviewTask] = relationship(back_populates="events")


class ReviewEvidence(Base):
    """OCR-block layout persisted alongside a review task (see
    settings.review_evidence_retention). One row per task -- a new table
    rather than a column on review_task since it's optional, can be sizable
    (every OCR block on the document), and is read separately from the task
    itself (GET .../evidence, not part of the default queue/detail payload).
    """

    __tablename__ = "review_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        ForeignKey("review_task.id"), unique=True, index=True
    )
    ocr_blocks_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    retention: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.UTC)
    )

    task: Mapped[ReviewTask] = relationship(back_populates="evidence")


_engine = None
_SessionLocal: sessionmaker[Session] | None = None

_REVIEW_TASK_MIGRATION_LOCK_KEY = 0x0D0C1E04


def ensure_review_task_validation_warnings_column(engine: Engine) -> bool:
    """Forward migration: add ``validation_warnings_json`` to an existing
    ``review_task`` table (mirrors ``ensure_placement_observed_columns`` in
    ``serving/catalog.py``). ``create_all`` never ALTERs an existing table, so
    a database that predates this column needs it added explicitly or
    ``enqueue_review`` throws ``UndefinedColumn`` on the first insert that
    carries a warning. Nullable-with-default, so no table rewrite / no lock
    pain. Returns whether the column was actually added.

    Same concurrency story as the placement migration: every process calling
    ``init_engine`` (api, worker, N replicas) may run this simultaneously, so
    PostgreSQL serializes via ``pg_advisory_xact_lock`` and uses
    ``ADD COLUMN IF NOT EXISTS``; sqlite (single-process dev/test only) falls
    back to inspect-then-ALTER. A fresh database already has the column via
    ``create_all`` and this is a no-op either way.
    """
    inspector = sa_inspect(engine)
    if not inspector.has_table("review_task"):
        return False  # create_all will create it complete
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _REVIEW_TASK_MIGRATION_LOCK_KEY},
            )
            existing = {
                row[0]
                for row in connection.execute(
                    sa_text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'review_task'"
                    )
                )
            }
            connection.execute(
                sa_text(
                    "ALTER TABLE review_task ADD COLUMN IF NOT EXISTS "
                    "validation_warnings_json JSON"
                )
            )
            added = "validation_warnings_json" not in existing
        else:
            existing = {column["name"] for column in inspector.get_columns("review_task")}
            added = "validation_warnings_json" not in existing
            if added:
                connection.execute(
                    sa_text("ALTER TABLE review_task ADD COLUMN validation_warnings_json JSON")
                )
    if added:
        logger.info("review_task migration: added validation_warnings_json")
    return added


_REVIEW_EVIDENCE_LOCK_KEY = 0x0D0C1E09
_SCHEMA_INIT_LOCK_KEY = 0x0D0C1E00


def ensure_review_evidence_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``review_evidence`` if missing.

    Same shape as ``ensure_seed_run_table`` (studio/seed_store.py):
    ``CREATE TABLE IF NOT EXISTS`` under ``pg_advisory_xact_lock`` on
    PostgreSQL, plus explicit ``if_not_exists=True`` index creation --
    ``Base.metadata.create_all()`` alone would race every process's
    concurrent ``init_engine`` (api, worker, N replicas) into a duplicate-
    table/index abort on whichever one loses.
    """
    from sqlalchemy.schema import CreateIndex, CreateTable

    evidence_table = ReviewEvidence.__table__
    assert isinstance(evidence_table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"), {"key": _REVIEW_EVIDENCE_LOCK_KEY}
            )
        existed = sa_inspect(connection).has_table("review_evidence")
        connection.execute(CreateTable(evidence_table, if_not_exists=True))
        for index in evidence_table.indexes:
            connection.execute(CreateIndex(index, if_not_exists=True))
    return not existed


def init_engine(database_url: str | None = None) -> None:
    global _engine, _SessionLocal
    resolved_url = database_url or get_settings().database_url
    if not resolved_url:
        return
    # Import model modules before creating metadata.
    import docie_bench.orchestrator.models  # noqa: F401
    import docie_bench.serving.catalog  # noqa: F401
    import docie_bench.studio.models  # noqa: F401

    _engine = create_engine(resolved_url, pool_pre_ping=True)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

    # Establish the dependency-ordered base schema before additive migrations.
    # This is load-bearing on a fresh PostgreSQL database: review_evidence has a
    # foreign key to review_task, so creating it individually before create_all
    # fails with UndefinedTable. api/serving/worker all initialize concurrently;
    # serialize create_all with a transaction-scoped advisory lock so its usual
    # inspect-then-CREATE sequence cannot race between processes. SQLAlchemy then
    # orders parent tables before their dependants within the locked transaction.
    with _engine.begin() as connection:
        if _engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _SCHEMA_INIT_LOCK_KEY},
            )
        Base.metadata.create_all(bind=connection)

    # Explicit forward migrations run AFTER create_all: create_all never ALTERs
    # an existing table, so a model_placement that predates the PR-1 observed
    # columns must gain them here or the reconciler's first publish throws
    # UndefinedColumn. On a fresh database create_all made the complete current
    # schema above, so every migration below is an idempotent no-op.
    from docie_bench.serving.catalog import (
        ensure_model_activity_table,
        ensure_placement_observed_columns,
        ensure_serving_node_table,
        ensure_size_bytes_bigint,
    )

    # Widen any pre-BigInteger `size_bytes` (model_store_entry + artifact tables)
    # from int4 to BIGINT so a multi-GB model/artifact size no longer overflows
    # on insert. No-op on fresh databases and on SQLite. See the caveat on
    # `ModelStoreEntry.size_bytes`.
    ensure_size_bytes_bigint(_engine)
    ensure_placement_observed_columns(_engine)
    # Same hazard, review_task's validation_warnings_json (added alongside the
    # arithmetic-reconciliation warning enrichment): a pre-existing table needs
    # it added explicitly, create_all only completes brand-new tables.
    ensure_review_task_validation_warnings_column(_engine)
    # review_evidence is a NEW table (evidence-aware Review workspace), same
    # concurrent-init_engine race as the other new tables below.
    ensure_review_evidence_table(_engine)
    # serving_node is a NEW table, which create_all would create — but the
    # api, serving service, and N workers all run init_engine concurrently at
    # stack-up, and create_all's inspect-then-CREATE can race into a
    # duplicate-table abort. The explicit CREATE TABLE IF NOT EXISTS (advisory
    # lock on PostgreSQL) makes the creation race-safe (PR-2, mirroring the
    # PR-1 observed-columns pattern above).
    ensure_serving_node_table(_engine)
    # Same race as serving_node above: model_activity is also new, also
    # created by every process's concurrent init_engine() at stack-up.
    ensure_model_activity_table(_engine)
    # Same race again: dynamic_schemas is also new.
    from docie_bench.studio.dynamic_schemas import ensure_dynamic_schema_table

    ensure_dynamic_schema_table(_engine)
    # Same race again: routing_policies is also new (registry for named,
    # reusable RoutingPolicy specs, alongside dynamic_schemas above).
    from docie_bench.studio.routing_policies import ensure_routing_policy_table

    ensure_routing_policy_table(_engine)
    # Same race again: seed_runs is also new (durable index for Ollama/HF
    # seed-download jobs, alongside dynamic_schemas/routing_policies above).
    from docie_bench.studio.seed_store import ensure_seed_run_table

    ensure_seed_run_table(_engine)
    # Same race again: batch_runs + batch_items are new (durable per-document
    # state for batch extraction). create_all above already ordered the child
    # FK after its parent; this is the idempotent race-safe belt-and-braces.
    from docie_bench.studio.batch_store import ensure_batch_tables

    ensure_batch_tables(_engine)


def get_session_factory() -> sessionmaker[Session] | None:
    return _SessionLocal


def database_enabled() -> bool:
    return _SessionLocal is not None


def dispose_engine() -> None:
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


@contextmanager
def session_scope() -> Iterator[Session | None]:
    if _SessionLocal is None:
        yield None
        return
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
