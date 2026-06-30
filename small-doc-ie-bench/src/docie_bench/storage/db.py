from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    create_engine,
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


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def init_engine(database_url: str | None = None) -> None:
    global _engine, _SessionLocal
    resolved_url = database_url or get_settings().database_url
    if not resolved_url:
        return
    # Import model modules before creating metadata.
    import docie_bench.orchestrator.models  # noqa: F401
    import docie_bench.serving.catalog  # noqa: F401

    _engine = create_engine(resolved_url, pool_pre_ping=True)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(bind=_engine)


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
