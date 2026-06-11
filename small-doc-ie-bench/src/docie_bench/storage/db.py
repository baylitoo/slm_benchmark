from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, MetaData, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

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


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def init_engine() -> None:
    global _engine, _SessionLocal
    settings = get_settings()
    if not settings.database_url:
        return
    _engine = create_engine(settings.database_url, pool_pre_ping=True)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(bind=_engine)


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
