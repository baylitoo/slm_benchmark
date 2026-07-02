"""Postgres-backed catalog of the local GGUF model store.

Blobs live on disk (the ``ModelStore`` under ``DOCIE_SERVING_HOME/models`` +
its ``index.json`` next to the weights, which the CLI/``serve_store_model``
use). This module adds a *queryable metadata catalog* in Postgres that the
Studio reads — so the API never has to read the on-disk flat file (avoiding
api/worker read/write races). The worker is the single writer (seed/deploy run
there); the API only reads.

Never store GGUF blobs in the DB — only metadata. See
``serving/model_store.py`` for the blob layer and ``docs/docie-studio.md``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import BigInteger, DateTime, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from docie_bench.serving.model_store import FAMILIES, StoreEntry
from docie_bench.storage.db import Base, session_scope


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class ModelStoreEntry(Base):
    __tablename__ = "model_store_entry"

    name: Mapped[str] = mapped_column(String(200), primary_key=True)
    family: Mapped[str] = mapped_column(String(64), index=True)
    model_path: Mapped[str] = mapped_column(Text)
    mmproj_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # BigInteger: GGUF blobs routinely exceed Postgres INTEGER's ~2.147 GB cap
    # (a 7B Q4 is ~4 GB), which would overflow on insert. This app uses
    # create_all (no migrations), so an EXISTING database must be migrated
    # manually: ALTER TABLE model_store_entry ALTER COLUMN size_bytes TYPE BIGINT.
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


def available_backends(family: str) -> list[str]:
    """Backends that can serve a model of ``family`` faithfully.

    llama-server can serve every family; Ollama only those whose template it
    does not silently drop (``ollama_faithful``).
    """
    contract = FAMILIES.get(family)
    backends = ["llama-server"]
    if contract is not None and contract.ollama_faithful:
        backends.append("ollama")
    return backends


def _to_view(row: ModelStoreEntry) -> dict[str, Any]:
    contract = FAMILIES.get(row.family)
    return {
        "name": row.name,
        "family": row.family,
        "vision": bool(contract and contract.vision),
        "available_backends": available_backends(row.family),
        "has_mmproj": row.mmproj_path is not None,
        "source": row.source,
        "size_bytes": row.size_bytes,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


class CatalogUnavailableError(RuntimeError):
    """Raised when the catalog is used but the database is not configured."""


class ModelCatalog:
    """CRUD over the Postgres model-store catalog (requires DATABASE_URL)."""

    def upsert(self, entry: StoreEntry, *, size_bytes: int | None = None) -> dict[str, Any]:
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelStoreEntry, entry.name)
            if row is None:
                row = ModelStoreEntry(name=entry.name)
                session.add(row)
            row.family = entry.family
            row.model_path = entry.model_path.as_posix()
            row.mmproj_path = entry.mmproj_path.as_posix() if entry.mmproj_path else None
            row.source = entry.source
            row.size_bytes = size_bytes
            session.flush()
            return _to_view(row)

    def list(self) -> list[dict[str, Any]]:
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            rows = session.scalars(select(ModelStoreEntry).order_by(ModelStoreEntry.name)).all()
            return [_to_view(row) for row in rows]

    def get(self, name: str) -> dict[str, Any] | None:
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelStoreEntry, name)
            return _to_view(row) if row is not None else None

    def delete(self, name: str) -> bool:
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelStoreEntry, name)
            if row is None:
                return False
            session.delete(row)
            return True


__all__ = ["ModelStoreEntry", "ModelCatalog", "CatalogUnavailableError", "available_backends"]
