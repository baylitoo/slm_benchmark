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
from docie_bench.storage.db import Base, get_session_factory, init_engine, session_scope


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


class ModelPlacement(Base):
    """Live serving binding for one deployment: where a model is reachable.

    One row per deployment ``name`` (the llama-server ``--alias`` / Ollama model
    name). ``model_name`` links back to the store entry being served (indexed so
    ``store:<name>`` resolution never scans), and stays NULL for deployments of
    raw model ids that have no store row. Operational fields (state/endpoint)
    churn on every reconcile, so they live here instead of on the immutable blob
    metadata row.

    Migration note: this is a brand-NEW table, so ``Base.metadata.create_all``
    (run by ``init_engine``) auto-creates it on existing databases — no manual
    step needed. The manual-ALTER caveat on ``ModelStoreEntry.size_bytes`` does
    NOT apply here; do not add a migration note for this table.
    """

    __tablename__ = "model_placement"

    name: Mapped[str] = mapped_column(String(200), primary_key=True)
    model_name: Mapped[str | None] = mapped_column(String(200), index=True, nullable=True)
    engine: Mapped[str] = mapped_column(String(32))  # "llama-server" | "ollama"
    endpoint: Mapped[str] = mapped_column(Text)  # advertised URL incl. /v1
    state: Mapped[str] = mapped_column(String(32))  # LifecycleState value
    # Probed known-good response-format style (PR "probe-at-deploy" fills it);
    # NULL means the resolver falls back to the engine default.
    negotiated_style: Mapped[str | None] = mapped_column(String(64), nullable=True)
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


def _placement_view(row: ModelPlacement) -> dict[str, Any]:
    return {
        "name": row.name,
        "model_name": row.model_name,
        "engine": row.engine,
        "endpoint": row.endpoint,
        "state": row.state,
        "negotiated_style": row.negotiated_style,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _to_view(row: ModelStoreEntry, placement: ModelPlacement | None = None) -> dict[str, Any]:
    contract = FAMILIES.get(row.family)
    return {
        "name": row.name,
        "family": row.family,
        "vision": bool(contract and contract.vision),
        "available_backends": available_backends(row.family),
        "has_mmproj": row.mmproj_path is not None,
        "source": row.source,
        "size_bytes": row.size_bytes,
        "placement": _placement_view(placement) if placement is not None else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


class CatalogUnavailableError(RuntimeError):
    """Raised when the catalog is used but the database is not configured."""


class ModelCatalog:
    """CRUD over the Postgres model-store catalog (requires DATABASE_URL)."""

    def __init__(self) -> None:
        # Lazily initialize the shared engine: host-native CLI entrypoints
        # (`docie up` / `docie stop`) never call init_engine(), so without this
        # every placement write/clear from the CLI silently hit session=None and
        # store:<name> kept resolving to endpoints of stopped deployments.
        # init_engine() is a no-op without DATABASE_URL (session_scope then
        # yields None and each method raises CatalogUnavailableError as before).
        if get_session_factory() is None:
            init_engine()

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
            # One placement fetch for the whole listing; ascending updated_at so
            # the freshest placement per model wins in the dict.
            placements: dict[str, ModelPlacement] = {
                placement.model_name: placement
                for placement in session.scalars(
                    select(ModelPlacement)
                    .where(ModelPlacement.model_name.is_not(None))
                    .order_by(ModelPlacement.updated_at)
                )
            }
            return [_to_view(row, placements.get(row.name)) for row in rows]

    def get(self, name: str) -> dict[str, Any] | None:
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelStoreEntry, name)
            if row is None:
                return None
            return _to_view(row, self._placement_row_for_model(session, name))

    def delete(self, name: str) -> bool:
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelStoreEntry, name)
            if row is None:
                return False
            session.delete(row)
            return True

    # ------------------------------------------------------------- placements
    def record_placement(
        self,
        name: str,
        *,
        model_name: str | None,
        engine: str,
        endpoint: str,
        state: str,
        negotiated_style: str | None = None,
    ) -> dict[str, Any]:
        """Upsert the live placement for deployment ``name`` (worker writes it)."""
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelPlacement, name)
            if row is None:
                row = ModelPlacement(name=name)
                session.add(row)
            row.model_name = model_name
            row.engine = engine
            row.endpoint = endpoint
            row.state = state
            row.negotiated_style = negotiated_style
            session.flush()
            return _placement_view(row)

    def set_placement_style(self, name: str, style: str | None) -> None:
        """Record the probed known-good response-format style (probe writer).

        No-op when the placement is gone (e.g. cleared by a concurrent stop) —
        a style without a live placement is meaningless.
        """
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelPlacement, name)
            if row is None:
                return
            row.negotiated_style = style

    def clear_placement(self, name: str) -> bool:
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelPlacement, name)
            if row is None:
                return False
            session.delete(row)
            return True

    def get_placement(self, name: str) -> dict[str, Any] | None:
        """Placement by deployment ``name`` (the table's primary key)."""
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelPlacement, name)
            return _placement_view(row) if row is not None else None

    def get_placement_for_model(self, model_name: str) -> dict[str, Any] | None:
        """Freshest placement serving store model ``model_name`` (indexed lookup)."""
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = self._placement_row_for_model(session, model_name)
            return _placement_view(row) if row is not None else None

    @staticmethod
    def _placement_row_for_model(session: Any, model_name: str) -> ModelPlacement | None:
        return session.scalars(
            select(ModelPlacement)
            .where(ModelPlacement.model_name == model_name)
            .order_by(ModelPlacement.updated_at.desc())
        ).first()


__all__ = [
    "ModelStoreEntry",
    "ModelPlacement",
    "ModelCatalog",
    "CatalogUnavailableError",
    "available_backends",
]
