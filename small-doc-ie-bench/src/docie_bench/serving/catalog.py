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
import logging
from collections.abc import Sequence
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import TypeEngine

from docie_bench.serving.model_store import FAMILIES, StoreEntry
from docie_bench.storage.db import Base, get_session_factory, init_engine, session_scope

logger = logging.getLogger(__name__)


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

    Migration note (PR-1 reconciler): the base table shipped before the
    *observed* columns below (``phase`` .. ``last_error``), and ``create_all``
    NEVER adds columns to an existing table — the same hazard documented on
    ``ModelStoreEntry.size_bytes``. ``init_engine`` therefore runs
    :func:`ensure_placement_observed_columns` (an explicit forward ``ALTER
    TABLE .. ADD COLUMN`` migration) right after ``create_all`` so an existing
    database gains the columns instead of throwing ``UndefinedColumn`` on the
    reconciler's first publish. Fresh databases get them via ``create_all``.

    Row lifecycle (PR-1): the reconciler is the sole observed-state writer and
    UPDATEs this row every cycle; ``stop``/future ``unload`` UPDATE it
    (``endpoint=""``, ``phase=cold``/``evicted``) so display + auto-reload
    metadata survive; only a real delete (``ControlPlane.remove``) DELETEs it.
    ``endpoint`` is NOT NULL, so a non-live row stores ``""`` (empty string)
    and every reader treats ``""`` as "no live endpoint".
    """

    __tablename__ = "model_placement"

    name: Mapped[str] = mapped_column(String(200), primary_key=True)
    model_name: Mapped[str | None] = mapped_column(String(200), index=True, nullable=True)
    engine: Mapped[str] = mapped_column(String(32))  # "llama-server" | "ollama"
    endpoint: Mapped[str] = mapped_column(Text)  # advertised URL incl. /v1 ("" when not live)
    state: Mapped[str] = mapped_column(String(32))  # LifecycleState value
    # Probed known-good response-format style (PR "probe-at-deploy" fills it);
    # NULL means the resolver falls back to the engine default.
    negotiated_style: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # ------------------------------------------------ observed state (PR-1)
    # Written ONLY by the serving-service reconciler each cycle. NULL means
    # "never observed" (e.g. the reconciler has not run since this row was
    # created). NOTE: lifecycle-control metadata (activation/pinned) is NOT
    # here by design — it lives in deployments.json so DB-optional routing
    # survives (design doc §1, fix #5).
    phase: Mapped[str | None] = mapped_column(String(32), nullable=True)  # hot|loading|cold|...
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)  # serving-ns pid, advisory
    pid_create_time: Mapped[float | None] = mapped_column(Float, nullable=True)  # reuse guard
    rss_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    health_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    last_probe_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# The observed columns added by PR-1 (see ModelPlacement docstring). Kept as a
# module-level tuple so the migration below and tests agree on one list.
_OBSERVED_COLUMNS: tuple[tuple[str, TypeEngine[Any]], ...] = (
    ("phase", String(32)),
    ("pid", Integer()),
    ("pid_create_time", Float()),
    ("rss_bytes", BigInteger()),
    ("health_ok", Boolean()),
    ("last_probe_at", DateTime(timezone=True)),
    ("last_error", Text()),
)


# Arbitrary-but-stable advisory-lock key for the model_placement observed-columns
# migration ("docie placement v1"). Any concurrent migrator serializes on it.
_PLACEMENT_MIGRATION_LOCK_KEY = 0x0D0C1E01


def _postgres_add_column_ddl(name: str, column_type: TypeEngine[Any], dialect: Any) -> str:
    """The race-safe PostgreSQL form of one observed-column ADD (design §0.1/P2)."""
    compiled = column_type.compile(dialect=dialect)
    return f"ALTER TABLE model_placement ADD COLUMN IF NOT EXISTS {name} {compiled}"


def ensure_placement_observed_columns(engine: Engine) -> list[str]:
    """Forward migration: add the PR-1 observed columns to an existing table.

    ``create_all`` never alters an existing table (the ``size_bytes`` caveat),
    so a database that predates the observed columns would make the
    reconciler's first publish throw ``UndefinedColumn``. Adds exactly the
    missing columns (all nullable, so no table rewrite / no lock pain).
    Returns the column names actually added.

    Concurrency/idempotence: every process that calls ``init_engine`` (api,
    serving, N scaled workers) runs this at startup, possibly simultaneously.
    On PostgreSQL each column is added with ``ADD COLUMN IF NOT EXISTS`` (the
    design doc's §0.1/P2 SQL) inside one transaction that first takes
    ``pg_advisory_xact_lock`` — so concurrent migrators serialize and a raced
    duplicate ADD degrades to a no-op instead of a DuplicateColumn abort. On
    other dialects (sqlite in tests — no IF NOT EXISTS support for ADD
    COLUMN) it falls back to inspect-then-ALTER, which is safe there because
    the sqlite path is single-process dev/test only. A fresh database created
    by ``create_all`` already has the columns and this is a no-op either way.
    """
    inspector = sa_inspect(engine)
    if not inspector.has_table("model_placement"):
        return []  # create_all will create it complete
    added: list[str] = []
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _PLACEMENT_MIGRATION_LOCK_KEY},
            )
            existing = {
                row[0]
                for row in connection.execute(
                    sa_text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'model_placement'"
                    )
                )
            }
            for name, column_type in _OBSERVED_COLUMNS:
                # IF NOT EXISTS even for columns the snapshot says are missing:
                # the snapshot + lock make `added` accurate, the SQL guard makes
                # the ALTER itself unconditionally race-safe.
                connection.execute(
                    sa_text(_postgres_add_column_ddl(name, column_type, engine.dialect))
                )
                if name not in existing:
                    added.append(name)
        else:
            existing = {
                column["name"] for column in inspector.get_columns("model_placement")
            }
            for name, column_type in _OBSERVED_COLUMNS:
                if name in existing:
                    continue
                compiled = column_type.compile(dialect=engine.dialect)
                connection.execute(
                    sa_text(f"ALTER TABLE model_placement ADD COLUMN {name} {compiled}")
                )
                added.append(name)
    if added:
        logger.info("model_placement migration: added observed columns %s", added)
    return added


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
        "phase": row.phase,
        "pid": row.pid,
        "pid_create_time": row.pid_create_time,
        "rss_bytes": row.rss_bytes,
        "health_ok": row.health_ok,
        "last_probe_at": row.last_probe_at.isoformat() if row.last_probe_at else None,
        "last_error": row.last_error,
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
                # The WHERE clause already excludes NULLs; the guard just
                # narrows str | None for the type checker.
                if placement.model_name is not None
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

    def publish_observed(
        self,
        name: str,
        *,
        engine: str,
        state: str,
        endpoint: str,
        phase: str,
        pid: int | None,
        pid_create_time: float | None,
        rss_bytes: int,
        health_ok: bool,
        last_error: str | None,
        probed_at: dt.datetime | None = None,
    ) -> dict[str, Any]:
        """Publish one reconciler observation (the reconciler is the sole caller).

        UPDATEs the existing row's observed fields; creates the row when it is
        missing (e.g. the deploy happened while the database was down) so the
        Board converges instead of staying blind forever. On create,
        ``model_name`` is linked iff a store entry with the deployment name
        exists — mirroring what the deploy path would have recorded.
        ``endpoint`` must already be ``""`` for any non-live observation (the
        reconciler enforces that; this method stores what it is given).
        """
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelPlacement, name)
            if row is None:
                row = ModelPlacement(
                    name=name,
                    model_name=(
                        name if session.get(ModelStoreEntry, name) is not None else None
                    ),
                    engine=engine,
                )
                session.add(row)
            row.state = state
            row.endpoint = endpoint
            row.phase = phase
            row.pid = pid
            row.pid_create_time = pid_create_time
            row.rss_bytes = rss_bytes
            row.health_ok = health_ok
            row.last_probe_at = probed_at or _utcnow()
            row.last_error = last_error
            session.flush()
            return _placement_view(row)

    def mark_placement_stopped(self, name: str, *, phase: str = "cold") -> bool:
        """UPDATE a stopped deployment's row instead of deleting it (fix #3).

        ``stop`` (and the future ``unload``) must RETAIN the row — with
        ``endpoint=""`` so nothing routes to it — because deletion is
        ``remove``'s job only. Returns False when no row exists (nothing was
        ever recorded; that is fine).
        """
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelPlacement, name)
            if row is None:
                return False
            row.state = "stopped"
            row.endpoint = ""
            row.phase = phase
            row.pid = None
            row.pid_create_time = None
            row.rss_bytes = 0
            row.health_ok = False
            return True

    # NOTE the Sequence return: inside this class body a bare ``list``
    # annotation resolves to the ``list`` METHOD above (mypy valid-type error),
    # so the abstract type is both correct and necessary here.
    def list_placements(self) -> Sequence[dict[str, Any]]:
        """All placement rows (the observed surface the Board reads)."""
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            rows = session.scalars(select(ModelPlacement).order_by(ModelPlacement.name)).all()
            return [_placement_view(row) for row in rows]

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
    def _placement_row_for_model(session: Session, model_name: str) -> ModelPlacement | None:
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
    "ensure_placement_observed_columns",
]
