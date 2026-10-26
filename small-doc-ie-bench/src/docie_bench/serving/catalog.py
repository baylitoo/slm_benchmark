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
import json
import logging
from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Table,
    Text,
    select,
    update,
)
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection, CursorResult, Engine
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.dml import Insert
from sqlalchemy.types import TypeEngine

from docie_bench.serving.model_store import FAMILIES, StoreEntry
from docie_bench.storage.db import (
    Base,
    get_session_factory,
    init_engine,
    isoformat,
    session_scope,
)

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
    # create_all (no migrations) and create_all never widens an EXISTING integer
    # column, so a pre-BigInteger database is repaired by ensure_size_bytes_bigint
    # (an explicit forward ALTER run in init_engine). Fresh databases get BIGINT
    # from create_all.
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
    *observed* columns below (everything from ``phase`` through
    ``throughput_source``), and ``create_all`` NEVER adds columns to an
    existing table — the same hazard documented on
    ``ModelStoreEntry.size_bytes``. ``init_engine`` therefore runs
    :func:`ensure_placement_observed_columns` (an explicit forward ``ALTER
    TABLE .. ADD COLUMN`` migration) right after ``create_all`` so an existing
    database gains the columns instead of throwing ``UndefinedColumn`` on the
    reconciler's first publish. Fresh databases get them via ``create_all``.
    Every later observed column must be added to BOTH this class and
    ``_OBSERVED_COLUMNS`` — mapping it here alone leaves existing databases
    behind.

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
    # Which node runs this deployment (#530). Set at creation by the
    # supervisor seam that launched the runtime, and rewritten every cycle by
    # that node's reconciler -- the process actually holding the handle -- so
    # ownership is observed rather than declared. NULL therefore means exactly
    # one thing: a row written before ownership was tracked, which on the
    # existing single-node deployment is every row. Indexed: the fleet views
    # filter by it.
    node_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
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
    # --------------------------------------------- measured throughput
    # Published from the per-model-path sidecar the reconciler's probe writes
    # (serving.throughput). NULL means "not measured" — never 0, which a reader
    # would render as a real and terrible number. ``throughput_source`` says
    # where the numbers came from ("timings" = llama.cpp's own server-side
    # measurement, "wall-clock" = client-timed over usage counts,
    # "not-applicable" = a runtime that generates no tokens, "unmeasured" = the
    # probe ran but yielded no defensible rate). ``ttft_ms`` is NULL on the
    # wall-clock source by construction: a non-streamed round-trip is not a
    # time-to-first-token. NOTHING here is ever predicted from hardware.
    tokens_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    throughput_measured_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    throughput_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # The real number of llama-server processing slots observed via GET
    # /slots (HealthResult.slot_count) -- NOT the --parallel launch value,
    # which can drift from what a manually-restarted process actually runs
    # with. NULL means "not live" or "undetermined" (a non-llamacpp engine,
    # an unhealthy deployment, --no-slots, an older build). Feeds the
    # concurrency-mismatch warning in model_gateway._state_for.
    slot_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# Primary key of the single serving_node row written before rows were keyed by
# node id (#529). Kept only so the first keyed publish can ADOPT that row and
# preserve the deployment's capacity history; nothing writes it any more.
LEGACY_SERVING_NODE_ROW_ID = "node"

# Deprecated alias: the fixed key is no longer where a snapshot is written.
SERVING_NODE_ROW_ID = LEGACY_SERVING_NODE_ROW_ID


class ServingNode(Base):
    """Per-node resource snapshot (PR-2, design doc §1 "Resources").

    Written ONLY by that node's serving-service reconciler each cycle (measured
    inside the serving container — the one whose cgroup actually bounds the
    runtimes); read by the api's ``/v1/serving/resources`` and the sizing
    engine. One row per node, keyed by ``node_identity.resolve_node_id()``
    (#529): a node's publish upserts ITS row, so N nodes reporting into one
    database stay distinguishable instead of overwriting each other — the
    capacity view would otherwise show whichever node reconciled last.
    ``source`` says whether the numbers came from an authoritative cgroup-v2
    limit (``cgroup``) or the soft VM fallback (``vm``) so the UI can badge
    soft measurements honestly.
    """

    __tablename__ = "serving_node"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    total_bytes: Mapped[int] = mapped_column(BigInteger)
    free_bytes: Mapped[int] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(16))  # "cgroup" | "vm"
    sum_rss_bytes: Mapped[int] = mapped_column(BigInteger)
    # Reclaimable page cache excluded from "used" when free_bytes was computed
    # (cgroup working-set accounting, see resources.read_cgroup_memory); 0
    # means no adjustment applied (vm source / nothing reclaimable).
    reclaimable_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    # The store models present in THIS node's blob store, as a JSON array of
    # names (#562). The blob store is node-local -- `DOCIE_SERVING_HOME/models`
    # on a volume #564 documents as unshareable -- while `model_store_entry` is
    # global. So the catalog saying a model exists does not mean the node
    # chosen to serve it has the file, and without this nothing shows which
    # nodes do. NULL means a node that has not reported its store yet.
    store_models: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Bumped by every desired-state write touching this node (#534). A node
    # compares it against the last generation it applied, which is what lets a
    # READER decide in one read whether the catalog is provably current for it
    # -- see serving/convergence.py for why readability is not enough. NULL
    # means no desired state has ever been written for this node.
    desired_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ServingDesiredDeployment(Base):
    """What a node SHOULD be running, owned by that node (#534, phase 2).

    Postgres becomes the authoritative global desired state while each node
    keeps ``deployments.json`` as its local snapshot -- the mechanism that makes
    the #527 non-negotiable work, since the reconciler repairs and saves without
    a database.

    Keyed ``(node_id, name)``: a desired row belongs to exactly one node, and
    moving a deployment between nodes is a delete here and an insert there
    rather than a mutation. ``generation`` is per-row and exists for WRITER
    concurrency -- every write carries the generation it read, and a mismatch
    loses, so two central API calls racing on one deployment cannot interleave.
    That is a different job from ``ServingNode.desired_generation``, which is
    per-node and exists for the READER.

    Nothing reads this yet; the reconciler side is a separate change.
    """

    __tablename__ = "serving_desired_deployment"

    node_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), primary_key=True)
    # Per-row optimistic concurrency (see the class docstring).
    generation: Mapped[int] = mapped_column(Integer, default=1)
    # "running" | "stopped" -- a removal is a DELETE, not a state, so a node
    # cannot mistake "should not run" for "no longer desired at all".
    desired_state: Mapped[str] = mapped_column(String(32))
    # The launch spec as JSON, opaque here: the catalog stores intent and the
    # node owns how to realise it.
    spec: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ModelActivity(Base):
    """Per-store-model request-activity signal, the seam an autoscale-up
    decision would read (design doc: autoscale-up research).

    One row per store model name (not per replica — replicas of the same
    model share load, so activity is tracked at the model level). Written
    best-effort from ``placement_resolver.resolve_store_profile`` on every
    live-placement pick; a write failure must never break routing (see the
    call site). ``window_count``/``window_started_at`` form a crude tumbling
    window: intentionally NOT a rolling rate or a request log — the
    reconciler is expected to read-then-zero the window each cycle (an
    autoscale decision only needs "how many requests since I last looked",
    not a precise rate). Read-only today (surfaced on ``GET
    /v1/serving/activity``, purely observational) — nothing acts on it yet;
    the reconciler wiring is deliberately a separate, later change.
    """

    __tablename__ = "model_activity"

    model_name: Mapped[str] = mapped_column(String(200), primary_key=True)
    window_count: Mapped[int] = mapped_column(Integer, default=0)
    window_started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_request_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# Arbitrary-but-stable advisory-lock key for the model_activity table
# creation. Distinct from every other migration's key so none of them
# serialize against each other needlessly.
_MODEL_ACTIVITY_LOCK_KEY = 0x0D0C1E05


def ensure_model_activity_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``model_activity`` if missing.

    Same pattern as :func:`ensure_serving_node_table` — ``CREATE TABLE IF
    NOT EXISTS`` under ``pg_advisory_xact_lock`` so concurrently starting
    processes (api, serving, N workers) can't race each other into a
    duplicate-table abort. No added-columns step: this table is new, so
    there is no pre-existing shape to migrate yet.
    """
    activity_table = ModelActivity.__table__
    assert isinstance(activity_table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _MODEL_ACTIVITY_LOCK_KEY},
            )
        existed = sa_inspect(connection).has_table("model_activity")
        connection.execute(CreateTable(activity_table, if_not_exists=True))
    return not existed


# Arbitrary-but-stable advisory-lock key for the serving_node table creation
# ("docie placement v2"). Distinct from the placement-columns key so the two
# migrations never serialize against each other needlessly.
_SERVING_NODE_LOCK_KEY = 0x0D0C1E02

# serving_node columns added AFTER the table first shipped on the PR-2 branch
# (create_all never ALTERs an existing table — the documented size_bytes
# hazard). Module-level tuple so the migration and tests agree on one list.
_SERVING_NODE_ADDED_COLUMNS: tuple[tuple[str, TypeEngine[Any]], ...] = (
    ("reclaimable_bytes", BigInteger()),
    # Which store models this node actually holds on disk (#562). JSON array.
    ("store_models", Text()),
    # Monotonic counter bumped by any desired-state write for this node (#534).
    ("desired_generation", Integer()),
)


def _ensure_serving_node_columns(connection: Connection, engine: Engine) -> None:
    """Add late serving_node columns to a pre-existing table (under the lock).

    PostgreSQL: ``ADD COLUMN IF NOT EXISTS .. NOT NULL DEFAULT 0`` — race-safe
    under the caller's advisory lock. Other dialects (sqlite tests):
    inspect-then-ALTER, safe there because sqlite is single-process dev/test.
    """
    if engine.dialect.name == "postgresql":
        for name, column_type in _SERVING_NODE_ADDED_COLUMNS:
            compiled = column_type.compile(dialect=engine.dialect)
            connection.execute(
                sa_text(
                    f"ALTER TABLE serving_node ADD COLUMN IF NOT EXISTS "
                    f"{name} {compiled} NOT NULL DEFAULT 0"
                )
            )
        return
    existing = {column["name"] for column in sa_inspect(connection).get_columns("serving_node")}
    for name, column_type in _SERVING_NODE_ADDED_COLUMNS:
        if name in existing:
            continue
        compiled = column_type.compile(dialect=engine.dialect)
        connection.execute(
            sa_text(f"ALTER TABLE serving_node ADD COLUMN {name} {compiled} NOT NULL DEFAULT 0")
        )


def ensure_serving_node_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``serving_node`` if missing (PR-2).

    Mirrors the PR-1 observed-columns pattern: ``create_all`` DOES create new
    tables, but concurrently starting processes (api, serving, N workers) each
    run inspect-then-CREATE and can race each other into a duplicate-table
    abort. Here the DDL is ``CREATE TABLE IF NOT EXISTS`` (supported by both
    PostgreSQL and the sqlite used in tests), taken under
    ``pg_advisory_xact_lock`` on PostgreSQL so concurrent migrators serialize
    and a raced CREATE degrades to a no-op. The ``existed`` inspection ALSO
    happens under that lock (mirroring the placement migration): a pre-lock
    snapshot could race a concurrent creator and misreport who created the
    table. Returns True when this call created the table; when it already
    existed, late-added columns are ensured (see
    ``_ensure_serving_node_columns``).
    """
    node_table = ServingNode.__table__
    assert isinstance(node_table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _SERVING_NODE_LOCK_KEY},
            )
        existed = sa_inspect(connection).has_table("serving_node")
        connection.execute(CreateTable(node_table, if_not_exists=True))
        if existed:
            _ensure_serving_node_columns(connection, engine)
    return not existed


_DESIRED_LOCK_KEY = 0x0D0C1E03


class DesiredStateConflictError(RuntimeError):
    """A desired-state write lost its optimistic-concurrency check (#534)."""


def ensure_serving_desired_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``serving_desired_deployment``.

    Same pattern and the same reasoning as :func:`ensure_serving_node_table` --
    ``CREATE TABLE IF NOT EXISTS`` under an advisory lock, with the inspection
    inside the lock so a pre-lock snapshot cannot misreport who created it.
    """
    table = ServingDesiredDeployment.__table__
    assert isinstance(table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _DESIRED_LOCK_KEY},
            )
        existed = sa_inspect(connection).has_table("serving_desired_deployment")
        connection.execute(CreateTable(table, if_not_exists=True))
    return not existed


# The observed columns added by PR-1 (see ModelPlacement docstring). Kept as a
# module-level tuple so the migration below and tests agree on one list. Any
# LATER observed column must be appended here too — ``create_all`` will give a
# fresh database the mapped column while an existing one silently keeps the old
# shape and throws ``UndefinedColumn`` on the next publish.
_OBSERVED_COLUMNS: tuple[tuple[str, TypeEngine[Any]], ...] = (
    ("phase", String(32)),
    ("pid", Integer()),
    ("pid_create_time", Float()),
    ("rss_bytes", BigInteger()),
    ("health_ok", Boolean()),
    ("last_probe_at", DateTime(timezone=True)),
    ("last_error", Text()),
    # Measured throughput (see the ModelPlacement docstring).
    ("tokens_per_second", Float()),
    ("ttft_ms", Float()),
    ("throughput_measured_at", DateTime(timezone=True)),
    ("throughput_source", String(32)),
    ("slot_count", Integer()),
    # Owning node (#530). Listed here, not just mapped above, or an existing
    # database would throw UndefinedColumn on the reconciler's first publish.
    ("node_id", String(32)),
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


_SIZE_BYTES_WIDEN_LOCK_KEY = 0x0D0C1E03

# Tables whose ``size_bytes`` column stores an on-disk model/artifact byte count
# that can exceed Postgres INTEGER's ~2.147 GB cap (a multi-GB GGUF blob or a
# benchmark artifact). Their ORM columns are ``BigInteger``, but ``create_all``
# never widens an EXISTING integer column — the documented ``size_bytes`` hazard
# — so a database created before the type was BigInteger keeps int4 and
# overflows on insert (``NumericValueOutOfRange``).
_SIZE_BYTES_BIGINT_TABLES = (
    "model_store_entry",
    "benchmark_run_artifacts",
    "studio_run_artifacts",
)


def ensure_size_bytes_bigint(engine: Engine) -> list[str]:
    """Forward migration: widen ``size_bytes`` from INTEGER to BIGINT in place.

    ``create_all`` creates the column as BIGINT from the ORM but never ALTERs an
    existing INTEGER column, so a long-lived database keeps int4 and any row with
    a >2.147 GB size (a multi-GB model blob) fails to insert with
    ``NumericValueOutOfRange``. This ALTERs each still-INTEGER column across the
    tables in ``_SIZE_BYTES_BIGINT_TABLES``.

    PostgreSQL-only. On SQLite (tests/dev) it is a no-op — SQLite INTEGER is a
    dynamic up-to-8-byte type with no 2 GB cap. Idempotent + concurrency-safe
    like the sibling migrations: it inspects the live column type and ALTERs only
    the ones still INTEGER, inside one transaction guarded by
    ``pg_advisory_xact_lock`` so concurrent migrators (api, serving, N workers)
    serialize. A fresh database created by ``create_all`` already has BIGINT and
    this is a no-op. Returns the table names actually widened.
    """
    if engine.dialect.name != "postgresql":
        return []
    inspector = sa_inspect(engine)
    widened: list[str] = []
    with engine.begin() as connection:
        connection.execute(
            sa_text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": _SIZE_BYTES_WIDEN_LOCK_KEY},
        )
        for table in _SIZE_BYTES_BIGINT_TABLES:
            if not inspector.has_table(table):
                continue  # create_all will create it complete (BIGINT)
            data_type = connection.execute(
                sa_text(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name = :t AND column_name = 'size_bytes'"
                ),
                {"t": table},
            ).scalar()
            if data_type == "integer":
                # Table name comes from the hard-coded tuple above, never input.
                connection.execute(
                    sa_text(f"ALTER TABLE {table} ALTER COLUMN size_bytes TYPE BIGINT")
                )
                widened.append(table)
    if widened:
        logger.info("size_bytes migration: widened %s to BIGINT", widened)
    return widened


def available_backends(family: str) -> list[str]:
    """Backends that can serve a model of ``family`` faithfully.

    Analyzer (encoder) families are served ONLY by the encoder runtime, never
    llama.cpp/Ollama. For everything else llama-server can serve every family;
    Ollama only those whose template it does not silently drop.
    """
    contract = FAMILIES.get(family)
    if contract is not None and contract.analyzer:
        return ["encoder"]
    # Transformers/AutoModel families are served ONLY by the transformers
    # runtime (no GGUF exists for them), never llama.cpp/Ollama.
    if contract is not None and contract.transformers_runtime:
        return ["transformers"]
    # Multi-vector (ColBERT) retrievers are served ONLY by the multi-vector
    # runtime (a safetensors snapshot, no GGUF), never llama.cpp/Ollama.
    if contract is not None and contract.multi_vector:
        return ["multi_vector"]
    if contract is not None and contract.dense_embedding:
        return ["dense_embedding"]
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
        "node_id": row.node_id,
        "phase": row.phase,
        "pid": row.pid,
        "pid_create_time": row.pid_create_time,
        "rss_bytes": row.rss_bytes,
        "health_ok": row.health_ok,
        "last_probe_at": isoformat(row.last_probe_at),
        "last_error": row.last_error,
        "tokens_per_second": row.tokens_per_second,
        "ttft_ms": row.ttft_ms,
        "throughput_measured_at": isoformat(row.throughput_measured_at),
        "throughput_source": row.throughput_source,
        "slot_count": row.slot_count,
        "created_at": isoformat(row.created_at),
        "updated_at": isoformat(row.updated_at),
    }


def _decode_store_models(raw: str | None) -> list[str] | None:
    """The node's reported store models, or None when it has not reported.

    None and [] mean different things: "this node has not told us" versus
    "this node holds nothing". A reader deciding whether a deploy can succeed
    there must not confuse them.
    """
    if raw is None:
        return None
    try:
        decoded = json.loads(raw)
    except ValueError:
        return None
    return [str(item) for item in decoded] if isinstance(decoded, list) else None


def _desired_view(row: ServingDesiredDeployment) -> dict[str, Any]:
    try:
        spec = json.loads(row.spec)
    except ValueError:
        spec = None
    return {
        "node_id": row.node_id,
        "name": row.name,
        "generation": row.generation,
        "desired_state": row.desired_state,
        "spec": spec if isinstance(spec, dict) else None,
        "updated_at": isoformat(row.updated_at),
    }


def _bump_desired_generation(session: Session, node_id: str) -> int:
    """Advance ``node_id``'s desired generation, creating no capacity numbers.

    The node row may not exist yet -- desired state can be written for a node
    before it has ever reported -- so this inserts a MINIMAL row rather than
    failing. The capacity fields stay at zero with ``source="unreported"`` and
    are overwritten by the node's first real publish; what matters here is that
    the counter exists and only ever moves forward.
    """
    row = session.get(ServingNode, node_id)
    if row is None:
        row = ServingNode(
            id=node_id,
            total_bytes=0,
            free_bytes=0,
            source="unreported",
            sum_rss_bytes=0,
            reclaimable_bytes=0,
            desired_generation=0,
        )
        session.add(row)
    row.desired_generation = (row.desired_generation or 0) + 1
    session.flush()
    return row.desired_generation


def _node_view(row: ServingNode) -> dict[str, Any]:
    return {
        # Which node these numbers describe. Published so a fleet reader can
        # never mistake one machine's capacity for the whole fleet's.
        "node_id": row.id,
        "store_models": _decode_store_models(row.store_models),
        "desired_generation": row.desired_generation,
        "total_bytes": row.total_bytes,
        "free_bytes": row.free_bytes,
        "source": row.source,
        "sum_rss_bytes": row.sum_rss_bytes,
        "reclaimable_bytes": row.reclaimable_bytes,
        "updated_at": isoformat(row.updated_at),
    }


def _activity_view(row: ModelActivity) -> dict[str, Any]:
    return {
        "model_name": row.model_name,
        "window_count": row.window_count,
        "window_started_at": isoformat(row.window_started_at),
        "last_request_at": isoformat(row.last_request_at),
        "updated_at": isoformat(row.updated_at),
    }


def _adopt_legacy_node_row(session: Session, node_id: str) -> bool:
    """Re-key the pre-fleet single ``"node"`` row to ``node_id`` (#529 migration).

    The deployment that ran before rows were keyed by node has one row under the
    fixed ``"node"`` key holding its capacity history. The node now publishing
    IS that machine, so adopting the row preserves the history instead of
    orphaning it beside a fresh one.

    One statement, decided by rowcount: two nodes sharing the central database
    both see the legacy row on their first publish, and a get-then-UPDATE would
    let both adopt it. The UPDATE is atomic, so the first wins and the second
    gets ``rowcount == 0`` and falls through to inserting its own row.

    Skipped when this node already has a row: during a rolling restart an
    outgoing pre-fleet container keeps re-creating the ``"node"`` row, and
    re-keying it onto an id that already exists would violate the primary key
    and fail every publish. The stale legacy row is then just another node in
    the fleet view until the staleness gate drops it, which is honest — the old
    container really is still publishing.

    Deliberately NOT in ``ensure_serving_node_table``: that runs at startup in
    every role, and a migration ordering mistake there is how the api last went
    down (#549). This runs on the serving reconciler's publish path only.
    """
    if node_id == LEGACY_SERVING_NODE_ROW_ID:
        return False
    already_mine = select(ServingNode.id).where(ServingNode.id == node_id).exists()
    result = cast(
        "CursorResult[Any]",
        session.execute(
            update(ServingNode)
            .where(ServingNode.id == LEGACY_SERVING_NODE_ROW_ID, ~already_mine)
            .values(id=node_id)
        ),
    )
    return bool(result.rowcount)


def _node_snapshot_upsert(
    dialect_name: str, node_id: str, values: dict[str, Any]
) -> Insert | None:
    """One-statement ``INSERT .. ON CONFLICT (id) DO UPDATE`` for serving_node.

    A node's publish must be a REAL upsert: get-then-INSERT lets two racing
    first publishes both observe "no row" and the loser abort on the duplicate
    key. Both production PostgreSQL and the sqlite used in tests support ``ON
    CONFLICT DO UPDATE`` natively; any other dialect returns ``None`` and the
    caller falls back to get-then-INSERT (single-writer semantics only).
    """
    if dialect_name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        pg_statement = pg_insert(ServingNode).values(id=node_id, **values)
        return pg_statement.on_conflict_do_update(
            index_elements=[ServingNode.id],
            set_={name: pg_statement.excluded[name] for name in values},
        )
    if dialect_name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        sqlite_statement = sqlite_insert(ServingNode).values(id=node_id, **values)
        return sqlite_statement.on_conflict_do_update(
            index_elements=[ServingNode.id],
            set_={name: sqlite_statement.excluded[name] for name in values},
        )
    return None


def _activity_upsert(dialect_name: str, model_name: str, now: dt.datetime) -> Insert | None:
    """One-statement ``INSERT .. ON CONFLICT (model_name) DO UPDATE`` that
    ATOMICALLY increments ``window_count``, for the same reason
    :func:`_node_snapshot_upsert` exists: this table is written from
    concurrent request-handling processes (every ``store:`` request), not a
    single writer, so a naive ``row.window_count += 1`` (Python-side
    read-modify-write) silently loses increments under READ COMMITTED — two
    concurrent requests both read N, both write N+1, one increment vanishes
    with no error. ``window_started_at`` is deliberately NOT in the
    conflict SET: it stays at its first-ever value until something (a
    future reconciler read) resets it — this call only ever advances
    ``window_count``/``last_request_at``/``updated_at``.
    """
    values = {
        "model_name": model_name,
        "window_count": 1,
        "window_started_at": now,
        "last_request_at": now,
        "updated_at": now,
    }
    if dialect_name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        pg_statement = pg_insert(ModelActivity).values(**values)
        return pg_statement.on_conflict_do_update(
            index_elements=[ModelActivity.model_name],
            set_={
                "window_count": ModelActivity.window_count + 1,
                "last_request_at": pg_statement.excluded.last_request_at,
                "updated_at": pg_statement.excluded.updated_at,
            },
        )
    if dialect_name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        sqlite_statement = sqlite_insert(ModelActivity).values(**values)
        return sqlite_statement.on_conflict_do_update(
            index_elements=[ModelActivity.model_name],
            set_={
                "window_count": ModelActivity.window_count + 1,
                "last_request_at": sqlite_statement.excluded.last_request_at,
                "updated_at": sqlite_statement.excluded.updated_at,
            },
        )
    return None


def _to_view(row: ModelStoreEntry, placement: ModelPlacement | None = None) -> dict[str, Any]:
    contract = FAMILIES.get(row.family)
    return {
        "name": row.name,
        "family": row.family,
        "vision": bool(contract and contract.vision),
        "available_backends": available_backends(row.family),
        "has_mmproj": row.mmproj_path is not None,
        # Launch paths (PR-3 sizing inputs): model_path keys the per-model
        # calibration sidecar and is the stat fallback for weights; mmproj_path
        # lets the fit table price the resident vision projector. SERVER-SIDE
        # inputs only: the unauthenticated /v1/serving/store endpoint strips
        # both before echoing entries to the browser (serving_api.list_store).
        "model_path": row.model_path,
        "mmproj_path": row.mmproj_path,
        "source": row.source,
        "size_bytes": row.size_bytes,
        "placement": _placement_view(placement) if placement is not None else None,
        "created_at": isoformat(row.created_at),
        "updated_at": isoformat(row.updated_at),
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
        node_id: str | None = None,
    ) -> dict[str, Any]:
        """Upsert the live placement for deployment ``name`` (worker writes it).

        ``node_id`` is optional here and required on
        :meth:`publish_observed`: the deploy path may not be running on the
        node that will own the runtime, while the reconciler always is. Passing
        ``None`` leaves whatever the owning node last recorded instead of
        erasing it.
        """
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
            if node_id is not None:
                row.node_id = node_id
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
        node_id: str,
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
        tokens_per_second: float | None = None,
        ttft_ms: float | None = None,
        throughput_measured_at: dt.datetime | None = None,
        throughput_source: str | None = None,
        slot_count: int | None = None,
    ) -> dict[str, Any]:
        """Publish one reconciler observation (the reconciler is the sole caller).

        UPDATEs the existing row's observed fields; creates the row when it is
        missing (e.g. the deploy happened while the database was down) so the
        Board converges instead of staying blind forever. On create,
        ``model_name`` is linked iff a store entry with the deployment name
        exists — mirroring what the deploy path would have recorded.
        ``endpoint`` must already be ``""`` for any non-live observation (the
        reconciler enforces that; this method stores what it is given).

        The throughput fields carry the reconciler's per-model-path sidecar
        sample forward each cycle (``None`` = not measured). They are written
        unconditionally alongside the rest of the observation so a measurement
        that stops being valid — model replaced, sample expired — clears
        instead of lingering as a number nothing stands behind.

        ``node_id`` records which node runs this deployment (#530). It is
        written with the rest of the observation, under the same monotonic
        guard: the publisher is the node holding the runtime's handle, so the
        freshest observation's node is the owner, and a stale publish cannot
        claim a deployment away from the node that actually has it.
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
                    node_id=node_id,
                )
                session.add(row)
            elif probed_at is not None and row.updated_at is not None:
                # Monotonic guard: the reconciler snapshots under the supervisor
                # lock but publishes OUTSIDE it, so a load/stop that updated
                # this row after the snapshot (mark_placement_ready et al.)
                # must not be clobbered by the older observation. The next
                # cycle's fresher observation applies normally.
                updated = row.updated_at
                if updated.tzinfo is None:  # sqlite naive round-trip
                    updated = updated.replace(tzinfo=dt.UTC)
                if updated > probed_at:
                    return _placement_view(row)
            row.node_id = node_id
            row.state = state
            row.endpoint = endpoint
            row.phase = phase
            row.pid = pid
            row.pid_create_time = pid_create_time
            row.rss_bytes = rss_bytes
            row.health_ok = health_ok
            row.last_probe_at = probed_at or _utcnow()
            row.last_error = last_error
            row.tokens_per_second = tokens_per_second
            row.ttft_ms = ttft_ms
            row.throughput_measured_at = throughput_measured_at
            row.throughput_source = throughput_source
            row.slot_count = slot_count
            session.flush()
            return _placement_view(row)

    # ----------------------------------------------------------- node snapshot
    def publish_node_snapshot(
        self,
        *,
        node_id: str,
        total_bytes: int,
        free_bytes: int,
        source: str,
        sum_rss_bytes: int,
        reclaimable_bytes: int = 0,
        store_models: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Upsert ``node_id``'s ``serving_node`` row (its reconciler is the sole caller).

        Keyed by node (#529), so repeated publishes from one node UPDATE in
        place while a second node adds its own row rather than overwriting this
        one. A REAL upsert (``INSERT .. ON CONFLICT (id) DO UPDATE``) — never
        get-then-INSERT, whose two racing first publishes (e.g. a replaced
        serving container overlapping the old one within the lease staleness
        window) both see "no row" and the loser aborts on the duplicate key.
        ``updated_at`` is stamped explicitly (not left to ``onupdate``) so an
        unchanged reading still proves the reconciler is alive.
        """
        stamped = _utcnow()
        values: dict[str, Any] = {
            "total_bytes": total_bytes,
            "free_bytes": free_bytes,
            "source": source,
            "sum_rss_bytes": sum_rss_bytes,
            "reclaimable_bytes": reclaimable_bytes,
            "updated_at": stamped,
        }
        if store_models is not None:
            values["store_models"] = json.dumps(sorted(set(store_models)))
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            _adopt_legacy_node_row(session, node_id)
            statement = _node_snapshot_upsert(
                session.get_bind().dialect.name, node_id, values
            )
            if statement is not None:
                session.execute(statement)
            else:  # pragma: no cover - no third dialect is in use
                # Fallback for dialects without ON CONFLICT support: the old
                # get-then-INSERT (single-writer semantics only).
                row = session.get(ServingNode, node_id)
                if row is None:
                    row = ServingNode(id=node_id)
                    session.add(row)
                for name, value in values.items():
                    setattr(row, name, value)
                session.flush()
            return {"node_id": node_id, **values, "updated_at": isoformat(stamped)}

    def get_node_snapshot(self, node_id: str | None = None) -> dict[str, Any] | None:
        """One node's published snapshot, or None when there is no single answer.

        With ``node_id``, that node's row. Without it, the row of the only node
        that has published — and ``None`` once several have. That is deliberate
        for the callers that PRICE with the result (RAM admission): picking one
        of N nodes' free RAM is the "whichever reconciled last" guess #529
        exists to remove, and it can refuse a deploy that fits on another node.
        The admission gate is documented fail-open on an absent snapshot, so
        ``None`` degrades to the serving container's own per-instance fit gate
        instead of a wrong refusal. Display callers want
        :meth:`list_node_snapshots` and must say which node they are showing.
        """
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            if node_id is not None:
                row = session.get(ServingNode, node_id)
                return _node_view(row) if row is not None else None
            rows = session.scalars(select(ServingNode).limit(2)).all()
            return _node_view(rows[0]) if len(rows) == 1 else None

    # ------------------------------------------------------- desired state
    def set_desired_deployment(
        self,
        node_id: str,
        name: str,
        *,
        desired_state: str,
        spec: dict[str, Any],
        expected_generation: int | None = None,
    ) -> dict[str, Any]:
        """Write what ``node_id`` should be running for ``name`` (#534).

        ``expected_generation`` is the per-row generation the caller read;
        ``None`` means "this row should not exist yet". A mismatch raises
        :class:`DesiredStateConflictError` rather than overwriting, so two central
        API calls racing on one deployment cannot interleave -- the loser
        re-reads and decides again.

        The node's ``desired_generation`` is bumped in the SAME transaction, so
        the per-row and per-node counters cannot disagree: a reader that sees a
        newer node generation is guaranteed to see the row that caused it.
        """
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ServingDesiredDeployment, (node_id, name))
            current = row.generation if row is not None else None
            if current != expected_generation:
                raise DesiredStateConflictError(
                    f"desired state for {name!r} on node {node_id!r} is at "
                    f"generation {current!r}, not {expected_generation!r}; "
                    f"re-read and retry"
                )
            if row is None:
                row = ServingDesiredDeployment(node_id=node_id, name=name, generation=1)
                session.add(row)
            else:
                row.generation = row.generation + 1
            row.desired_state = desired_state
            row.spec = json.dumps(spec, sort_keys=True)
            node_generation = _bump_desired_generation(session, node_id)
            session.flush()
            return {
                "node_id": node_id,
                "name": name,
                "generation": row.generation,
                "desired_state": desired_state,
                "node_desired_generation": node_generation,
            }

    def delete_desired_deployment(
        self, node_id: str, name: str, *, expected_generation: int
    ) -> int:
        """Remove the desired row, returning the node's new generation.

        A removal is a DELETE rather than a state, so a node cannot confuse
        "should not be running" with "no longer desired at all" -- the two call
        for different actions.
        """
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ServingDesiredDeployment, (node_id, name))
            found = row.generation if row is not None else None
            if found != expected_generation:
                raise DesiredStateConflictError(
                    f"desired state for {name!r} on node {node_id!r} is at "
                    f"generation {found!r}, not {expected_generation!r}; "
                    f"re-read and retry"
                )
            session.delete(row)
            return _bump_desired_generation(session, node_id)

    def desired_snapshot(self, node_id: str) -> dict[str, Any]:
        """Everything ``node_id`` should be running, read in ONE transaction.

        The generation and the rows come from the same snapshot deliberately.
        A node derives DELETIONS from this, and a partial or interrupted read
        would otherwise look like "these deployments are gone" -- so the two
        must never be fetched separately.

        ``generation`` is None when no desired state has ever been written for
        this node, which ``convergence.decide_convergence`` treats differently
        from a node that had rows and lost them.
        """
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            node = session.get(ServingNode, node_id)
            rows = session.scalars(
                select(ServingDesiredDeployment)
                .where(ServingDesiredDeployment.node_id == node_id)
                .order_by(ServingDesiredDeployment.name)
            ).all()
            return {
                "node_id": node_id,
                # Whether the catalog knows this node AT ALL, reported apart
                # from the generation because they mean different things and a
                # reader must not conflate them: a known node with no desired
                # state yet is ordinary, while an unknown node that is running
                # deployments suggests the wrong database. Both show
                # generation=None. See convergence.decide_convergence.
                "node_known": node is not None,
                "generation": node.desired_generation if node is not None else None,
                "deployments": [_desired_view(row) for row in rows],
            }

    def list_node_snapshots(self) -> Sequence[dict[str, Any]]:
        """Every node's published snapshot, newest measurement first."""
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            rows = session.scalars(
                select(ServingNode).order_by(ServingNode.updated_at.desc())
            ).all()
            return [_node_view(row) for row in rows]

    def mark_placement_ready(self, name: str, *, endpoint: str) -> bool:
        """UPDATE an existing row to ready/hot after a load (PR-4, no create).

        The load-side sibling of ``mark_placement_stopped``: an evicted row
        holds ``endpoint=""`` and would keep ``store:<name>`` refusing to
        route for up to one reconcile cycle after the model is actually hot —
        this closes that window. Deliberately never CREATEs a row: the
        reconciler owns row creation for never-recorded deployments (its
        ``publish_observed`` fills engine/model_name properly). Returns False
        when no row exists.
        """
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelPlacement, name)
            if row is None:
                return False
            row.state = "ready"
            row.endpoint = endpoint
            row.phase = "hot"
            row.health_ok = True
            return True

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

    def list_placements_for_model(self, model_name: str) -> Sequence[dict[str, Any]]:
        """ALL placements serving store model ``model_name`` (its scaled
        replicas), freshest first. The load-balancing resolver picks one live
        replica per request; a single-instance model returns one row exactly as
        before."""
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            rows = session.scalars(
                select(ModelPlacement)
                .where(ModelPlacement.model_name == model_name)
                .order_by(ModelPlacement.updated_at.desc())
            ).all()
            return [_placement_view(row) for row in rows]

    # --------------------------------------------------------------- activity
    def record_activity(self, model_name: str) -> None:
        """Bump ``model_activity`` for ``model_name`` by one request.

        A REAL upsert (:func:`_activity_upsert`), not get-then-INSERT: this
        is called from the hot request-resolution path
        (``resolve_store_profile``) by every concurrent request for a store
        model, not a single writer, so a naive read-modify-write would
        silently lose increments under concurrent traffic (see
        ``_activity_upsert``'s docstring) — exactly the accuracy an
        eventual autoscale-up read needs.

        A no-op (not an error) when DATABASE_URL is not configured — unlike
        every other write on this class, this one is inherently best-effort:
        a missing/unhealthy DB must never block routing. A genuine DB error
        still raises; the caller is responsible for swallowing it (see
        ``placement_resolver``'s call site, which wraps this broadly and
        logs at debug level).
        """
        with session_scope() as session:
            if session is None:
                return
            now = _utcnow()
            statement = _activity_upsert(session.get_bind().dialect.name, model_name, now)
            if statement is not None:
                session.execute(statement)
                return
            # Fallback for a dialect without ON CONFLICT support: the old
            # get-then-INSERT (single-writer semantics only). No third
            # dialect is in use.
            row = session.get(ModelActivity, model_name)  # pragma: no cover
            if row is None:  # pragma: no cover
                session.add(
                    ModelActivity(
                        model_name=model_name,
                        window_count=1,
                        window_started_at=now,
                        last_request_at=now,
                        updated_at=now,
                    )
                )
                return
            row.window_count += 1  # pragma: no cover
            row.last_request_at = now  # pragma: no cover

    def get_activity(self, model_name: str) -> dict[str, Any] | None:
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            row = session.get(ModelActivity, model_name)
            return _activity_view(row) if row is not None else None

    def list_activity(self) -> Sequence[dict[str, Any]]:
        # Sequence, not the bare `list[...]` used elsewhere in this class —
        # this class already has a method literally named `list`, which
        # shadows the builtin for any bare `list[...]` annotation appearing
        # after it (a pre-existing mypy false-positive on
        # list_placements_for_model, left as-is; not reproducing it here).
        with session_scope() as session:
            if session is None:
                raise CatalogUnavailableError("DATABASE_URL is not configured")
            rows = session.scalars(
                select(ModelActivity).order_by(ModelActivity.last_request_at.desc())
            ).all()
            # One placement fetch for the whole listing (mirrors `list()`'s own
            # comment above) — replica counts are a display enrichment, not
            # worth a query per activity row. A model with activity but
            # live_replica_count == 0 is a real, useful signal on its own:
            # it was hot recently and has since been evicted/removed
            # entirely, not just idle.
            placements = session.scalars(
                select(ModelPlacement).where(ModelPlacement.model_name.is_not(None))
            ).all()
            live_counts: dict[str, int] = {}
            total_counts: dict[str, int] = {}
            for placement in placements:
                if placement.model_name is None:
                    continue
                total_counts[placement.model_name] = total_counts.get(placement.model_name, 0) + 1
                if placement.state == "ready" and (placement.endpoint or "").strip():
                    live_counts[placement.model_name] = live_counts.get(placement.model_name, 0) + 1
            return [
                {
                    **_activity_view(row),
                    "live_replica_count": live_counts.get(row.model_name, 0),
                    "total_replica_count": total_counts.get(row.model_name, 0),
                }
                for row in rows
            ]

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
    "ModelActivity",
    "ModelCatalog",
    "CatalogUnavailableError",
    "DesiredStateConflictError",
    "LEGACY_SERVING_NODE_ROW_ID",
    "SERVING_NODE_ROW_ID",
    "ServingDesiredDeployment",
    "ServingNode",
    "available_backends",
    "ensure_placement_observed_columns",
    "ensure_serving_desired_table",
    "ensure_serving_node_table",
    "ensure_model_activity_table",
    "ensure_size_bytes_bigint",
]
