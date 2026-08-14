"""Postgres-backed model-store catalog: large blob sizes must not overflow.

``ModelStoreEntry.size_bytes`` must be a 64-bit column. GGUF blobs routinely
exceed the ~2.147 GB Postgres INTEGER cap (a 7B Q4 is ~4 GB), which would
overflow on insert and wedge the seed job in a retry loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import BigInteger, create_engine

import docie_bench.storage.db as db
from docie_bench.serving.catalog import (
    ModelActivity,
    ModelCatalog,
    ModelStoreEntry,
    ensure_model_activity_table,
)
from docie_bench.serving.model_store import StoreEntry

# Larger than a signed 32-bit int (Postgres INTEGER max ~2.147 GB) — the size
# that used to overflow. Roughly a 3 GB blob.
_BIG_SIZE = 3_000_000_000


def test_size_bytes_column_is_bigint() -> None:
    # The load-bearing guard: BigInteger subclasses Integer, so this is False
    # for the old (overflowing) Integer column and True only after the fix.
    # (SQLite stores all ints as 64-bit, so a round-trip alone can't catch it.)
    assert isinstance(ModelStoreEntry.__table__.c.size_bytes.type, BigInteger)
    assert _BIG_SIZE > 2**31


@pytest.fixture
def _sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


def test_large_size_bytes_round_trips_through_catalog(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    entry = StoreEntry(
        name="qwen2.5-7b-q4",
        family="openai_chat",
        model_path=Path("/models/qwen2.5-7b-q4.gguf"),
    )

    upserted = catalog.upsert(entry, size_bytes=_BIG_SIZE)
    assert upserted["size_bytes"] == _BIG_SIZE

    listed = catalog.list()
    assert [row["size_bytes"] for row in listed] == [_BIG_SIZE]
    assert catalog.get("qwen2.5-7b-q4")["size_bytes"] == _BIG_SIZE


def test_store_view_includes_placement_or_null(_sqlite_catalog: None) -> None:
    """GET /v1/serving/store entries carry where (and whether) a model is served,
    so the Playground/any UI can pick a live deployment without guessing."""
    catalog = ModelCatalog()
    catalog.upsert(
        StoreEntry(
            name="qwen2.5-7b-q4",
            family="openai_chat",
            model_path=Path("/models/qwen2.5-7b-q4.gguf"),
        )
    )

    assert catalog.get("qwen2.5-7b-q4")["placement"] is None
    assert [row["placement"] for row in catalog.list()] == [None]

    catalog.record_placement(
        "qwen2.5-7b-q4",
        model_name="qwen2.5-7b-q4",
        engine="llama-server",
        endpoint="http://127.0.0.1:8088/v1",
        state="ready",
    )

    placement = catalog.get("qwen2.5-7b-q4")["placement"]
    assert placement is not None
    assert placement["engine"] == "llama-server"
    assert placement["endpoint"] == "http://127.0.0.1:8088/v1"
    assert placement["state"] == "ready"
    assert placement["negotiated_style"] is None
    assert placement["updated_at"] is not None
    listed = catalog.list()
    assert listed[0]["placement"]["state"] == "ready"


# ── model_activity: the autoscale-up signal (write side only — nothing acts
# on it yet, see catalog.ModelActivity's docstring) ─────────────────────────


def test_activity_starts_absent(_sqlite_catalog: None) -> None:
    assert ModelCatalog().get_activity("qwen2.5-7b-q4") is None
    assert ModelCatalog().list_activity() == []


def test_activity_first_record_creates_the_row(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    catalog.record_activity("qwen2.5-7b-q4")

    activity = catalog.get_activity("qwen2.5-7b-q4")
    assert activity is not None
    assert activity["model_name"] == "qwen2.5-7b-q4"
    assert activity["window_count"] == 1
    assert activity["window_started_at"] is not None
    assert activity["last_request_at"] is not None


def test_activity_repeated_records_increment_the_same_row(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    for _ in range(3):
        catalog.record_activity("qwen2.5-7b-q4")

    activity = catalog.get_activity("qwen2.5-7b-q4")
    assert activity["window_count"] == 3
    # One row per model, not one per request.
    assert len(catalog.list_activity()) == 1


def test_activity_tracks_independently_per_model(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    catalog.record_activity("qwen2.5-7b-q4")
    catalog.record_activity("qwen2.5-7b-q4")
    catalog.record_activity("lfm2.5-350m")

    by_name = {row["model_name"]: row["window_count"] for row in catalog.list_activity()}
    assert by_name == {"qwen2.5-7b-q4": 2, "lfm2.5-350m": 1}


# ── list_activity replica-count enrichment: the missing half of the "how hot
# is this model" picture. window_count alone can't tell an operator whether
# a busy model is already scaled out or still a single point of contention —
# that's exactly the judgment call a human makes before clicking the
# existing scale stepper (Deployments tab, #92-95). This is a pure display
# join, nothing here decides or acts on its own. ───────────────────────────


def test_activity_reports_zero_replicas_when_never_placed(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    catalog.record_activity("lfm2.5-350m")

    row = catalog.list_activity()[0]
    assert row["live_replica_count"] == 0
    assert row["total_replica_count"] == 0


def test_activity_reports_live_and_evicted_replicas_separately(_sqlite_catalog: None) -> None:
    """A model that WAS hot and has since had its only placement evicted
    (stopped, endpoint cleared) must read as 0 live replicas -- not "still
    running" -- while still counting toward total_replica_count so the UI
    can tell "never deployed" apart from "was deployed, now idle-unloaded"."""
    catalog = ModelCatalog()
    catalog.record_activity("lfm2.5-350m")
    catalog.record_placement(
        "lfm2.5-350m",
        model_name="lfm2.5-350m",
        engine="llama-server",
        endpoint="",
        state="stopped",
    )

    row = catalog.list_activity()[0]
    assert row["live_replica_count"] == 0
    assert row["total_replica_count"] == 1


def test_activity_counts_scaled_replicas_correctly(_sqlite_catalog: None) -> None:
    """A scaled model has several placement rows sharing model_name (one per
    replica record name, e.g. <base>/<base>-2) -- live_replica_count must
    count only the ones actually serving, total_replica_count all of them."""
    catalog = ModelCatalog()
    catalog.record_activity("lfm2.5-350m")
    catalog.record_placement(
        "lfm2.5-350m",
        model_name="lfm2.5-350m",
        engine="llama-server",
        endpoint="http://worker:8091/v1",
        state="ready",
    )
    catalog.record_placement(
        "lfm2.5-350m-2",
        model_name="lfm2.5-350m",
        engine="llama-server",
        endpoint="",
        state="starting",
    )

    row = catalog.list_activity()[0]
    assert row["live_replica_count"] == 1
    assert row["total_replica_count"] == 2


@pytest.mark.parametrize("dialect_name", ["postgresql", "sqlite"])
def test_record_activity_is_a_native_atomic_upsert(dialect_name: str) -> None:
    """record_activity must not be get-then-INSERT: two concurrent requests
    for the SAME store model both reading window_count=N and both writing
    N+1 would silently lose an increment under READ COMMITTED — the exact
    accuracy an autoscale-up read needs. Verified at statement-shape level,
    mirroring test_publish_node_snapshot_is_a_native_on_conflict_upsert:
    the SET clause must reference the column (an atomic SQL increment),
    never a literal or excluded.window_count (which would just overwrite)."""
    import datetime as dt

    from sqlalchemy.dialects import postgresql, sqlite

    from docie_bench.serving.catalog import _activity_upsert

    now = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    statement = _activity_upsert(dialect_name, "qwen2.5-7b-q4", now)
    assert statement is not None
    dialect = postgresql.dialect() if dialect_name == "postgresql" else sqlite.dialect()
    sql = str(statement.compile(dialect=dialect))
    assert "INSERT INTO model_activity" in sql
    assert "ON CONFLICT (model_name) DO UPDATE" in sql
    # The atomic increment: window_count = model_activity.window_count + :window_count_1
    # (or the sqlite-quoted equivalent) -- NOT "= excluded.window_count".
    assert "window_count = model_activity.window_count +" in sql or (
        "window_count = " in sql and "excluded.window_count" not in sql
    )
    assert "last_request_at = excluded.last_request_at" in sql
    # window_started_at is set on first INSERT only -- never touched by the
    # conflict SET, so an existing model's window never silently resets.
    assert "window_started_at = excluded.window_started_at" not in sql


def test_ensure_model_activity_table_is_race_safe_and_idempotent(tmp_path: Path) -> None:
    """Mirrors ensure_serving_node_table's own test: CREATE TABLE IF NOT
    EXISTS (advisory lock on PostgreSQL), so concurrently starting processes
    (api, serving, N workers) cannot abort each other; a second run no-ops."""
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    assert ensure_model_activity_table(engine) is True
    assert ensure_model_activity_table(engine) is False  # already there: no-op

    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    ddl = str(
        CreateTable(ModelActivity.__table__, if_not_exists=True).compile(
            dialect=postgresql.dialect()
        )
    )
    assert "CREATE TABLE IF NOT EXISTS model_activity" in ddl
