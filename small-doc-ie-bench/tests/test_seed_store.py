"""``docie_bench.studio.seed_store`` -- the durable claim/complete/fail/list
lifecycle for seed-download jobs. Before this module, a seed's outcome
(success or failure, and any error text) had no record beyond the ephemeral
realtime topic and a pollable-but-cleared-on-settle progress sidecar."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text as sa_text

from docie_bench.storage.db import dispose_engine, get_session_factory, init_engine
from docie_bench.studio.seed_store import (
    claim_seed_run,
    complete_seed_run,
    ensure_seed_run_table,
    fail_seed_run,
    list_seed_runs,
)


@pytest.fixture(autouse=True)
def seed_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'seeds.db'}")
    yield
    dispose_engine()


def test_claim_then_complete_round_trips() -> None:
    outcome, row = claim_seed_run(
        event_id="ev1", channel="seed:abc", tenant_id="t1", kind="ollama", name="qwen",
        reference="qwen2.5:1.5b",
    )

    assert outcome == "claimed"
    assert row["status"] == "running"

    completed = complete_seed_run(event_id="ev1", result={"name": "qwen", "size_bytes": 123})

    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == {"name": "qwen", "size_bytes": 123}
    assert completed["error"] is None


def test_complete_overwrites_name_from_result() -> None:
    # HF seeds may have no explicit name at claim time (derived mid-job).
    claim_seed_run(event_id="ev1", channel="seed:abc", tenant_id="t1", kind="hf", name="")

    completed = complete_seed_run(event_id="ev1", result={"name": "derived-name"})

    assert completed is not None
    assert completed["name"] == "derived-name"


def test_claim_redelivery_of_completed_run_returns_exists_without_resetting_status() -> None:
    claim_seed_run(event_id="ev1", channel="seed:abc", tenant_id="t1", kind="ollama", name="qwen")
    complete_seed_run(event_id="ev1", result={"name": "qwen"})

    outcome, row = claim_seed_run(
        event_id="ev1", channel="seed:abc", tenant_id="t1", kind="ollama", name="qwen"
    )

    assert outcome == "exists"
    assert row["status"] == "completed"


def test_claim_retry_of_failed_run_resets_to_running() -> None:
    claim_seed_run(event_id="ev1", channel="seed:abc", tenant_id="t1", kind="ollama", name="qwen")
    fail_seed_run(event_id="ev1", error="connection reset")

    outcome, row = claim_seed_run(
        event_id="ev1", channel="seed:abc", tenant_id="t1", kind="ollama", name="qwen"
    )

    assert outcome == "claimed"
    assert row["status"] == "running"
    assert row["error"] is None


def test_fail_records_truncated_error_text() -> None:
    claim_seed_run(event_id="ev1", channel="seed:abc", tenant_id="t1", kind="ollama", name="qwen")

    failed = fail_seed_run(event_id="ev1", error="x" * 5000)

    assert failed is not None
    assert failed["status"] == "failed"
    assert len(failed["error"]) == 4000


def test_list_is_tenant_scoped_and_ordered_newest_first() -> None:
    claim_seed_run(event_id="ev1", channel="seed:a", tenant_id="tenant-a", kind="ollama", name="a")
    claim_seed_run(event_id="ev2", channel="seed:b", tenant_id="tenant-a", kind="ollama", name="b")
    claim_seed_run(event_id="ev3", channel="seed:c", tenant_id="tenant-b", kind="ollama", name="c")

    mine = list_seed_runs(tenant_id="tenant-a")

    assert [r["event_id"] for r in mine] == ["ev2", "ev1"]
    assert all(r["tenant_id"] == "tenant-a" for r in mine)


def test_duplicate_trigger_different_event_id_same_channel_resolves_to_existing() -> None:
    claim_seed_run(event_id="ev1", channel="seed:dup", tenant_id="t1", kind="ollama", name="qwen")

    outcome, row = claim_seed_run(
        event_id="ev2", channel="seed:dup", tenant_id="t1", kind="ollama", name="qwen"
    )

    assert outcome == "exists"
    assert row["event_id"] == "ev1"


def test_migration_actually_creates_the_unique_index_not_just_the_bare_table() -> None:
    factory = get_session_factory()
    assert factory is not None
    engine = factory.kw["bind"]
    ensure_seed_run_table(engine)  # idempotent; init_engine already ran it once

    with engine.connect() as connection:
        indexes = connection.execute(
            sa_text("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='seed_runs'")
        ).fetchall()

    assert any("UNIQUE" in (row[0] or "") for row in indexes)


def test_no_database_degrades_gracefully() -> None:
    dispose_engine()

    outcome, row = claim_seed_run(
        event_id="ev1", channel="seed:abc", tenant_id="t1", kind="ollama", name="qwen"
    )
    assert outcome == "unavailable"
    assert row == {}
    assert complete_seed_run(event_id="ev1", result={}) is None
    assert fail_seed_run(event_id="ev1", error="x") is None
    assert list_seed_runs(tenant_id="t1") == []
