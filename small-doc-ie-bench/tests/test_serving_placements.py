"""Deployment placements: the catalog binding that makes deploys discoverable.

The deploy job records *where* a model is served (``ModelPlacement``); stopping
the deployment clears it. Without this table an extraction has no way to find
the endpoint a deploy just created — the deploy/extraction disconnect.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog, ModelPlacement, ModelStoreEntry
from docie_bench.serving.control_plane import _DefaultSupervisor


@pytest.fixture
def _sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


def _record(catalog: ModelCatalog, *, state: str = "ready") -> dict[str, Any]:
    return catalog.record_placement(
        "invoice-extractor",
        model_name="invoice-extractor",
        engine="llama-server",
        endpoint="http://127.0.0.1:8088/v1",
        state=state,
    )


def test_placement_table_is_separate() -> None:
    # A brand-new table is auto-created by create_all (no manual ALTER, unlike
    # widening a column on model_store_entry) and keeps churny operational state
    # off the immutable blob-metadata row.
    assert ModelPlacement.__tablename__ == "model_placement"
    assert ModelPlacement.__table__ is not ModelStoreEntry.__table__
    assert "created_at" in ModelPlacement.__table__.c
    assert "updated_at" in ModelPlacement.__table__.c


def test_record_placement_roundtrip(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    recorded = _record(catalog)
    assert recorded["engine"] == "llama-server"

    placement = catalog.get_placement("invoice-extractor")
    assert placement is not None
    assert placement["engine"] == "llama-server"
    assert placement["endpoint"] == "http://127.0.0.1:8088/v1"
    assert placement["state"] == "ready"
    assert placement["negotiated_style"] is None


def _ts(value: str) -> dt.datetime:
    """Parse a view timestamp as naive UTC (sqlite drops the tz offset)."""
    parsed = dt.datetime.fromisoformat(value)
    return parsed.replace(tzinfo=None)


def test_record_placement_upsert_updates_state_and_updated_at(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    before = _record(catalog, state="starting")
    after = _record(catalog, state="ready")

    assert after["state"] == "ready"
    assert _ts(after["created_at"]) == _ts(before["created_at"])
    assert _ts(after["updated_at"]) >= _ts(before["updated_at"])
    # Still one row: the upsert is keyed by deployment name.
    assert catalog.get_placement("invoice-extractor")["state"] == "ready"


def test_clear_placement_removes_binding(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    _record(catalog)

    assert catalog.clear_placement("invoice-extractor") is True
    assert catalog.get_placement("invoice-extractor") is None
    assert catalog.clear_placement("invoice-extractor") is False


def test_set_placement_style_updates_and_noops_when_missing(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    _record(catalog)

    catalog.set_placement_style("invoice-extractor", "json_object")
    assert catalog.get_placement("invoice-extractor")["negotiated_style"] == "json_object"
    catalog.set_placement_style("no-such-deployment", "json_object")  # must not raise


def test_run_deploy_records_placement(_sqlite_catalog: None, monkeypatch) -> None:
    from docie_bench.inngest import functions

    record = {
        "spec": {"name": "invoice-extractor", "launch": {"runtime": "llamacpp"}},
        "endpoint": "http://127.0.0.1:8088/v1",
        "state": "ready",
    }

    class _FakeControlPlane:
        async def up(self, name: str, *, port: int, context_length: int) -> dict[str, Any]:
            return record

    monkeypatch.setattr(functions, "_serving_control_plane", lambda: _FakeControlPlane())
    result = asyncio.run(functions._run_deploy({"model": "invoice-extractor"}))
    assert result is record

    placement = ModelCatalog().get_placement("invoice-extractor")
    assert placement is not None
    assert placement["engine"] == "llama-server"  # llamacpp runtime -> llama-server engine
    assert placement["model_name"] == "invoice-extractor"
    assert placement["endpoint"] == "http://127.0.0.1:8088/v1"
    assert placement["state"] == "ready"


def test_run_deploy_survives_missing_database(monkeypatch) -> None:
    """No DATABASE_URL: the deploy still succeeds, it is just not discoverable."""
    from docie_bench.inngest import functions

    db.dispose_engine()
    record = {
        "spec": {"name": "invoice-extractor", "launch": {"runtime": "llamacpp"}},
        "endpoint": "http://127.0.0.1:8088/v1",
        "state": "ready",
    }

    class _FakeControlPlane:
        async def up(self, name: str, *, port: int, context_length: int) -> dict[str, Any]:
            return record

    monkeypatch.setattr(functions, "_serving_control_plane", lambda: _FakeControlPlane())
    assert asyncio.run(functions._run_deploy({"model": "invoice-extractor"})) is record


class _FakeSupervisorBackend:
    def __init__(self) -> None:
        self.stopped: list[str] = []
        self.removed: list[str] = []

    def stop(self, name: str) -> dict[str, str]:
        self.stopped.append(name)
        return {"name": name, "state": "stopped"}

    def remove(self, name: str) -> None:
        self.removed.append(name)


def test_stop_clears_placement(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    _record(catalog)
    supervisor = _DefaultSupervisor(_FakeSupervisorBackend(), planner=None)

    supervisor.stop("invoice-extractor")
    assert catalog.get_placement("invoice-extractor") is None


def test_remove_clears_placement(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    _record(catalog)
    supervisor = _DefaultSupervisor(_FakeSupervisorBackend(), planner=None)

    supervisor.remove("invoice-extractor")
    assert catalog.get_placement("invoice-extractor") is None


def test_stop_without_database_is_best_effort() -> None:
    """Stopping a local process must not require (or fail on) a database."""
    db.dispose_engine()
    backend = _FakeSupervisorBackend()
    supervisor = _DefaultSupervisor(backend, planner=None)

    supervisor.stop("invoice-extractor")  # must not raise
    assert backend.stopped == ["invoice-extractor"]
