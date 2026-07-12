"""Deployment placements: the catalog binding that makes deploys discoverable.

The deploy job records *where* a model is served (``ModelPlacement``). PR-1
row lifecycle (design fix #3): stopping a deployment RETAINS the row and
UPDATEs it non-routable (``state=stopped``, ``endpoint=""``, ``phase=cold``)
so display/auto-reload metadata survive; only ``remove`` DELETEs it. Without
this table an extraction has no way to find the endpoint a deploy just
created — the deploy/extraction disconnect.
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


_SERVE_RECORD: dict[str, Any] = {
    "spec": {"name": "invoice-extractor", "launch": {"runtime": "llamacpp"}},
    "endpoint": "http://127.0.0.1:8088/v1",
    "state": "ready",
}


class _FakeServeControlPlane:
    async def serve(
        self, model: str, *, name: str | None, runtime: str | None, replicas: int
    ) -> dict[str, Any]:
        return _SERVE_RECORD


def test_run_deploy_records_placement_for_runtime_deploys(
    _sqlite_catalog: None, monkeypatch
) -> None:
    """The explicit-runtime `serve` path bypasses serve_store_model, so the
    worker job records its placement itself (the `up` path records at the
    control-plane seam instead — see the tests below)."""
    from docie_bench.inngest import functions

    monkeypatch.setattr(functions, "_serving_control_plane", lambda: _FakeServeControlPlane())
    result = asyncio.run(
        functions._run_deploy({"model": "invoice-extractor", "runtime": "llamacpp"})
    )
    assert result is _SERVE_RECORD

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
    monkeypatch.setattr(functions, "_serving_control_plane", lambda: _FakeServeControlPlane())
    result = asyncio.run(
        functions._run_deploy({"model": "invoice-extractor", "runtime": "llamacpp"})
    )
    assert result is _SERVE_RECORD


def _up_plane(tmp_path: Path) -> Any:
    """A real ControlPlane over serve_store_model (seeded store + fake adapter)."""
    from test_serving_control_plane import _seed_nuextract3_store
    from test_serving_supervisor import FakeAdapter

    from docie_bench.serving.control_plane import ControlPlane
    from docie_bench.serving.runtime import RuntimeKind
    from docie_bench.serving.supervisor import PersistentSupervisor

    root = tmp_path / "models"
    _seed_nuextract3_store(root)
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.LLAMACPP: FakeAdapter()}
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None, model_store_root=root)
    return ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]


def test_docie_up_records_placement_at_the_control_plane_seam(
    _sqlite_catalog: None, tmp_path: Path
) -> None:
    """Placement recording lives in serve_store_model — the seam host-native
    `docie up` and the worker deploy share, symmetric with stop() clearing it —
    so the resolver's 'Deploy it first (... `docie up <name>`)' hint is
    truthful for CLI users, not only for Inngest deploys."""
    plane = _up_plane(tmp_path)

    asyncio.run(plane.up("nuextract3", port=8088))

    placement = ModelCatalog().get_placement("nuextract3")
    assert placement is not None
    assert placement["model_name"] == "nuextract3"
    assert placement["engine"] == "llama-server"
    assert placement["endpoint"] == "http://127.0.0.1:8088/v1"
    assert placement["state"] == "ready"


def test_up_without_database_is_best_effort(tmp_path: Path) -> None:
    """A deploy that succeeded must never fail on placement recording."""
    db.dispose_engine()
    plane = _up_plane(tmp_path)

    record = asyncio.run(plane.up("nuextract3", port=8088))
    assert record["state"] == "ready"


def test_cli_stop_lazily_initializes_engine(tmp_path: Path, monkeypatch) -> None:
    """`docie stop` runs in a process that never calls init_engine(): the
    catalog seam lazily initializes it from DATABASE_URL, otherwise the
    staleness cleanup silently no-ops and store:<name> keeps resolving to a
    dead endpoint after every stop — the exact staleness the stop/remove hooks
    exist to prevent. PR-1: the cleanup is now an UPDATE (endpoint="") that
    retains the row, not a delete."""
    from docie_bench.settings import get_settings

    db.dispose_engine()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'catalog.db'}")
    get_settings.cache_clear()
    try:
        _record(ModelCatalog())  # ModelCatalog() itself lazy-initializes the engine
        supervisor = _DefaultSupervisor(_FakeSupervisorBackend(), planner=None)

        supervisor.stop("invoice-extractor")
        placement = ModelCatalog().get_placement("invoice-extractor")
        assert placement is not None  # retained (delete is remove()'s job only)
        assert placement["endpoint"] == ""  # but no longer routable
        assert placement["state"] == "stopped"
    finally:
        get_settings.cache_clear()
        db.dispose_engine()


class _FakeSupervisorBackend:
    def __init__(self) -> None:
        self.stopped: list[str] = []
        self.removed: list[str] = []

    def stop(self, name: str) -> dict[str, str]:
        self.stopped.append(name)
        return {"name": name, "state": "stopped"}

    def remove(self, name: str) -> None:
        self.removed.append(name)


def test_stop_updates_placement_instead_of_clearing(_sqlite_catalog: None) -> None:
    """PR-1 (design fix #3): stop UPDATEs the row cold/non-routable; only
    remove() deletes. An evicted/stopped deployment must keep its row so the
    Board (and PR-4's auto-reload) still see it."""
    catalog = ModelCatalog()
    _record(catalog)
    supervisor = _DefaultSupervisor(_FakeSupervisorBackend(), planner=None)

    supervisor.stop("invoice-extractor")
    placement = catalog.get_placement("invoice-extractor")
    assert placement is not None
    assert placement["state"] == "stopped"
    assert placement["endpoint"] == ""  # readers treat "" as "no live endpoint"
    assert placement["phase"] == "cold"
    assert placement["health_ok"] is False


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
