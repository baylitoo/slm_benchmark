"""Traffic served from one machine keeps a deployment on another machine warm.

The recency sidecar lives under the caller's serving home, so a central worker
serving a deployment on a remote node used to leave no trace that node could
see: its reconciler idle-unloaded the model mid-use. The catalog stamp closes
that gap.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.serving import recency
from docie_bench.serving.catalog import ModelCatalog
from tests.test_serving_lifecycle import ScriptedAdapter, _idle_reconciler, _spec, _supervisor


@pytest.fixture
def catalog(tmp_path: Path) -> Iterator[ModelCatalog]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield ModelCatalog()
    finally:
        db.dispose_engine()


def _place(catalog: ModelCatalog, name: str, model_name: str) -> None:
    catalog.record_placement(
        name,
        model_name=model_name,
        engine="llama-server",
        endpoint="http://rocky-llm:8088/v1",
        state="ready",
        node_id="rocky-llm",
    )


def _publish(catalog: ModelCatalog, name: str) -> None:
    catalog.publish_observed(
        name,
        node_id="rocky-llm",
        engine="llama-server",
        state="ready",
        endpoint="http://rocky-llm:8088/v1",
        phase="hot",
        pid=None,
        pid_create_time=None,
        rss_bytes=0,
        health_ok=True,
        last_error=None,
    )


def test_a_store_profile_stamps_every_replica_from_a_machine_without_the_record(
    catalog: ModelCatalog, tmp_path: Path
) -> None:
    _place(catalog, "lfm3-1.7b-f09ae4", "lfm3-1.7b")
    _place(catalog, "lfm3-1.7b-77c1d2", "lfm3-1.7b")
    central = tmp_path / "central"  # no deployments.json for these names

    recency.stamp_served_profile("store:lfm3-1.7b", timestamp=5000.0, home=central)

    served = catalog.last_served_for(["lfm3-1.7b-f09ae4", "lfm3-1.7b-77c1d2"])
    assert served == {"lfm3-1.7b-f09ae4": 5000.0, "lfm3-1.7b-77c1d2": 5000.0}
    assert not (central / "recency").exists()


def test_the_stamp_is_max_wins_and_survives_the_owners_publish(
    catalog: ModelCatalog, tmp_path: Path
) -> None:
    _place(catalog, "lfm3-1.7b-f09ae4", "lfm3-1.7b")
    recency.stamp_served_profile("lfm3-1.7b-f09ae4", timestamp=5000.0, home=tmp_path)
    recency.stamp_served_profile("lfm3-1.7b-f09ae4", timestamp=4000.0, home=tmp_path)
    _publish(catalog, "lfm3-1.7b-f09ae4")
    _place(catalog, "lfm3-1.7b-f09ae4", "lfm3-1.7b")

    assert catalog.last_served_for(["lfm3-1.7b-f09ae4"]) == {"lfm3-1.7b-f09ae4": 5000.0}


def test_remote_traffic_keeps_the_owner_from_idle_unloading(
    catalog: ModelCatalog, tmp_path: Path
) -> None:
    owner = tmp_path / "rocky"
    owner.mkdir()
    now = [1000.0]
    supervisor = _supervisor(owner, ScriptedAdapter(), clock=lambda: now[0])
    supervisor.deploy(_spec())
    _place(catalog, "invoice", "invoice")
    reconciler, _ = _idle_reconciler(supervisor, owner, now)

    now[0] = 1290.0
    recency.stamp_served_profile("invoice", timestamp=1290.0, home=tmp_path / "central")
    now[0] = 1400.0  # 400 s since load, 110 s since remote traffic; TTL 300 s

    assert reconciler.run_cycle()[0].phase == "hot"
    assert supervisor.get("invoice").last_served == 1290.0


def test_an_unreadable_catalog_postpones_idle_unload(tmp_path: Path) -> None:
    now = [1000.0]
    supervisor = _supervisor(tmp_path, ScriptedAdapter(), clock=lambda: now[0])
    supervisor.deploy(_spec())
    reconciler, _ = _idle_reconciler(supervisor, tmp_path, now)

    def broken(names: list[str]) -> dict[str, float]:
        raise RuntimeError("database unreachable")

    reconciler._catalog_recency = broken
    now[0] = 1400.0
    assert reconciler.run_cycle()[0].phase == "hot"

    reconciler._catalog_recency = lambda names: {}
    assert reconciler.run_cycle()[0].phase == "evicted"
