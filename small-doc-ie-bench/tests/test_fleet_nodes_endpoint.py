"""The fleet listing: every serving node, including the silent ones (#536).

`/resources` answers "how much room is there", about ONE node, and is what the
sizing engine consumes. Nothing answered "what is the fleet" -- the per-node
rows lived in `serving_node` and no endpoint exposed them, so the Studio could
only ever show the node that happened to serve the request.

The property these tests pin is that absence has its own state. A node can have
a row before it ever publishes a measurement (desired state staged for a machine
that has not started), and such a node is the one an operator most needs to see.
Dropping it would make "expected but silent" look exactly like "does not exist".
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.settings import get_settings

GIB = 1024**3


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    home = tmp_path / "serving"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    get_settings.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()
        get_settings.cache_clear()


def _publish(node_id: str, *, store_models: list[str] | None = None) -> None:
    ModelCatalog().publish_node_snapshot(
        node_id=node_id,
        total_bytes=8 * GIB,
        free_bytes=4 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
        store_models=store_models,
    )


def _nodes() -> dict[str, dict[str, Any]]:
    from docie_bench.inngest.serving_api import serving_nodes

    result = asyncio.run(serving_nodes())
    return {str(row["node_id"]): row for row in result["nodes"]}


def test_a_node_that_has_never_published_is_listed_as_silent(fleet: None) -> None:
    """Staging desired state for a machine creates its row. Until that machine
    starts it reports nothing, and an operator needs to see it sitting there --
    not an empty fleet."""
    ModelCatalog().set_desired_deployment(
        "node-b",
        "lfm2.5-350m-1",
        desired_state="running",
        spec={"model": "lfm2.5-350m"},
    )

    rows = _nodes()

    assert "node-b" in rows
    assert rows["node-b"]["never_reported"] is True
    assert rows["node-b"]["reporting"] is False
    # Zeroes that no reader should price against -- `never_reported` is what
    # says so, since the numbers themselves look like a full machine.
    assert rows["node-b"]["age_s"] is None


def test_a_reporting_node_counts_the_placements_it_owns(fleet: None) -> None:
    """Ownership is counted server-side so one definition serves every reader
    (the Studio was deriving its own from the rows it happened to receive)."""
    _publish("node-a", store_models=["lfm2.5-350m"])
    catalog = ModelCatalog()
    catalog.record_placement(
        "lfm2.5-350m-1",
        model_name="lfm2.5-350m",
        engine="llamacpp",
        endpoint="http://node-a:8088",
        state="ready",
        node_id="node-a",
    )
    catalog.record_placement(
        "lfm2.5-350m-2",
        model_name="lfm2.5-350m",
        engine="llamacpp",
        endpoint="http://node-a:8089",
        state="ready",
        node_id="node-a",
    )
    # A placement with no owner must not be attributed to anyone: a NULL owner
    # is an unknown one, not this node's.
    catalog.record_placement(
        "nuextract3-1",
        model_name="nuextract3",
        engine="llamacpp",
        endpoint="http://unknown:8090",
        state="ready",
        node_id=None,
    )

    rows = _nodes()

    assert rows["node-a"]["reporting"] is True
    assert rows["node-a"]["never_reported"] is False
    assert rows["node-a"]["deployments"] == 2


def test_a_silent_node_and_a_live_one_are_both_listed(fleet: None) -> None:
    """The mixed fleet is the case the view exists for: one machine serving,
    one expected and not answering."""
    _publish("node-a", store_models=["lfm2.5-350m"])
    ModelCatalog().set_desired_deployment(
        "node-b",
        "lfm2.5-350m-1",
        desired_state="running",
        spec={"model": "lfm2.5-350m"},
    )

    rows = _nodes()

    assert sorted(rows) == ["node-a", "node-b"]
    assert rows["node-a"]["reporting"] is True
    assert rows["node-b"]["reporting"] is False


def test_the_fleet_view_degrades_instead_of_failing(fleet: None) -> None:
    """Observed state lives in Postgres, which is optional. A single-node
    deployment without DATABASE_URL must get an empty fleet and a reason, not a
    500 on a page it has no use for."""
    db.dispose_engine()

    from docie_bench.inngest.serving_api import serving_nodes

    result = asyncio.run(serving_nodes())

    assert result["nodes"] == []
    assert result["observed_available"] is False
    assert "not configured" in str(result["detail"])


def test_the_staleness_window_is_reported_so_the_ui_need_not_guess(
    fleet: None,
) -> None:
    """The UI shows "last seen 12s ago" against a threshold it must not
    hardcode -- the router demotes on the same rule, and two copies of it would
    drift."""
    _publish("node-a")

    from docie_bench.inngest.serving_api import serving_nodes
    from docie_bench.serving.node_liveness import snapshot_stale_after_s

    result = asyncio.run(serving_nodes())

    assert result["stale_after_s"] == snapshot_stale_after_s()
