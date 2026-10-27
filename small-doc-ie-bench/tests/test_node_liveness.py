"""A node that stopped reporting leaves the rotation (#531, phase 1 of #527).

Nothing expired a placement row before this: servability was `state == "ready"
and endpoint`, with no time component, so a node that was switched off kept its
rows looking live and kept receiving its share of round-robin traffic forever.

The rule is deliberately a DEMOTION, not an exclusion, and the second half of
this file is about why: a stale snapshot proves the reconciler stopped
publishing, not that the runtime died, and #527 requires an already-running
model to keep serving.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog, ServingNode
from docie_bench.serving.model_store import StoreEntry
from docie_bench.serving.node_liveness import snapshot_stale_after_s, stale_node_ids
from docie_bench.serving.placement_resolver import resolve_store_profile

MODEL = "lfm2.5-350m"
GIB = 1024**3


@pytest.fixture
def sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


@pytest.fixture(autouse=True)
def _empty_caches() -> Iterator[None]:
    from docie_bench.serving import placement_resolver

    placement_resolver._RESOLUTION_CACHE.clear()
    placement_resolver._ROUND_ROBIN_COUNTERS.clear()
    try:
        yield
    finally:
        placement_resolver._RESOLUTION_CACHE.clear()
        placement_resolver._ROUND_ROBIN_COUNTERS.clear()


def _seed_store() -> None:
    ModelCatalog().upsert(
        StoreEntry(
            name=MODEL,
            family="openai_chat",
            model_path=Path(f"/models/{MODEL}/model.gguf"),
        )
    )


def _place(record: str, *, port: int, node_id: str) -> None:
    ModelCatalog().publish_observed(
        record,
        node_id=node_id,
        engine="llama-server",
        state="ready",
        endpoint=f"http://{node_id}:{port}/v1",
        phase="hot",
        pid=1,
        pid_create_time=1.0,
        rss_bytes=GIB,
        health_ok=True,
        last_error=None,
    )
    ModelCatalog().record_placement(
        record,
        model_name=MODEL,
        engine="llama-server",
        endpoint=f"http://{node_id}:{port}/v1",
        state="ready",
        node_id=node_id,
    )


def _node(node_id: str, *, age_s: float) -> None:
    """Publish a node snapshot, then age its stamp by ``age_s`` seconds."""
    ModelCatalog().publish_node_snapshot(
        node_id=node_id,
        total_bytes=8 * GIB,
        free_bytes=4 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
    )
    with db.session_scope() as session:
        assert session is not None
        row = session.get(ServingNode, node_id)
        assert row is not None
        row.updated_at = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=age_s)


# --- the rule ------------------------------------------------------------


def test_stale_node_ids_names_only_nodes_that_published_then_stopped() -> None:
    fresh = {"node_id": "live", "updated_at": dt.datetime.now(dt.UTC).isoformat()}
    gone = {
        "node_id": "gone",
        "updated_at": (
            dt.datetime.now(dt.UTC) - dt.timedelta(seconds=snapshot_stale_after_s() * 10)
        ).isoformat(),
    }
    unparseable = {"node_id": "weird", "updated_at": "not-a-date"}

    assert stale_node_ids([fresh, gone, unparseable]) == frozenset({"gone"})


def test_a_node_that_never_published_is_not_stale() -> None:
    """It is new, not gone: a deploy records its placement at once while the
    first reconcile publish is up to one cycle away."""
    assert stale_node_ids([]) == frozenset()


# --- routing -------------------------------------------------------------


def test_traffic_stops_going_to_a_node_that_stopped_reporting(
    sqlite_catalog: None,
) -> None:
    """The bug: the switched-off node's row stays `ready` with a live-looking
    endpoint, so round-robin keeps handing it a share of requests."""
    _seed_store()
    _place(MODEL, port=8088, node_id="node-a")
    _place(f"{MODEL}-2", port=8088, node_id="node-b")
    _node("node-a", age_s=0)
    _node("node-b", age_s=snapshot_stale_after_s() * 10)  # switched off

    urls = {resolve_store_profile(MODEL).base_url for _ in range(8)}

    assert urls == {"http://node-a:8088/v1"}


def test_a_rejoining_node_returns_to_the_rotation(sqlite_catalog: None) -> None:
    """Acceptance (#533 check 6): no operator action, just a fresh publish."""
    _seed_store()
    _place(MODEL, port=8088, node_id="node-a")
    _place(f"{MODEL}-2", port=8088, node_id="node-b")
    _node("node-a", age_s=0)
    _node("node-b", age_s=snapshot_stale_after_s() * 10)

    assert {resolve_store_profile(MODEL).base_url for _ in range(8)} == {
        "http://node-a:8088/v1"
    }

    _node("node-b", age_s=0)  # the node comes back and republishes

    assert {resolve_store_profile(MODEL).base_url for _ in range(8)} == {
        "http://node-a:8088/v1",
        "http://node-b:8088/v1",
    }


# --- why it is a demotion and not an exclusion ---------------------------


def test_the_only_replica_is_still_served_when_its_node_stops_reporting(
    sqlite_catalog: None,
) -> None:
    """The #527 non-negotiable, and the reason this is not an exclusion.

    A stale snapshot proves the RECONCILER stopped publishing, not that the
    runtime died -- a crashed reconciler leaves llama-server serving happily.
    Refusing here would turn a reconciler crash into an outage for a model that
    is still answering.
    """
    _seed_store()
    _place(MODEL, port=8088, node_id="node-a")
    _node("node-a", age_s=snapshot_stale_after_s() * 10)

    assert resolve_store_profile(MODEL).base_url == "http://node-a:8088/v1"


def test_every_replica_on_stale_nodes_still_routes(sqlite_catalog: None) -> None:
    """Same rule with several replicas: demote, and if there is nothing left to
    prefer, keep them all rather than refuse."""
    _seed_store()
    _place(MODEL, port=8088, node_id="node-a")
    _place(f"{MODEL}-2", port=8088, node_id="node-b")
    _node("node-a", age_s=snapshot_stale_after_s() * 10)
    _node("node-b", age_s=snapshot_stale_after_s() * 10)

    urls = {resolve_store_profile(MODEL).base_url for _ in range(8)}

    assert urls == {"http://node-a:8088/v1", "http://node-b:8088/v1"}


def test_a_placement_without_an_owner_is_never_demoted(sqlite_catalog: None) -> None:
    """Rows written before ownership was tracked have an unknown owner, not a
    dead one -- the migration must not unroute them."""
    _seed_store()
    ModelCatalog().record_placement(
        MODEL,
        model_name=MODEL,
        engine="llama-server",
        endpoint="http://legacy:8088/v1",
        state="ready",
    )
    _place(f"{MODEL}-2", port=8088, node_id="node-b")
    _node("node-b", age_s=snapshot_stale_after_s() * 10)

    urls = {resolve_store_profile(MODEL).base_url for _ in range(8)}

    assert "http://legacy:8088/v1" in urls

def test_the_node_table_is_read_once_per_resolution(sqlite_catalog: None) -> None:
    """Both node rules need this table -- liveness to demote a node that
    stopped reporting, drain to exclude one an operator took out -- and each
    used to fetch it for itself. That is two SELECTs on the hot path of every
    request for an answer that cannot change between them.

    Counted rather than reasoned about, because the second read was invisible:
    it changed no behaviour, so nothing else in this suite could notice it.
    """
    _seed_store()
    _node("node-a", age_s=0)
    _place(f"{MODEL}-1", port=8088, node_id="node-a")

    catalog = ModelCatalog()
    reads: list[int] = []
    unwrapped = catalog.list_node_snapshots

    def counting() -> object:
        reads.append(1)
        return unwrapped()

    catalog.list_node_snapshots = counting  # type: ignore[method-assign]

    resolve_store_profile(MODEL, catalog=catalog)

    assert len(reads) == 1


def test_an_unreadable_node_table_still_routes(sqlite_catalog: None) -> None:
    """The property that lets the read move out of both rules: a failure leaves
    an empty list, and each rule degrades to a no-op on it exactly as it did
    when it swallowed its own exception. Routing is never withheld because node
    state could not be established."""
    _seed_store()
    _node("node-a", age_s=0)
    _place(f"{MODEL}-1", port=8088, node_id="node-a")

    catalog = ModelCatalog()

    def boom() -> object:
        raise RuntimeError("node table is on fire")

    catalog.list_node_snapshots = boom  # type: ignore[method-assign]

    profile = resolve_store_profile(MODEL, catalog=catalog)

    assert profile.base_url == "http://node-a:8088/v1"
