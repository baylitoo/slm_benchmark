"""The worker's deployment reads cover the fleet, not one node's disk (#562).

Sibling of ``test_fleet_deployment_listing.py``, which covers the Studio's
read path. These are the two readers the issue's own table still lists as
node-local: ``_read_deployment`` (recency stamping, autoload detection,
readiness polling) and ``_read_all_deployments`` (batch fan-out capacity).

Both failed silently rather than loudly. A deployment running perfectly well
on another machine read as "no such deployment", and fan-out priced a
deployment that #589 had spread across the fleet at one node's share of it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import docie_bench.storage.db as db
from docie_bench.inngest import functions
from docie_bench.serving.catalog import ModelCatalog, ServingNode
from docie_bench.serving.node_liveness import snapshot_stale_after_s
from docie_bench.settings import get_settings

GIB = 1024**3
LOCAL = "lfm2.5-350m"
REMOTE = "nuextract3"


@pytest.fixture
def serving_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "serving"
    home.mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    get_settings.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield home
    finally:
        db.dispose_engine()
        get_settings.cache_clear()


def _write_local(home: Path, records: dict[str, dict[str, Any]]) -> None:
    """This node's own ``deployments.json`` — the only file the worker reads."""
    (home / "deployments.json").write_text(
        json.dumps({"deployments": records}), encoding="utf-8"
    )


def _local_record(*, n_parallel: int = 1, state: str = "ready", port: int = 8088) -> dict[str, Any]:
    return {
        "state": state,
        "endpoint": f"http://local:{port}/v1" if state == "ready" else "",
        "spec": {"launch": {"n_parallel": n_parallel}},
    }


def _observe(
    name: str,
    *,
    node_id: str,
    state: str = "ready",
    slot_count: int | None = None,
) -> None:
    """A placement row as the owning node's reconciler would publish it."""
    ModelCatalog().publish_observed(
        name,
        node_id=node_id,
        engine="llama-server",
        state=state,
        endpoint=f"http://{node_id}:8088/v1" if state == "ready" else "",
        phase="hot" if state == "ready" else "cold",
        pid=None,
        pid_create_time=None,
        rss_bytes=0,
        health_ok=state == "ready",
        last_error=None,
        slot_count=slot_count,
    )


def _node_snapshot(node_id: str, *, age_s: float = 0.0) -> None:
    ModelCatalog().publish_node_snapshot(
        node_id=node_id,
        total_bytes=8 * GIB,
        free_bytes=4 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
    )
    if age_s:
        with db.session_scope() as session:
            assert session is not None
            row = session.get(ServingNode, node_id)
            assert row is not None
            row.updated_at = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=age_s)


# --- a deployment on another node exists -----------------------------------


def test_a_deployment_on_another_node_is_not_read_as_nonexistent(
    serving_home: Path,
) -> None:
    _write_local(serving_home, {LOCAL: _local_record()})
    _observe(REMOTE, node_id="node-b")

    record = functions._read_deployment(REMOTE)

    assert record is not None, "a live deployment on node-b read as no such deployment"
    assert functions._deployment_is_live(record)
    assert record["node_id"] == "node-b"
    assert record["record_source"] == "catalog"


def test_polling_sees_a_replica_that_was_ready_on_another_node_all_along(
    serving_home: Path,
) -> None:
    """``_poll_deployment_live`` waited out its whole budget and then reported
    not-ready, with no error to show, for a deployment that was ready the
    entire time."""
    _write_local(serving_home, {})
    _observe(REMOTE, node_id="node-b")

    result = asyncio.run(functions._poll_deployment_live(REMOTE, timeout_s=0.3))

    assert result["ready"] is True
    assert result["endpoint"] == "http://node-b:8088/v1"


def test_this_nodes_own_record_still_wins(serving_home: Path) -> None:
    """The local file is fresher (saved every reconcile cycle, no database in
    the path) and carries launch detail a placement row does not."""
    _write_local(serving_home, {LOCAL: _local_record(n_parallel=4)})
    _observe(LOCAL, node_id="node-a")

    record = functions._read_deployment(LOCAL)

    assert record is not None
    assert "record_source" not in record
    assert (record["spec"]["launch"])["n_parallel"] == 4


def test_a_remote_record_carries_no_desired_state_so_autoload_declines(
    serving_home: Path,
) -> None:
    """The scope line. A placement row has no ``desired_state`` or
    ``activation``, so a cold deployment on another node still does not
    autoload -- but now because the record says nothing about desired state,
    not because the deployment looked nonexistent. Loading on another node
    needs a desired state that can express it (#590).
    """
    _write_local(serving_home, {})
    _observe(REMOTE, node_id="node-b", state="cold")

    assert functions._read_deployment(REMOTE) is not None
    assert functions._autoload_target({"deployment": REMOTE}) is None


# --- fan-out prices the fleet ----------------------------------------------


def test_fanout_prices_replicas_on_other_nodes(serving_home: Path) -> None:
    """#589 spreads replicas across nodes, so this node's file is one share of
    the deployment. Pricing only that share ran a third of what the fleet
    could at once."""
    _write_local(serving_home, {LOCAL: _local_record(n_parallel=4)})
    _observe(f"{LOCAL}-2", node_id="node-b", slot_count=4)
    _observe(f"{LOCAL}-3", node_id="node-c", slot_count=4)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 12


def test_fanout_counts_a_remote_replica_even_with_no_local_file(
    serving_home: Path,
) -> None:
    """The whole deployment can live elsewhere; the early "no local records ->
    width 1" return made that case permanently sequential."""
    _write_local(serving_home, {})
    _observe(LOCAL, node_id="node-b", slot_count=6)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 6


def test_fanout_skips_replicas_on_nodes_that_stopped_reporting(
    serving_home: Path,
) -> None:
    """Where pricing parts company with routing.

    ``placement_resolver`` only ever uses staleness to prefer a better
    replica, never to refuse the only one, because a crashed reconciler
    leaves llama-server serving. A width, though, decides how many documents
    to launch at a machine nobody can confirm -- the same rule the node table
    already states for unreported capacity. Costing throughput is
    recoverable; a batch failing mid-flight is not.
    """
    _write_local(serving_home, {LOCAL: _local_record(n_parallel=2)})
    _observe(f"{LOCAL}-2", node_id="node-b", slot_count=8)
    _node_snapshot("node-b", age_s=snapshot_stale_after_s() * 10)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 2


def test_a_node_that_never_reported_is_priced_rather_than_skipped(
    serving_home: Path,
) -> None:
    """``stale_node_ids`` is deliberately the STALE set, not the live one: a
    node with no snapshot yet is new, not gone."""
    _write_local(serving_home, {})
    _observe(LOCAL, node_id="node-new", slot_count=3)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 3


def test_a_local_record_is_not_counted_twice_from_its_own_placement_row(
    serving_home: Path,
) -> None:
    _write_local(serving_home, {LOCAL: _local_record(n_parallel=4)})
    _observe(LOCAL, node_id="node-a", slot_count=4)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 4


def test_a_local_replica_that_is_not_live_is_not_revived_by_its_row(
    serving_home: Path,
) -> None:
    """A record this node owns and knows is cold must not be counted from a
    stale placement row the reconciler has not caught up on."""
    _write_local(serving_home, {LOCAL: _local_record(n_parallel=4, state="cold")})
    _observe(LOCAL, node_id="node-a", slot_count=4)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 1


def test_a_remote_replica_with_no_observed_slots_counts_as_one(
    serving_home: Path,
) -> None:
    """NULL ``slot_count`` (a non-llamacpp engine, ``--no-slots``, an older
    build) contributes the local path's default of 1, never a guess."""
    _write_local(serving_home, {})
    _observe(LOCAL, node_id="node-b", slot_count=None)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 1


# --- one node is unchanged, by construction --------------------------------


def test_a_record_on_this_node_costs_no_catalog_query(
    serving_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance criterion "with one node, behaviour and freshness are
    unchanged", proved rather than asserted: the catalog is made to explode,
    and a read that finds its answer locally never touches it."""
    import docie_bench.serving.catalog as catalog_module

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the local record was present; no catalog read is allowed")

    _write_local(serving_home, {LOCAL: _local_record(n_parallel=4)})
    monkeypatch.setattr(catalog_module, "ModelCatalog", explode)

    record = functions._read_deployment(LOCAL)

    assert record is not None
    assert record["state"] == "ready"


def test_a_catalog_outage_leaves_the_local_answer_intact(
    serving_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-effort in both readers: no DATABASE_URL degrades to exactly the
    single-node behaviour, never to an error on the extraction path."""
    _write_local(serving_home, {LOCAL: _local_record(n_parallel=4)})
    db.dispose_engine()
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 4
    assert functions._read_deployment(REMOTE) is None

# --- replicas are grouped by alias, not by a name pattern (#530) -----------

MINTED = (f"{LOCAL}-a1b2c3", f"{LOCAL}-d4e5f6", f"{LOCAL}-7a8b9c")


def _write_minted(home: Path, names: tuple[str, ...], *, n_parallel: int = 4) -> None:
    """Replicas named the way mint_deployment_name() actually names them."""
    _write_local(
        home,
        {
            name: {
                "state": "ready",
                "endpoint": f"http://local:{8088 + i}/v1",
                "spec": {
                    "name": name,
                    "launch": {"n_parallel": n_parallel, "alias": LOCAL, "model": LOCAL},
                },
            }
            for i, name in enumerate(names)
        },
    )


def _place_with_model(name: str, *, node_id: str, model: str, slots: int) -> None:
    """A placement row that carries the launch alias as its model_name, which
    is what groups a store model's replicas across nodes."""
    ModelCatalog().record_placement(
        name,
        model_name=model,
        engine="llama-server",
        endpoint=f"http://{node_id}:8088/v1",
        state="ready",
        node_id=node_id,
    )
    ModelCatalog().publish_observed(
        name,
        node_id=node_id,
        engine="llama-server",
        state="ready",
        endpoint=f"http://{node_id}:8088/v1",
        phase="hot",
        pid=None,
        pid_create_time=None,
        rss_bytes=0,
        health_ok=True,
        last_error=None,
        slot_count=slots,
    )


def test_minted_replica_names_are_priced_by_their_alias(serving_home: Path) -> None:
    """#530 mints record names as ``<base>-<6 hex>``, which the positional
    ``<base>-<n>`` pattern cannot match -- so every scaled deployment priced
    itself at a single replica. Measured before: 12 real slots, width 1."""
    _write_minted(serving_home, MINTED)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 12


def test_minted_replicas_on_other_nodes_are_priced_too(serving_home: Path) -> None:
    """The same rule across the fleet: a placement row's ``model_name`` is the
    launch alias, so it groups with this node's replicas of that model."""
    _write_minted(serving_home, MINTED[:1])
    _place_with_model(f"{LOCAL}-b2c3d4", node_id="node-b", model=LOCAL, slots=4)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 8


def test_a_selector_naming_one_minted_record_prices_only_that_record(
    serving_home: Path,
) -> None:
    """Grouping by alias must not make an explicit record selector mean its
    siblings. A minted name is nobody's alias, so it stands alone."""
    _write_minted(serving_home, MINTED)

    assert functions._batch_fanout_width({"deployment": MINTED[0]}) == 4


def test_a_drained_node_is_not_priced(serving_home: Path) -> None:
    """Drain EXCLUDES where staleness only demotes: an operator has declared
    the node out of rotation, so fanning a batch onto it is the one thing a
    width must not do."""
    _write_minted(serving_home, MINTED[:1])
    _place_with_model(f"{LOCAL}-b2c3d4", node_id="node-b", model=LOCAL, slots=8)
    _node_snapshot("node-b")
    ModelCatalog().set_node_drain("node-b", drained=True)

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 4


def test_an_undrained_node_is_still_priced(serving_home: Path) -> None:
    """The negative half, so the drain filter is not just always-on."""
    _write_minted(serving_home, MINTED[:1])
    _place_with_model(f"{LOCAL}-b2c3d4", node_id="node-b", model=LOCAL, slots=8)
    _node_snapshot("node-b")

    assert functions._batch_fanout_width({"deployment": LOCAL}) == 12

