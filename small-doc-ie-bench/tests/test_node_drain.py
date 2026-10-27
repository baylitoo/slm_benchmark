"""Taking a node out of the rotation (#536).

Drain is the acceptance criterion of #536 that nothing could express before:
every surface reached a node through its own in-process control plane, so
"stop sending this machine work" had no representation at all.

Two properties carry most of the risk and most of these tests:

* A drain is DECLARED, so routing excludes the node even when it holds the last
  replica -- the opposite of the staleness rule in ``test_node_liveness``,
  which is inferred and therefore only ever demotes.
* The node stops its runtimes only after a grace period, and only through the
  same generation gate as every other convergence action. A flag read outside
  that gate would unload a node's whole working set when it was pointed at the
  wrong database, which is the failure ``convergence.decide_convergence``
  exists to prevent.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog, ServingNode
from docie_bench.serving.drain import (
    DRAIN_GRACE_CEILING_S,
    DRAIN_GRACE_FLOOR_S,
    DrainPhase,
    drain_grace_seconds,
    drain_phase,
    parse_drained_at,
)
from docie_bench.serving.model_store import StoreEntry
from docie_bench.settings import get_settings

MODEL = "lfm2.5-350m"
OTHER = "nuextract3"
NODE = "ovh-cpu-02"
GIB = 1024**3


# --- the pure rules -------------------------------------------------------


def test_grace_is_the_longest_request_timeout_on_the_node() -> None:
    """A request admitted an instant before the drain may run until its own
    timeout, so that is how long the node waits before killing the runtime."""
    assert drain_grace_seconds([45.0, 120.0, None]) == 120.0


def test_grace_falls_back_to_the_floor_rather_than_to_zero() -> None:
    """No declared timeout must not mean "stop immediately" -- that would drop
    whatever was in flight on a node nobody had configured a timeout for."""
    assert drain_grace_seconds([]) == DRAIN_GRACE_FLOOR_S
    assert drain_grace_seconds([None, None]) == DRAIN_GRACE_FLOOR_S
    # Shorter than the floor still gets the floor.
    assert drain_grace_seconds([1.0]) == DRAIN_GRACE_FLOOR_S


def test_grace_is_capped_so_a_batch_timeout_cannot_wedge_a_drain() -> None:
    """A deployment may declare an hour. Honouring that literally leaves a
    drain indistinguishable from a broken reconciler."""
    assert drain_grace_seconds([36_000.0]) == DRAIN_GRACE_CEILING_S


def test_grace_ignores_values_it_cannot_read() -> None:
    assert drain_grace_seconds(["nonsense", None, 90.0]) == 90.0  # type: ignore[list-item]


def test_phase_is_admitting_only_when_no_drain_was_declared() -> None:
    now = dt.datetime.now(dt.UTC)
    assert drain_phase(drained_at=None, now=now, grace_s=30) is DrainPhase.ADMITTING


def test_phase_waits_out_the_grace_before_allowing_a_stop() -> None:
    now = dt.datetime.now(dt.UTC)
    just_now = now - dt.timedelta(seconds=5)
    long_ago = now - dt.timedelta(seconds=90)

    assert drain_phase(drained_at=just_now, now=now, grace_s=30) is DrainPhase.DRAINING
    assert drain_phase(drained_at=long_ago, now=now, grace_s=30) is DrainPhase.DRAINED


def test_a_drain_stamped_in_the_future_is_not_already_elapsed() -> None:
    """Clock skew between the API that wrote the stamp and the node reading it.
    Read the wrong way round, the node unloads at once and drops exactly the
    in-flight requests the grace period exists for."""
    now = dt.datetime.now(dt.UTC)
    skewed = now + dt.timedelta(seconds=120)

    assert drain_phase(drained_at=skewed, now=now, grace_s=30) is DrainPhase.DRAINING


def test_an_unreadable_drain_stamp_reads_as_not_drained() -> None:
    """Inventing a drain from a malformed timestamp unloads a healthy node;
    ignoring one leaves it serving. #527 says err towards serving."""
    assert parse_drained_at("not-a-date") is None
    assert parse_drained_at("") is None
    assert parse_drained_at(None) is None


def test_a_naive_stamp_is_read_as_utc() -> None:
    """SQLite stores no offset, so the value comes back naive and subtracting
    it from an aware `now` would raise -- inside the try that swallows
    convergence errors, which would disable drain silently."""
    naive = dt.datetime(2026, 10, 26, 12, 0, 0)  # noqa: DTZ001 - the case under test

    parsed = parse_drained_at(naive)

    assert parsed is not None
    assert parsed.tzinfo is not None


# --- the catalog column ---------------------------------------------------


@pytest.fixture
def catalog_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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


def _snapshot(node_id: str) -> dict[str, object]:
    rows = {
        str(row["node_id"]): row for row in ModelCatalog().list_node_snapshots()
    }
    return rows[node_id]


def test_a_drain_survives_the_nodes_own_heartbeat(catalog_db: None) -> None:
    """THE correctness property of putting this column on serving_node.

    That row is documented as written only by its own node's reconciler, which
    republishes it every cycle. A full-row upsert would therefore erase a
    centrally-written drain within one reconcile interval, and the operator
    would watch the node quietly rejoin the rotation.
    """
    _publish(NODE)
    ModelCatalog().set_node_drain(NODE, drained=True)

    _publish(NODE)  # the node's next heartbeat

    assert _snapshot(NODE)["drained_at"] is not None


def test_a_drain_bumps_the_desired_generation(catalog_db: None) -> None:
    """How the flag reaches the node at all. Desired state is the only channel
    a node trusts, and it trusts it because of the generation -- a drain that
    did not bump it would either be ignored or have to be read outside the gate
    that makes convergence safe."""
    _publish(NODE)
    before = _snapshot(NODE)["desired_generation"] or 0

    result = ModelCatalog().set_node_drain(NODE, drained=True)

    assert result["desired_generation"] > before
    assert _snapshot(NODE)["desired_generation"] == result["desired_generation"]


def test_resuming_clears_the_drain_and_bumps_again(catalog_db: None) -> None:
    _publish(NODE)
    ModelCatalog().set_node_drain(NODE, drained=True)
    drained_generation = _snapshot(NODE)["desired_generation"]

    ModelCatalog().set_node_drain(NODE, drained=False)

    assert _snapshot(NODE)["drained_at"] is None
    assert _snapshot(NODE)["desired_generation"] > drained_generation


def test_a_node_that_never_reported_can_still_be_drained(catalog_db: None) -> None:
    """An operator takes a machine out precisely when it is misbehaving.
    Requiring a healthy publish first would block the case drain is for."""
    ModelCatalog().set_node_drain("never-started", drained=True)

    assert _snapshot("never-started")["drained_at"] is not None


def test_the_drain_reaches_the_node_in_its_desired_snapshot(catalog_db: None) -> None:
    """Carried in the same read as the generation, deliberately: a node must
    not be able to see a drain without the generation that proves the catalog
    is describing it."""
    _publish(NODE)
    ModelCatalog().set_node_drain(NODE, drained=True)

    snapshot = ModelCatalog().desired_snapshot(NODE)

    assert snapshot["drained_at"] is not None
    assert snapshot["generation"] is not None


# --- routing --------------------------------------------------------------


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


def _seed_store(name: str = MODEL) -> None:
    ModelCatalog().upsert(
        StoreEntry(
            name=name,
            family="openai_chat",
            model_path=Path(f"/models/{name}/model.gguf"),
        )
    )


def _place(record: str, *, node_id: str, port: int, model: str = MODEL) -> None:
    ModelCatalog().record_placement(
        record,
        model_name=model,
        engine="llama-server",
        endpoint=f"http://{node_id}:{port}/v1",
        state="ready",
        node_id=node_id,
    )


def test_routing_skips_a_drained_node_when_another_admits(catalog_db: None) -> None:
    from docie_bench.serving.placement_resolver import resolve_store_profile

    _seed_store()
    _publish("node-a")
    _publish("node-b")
    _place(f"{MODEL}-1", node_id="node-a", port=8088)
    _place(f"{MODEL}-2", node_id="node-b", port=8089)
    ModelCatalog().set_node_drain("node-b", drained=True)

    # Several resolutions: a round-robin that merely deprioritised the drained
    # node would still land on it eventually.
    endpoints = {resolve_store_profile(MODEL).base_url for _ in range(6)}

    assert endpoints == {"http://node-a:8088/v1"}


def test_draining_the_last_replica_refuses_with_a_reason(catalog_db: None) -> None:
    """The deliberate difference from staleness. An operator asked for this
    machine to be out; "unless it is the only one" is not something they asked
    for, and a drain that kept serving would never let the node report empty.
    """
    from docie_bench.serving.placement_resolver import (
        PlacementNotReadyError,
        resolve_store_profile,
    )

    _seed_store()
    _publish("node-a")
    _place(f"{MODEL}-1", node_id="node-a", port=8088)
    ModelCatalog().set_node_drain("node-a", drained=True)

    with pytest.raises(PlacementNotReadyError) as excinfo:
        resolve_store_profile(MODEL)

    message = str(excinfo.value)
    assert "drained" in message
    assert "resume the node" in message


def test_a_placement_with_no_owner_is_never_excluded(catalog_db: None) -> None:
    """Written before ownership was tracked: its owner is unknown, not
    drained."""
    from docie_bench.serving.placement_resolver import resolve_store_profile

    _seed_store()
    _publish("node-a")
    _place(f"{MODEL}-1", node_id=None, port=8088)  # type: ignore[arg-type]
    ModelCatalog().set_node_drain("node-a", drained=True)

    assert resolve_store_profile(MODEL).base_url.endswith("/v1")


# --- the node side --------------------------------------------------------


class _Launch:
    def __init__(self, timeout_seconds: float | None) -> None:
        self.timeout_seconds = timeout_seconds


class _Spec:
    def __init__(self, timeout_seconds: float | None) -> None:
        self.launch = _Launch(timeout_seconds)


class _Record:
    def __init__(self, timeout_seconds: float | None = None) -> None:
        self.spec = _Spec(timeout_seconds)


class _FakeSupervisor:
    """Just the surface convergence and drain touch."""

    def __init__(self, names: list[str]) -> None:
        self._records = {name: _Record() for name in names}
        self.stopped: list[str] = []
        self.removed: list[str] = []

    class _Lock:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    lock = _Lock()

    def records(self) -> dict[str, _Record]:
        return self._records

    def stop(self, name: str) -> None:
        self.stopped.append(name)

    def remove(self, name: str) -> None:
        self.removed.append(name)


@pytest.fixture
def node_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    monkeypatch.setenv("DOCIE_SERVING_NODE_ID", NODE)
    monkeypatch.delenv("DOCIE_SERVING_DESIRED_APPLY", raising=False)
    from docie_bench.serving.node_identity import resolve_node_id

    resolve_node_id.cache_clear()
    get_settings.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()
        resolve_node_id.cache_clear()
        get_settings.cache_clear()


def _converge(supervisor: _FakeSupervisor) -> object:
    from docie_bench.serving.reconciler import ServingReconciler

    reconciler = ServingReconciler(
        supervisor,  # type: ignore[arg-type]
        snapshot_publisher=lambda _snapshot: None,
    )
    return reconciler._converge_desired_state()


def _backdate_drain(node_id: str, *, seconds: float) -> None:
    """Declare a drain and move its stamp into the past, so the grace has
    elapsed without the test having to wait it out."""
    ModelCatalog().set_node_drain(node_id, drained=True)
    with db.session_scope() as session:
        assert session is not None
        row = session.get(ServingNode, node_id)
        assert row is not None
        row.drained_at = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=seconds)


def test_inside_the_grace_nothing_is_stopped(node_home: None) -> None:
    """Routing already excludes the node centrally, so it is admitting nothing;
    the runtimes stay up for whatever is still running on them."""
    monkeypatched = _FakeSupervisor([f"{MODEL}-1"])
    ModelCatalog().set_desired_deployment(
        NODE, f"{MODEL}-1", desired_state="running", spec={}
    )
    ModelCatalog().set_node_drain(NODE, drained=True)

    _converge(monkeypatched)

    assert monkeypatched.stopped == []


def test_inside_the_grace_the_generation_is_not_recorded(node_home: None) -> None:
    """The subtle half. Recording it would make the next cycle decide NOTHING,
    and the node would sit in the grace period forever without ever stopping
    anything -- a drain that never completes."""
    from docie_bench.serving.applied_state import read_applied_state

    ModelCatalog().set_desired_deployment(
        NODE, f"{MODEL}-1", desired_state="running", spec={}
    )
    ModelCatalog().set_node_drain(NODE, drained=True)

    _converge(_FakeSupervisor([f"{MODEL}-1"]))

    assert read_applied_state().generation == 0


def test_a_drained_node_stops_everything_once_the_grace_elapses(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    ModelCatalog().set_desired_deployment(
        NODE, f"{MODEL}-1", desired_state="running", spec={}
    )
    _backdate_drain(NODE, seconds=DRAIN_GRACE_CEILING_S * 2)
    supervisor = _FakeSupervisor([f"{MODEL}-1"])

    _converge(supervisor)

    assert supervisor.stopped == [f"{MODEL}-1"]


def test_a_drain_stops_deployments_the_catalog_never_knew_about(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drain is about the MACHINE. `docie up` creates deployments the catalog
    was never told about, and a node still running one has not reported empty
    -- so the acceptance criterion is not met by stopping only desired rows.
    """
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    ModelCatalog().set_desired_deployment(
        NODE, f"{MODEL}-1", desired_state="running", spec={}
    )
    _backdate_drain(NODE, seconds=DRAIN_GRACE_CEILING_S * 2)
    # The second name exists only on the node.
    supervisor = _FakeSupervisor([f"{MODEL}-1", f"{OTHER}-local"])

    _converge(supervisor)

    assert supervisor.stopped == [f"{MODEL}-1", f"{OTHER}-local"]


def test_a_drain_does_not_make_a_local_deployment_a_removal_candidate(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AppliedState.names`` is the removal whitelist. Writing a name this
    drain merely stopped into it would let a later desired-row deletion remove
    a deployment the catalog was never told about -- the exact teardown that
    whitelist exists to prevent."""
    from docie_bench.serving.applied_state import read_applied_state

    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    ModelCatalog().set_desired_deployment(
        NODE, f"{MODEL}-1", desired_state="running", spec={}
    )
    _backdate_drain(NODE, seconds=DRAIN_GRACE_CEILING_S * 2)

    _converge(_FakeSupervisor([f"{MODEL}-1", f"{OTHER}-local"]))

    assert read_applied_state().names == frozenset({f"{MODEL}-1"})


def test_a_drain_stops_nothing_without_the_apply_opt_in(node_home: None) -> None:
    """Same opt-in as every other convergence action: this path can stop a
    model nobody asked it to stop."""
    ModelCatalog().set_desired_deployment(
        NODE, f"{MODEL}-1", desired_state="running", spec={}
    )
    _backdate_drain(NODE, seconds=DRAIN_GRACE_CEILING_S * 2)
    supervisor = _FakeSupervisor([f"{MODEL}-1"])

    _converge(supervisor)

    assert supervisor.stopped == []


# --- the endpoints --------------------------------------------------------


def _drain(node_id: str, *, acknowledge: bool = False) -> dict[str, object]:
    from docie_bench.inngest.serving_api import NodeDrainRequest, drain_node

    return asyncio.run(
        drain_node(
            node_id,
            NodeDrainRequest(acknowledge_outage=acknowledge),
            tenant=None,  # type: ignore[arg-type]
        )
    )


def _nodes() -> dict[str, dict[str, object]]:
    from docie_bench.inngest.serving_api import serving_nodes

    result = asyncio.run(serving_nodes())
    return {str(row["node_id"]): row for row in result["nodes"]}


def test_draining_the_last_node_serving_a_model_is_refused(catalog_db: None) -> None:
    """Told before it happens, not discovered from failing requests -- the
    same pre-flight discipline as the targeted-deploy refusals."""
    from fastapi import HTTPException

    _seed_store()
    _publish("node-a")
    _place(f"{MODEL}-1", node_id="node-a", port=8088)

    with pytest.raises(HTTPException) as excinfo:
        _drain("node-a")

    assert excinfo.value.status_code == 409
    assert MODEL in str(excinfo.value.detail)


def test_the_refusal_is_overridable_because_only_the_operator_knows(
    catalog_db: None,
) -> None:
    _seed_store()
    _publish("node-a")
    _place(f"{MODEL}-1", node_id="node-a", port=8088)

    result = _drain("node-a", acknowledge=True)

    assert result["models_offline"] == [MODEL]
    assert result["drained_at"] is not None


def test_draining_a_node_with_a_replica_elsewhere_needs_no_override(
    catalog_db: None,
) -> None:
    _seed_store()
    _publish("node-a")
    _publish("node-b")
    _place(f"{MODEL}-1", node_id="node-a", port=8088)
    _place(f"{MODEL}-2", node_id="node-b", port=8089)

    result = _drain("node-a")

    assert result["models_offline"] == []


def test_an_already_drained_node_is_not_somewhere_else_to_route(
    catalog_db: None,
) -> None:
    """Draining the second of two nodes must still warn: routing excludes the
    first one too, so it is not a fallback."""
    from fastapi import HTTPException

    _seed_store()
    _publish("node-a")
    _publish("node-b")
    _place(f"{MODEL}-1", node_id="node-a", port=8088)
    _place(f"{MODEL}-2", node_id="node-b", port=8089)
    _drain("node-a", acknowledge=True)

    with pytest.raises(HTTPException) as excinfo:
        _drain("node-b")

    assert excinfo.value.status_code == 409


def test_draining_an_unknown_node_is_a_404(catalog_db: None) -> None:
    from fastapi import HTTPException

    _publish("node-a")

    with pytest.raises(HTTPException) as excinfo:
        _drain("not-a-node")

    assert excinfo.value.status_code == 404


def test_the_fleet_view_separates_draining_from_drained(catalog_db: None) -> None:
    """An operator waiting on a machine needs to know which one they are in:
    draining means requests are still finishing on it, drained means it has
    released everything."""
    _seed_store()
    _publish("node-a")
    _publish("node-b")
    _place(f"{MODEL}-1", node_id="node-a", port=8088)
    _place(f"{MODEL}-2", node_id="node-b", port=8089)

    _drain("node-a", acknowledge=True)
    rows = _nodes()
    assert rows["node-a"]["drain_state"] == "draining"
    assert rows["node-b"]["drain_state"] == "admitting"

    # Its placement is gone: the node has reported empty.
    ModelCatalog().clear_placement(f"{MODEL}-1")

    assert _nodes()["node-a"]["drain_state"] == "drained"


def test_resuming_puts_the_node_back_in_the_rotation(catalog_db: None) -> None:
    from docie_bench.inngest.serving_api import resume_node
    from docie_bench.serving.placement_resolver import resolve_store_profile

    _seed_store()
    _publish("node-a")
    _place(f"{MODEL}-1", node_id="node-a", port=8088)
    _drain("node-a", acknowledge=True)

    asyncio.run(resume_node("node-a", tenant=None))  # type: ignore[arg-type]

    assert resolve_store_profile(MODEL).base_url == "http://node-a:8088/v1"
