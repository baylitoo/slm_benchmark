"""The reconciler acting on its own desired state (#534/#535).

The only part of phase 2 that changes behaviour, so the tests are mostly about
what it does NOT do: nothing without an explicit opt-in, nothing to a
deployment it did not create from desired state, and nothing at all to an
existing single-node stack.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.applied_state import AppliedState, read_applied_state, write_applied_state
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.serving.convergence import ConvergeAction

NODE = "ovh-cpu-02"


class _FakeSupervisor:
    """Just the surface convergence touches."""

    def __init__(self, names: list[str]) -> None:
        self._records = dict.fromkeys(names, object())
        self.stopped: list[str] = []
        self.removed: list[str] = []

    class _Lock:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    lock = _Lock()

    def records(self) -> dict[str, object]:
        return self._records

    def stop(self, name: str) -> None:
        self.stopped.append(name)

    def remove(self, name: str) -> None:
        self.removed.append(name)


def _reconciler(
    supervisor: _FakeSupervisor,
    *,
    deployer: object = None,
    lease: object = None,
) -> object:
    """Build a reconciler through its REAL constructor.

    These tests used ``ServingReconciler.__new__`` to skip ``__init__`` and set
    only the attributes convergence reads. That broke three times as
    convergence grew -- ``_deployer``, then ``_lease``, then ``_last_snapshot``
    -- each time surfacing as an AttributeError in a test rather than as a
    signal about the code. Going through ``__init__`` means a new field arrives
    initialised, while a field convergence genuinely needs and nobody wires
    still fails loudly.

    ``snapshot_publisher`` is a no-op: these tests are about the convergence
    decision, not about publishing capacity.
    """
    from docie_bench.serving.reconciler import ServingReconciler

    return ServingReconciler(
        supervisor,  # type: ignore[arg-type]
        snapshot_publisher=lambda _snapshot: None,
        deployer=deployer,  # type: ignore[arg-type]
        lease=lease,  # type: ignore[arg-type]
    )


@pytest.fixture
def node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    monkeypatch.setenv("DOCIE_SERVING_NODE_ID", NODE)
    monkeypatch.delenv("DOCIE_SERVING_DESIRED_APPLY", raising=False)
    from docie_bench.serving.node_identity import resolve_node_id

    resolve_node_id.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()
        resolve_node_id.cache_clear()


def _converge(supervisor: _FakeSupervisor) -> object:
    """Call the convergence step with a stand-in supervisor."""

    reconciler = _reconciler(supervisor)
    return reconciler._converge_desired_state()


def _desire(name: str, state: str = "running") -> None:
    ModelCatalog().set_desired_deployment(NODE, name, desired_state=state, spec={})


def _publish_capacity() -> None:
    ModelCatalog().publish_node_snapshot(
        node_id=NODE, total_bytes=1, free_bytes=1, source="cgroup", sum_rss_bytes=0
    )


# --- inert by default ----------------------------------------------------


def test_an_existing_stack_with_no_desired_state_does_nothing(node: None) -> None:
    """What every node looks like the moment this ships: capacity published,
    nothing desired. It must not touch the deployments it is running."""
    _publish_capacity()
    supervisor = _FakeSupervisor(["lfm2.5-350m"])

    decision = _converge(supervisor)

    assert decision.action is ConvergeAction.NOTHING
    assert decision.should_alarm is False
    assert supervisor.stopped == []
    assert supervisor.removed == []


def test_an_unreachable_catalog_does_nothing(node: None) -> None:
    """The #527 non-negotiable, at the one call site that could violate it."""
    db.dispose_engine()
    supervisor = _FakeSupervisor(["lfm2.5-350m"])

    decision = _converge(supervisor)

    assert decision.action is ConvergeAction.HOLD
    assert supervisor.stopped == []
    assert supervisor.removed == []


def test_a_plan_is_not_carried_out_without_the_opt_in(node: None) -> None:
    """Execution opts in explicitly: this is the only path in the serving role
    that can stop a model nobody asked it to stop."""
    _publish_capacity()
    _desire("wanted")
    supervisor = _FakeSupervisor(["unwanted"])
    write_applied_state(AppliedState(generation=0, names=frozenset({"unwanted"})))

    decision = _converge(supervisor)

    assert decision.action is ConvergeAction.CONVERGE
    assert supervisor.removed == []  # planned, not applied
    # And the generation is NOT recorded, so enabling the flag still converges.
    assert read_applied_state().generation == 0


# --- with the opt-in -----------------------------------------------------


def test_a_deployment_no_longer_desired_is_removed(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("wanted")
    supervisor = _FakeSupervisor(["wanted", "retired"])
    write_applied_state(AppliedState(generation=0, names=frozenset({"wanted", "retired"})))

    _converge(supervisor)

    assert supervisor.removed == ["retired"]


def test_a_locally_created_deployment_is_left_alone(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safety property, end to end: `docie up` created this and the
    catalog never knew about it, so convergence must not remove it."""
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("wanted")
    supervisor = _FakeSupervisor(["wanted", "made-by-hand"])
    write_applied_state(AppliedState(generation=0, names=frozenset({"wanted"})))

    _converge(supervisor)

    assert supervisor.removed == []


def test_a_deployment_desired_stopped_is_stopped(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("paused", state="stopped")
    supervisor = _FakeSupervisor(["paused"])
    write_applied_state(AppliedState(generation=0, names=frozenset({"paused"})))

    _converge(supervisor)

    assert supervisor.stopped == ["paused"]
    assert supervisor.removed == []


def test_the_applied_generation_is_recorded_so_it_converges_once(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise every cycle would decide CONVERGE forever and re-apply."""
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("wanted")
    supervisor = _FakeSupervisor(["wanted"])

    first = _converge(supervisor)
    assert first.action is ConvergeAction.CONVERGE
    assert read_applied_state().generation == 1

    second = _converge(supervisor)
    assert second.action is ConvergeAction.NOTHING


def test_an_already_satisfied_plan_still_records_the_generation(
    node: None,
) -> None:
    """A no-op plan is still an applied intent. Not recording it would
    re-decide CONVERGE on every cycle forever, even with nothing to do -- and
    that happens WITHOUT the apply flag, so it must work either way."""
    _publish_capacity()
    _desire("wanted")
    supervisor = _FakeSupervisor(["wanted"])
    write_applied_state(AppliedState(generation=0, names=frozenset({"wanted"})))

    _converge(supervisor)

    assert read_applied_state().generation == 1


def test_one_failing_action_does_not_stall_the_rest(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    supervisor = _FakeSupervisor(["bad", "good"])
    write_applied_state(AppliedState(generation=0, names=frozenset({"bad", "good"})))
    _desire("keep")

    def explode(name: str) -> None:
        if name == "bad":
            raise RuntimeError("will not remove")
        supervisor.removed.append(name)

    monkeypatch.setattr(supervisor, "remove", explode)

    _converge(supervisor)

    assert supervisor.removed == ["good"]


# --- starts (#535) -------------------------------------------------------


def _converge_with_deployer(
    supervisor: _FakeSupervisor, started: list[tuple[str, dict[str, object]]]
) -> object:

    reconciler = _reconciler(supervisor, deployer=lambda name, spec: started.append((name, spec)))
    return reconciler._converge_desired_state()


def test_a_desired_deployment_is_started_with_its_recorded_spec(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spec is the intent the central side recorded; the node must pass it
    through rather than guess a port or context length of its own."""
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    ModelCatalog().set_desired_deployment(
        NODE,
        "lfm2.5-350m-ovh-02",
        desired_state="running",
        spec={"model": "lfm2.5-350m", "context_length": 8192},
    )
    started: list[tuple[str, dict[str, object]]] = []

    _converge_with_deployer(_FakeSupervisor([]), started)

    assert started == [
        ("lfm2.5-350m-ovh-02", {"model": "lfm2.5-350m", "context_length": 8192})
    ]


def test_a_start_is_not_attempted_without_the_opt_in(
    node: None,
) -> None:
    _publish_capacity()
    _desire("wanted")
    started: list[tuple[str, dict[str, object]]] = []

    _converge_with_deployer(_FakeSupervisor([]), started)

    assert started == []


def test_a_failing_start_does_not_stall_the_plan(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("first")
    _desire("second")
    supervisor = _FakeSupervisor([])
    started: list[str] = []

    def deployer(name: str, spec: dict[str, object]) -> None:
        if name == "first":
            raise RuntimeError("port exhausted")
        started.append(name)

    reconciler = _reconciler(supervisor, deployer=deployer)
    reconciler._converge_desired_state()

    assert started == ["second"]


def test_starts_happen_before_stops_and_removals(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Converging towards MORE capacity first means a generation that replaces
    one deployment with another never leaves the model unserved in between."""

    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("replacement")
    supervisor = _FakeSupervisor(["retired"])
    write_applied_state(AppliedState(generation=0, names=frozenset({"retired"})))
    order: list[str] = []

    reconciler = _reconciler(supervisor, deployer=lambda name, spec: order.append(f"start:{name}"))
    monkeypatch.setattr(
        supervisor, "remove", lambda name: order.append(f"remove:{name}")
    )
    reconciler._converge_desired_state()

    assert order == ["start:replacement", "remove:retired"]
def test_a_failed_action_does_not_record_the_generation(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reconciler that gives up after one attempt is not reconciling.

    Recording the generation after a failure would make the generations match
    on the next cycle, so the decision becomes NOTHING forever and a single
    transient failure -- a port briefly held, a GGUF on a slow disk -- wedges
    the deployment until somebody writes desired state again.
    """

    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("keep")
    supervisor = _FakeSupervisor(["doomed"])
    write_applied_state(AppliedState(generation=0, names=frozenset({"doomed"})))

    def explode(name: str) -> None:
        raise RuntimeError("will not remove")

    monkeypatch.setattr(supervisor, "remove", explode)
    reconciler = _reconciler(supervisor)
    reconciler._converge_desired_state()

    # Not recorded, so the next cycle re-plans and tries again.
    assert read_applied_state().generation == 0


def test_a_start_that_cannot_be_performed_does_not_record_the_generation(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no deployer wired a start is logged, not performed -- so the plan
    was not applied and must not be marked as done."""
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("wanted")

    _converge(_FakeSupervisor([]))

    assert read_applied_state().generation == 0


def test_a_fully_applied_plan_does_record_the_generation(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half: success must still converge exactly once."""
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("keep")
    # "keep" is already running, so the plan has no start -- with no deployer
    # wired a start would (correctly) block the generation from being recorded.
    supervisor = _FakeSupervisor(["keep", "retired"])
    write_applied_state(AppliedState(generation=0, names=frozenset({"keep", "retired"})))

    _converge(supervisor)

    assert supervisor.removed == ["retired"]
    assert read_applied_state().generation == 1


def test_the_lease_is_heartbeated_between_actions(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start blocks until the model is serving, which for a multi-GB GGUF
    exceeds the lease staleness window. Without a mid-plan heartbeat a plan
    with several starts leaves the lease unrefreshed long enough for a newly
    booting serving container to judge it abandoned and claim it.
    """

    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("first")
    _desire("second")
    beats: list[str] = []

    class _Lease:
        def refresh(self) -> None:
            beats.append("beat")

    reconciler = _reconciler(_FakeSupervisor([]), deployer=lambda name, spec: None, lease=_Lease())
    reconciler._converge_desired_state()

    # One per action, not one per plan.
    assert len(beats) == 2


def test_a_failing_heartbeat_does_not_fail_the_plan(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next cycle's own refresh is the backstop, so a lease write that
    fails must not abort a convergence that is otherwise succeeding."""

    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("wanted")
    started: list[str] = []

    class _BrokenLease:
        def refresh(self) -> None:
            raise OSError("volume is read-only")

    reconciler = _reconciler(
        _FakeSupervisor([]),
        deployer=lambda name,
        spec: started.append(name),
        lease=_BrokenLease(),
    )
    reconciler._converge_desired_state()

    assert started == ["wanted"]
    assert read_applied_state().generation == 1


def test_the_node_snapshot_is_refreshed_between_actions(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start blocks until the model is serving, which delays the NEXT cycle
    and with it the snapshot publish. Once this node's `serving_node` row goes
    stale (30s by default) the router demotes placements on it (#531) -- so
    loading a model on a busy node drained traffic from that node's OTHER,
    healthy models onto its peers, for the whole duration of the load.
    """
    from docie_bench.serving.reconciler import ServingReconciler
    from docie_bench.serving.resources import NodeSnapshot

    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("first")
    _desire("second")
    published: list[NodeSnapshot] = []

    reconciler = ServingReconciler(
        _FakeSupervisor([]),  # type: ignore[arg-type]
        snapshot_publisher=published.append,
        deployer=lambda name, spec: None,
    )
    # Stand in for a completed earlier cycle, which is what supplies the RSS
    # sum a mid-plan refresh carries forward.
    reconciler._last_snapshot = NodeSnapshot(
        total_bytes=8 * 1024**3,
        free_bytes=4 * 1024**3,
        source="cgroup",
        sum_rss_bytes=1024**3,
    )
    reconciler._converge_desired_state()

    # One per action: the node keeps saying it is alive while it loads.
    assert len(published) == 2
    # And it carries the measured RSS forward rather than claiming zero.
    assert all(snap.sum_rss_bytes == 1024**3 for snap in published)


def test_no_snapshot_is_published_before_a_full_cycle_has_run(
    node: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no previous snapshot there is no RSS sum to carry forward, and
    inventing one would publish a measurement nothing took."""
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    _publish_capacity()
    _desire("wanted")
    published: list[object] = []

    reconciler = _reconciler(_FakeSupervisor([]), deployer=lambda name, spec: None)
    reconciler._snapshot_publisher = published.append  # type: ignore[attr-defined]
    reconciler._converge_desired_state()  # type: ignore[attr-defined]

    assert published == []
