"""What a node does with the desired state it read (#534).

The removal rule is the interesting half. Convergence must be able to remove a
deployment the catalog no longer wants, and a node also holds deployments the
catalog was never told about -- `docie up` creates one directly, which is the
documented way to deploy on a chosen node while lifecycle events are not
node-addressable (#574, #576). Getting that boundary wrong means the first
convergence pass tears down exactly the deployments the runbook just created.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docie_bench.serving.applied_state import (
    AppliedState,
    applied_state_path,
    read_applied_state,
    write_applied_state,
)
from docie_bench.serving.convergence import plan_convergence


@pytest.fixture
def node_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "serving"
    home.mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    return home


# --- the plan ------------------------------------------------------------


def test_a_desired_deployment_not_running_is_started() -> None:
    plan = plan_convergence(
        desired={"lfm2.5-350m": "running"}, local_names=[], previously_applied=[]
    )

    assert plan.to_start == ("lfm2.5-350m",)
    assert plan.to_stop == ()
    assert plan.to_remove == ()


def test_a_deployment_desired_stopped_is_stopped_not_removed() -> None:
    """"Should not be running" and "no longer desired" are different
    instructions, which is why the catalog deletes a row to mean the latter."""
    plan = plan_convergence(
        desired={"lfm2.5-350m": "stopped"},
        local_names=["lfm2.5-350m"],
        previously_applied=["lfm2.5-350m"],
    )

    assert plan.to_stop == ("lfm2.5-350m",)
    assert plan.to_remove == ()


def test_a_previously_applied_deployment_no_longer_desired_is_removed() -> None:
    plan = plan_convergence(
        desired={}, local_names=["lfm2.5-350m"], previously_applied=["lfm2.5-350m"]
    )

    assert plan.to_remove == ("lfm2.5-350m",)


def test_a_locally_created_deployment_is_never_removed() -> None:
    """The safety property. `docie up` creates a record the catalog knows
    nothing about; "local but not desired" would tear it down on the first
    pass, which would break the documented per-node deploy path.
    """
    plan = plan_convergence(
        desired={},
        local_names=["lfm2.5-350m-ovh-02"],
        previously_applied=[],  # this node never applied it from desired state
    )

    assert plan.to_remove == ()
    assert plan.is_empty


def test_a_mixed_node_removes_only_what_it_applied() -> None:
    plan = plan_convergence(
        desired={"wanted": "running"},
        local_names=["applied-before", "created-locally"],
        previously_applied=["applied-before"],
    )

    assert plan.to_start == ("wanted",)
    assert plan.to_remove == ("applied-before",)
    assert "created-locally" not in plan.to_remove


def test_an_applied_name_already_gone_is_not_a_removal() -> None:
    """Idempotence: a second pass over the same desired state asks for
    nothing, including when the node already removed it."""
    plan = plan_convergence(
        desired={}, local_names=[], previously_applied=["already-removed"]
    )

    assert plan.is_empty


def test_an_already_satisfied_desired_state_plans_nothing() -> None:
    plan = plan_convergence(
        desired={"a": "running", "b": "running"},
        local_names=["a", "b"],
        previously_applied=["a", "b"],
    )

    assert plan.is_empty


def test_the_plan_is_deterministic() -> None:
    """Sorted, so a log line or a test can be compared across passes."""
    plan = plan_convergence(
        desired={"z": "running", "a": "running", "m": "running"},
        local_names=[],
        previously_applied=[],
    )

    assert plan.to_start == ("a", "m", "z")


# --- the applied-state file ---------------------------------------------


def test_applied_state_round_trips(node_home: Path) -> None:
    write_applied_state(AppliedState(generation=7, names=frozenset({"b", "a"})))

    state = read_applied_state()

    assert state.generation == 7
    assert state.names == frozenset({"a", "b"})


def test_a_missing_file_reads_as_nothing_applied(node_home: Path) -> None:
    """Generation 0 and no names is the safe direction: the convergence table
    then treats an empty catalog as "nothing written yet" rather than
    "everything was lost", and nothing is a removal candidate."""
    state = read_applied_state()

    assert state.generation == 0
    assert state.names == frozenset()


def test_a_corrupt_file_reads_as_nothing_applied(node_home: Path) -> None:
    applied_state_path().write_text("{not json", encoding="utf-8")

    assert read_applied_state() == AppliedState()


def test_a_nonsense_generation_reads_as_zero(node_home: Path) -> None:
    applied_state_path().write_text('{"generation": -5, "names": "oops"}', encoding="utf-8")

    state = read_applied_state()

    assert state.generation == 0
    assert state.names == frozenset()


def test_the_write_is_atomic(node_home: Path) -> None:
    """Temp-then-replace, so a reader never sees a half-written file -- which
    would read as "nothing applied" and make the node re-converge."""
    write_applied_state(AppliedState(generation=1, names=frozenset({"a"})))

    leftovers = [p.name for p in node_home.iterdir() if p.name.endswith(".tmp")]

    assert leftovers == []
    assert read_applied_state().generation == 1


def test_it_lives_beside_the_other_node_state(node_home: Path) -> None:
    """Shares the serving home, and therefore the durability, of the node id
    and deployments.json -- a node that lost this has lost those too, so it has
    nothing to tear down and no history to contradict."""
    assert applied_state_path().parent == node_home
