"""The convergence decision table (#534).

Every case here is a situation that, got wrong, tears down models that are
serving correctly. They are unit tests with no database and no partition to
simulate precisely because the decision is a pure function -- which is the
argument for keeping it one.
"""

from __future__ import annotations

import pytest

from docie_bench.serving.convergence import (
    ConvergeAction,
    decide_convergence,
)


def _decide(**overrides: object) -> object:
    base: dict[str, object] = {
        "catalog_reachable": True,
        "node_row_present": True,
        "catalog_generation": 5,
        "applied_generation": 5,
        "local_deployment_count": 2,
    }
    base.update(overrides)
    return decide_convergence(**base)  # type: ignore[arg-type]


def test_a_newer_generation_converges() -> None:
    decision = _decide(catalog_generation=6, applied_generation=5)

    assert decision.action is ConvergeAction.CONVERGE


def test_the_same_generation_does_nothing() -> None:
    decision = _decide(catalog_generation=5, applied_generation=5)

    assert decision.action is ConvergeAction.NOTHING


def test_an_unreachable_catalog_keeps_serving() -> None:
    """The #527 non-negotiable: the global control plane must never become a
    dependency for an already-running model to continue serving."""
    decision = _decide(catalog_reachable=False)

    assert decision.action is ConvergeAction.HOLD
    # Ordinary, not alarming -- this is the designed behaviour.
    assert decision.should_alarm is False


def test_an_older_generation_refuses_to_go_backwards() -> None:
    """Replica lag, a restored dump, or the wrong database. None of them is a
    reason to undo work this node has already applied."""
    decision = _decide(catalog_generation=3, applied_generation=5)

    assert decision.action is ConvergeAction.HOLD
    assert decision.should_alarm is True


def test_a_missing_node_row_while_running_deployments_holds() -> None:
    """The expensive case. A `DOCIE_DB_HOST` typo reaching another stack's
    Postgres looks exactly like "someone deleted my row": the catalog is
    readable and says this node should be running nothing.

    Converging there would tear down the node's whole working set with no error
    anywhere, so the only safe reading is to hold and alarm.
    """
    decision = _decide(node_row_present=False, local_deployment_count=3)

    assert decision.action is ConvergeAction.HOLD
    assert decision.should_alarm is True


def test_a_missing_node_row_on_a_fresh_node_is_not_suspicious() -> None:
    """A genuinely new node has nothing to lose, so the same observation is
    unremarkable -- holding here would stop a node ever starting."""
    decision = _decide(node_row_present=False, local_deployment_count=0)

    assert decision.action is ConvergeAction.NOTHING
    assert decision.should_alarm is False


def test_an_unreadable_generation_holds() -> None:
    decision = _decide(catalog_generation=None)

    assert decision.action is ConvergeAction.HOLD


@pytest.mark.parametrize(
    ("catalog_generation", "applied_generation"),
    [(1, 0), (2, 1), (100, 99), (1_000_000, 1)],
)
def test_any_strictly_newer_generation_converges(
    catalog_generation: int, applied_generation: int
) -> None:
    assert (
        _decide(
            catalog_generation=catalog_generation,
            applied_generation=applied_generation,
        ).action
        is ConvergeAction.CONVERGE
    )


def test_teardown_is_never_reachable_from_a_hold() -> None:
    """The invariant worth stating as a test: no combination that holds can be
    read as permission to converge. A reader that treats HOLD as "nothing to
    do, so the desired set is empty" would reintroduce the whole bug.
    """
    holds = [
        _decide(catalog_reachable=False),
        _decide(catalog_generation=3, applied_generation=5),
        _decide(node_row_present=False, local_deployment_count=3),
        _decide(catalog_generation=None),
    ]

    assert all(decision.action is ConvergeAction.HOLD for decision in holds)
    assert all(decision.action is not ConvergeAction.CONVERGE for decision in holds)
    # And each says why, because a hold an operator cannot explain is a hold
    # they will override.
    assert all(decision.reason for decision in holds)


def test_the_fresh_node_case_is_distinguishable_from_the_wiped_case() -> None:
    """Both have `node_row_present=False`; only the deployment count tells them
    apart. If that input were ever dropped, the dangerous case would silently
    take the safe case's path."""
    fresh = _decide(node_row_present=False, local_deployment_count=0)
    wiped = _decide(node_row_present=False, local_deployment_count=1)

    assert fresh.action is not wiped.action
