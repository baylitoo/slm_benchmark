"""Waiting until a drained node is free to stop (#634).

Table-driven over one `/v1/serving/nodes` row and an injected fetch, with no
HTTP and no node: the cases that drop live traffic when wrong are the ones that
are expensive to reproduce on a machine somebody is waiting on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from docie_bench.serving.drain_wait import (
    DrainReadiness,
    assess_node,
    find_node,
    wait_for_drain,
)

NODE = "ovh-cpu-02"


def row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "node_id": NODE,
        "drain_state": "admitting",
        "reporting": True,
        "deployments": 1,
    }
    base.update(overrides)
    return base


# --- reading one row -------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "reporting", "expected"),
    [
        ("admitting", True, DrainReadiness.ADMITTING),
        ("draining", True, DrainReadiness.ROUTING_EXCLUDED),
        ("drained", True, DrainReadiness.RUNTIMES_STOPPED),
        # Drained, still holding live runtimes, and no longer publishing: it is
        # not converging, so waiting may never finish. Worth its own state
        # rather than reading as "nearly there".
        ("draining", False, DrainReadiness.STALE),
        # Rows say nothing is live, published BEFORE the node went quiet --
        # which is evidence of the stop, not an absence of evidence.
        ("drained", False, DrainReadiness.RUNTIMES_STOPPED),
        ("", True, DrainReadiness.UNKNOWN),
    ],
)
def test_readiness_of_one_row(
    state: str, reporting: bool, expected: DrainReadiness
) -> None:
    assert assess_node(row(drain_state=state, reporting=reporting)) is expected


def test_a_missing_row_is_unknown_not_drained() -> None:
    """The direction that matters: an absent node must never read as free to
    replace."""
    from docie_bench.serving.drain_wait import SAFE_TO_REPLACE

    assert assess_node(None) is DrainReadiness.UNKNOWN
    assert assess_node(None) not in SAFE_TO_REPLACE


def test_only_runtimes_stopped_is_safe_to_replace() -> None:
    from docie_bench.serving.drain_wait import SAFE_TO_REPLACE

    assert frozenset({DrainReadiness.RUNTIMES_STOPPED}) == SAFE_TO_REPLACE


def test_find_node_picks_the_right_row() -> None:
    rows = [row(node_id="other"), row(node_id=NODE, drain_state="drained")]

    assert find_node(rows, NODE) is rows[1]
    assert find_node(rows, "absent") is None


# --- the wait loop ---------------------------------------------------------


class _Clock:
    """Monotonic seconds the test advances by hand, through `sleep`."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _scripted(
    states: Sequence[str], *, reporting: bool = True
) -> tuple[list[int], Any]:
    """A fetch that walks `states`, then repeats the last one for ever."""
    calls: list[int] = []

    def fetch() -> list[Mapping[str, Any]]:
        calls.append(1)
        index = min(len(calls) - 1, len(states) - 1)
        return [row(drain_state=states[index], reporting=reporting)]

    return calls, fetch


def test_it_returns_as_soon_as_the_runtimes_are_stopped() -> None:
    clock = _Clock()
    _calls, fetch = _scripted(["draining", "draining", "drained"])

    result = wait_for_drain(
        fetch, NODE, timeout_s=600, poll_interval_s=3, sleep=clock.sleep, clock=clock
    )

    assert result.readiness is DrainReadiness.RUNTIMES_STOPPED
    assert result.safe_to_replace
    assert result.polls == 3
    # Two sleeps between three polls, not a fixed wait.
    assert result.elapsed_s == 6


def test_an_already_drained_node_returns_on_the_first_poll() -> None:
    """Idempotence from the caller's side: a hook that runs twice must not
    wait out a second grace period."""
    clock = _Clock()
    _calls, fetch = _scripted(["drained"])

    result = wait_for_drain(
        fetch, NODE, timeout_s=600, poll_interval_s=3, sleep=clock.sleep, clock=clock
    )

    assert result.polls == 1
    assert result.elapsed_s == 0


def test_a_timeout_is_not_safe_to_replace() -> None:
    """#634: a timeout must fail, and must not trigger an automatic
    replacement or quietly add acknowledge_outage."""
    clock = _Clock()
    _calls, fetch = _scripted(["draining"])

    result = wait_for_drain(
        fetch, NODE, timeout_s=10, poll_interval_s=3, sleep=clock.sleep, clock=clock
    )

    assert result.readiness is DrainReadiness.ROUTING_EXCLUDED
    assert not result.safe_to_replace
    assert result.elapsed_s <= 10
    assert "still holding live runtimes" in result.detail


def test_it_stops_waiting_on_a_node_that_is_admitting() -> None:
    """Either the drain never reached the catalog or something resumed the
    node. Neither resolves by waiting, and burning the deadline would hide it
    behind a timeout."""
    clock = _Clock()
    _calls, fetch = _scripted(["admitting"])

    result = wait_for_drain(
        fetch, NODE, timeout_s=600, poll_interval_s=3, sleep=clock.sleep, clock=clock
    )

    assert result.readiness is DrainReadiness.ADMITTING
    assert not result.safe_to_replace
    assert result.polls == 1
    assert "no drain is in effect" in result.detail


def test_a_stale_node_times_out_with_its_own_reason() -> None:
    clock = _Clock()
    _calls, fetch = _scripted(["draining"], reporting=False)

    result = wait_for_drain(
        fetch, NODE, timeout_s=10, poll_interval_s=3, sleep=clock.sleep, clock=clock
    )

    assert result.readiness is DrainReadiness.STALE
    assert not result.safe_to_replace
    assert "stopped reporting" in result.detail


def test_a_dropped_read_is_retried_rather_than_failing_the_deploy() -> None:
    clock = _Clock()
    calls: list[int] = []

    def flaky() -> list[Mapping[str, Any]]:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("reset by peer")
        return [row(drain_state="drained")]

    result = wait_for_drain(
        flaky, NODE, timeout_s=600, poll_interval_s=3, sleep=clock.sleep, clock=clock
    )

    assert result.readiness is DrainReadiness.RUNTIMES_STOPPED
    assert result.polls == 3


def test_a_read_that_never_succeeds_is_unknown_and_unsafe() -> None:
    clock = _Clock()

    def broken() -> list[Mapping[str, Any]]:
        raise ConnectionError("no route to host")

    result = wait_for_drain(
        broken, NODE, timeout_s=10, poll_interval_s=3, sleep=clock.sleep, clock=clock
    )

    assert result.readiness is DrainReadiness.UNKNOWN
    assert not result.safe_to_replace


def test_a_node_that_vanishes_mid_wait_is_unsafe() -> None:
    """A fleet listing that stops naming the node is not a completed drain."""
    clock = _Clock()
    calls: list[int] = []

    def vanishing() -> list[Mapping[str, Any]]:
        calls.append(1)
        return [] if len(calls) > 1 else [row(drain_state="draining")]

    result = wait_for_drain(
        vanishing, NODE, timeout_s=10, poll_interval_s=3, sleep=clock.sleep, clock=clock
    )

    assert result.readiness is DrainReadiness.UNKNOWN
    assert not result.safe_to_replace


def test_success_does_not_claim_every_request_finished() -> None:
    """The guarantee is bounded by the deployments' declared timeouts, capped.
    Saying otherwise would be wrong exactly when it matters."""
    clock = _Clock()
    _calls, fetch = _scripted(["drained"])

    result = wait_for_drain(
        fetch, NODE, timeout_s=600, poll_interval_s=3, sleep=clock.sleep, clock=clock
    )

    assert "does not prove every in-flight request finished" in result.detail
