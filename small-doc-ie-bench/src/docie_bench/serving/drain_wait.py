"""Waiting until a drained node is free to stop, and saying what that proves (#634).

A drain returns as soon as the flag is written. The second half -- the node
unloading its runtimes -- happens on the node's own reconcile cycle, after a
grace period it computes from its deployments' request timeouts. So a caller
that wants "this machine is now safe to replace" has to poll, and #634's
observation was that every automated caller will otherwise reimplement that
loop, each with its own idea of what counts as done.

WHAT IS AND IS NOT OBSERVABLE CENTRALLY. #634 asks for four states to be kept
apart. Three of them are visible from the catalog, and the fourth is not
visible anywhere:

===========================  =============================================
routing excluded             ``drained_at`` is set. Immediate, central, and
                             true even for a machine that is unreachable.
grace elapsed                NOT centrally observable -- the grace is
                             computed ON the node from its own deployments'
                             ``timeout_seconds``. But it does not need to
                             be: a node only stops its runtimes AFTER its
                             grace, so observing the next state proves this
                             one happened.
runtimes stopped             No placement of this node's has a live
                             endpoint. This is the state worth waiting for.
safe replacement confirmed   NOT OBSERVABLE AT ALL. See below.
===========================  =============================================

THE GUARANTEE IS BOUNDED, NOT COMPLETE. "Runtimes stopped" means the node
waited out the longest request timeout its deployments declared and then
stopped them. It does **not** prove every request finished:

* the grace is capped at ``DRAIN_GRACE_CEILING_S``, so a deployment declaring
  a longer timeout than the cap is cut short by design;
* a request admitted an instant before the drain, by a caller whose own
  timeout is longer than the deployment's, is outside what the grace bounds;
* queued work that had not yet reached a runtime is not counted at all.

So a successful wait says *the bounded drain completed*, and a caller that
needs more than that bound has to raise the deployments' ``timeout_seconds``
or accept the residual risk explicitly. Reporting it as "no requests were
dropped" would be the kind of claim that is wrong exactly when it matters.

Pure over one ``/v1/serving/nodes`` row and an injected fetch, for the reason
:mod:`docie_bench.serving.drain` is pure: the cases that drop live traffic when
wrong are cheaper to be wrong about in a table-driven test than against a
machine somebody is waiting on.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

#: How often to re-read the fleet while waiting. The thing being waited for is
#: a grace period measured in tens of seconds and a process shutdown, so a
#: faster poll buys nothing and a slower one adds latency to every deploy.
DEFAULT_POLL_INTERVAL_S = 3.0

#: Default deadline. Comfortably past ``DRAIN_GRACE_CEILING_S`` (600) plus the
#: time to stop the runtimes, so the common case never hits it -- a timeout
#: should mean something is wrong, not that the caller was impatient.
DEFAULT_WAIT_TIMEOUT_S = 900.0


class DrainReadiness(enum.StrEnum):
    """How far a node is from being free to stop, as the catalog can see it."""

    ADMITTING = "admitting"
    """Not drained. Still taking new requests."""

    ROUTING_EXCLUDED = "routing_excluded"
    """Drained, and still holding live runtimes.

    Nothing new is being routed here; what was already admitted is finishing.
    The node's own grace period is running, or its reconciler has not reached
    the cycle that stops things.
    """

    RUNTIMES_STOPPED = "runtimes_stopped"
    """Drained, with no live placement left. The bounded drain is complete."""

    STALE = "stale"
    """Drained, still holding live runtimes, and no longer reporting.

    The distinction that matters to a caller: the node is not converging, so
    waiting may never finish. Its placement rows are the last thing it
    published, not what it holds now -- routing demotes such a node but the
    rows are not evidence of a stop that never happened.
    """

    UNKNOWN = "unknown"
    """No row for this node id, or a row that cannot be read."""


#: The only readiness that permits replacing the container.
SAFE_TO_REPLACE = frozenset({DrainReadiness.RUNTIMES_STOPPED})


def assess_node(row: Mapping[str, object] | None) -> DrainReadiness:
    """Readiness of one ``/v1/serving/nodes`` row.

    Reads the API's own ``drain_state`` rather than recomputing it, so this and
    the Fleet page cannot disagree about what "drained" means.

    A node that is not reporting is only called ``STALE`` while it still holds
    live placements. Once the rows say nothing is live, they say so because the
    node published that before it went quiet -- which is evidence of the stop,
    not an absence of evidence.
    """
    if not row:
        return DrainReadiness.UNKNOWN
    state = str(row.get("drain_state") or "")
    if state == "admitting":
        return DrainReadiness.ADMITTING
    if state == "drained":
        return DrainReadiness.RUNTIMES_STOPPED
    if state != "draining":
        return DrainReadiness.UNKNOWN
    # `reporting` is the same staleness rule the router uses to demote a node's
    # placements, so a caller and the routing layer agree about who is alive.
    if row.get("reporting") is False:
        return DrainReadiness.STALE
    return DrainReadiness.ROUTING_EXCLUDED


def find_node(
    rows: Sequence[Mapping[str, object]], node_id: str
) -> Mapping[str, object] | None:
    for row in rows:
        if str(row.get("node_id") or "") == node_id:
            return row
    return None


@dataclass(frozen=True)
class DrainWaitResult:
    """What the wait saw, and whether the caller may proceed."""

    readiness: DrainReadiness
    elapsed_s: float
    polls: int

    @property
    def safe_to_replace(self) -> bool:
        return self.readiness in SAFE_TO_REPLACE

    @property
    def detail(self) -> str:
        """One sentence, written for whoever reads a failed deploy's log."""
        if self.readiness is DrainReadiness.RUNTIMES_STOPPED:
            return (
                "the node holds no live runtime: the bounded drain completed "
                "(its grace period elapsed and its runtimes stopped). This does "
                "not prove every in-flight request finished -- the grace bounds "
                "them by the deployments' declared timeouts, capped"
            )
        if self.readiness is DrainReadiness.ROUTING_EXCLUDED:
            return (
                f"still holding live runtimes after {self.elapsed_s:.0f}s. The "
                f"node is out of the routing rotation, so nothing new is being "
                f"admitted, but its reconciler has not stopped them yet -- "
                f"check that it is running and that "
                f"DOCIE_SERVING_DESIRED_APPLY is set on it"
            )
        if self.readiness is DrainReadiness.STALE:
            return (
                "the node stopped reporting while still holding live runtimes. "
                "It is not converging, so the drain cannot complete: its "
                "placement rows are the last thing it published, not what it "
                "holds now"
            )
        if self.readiness is DrainReadiness.ADMITTING:
            return (
                "the node is admitting traffic: no drain is in effect. Was it "
                "resumed, or did the drain not reach the catalog?"
            )
        return "no readable row for this node"


def wait_for_drain(
    fetch_nodes: Callable[[], Sequence[Mapping[str, object]]],
    node_id: str,
    *,
    timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> DrainWaitResult:
    """Poll until ``node_id`` is free to stop, or the deadline passes.

    Returns rather than raising, including on timeout: the caller decides what a
    timeout means, and #634 is explicit that it must never be turned into an
    automatic ``acknowledge_outage`` or an automatic replacement.

    ``ADMITTING`` returns immediately instead of waiting out the deadline.
    Either the drain never reached the catalog or something resumed the node,
    and neither resolves by waiting -- both want a human to look.

    A fetch that raises is treated as a transient read failure and retried
    until the deadline, because the alternative is failing a deploy over one
    dropped connection. If it never succeeds the result is ``UNKNOWN``, which
    is not safe to replace.
    """
    started = clock()
    polls = 0
    readiness = DrainReadiness.UNKNOWN
    while True:
        polls += 1
        try:
            readiness = assess_node(find_node(list(fetch_nodes()), node_id))
        except Exception:  # noqa: BLE001 - a dropped read must not fail a deploy
            readiness = DrainReadiness.UNKNOWN
        if readiness in SAFE_TO_REPLACE or readiness is DrainReadiness.ADMITTING:
            break
        elapsed = clock() - started
        if elapsed + poll_interval_s > timeout_s:
            break
        sleep(poll_interval_s)
    return DrainWaitResult(
        readiness=readiness, elapsed_s=clock() - started, polls=polls
    )


__all__ = [
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_WAIT_TIMEOUT_S",
    "SAFE_TO_REPLACE",
    "DrainReadiness",
    "DrainWaitResult",
    "assess_node",
    "find_node",
    "wait_for_drain",
]
