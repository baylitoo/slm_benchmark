"""When may a node converge on the catalog's desired state? (#534)

Postgres is the authoritative global desired state; each node keeps
``deployments.json`` as its local snapshot. The question this module answers is
the one #534 left open: what wins on disagreement, and when.

"The catalog is authoritative" is the right principle and is not sufficient as
an implementation. Three situations look identical to a node -- the catalog is
readable and lists fewer deployments than this node is running:

1. An operator genuinely removed one while this node was partitioned.
   **Converge.**
2. The node is pointed at the wrong database, or a fresh one -- a
   ``DOCIE_DB_HOST`` typo reaching another stack's Postgres.
3. The rows were lost: a restore from an older dump, a migration that dropped
   them.

A node that treats "readable" as "authoritative" tears down every model it is
serving in cases 2 and 3, with no error anywhere, because it believes it is
converging correctly. One mistyped host would take out a node's whole working
set. So readability cannot be the test.

The test is whether the catalog is provably *current for this node*, which
needs a per-node generation counter the node compares against the last
generation it applied. A wrong database cannot produce a generation newer than
what this node has already seen, because it has never seen this node's id.

Deliberately a pure function over four facts, so every case is a table-driven
unit test with no database, no node and no partition to simulate. The live
two-node run (#533) then verifies the wiring rather than the logic.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Mapping
from dataclasses import dataclass


class ConvergeAction(enum.Enum):
    """What the reconciler should do with the catalog's desired state."""

    CONVERGE = "converge"
    """Apply it, deletions included: a strictly newer generation for this node."""

    NOTHING = "nothing"
    """Already applied; the catalog and this node agree."""

    HOLD = "hold"
    """Keep serving, change nothing, and say why.

    Not an error and not a no-op: it means the catalog cannot be trusted to
    describe this node right now. Teardown is never reached from here.
    """


@dataclass(frozen=True)
class ConvergeDecision:
    action: ConvergeAction
    reason: str

    @property
    def should_alarm(self) -> bool:
        """Whether this state deserves an operator's attention.

        A hold caused by an unreachable catalog is ordinary -- the #527
        non-negotiable is that a node keeps serving through it. A hold caused
        by a generation going BACKWARDS, or by this node's row vanishing, means
        the node is probably talking to the wrong database, which is worth
        shouting about.
        """
        return self.action is ConvergeAction.HOLD and "unreachable" not in self.reason


def decide_convergence(
    *,
    catalog_reachable: bool,
    node_row_present: bool,
    catalog_generation: int | None,
    applied_generation: int,
    local_deployment_count: int,
) -> ConvergeDecision:
    """Whether this node may converge on the catalog's desired state.

    ``applied_generation`` is the last generation this node applied, persisted
    beside its local snapshot. ``local_deployment_count`` is how many
    deployments it is currently running, which is what makes a missing node row
    suspicious rather than merely new.
    """
    if not catalog_reachable:
        # The #527 non-negotiable: an unreachable control plane must never stop
        # an already-running model from serving.
        return ConvergeDecision(
            ConvergeAction.HOLD, "catalog unreachable; keeping what is loaded"
        )

    if not node_row_present:
        if local_deployment_count == 0:
            # A genuinely new node: nothing to lose, nothing to tear down.
            return ConvergeDecision(
                ConvergeAction.NOTHING, "no desired state for this node yet"
            )
        # This node is running deployments the catalog has never heard of. The
        # benign reading (someone deleted the row) is indistinguishable from
        # the dangerous one (wrong or wiped database), and only one of them is
        # recoverable, so hold.
        return ConvergeDecision(
            ConvergeAction.HOLD,
            "this node has no row in the catalog but is running deployments; "
            "refusing to converge on a catalog that may not be ours",
        )

    if catalog_generation is None:
        if applied_generation == 0:
            # The ordinary state before any desired state is written for this
            # node -- including every node of an existing deployment the
            # moment this ships. Holding here would alarm on every cycle of a
            # perfectly healthy single-node stack.
            return ConvergeDecision(
                ConvergeAction.NOTHING, "no desired state written for this node yet"
            )
        # This node HAS applied a generation, and the catalog now reports none.
        # Something removed what we converged on: a restore, a wipe, or the
        # wrong database. Never a reason to undo work.
        return ConvergeDecision(
            ConvergeAction.HOLD,
            f"catalog reports no desired generation but this node applied "
            f"{applied_generation}; refusing to converge on a catalog that may "
            f"not be ours",
        )

    if catalog_generation > applied_generation:
        return ConvergeDecision(
            ConvergeAction.CONVERGE,
            f"generation {catalog_generation} is newer than the applied "
            f"{applied_generation}",
        )

    if catalog_generation == applied_generation:
        return ConvergeDecision(
            ConvergeAction.NOTHING, f"generation {catalog_generation} already applied"
        )

    # Strictly older than what this node has already applied: replica lag, a
    # restored dump, or the wrong database. Never a reason to undo work.
    return ConvergeDecision(
        ConvergeAction.HOLD,
        f"catalog generation {catalog_generation} is older than the applied "
        f"{applied_generation}; refusing to go backwards",
    )


@dataclass(frozen=True)
class ConvergencePlan:
    """What a node must do to match the desired state it just read.

    Separate from carrying it out, and pure, for the same reason
    :func:`decide_convergence` is: these are the cases that destroy running
    capacity when wrong, and a table-driven test is a cheaper place to be wrong
    than a two-node hardware session.
    """

    to_start: tuple[str, ...] = ()
    """Desired, running, and not present locally."""

    to_stop: tuple[str, ...] = ()
    """Desired to be stopped, and currently present locally."""

    to_remove: tuple[str, ...] = ()
    """No longer desired, AND previously applied from desired state."""

    @property
    def is_empty(self) -> bool:
        return not (self.to_start or self.to_stop or self.to_remove)


def plan_convergence(
    *,
    desired: Mapping[str, str],
    local_names: Collection[str],
    previously_applied: Collection[str],
) -> ConvergencePlan:
    """Diff desired state against what this node is running.

    ``desired`` maps deployment name to desired state ("running"/"stopped").
    ``previously_applied`` is what this node created from desired state on an
    earlier pass -- see :mod:`docie_bench.serving.applied_state` for why that
    is tracked rather than inferred.

    REMOVAL IS NARROW BY CONSTRUCTION. A node also holds deployments the
    catalog was never told about: ``docie up`` creates one directly, which is
    the documented way to deploy on a chosen node while lifecycle events are
    not node-addressable. "Local but not desired" would therefore tear those
    down on the first pass. Only names this node previously applied FROM
    desired state are removal candidates; anything else is left alone, and a
    node's own deployments survive a convergence it was never part of.

    Note that an absent desired row and a ``stopped`` one produce different
    actions -- remove versus stop -- which is why the catalog represents a
    removal as a deleted row rather than a state.
    """
    local = {str(name) for name in local_names}
    applied = {str(name) for name in previously_applied}

    to_start = sorted(
        name
        for name, state in desired.items()
        if state == "running" and name not in local
    )
    to_stop = sorted(
        name for name, state in desired.items() if state != "running" and name in local
    )
    # Previously applied, no longer desired at all, and still here to remove.
    to_remove = sorted((applied - set(desired)) & local)
    return ConvergencePlan(
        to_start=tuple(to_start),
        to_stop=tuple(to_stop),
        to_remove=tuple(to_remove),
    )


__all__ = [
    "ConvergeAction",
    "ConvergeDecision",
    "ConvergencePlan",
    "decide_convergence",
    "plan_convergence",
]
