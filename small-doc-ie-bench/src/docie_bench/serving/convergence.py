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
        return ConvergeDecision(
            ConvergeAction.HOLD, "desired generation is unreadable for this node"
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


__all__ = [
    "ConvergeAction",
    "ConvergeDecision",
    "decide_convergence",
]
