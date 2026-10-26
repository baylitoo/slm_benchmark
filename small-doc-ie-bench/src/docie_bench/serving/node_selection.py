"""Which node should run a model, and which nodes should not (#536).

Until now a deploy either landed on whichever node Inngest happened to pick, or
named one explicitly. Neither is what an operator wants most of the time: they
want "put this somewhere it fits", and for a scaled model they want the
replicas on *different* machines, which is the only arrangement that survives
losing one.

Answering that is a ranking problem over four hard gates and one preference,
and the gates matter more than the ranking. Three of them are the same
conditions the targeted-deploy path already refuses on, and they are repeated
here rather than discovered later because an auto-placed deploy has no operator
watching to read a 409:

* a node that has never published a measurement -- its capacity numbers are
  zeroes, so it would otherwise rank as infinitely roomy and win every time;
* a node that published and stopped;
* a node an operator has drained;
* a node whose blob store does not hold the model, since the store is
  node-local while the catalog is global.

The fit itself is NOT recomputed here. :func:`docie_bench.serving.sizing.compute_sizing`
already prices a model against one node's snapshot, with the safety margin, the
mid-load reservation and the calibrated-over-predicted rule that took their own
issues to get right. This module calls it once per node and reads ``fits_now``.
A second opinion about what fits would be a second set of bugs.

Honest about not knowing. A node whose snapshot cannot price the model is not
treated as roomy OR as full: it stays eligible, and ranks below every node
whose fit is known. Picking an unpriceable node over a measured one would mean
preferring ignorance.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from docie_bench.serving.sizing import compute_sizing

#: ``source`` value of a node row that exists without a measurement.
UNREPORTED_CAPACITY = "unreported"


@dataclass(frozen=True)
class NodeCandidate:
    """One node, judged for one model."""

    node_id: str
    eligible: bool
    #: Why this node cannot take the model; "" when it can. Carried per node
    #: because the useful error for a failed auto-placement is not "no node
    #: fits" but which node failed which gate.
    reason: str
    #: How many more instances fit, from the sizing engine. None when the model
    #: cannot be priced against this node -- unknown, never zero.
    fits_now: int | None
    #: Budget after margin and mid-load reservations. Ranking tiebreak only.
    free_effective_bytes: int | None
    #: Placements this node already owns. The spread preference.
    owned: int

    @property
    def priced(self) -> bool:
        return self.fits_now is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "eligible": self.eligible,
            "reason": self.reason or None,
            "fits_now": self.fits_now,
            "free_effective_bytes": self.free_effective_bytes,
            "owned": self.owned,
        }


def _gate(
    snapshot: Mapping[str, Any],
    *,
    model: str,
    stale: Collection[str],
) -> str:
    """The first hard reason this node cannot take ``model``, or ""."""
    node_id = str(snapshot.get("node_id") or "")
    if snapshot.get("source") == UNREPORTED_CAPACITY:
        return "has never published a capacity measurement"
    if node_id in stale:
        return "has stopped reporting"
    if snapshot.get("drained_at"):
        return "is drained and out of the routing rotation"
    held = snapshot.get("store_models")
    # A list that lacks the model is a definite no. None means the node has not
    # reported its store, which is not the same answer -- the targeted path
    # accepts it for exactly that reason, so this one does too rather than
    # silently excluding a node that may well hold the blob.
    if isinstance(held, list) and model not in held:
        return "does not hold this model's blob"
    return ""


def rank_nodes(
    model: str,
    *,
    models: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    placements: Sequence[Mapping[str, Any]],
    stale: Collection[str] = (),
    context_length: int | None = None,
    n_parallel: int = 1,
) -> list[NodeCandidate]:
    """Every known node, judged for ``model``, best first.

    Ineligible nodes are RETURNED rather than dropped, carrying the gate they
    failed. The caller needs them: "no node can take this model" is not a
    useful thing to tell an operator, while "node-a has no room, node-b does
    not hold the blob" tells them what to do next.

    Ordering among eligible nodes, in priority order:

    1. fewest placements already owned -- the spread preference, so a second
       replica goes to a different machine rather than stacking on the roomiest
       one. Spreading is the point of having a fleet: two replicas on one node
       share its fate.
    2. most instances that still fit, then most free budget -- headroom, so a
       node that is nearly full is not chosen while a roomier one idles.
    3. priced before unpriceable, then node id for a deterministic result.
    """
    owned: dict[str, int] = {}
    for row in placements:
        node_id = str(row.get("node_id") or "")
        if node_id:
            owned[node_id] = owned.get(node_id, 0) + 1

    candidates: list[NodeCandidate] = []
    for snapshot in snapshots:
        node_id = str(snapshot.get("node_id") or "")
        if not node_id:
            continue
        reason = _gate(snapshot, model=model, stale=stale)
        fits_now: int | None = None
        free_effective: int | None = None
        if not reason:
            # One node's view: the engine takes a single snapshot, and the
            # placements are filtered to this node so its running instances and
            # mid-load reservations are its own.
            report = compute_sizing(
                models,
                snapshot,
                [
                    row
                    for row in placements
                    if str(row.get("node_id") or "") == node_id
                ],
                context_length=context_length,
                n_parallel=n_parallel,
            )
            free_effective = report.free_effective_bytes
            for fit in report.per_model:
                if fit.name == model:
                    fits_now = fit.fits_now
                    break
            if fits_now == 0:
                reason = "has no room for another instance of this model"
        candidates.append(
            NodeCandidate(
                node_id=node_id,
                eligible=not reason,
                reason=reason,
                fits_now=fits_now,
                free_effective_bytes=free_effective,
                owned=owned.get(node_id, 0),
            )
        )

    candidates.sort(
        key=lambda c: (
            not c.eligible,
            c.owned,
            -(c.fits_now or 0),
            -(c.free_effective_bytes or 0),
            not c.priced,
            c.node_id,
        )
    )
    return candidates


def select_nodes(
    model: str,
    *,
    count: int = 1,
    models: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    placements: Sequence[Mapping[str, Any]],
    stale: Collection[str] = (),
    context_length: int | None = None,
    n_parallel: int = 1,
) -> tuple[list[str], list[NodeCandidate]]:
    """``(chosen, all_candidates)`` -- up to ``count`` DISTINCT nodes.

    Distinct is the whole contract for a scaled deploy. Two replicas on one
    machine share its memory, its cgroup limit and its failure, which is not
    what asking for two replicas means. So this returns fewer nodes than asked
    rather than doubling up, and the caller decides whether that is a refusal
    or a partial placement -- a decision that belongs to the API, not here.

    The returned candidates are the full ranking, ineligible nodes included, so
    a caller refusing can say which node failed which gate.
    """
    candidates = rank_nodes(
        model,
        models=models,
        snapshots=snapshots,
        placements=placements,
        stale=stale,
        context_length=context_length,
        n_parallel=n_parallel,
    )
    chosen = [c.node_id for c in candidates if c.eligible][: max(0, count)]
    return chosen, candidates


__all__ = [
    "UNREPORTED_CAPACITY",
    "NodeCandidate",
    "rank_nodes",
    "select_nodes",
]
