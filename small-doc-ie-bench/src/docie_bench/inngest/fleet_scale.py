"""Replica counting and placement across the fleet (#664 A4).

The api's ``deployments.json`` is one node's file, so replica counts read from
it ignore every other node, and deploy/delete events reach whichever node
Inngest picks. Here the count comes from ``model_placement`` plus desired
state, and once more than one node has reported, new replicas are placed with
``select_nodes`` and surplus ones removed through the owner's desired state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException

from docie_bench.serving.control_plane import (
    deployment_name_of,
    mint_deployment_name,
)


@dataclass
class FleetView:
    """What the catalog says about one store model's replicas."""

    #: Replica name -> owning node id ("" when unknown).
    owners: dict[str, str] = field(default_factory=dict)
    #: Replica name -> (node id, generation) of its desired-state row.
    desired: dict[str, tuple[str, int]] = field(default_factory=dict)
    node_ids: list[str] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    placements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_fleet(self) -> bool:
        """More than one node that matters: a live one, or one owning a replica.

        A leftover row from a replaced node does not make a fleet, or a
        single-node install would switch to desired state nobody applies.
        """
        from docie_bench.serving.node_liveness import stale_node_ids
        from docie_bench.serving.node_selection import UNREPORTED_CAPACITY

        stale = stale_node_ids(self.snapshots)
        live = {
            str(s.get("node_id") or "")
            for s in self.snapshots
            if s.get("source") != UNREPORTED_CAPACITY
        } - stale - {""}
        holders = {node for node in self.owners.values() if node}
        return len(live | holders) > 1


def _spec_model(row: Mapping[str, Any]) -> str:
    spec = row.get("spec")
    return str(spec.get("model") or "") if isinstance(spec, dict) else ""


def collect_fleet_view(
    base: str,
    *,
    snapshots: Sequence[Mapping[str, Any]],
    placements: Sequence[Mapping[str, Any]],
    desired_snapshots: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]] = (),
) -> FleetView:
    """Replicas of ``base`` from placements and desired rows (pure)."""
    view = FleetView(
        node_ids=sorted({str(s.get("node_id") or "") for s in snapshots} - {""}),
        models=[dict(m) for m in models],
        snapshots=[dict(s) for s in snapshots],
        placements=[dict(p) for p in placements],
    )
    for row in placements:
        name = str(row.get("name") or "")
        if name and str(row.get("model_name") or "") == base:
            view.owners[name] = str(row.get("node_id") or "")
    for snapshot in desired_snapshots:
        rows = snapshot.get("deployments")
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or _spec_model(row) != base:
                continue
            name = str(row.get("name") or "")
            node_id = str(row.get("node_id") or snapshot.get("node_id") or "")
            if not name:
                continue
            # A desired row not yet converged has no placement, but it is a
            # replica already asked for: counting it keeps scale idempotent.
            view.owners.setdefault(name, node_id)
            view.desired[name] = (node_id, int(row.get("generation") or 0))
    return view


def read_fleet_view(base: str) -> FleetView | None:
    """The catalog's view of ``base``, or None when the catalog is unreadable."""
    from docie_bench.serving.catalog import ModelCatalog

    try:
        catalog = ModelCatalog()
        snapshots = list(catalog.list_node_snapshots())
        placements = list(catalog.list_placements())
        node_ids = sorted({str(s.get("node_id") or "") for s in snapshots} - {""})
        desired = [catalog.desired_snapshot(node_id) for node_id in node_ids]
        models = list(catalog.list()) if len(node_ids) > 1 else []
    except Exception:  # noqa: BLE001 - no catalog = the api-local file, as before
        return None
    return collect_fleet_view(
        base,
        snapshots=snapshots,
        placements=placements,
        desired_snapshots=desired,
        models=models,
    )


def merged_records(
    base: str, local: Sequence[dict[str, Any]], view: FleetView | None
) -> list[dict[str, Any]]:
    """Local records plus a stand-in for each replica only the catalog knows.

    The stand-ins carry just the name and alias, which is all the replica math
    in ``control_plane`` reads.
    """
    records = list(local)
    if view is None:
        return records
    known = {deployment_name_of(record) for record in records}
    for name in sorted(view.owners):
        if name not in known:
            records.append({"spec": {"name": name, "launch": {"alias": base}}})
    return records


def place_replicas(
    base: str,
    count: int,
    view: FleetView,
    *,
    context_length: int | None,
    n_parallel: int = 1,
    launch: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Record ``count`` new replicas as desired state on distinct fitting nodes.

    Nodes already running a replica of ``base`` are skipped (two replicas on
    one machine share its failure), and a placement short of ``count`` is
    refused with the per-node reasons, the auto-place contract.
    """
    from docie_bench.serving.catalog import DesiredStateConflictError, ModelCatalog
    from docie_bench.serving.node_liveness import stale_node_ids
    from docie_bench.serving.node_selection import select_nodes

    holders = {node for node in view.owners.values() if node}
    open_snapshots = [s for s in view.snapshots if str(s.get("node_id") or "") not in holders]
    chosen, candidates = select_nodes(
        base,
        count=count,
        models=view.models,
        snapshots=open_snapshots,
        placements=view.placements,
        stale=stale_node_ids(view.snapshots),
        context_length=context_length,
        n_parallel=n_parallel,
    )
    if len(chosen) < count:
        reasons = [f"{node}: already runs a replica" for node in sorted(holders)]
        reasons += [f"{c.node_id}: {c.reason}" for c in candidates if not c.eligible]
        raise HTTPException(
            status_code=409,
            detail=(
                f"{count} more replica(s) of {base!r} need {count} more distinct "
                f"nodes and only {len(chosen)} can take one. Replicas are never "
                f"stacked on one node. Rejected -- {'; '.join(reasons) or 'none'}"
            ),
        )
    catalog = ModelCatalog()
    # ``launch`` is the deploy's own desired spec (all launch settings); a
    # scale call without one carries model and context only.
    spec: dict[str, Any] = {**(launch or {}), "model": base}
    spec.pop("port", None)  # each replica gets its own port on its node
    if context_length is not None:
        spec["context_length"] = context_length
    placed: list[dict[str, Any]] = []
    for node_id in chosen:
        name = mint_deployment_name(base)
        try:
            catalog.set_desired_deployment(node_id, name, desired_state="running", spec=spec)
        except DesiredStateConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"desired state for {name!r} on {node_id!r} changed under us: "
                f"{exc}. Already placed: {placed or 'nothing'}",
            ) from exc
        placed.append({"name": name, "node_id": node_id})
    return placed


def remove_replicas(base: str, surplus: int, view: FleetView) -> list[dict[str, Any]]:
    """Remove ``surplus`` replicas through their owner's desired state.

    Only a replica its node created from desired state can be removed that way
    (#590). Those go first; if the surplus needs one that cannot, nothing is
    removed and the 409 names its owner, because an untargeted delete event
    may reach a node that does not own it.
    """
    from docie_bench.serving.catalog import DesiredStateConflictError, ModelCatalog

    names = sorted(view.owners, reverse=True)
    removable = [name for name in names if name in view.desired]
    blocked = [name for name in names if name not in view.desired]
    victims = (removable + blocked)[:surplus]
    stuck = [name for name in victims if name not in view.desired]
    if stuck:
        owners = ", ".join(f"{name} on {view.owners[name] or 'an unknown node'}" for name in stuck)
        raise HTTPException(
            status_code=409,
            detail=(
                f"scaling {base!r} down needs to remove {owners}, which was not "
                f"deployed through desired state, so it can only be removed on "
                f"its node (`docie-serving stop <name>`); nothing was removed"
            ),
        )
    catalog = ModelCatalog()
    removed: list[dict[str, Any]] = []
    for name in victims:
        node_id, generation = view.desired[name]
        try:
            catalog.delete_desired_deployment(node_id, name, expected_generation=generation)
        except DesiredStateConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"desired state for {name!r} changed under us: {exc}. "
                f"Already removed: {removed or 'nothing'}",
            ) from exc
        removed.append({"name": name, "node_id": node_id})
    return removed


__all__ = [
    "FleetView",
    "collect_fleet_view",
    "merged_records",
    "place_replicas",
    "read_fleet_view",
    "remove_replicas",
]
