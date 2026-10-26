"""Is a serving node still reporting? (#531, phase 1 of #527)

The staleness rule already existed for capacity — a published ``serving_node``
snapshot is trusted only while it is at most ``SNAPSHOT_STALE_INTERVALS``
reconcile intervals old, so a dead reconciler degrades to "observed
unavailable" rather than backing sizing with a number from a dead past.

It lived in ``inngest/serving_api.py``, which routing cannot import (api
depends on serving, not the other way round). Moved here so the same rule can
answer the question #531 asks: *should this node's placements still receive
traffic?* One definition of "reporting", used by both the capacity view and
the router, rather than two that can drift.

Fail-open throughout. A node is only treated as gone when that is **provable**
— it published before and has stopped. A node with no snapshot at all is not
dead, it is new: a deploy records its placement immediately while the first
reconcile publish is up to one cycle away, and filtering on absence would
unroute a brand-new deployment during exactly that window.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from typing import Any

# A published node snapshot is trusted while it is at most this many reconcile
# intervals old. If the reconciler dies, its last snapshot must not keep
# backing "observed_available: true" forever.
SNAPSHOT_STALE_INTERVALS = 3.0
# Floor so a very short dev interval (e.g. 1s) does not flap the gate on one
# slow DB round-trip.
SNAPSHOT_STALE_FLOOR_S = 30.0


def reconcile_interval_s() -> float:
    """The reconciler's cycle interval — same env knob ``worker.py`` reads."""
    try:
        interval = float(os.getenv("DOCIE_SERVING_RECONCILE_INTERVAL", "10"))
    except ValueError:
        return 10.0
    return interval if interval > 0 else 10.0


def snapshot_stale_after_s() -> float:
    return max(SNAPSHOT_STALE_INTERVALS * reconcile_interval_s(), SNAPSHOT_STALE_FLOOR_S)


def snapshot_age_s(
    snapshot: Mapping[str, Any], *, now: dt.datetime | None = None
) -> float | None:
    """Seconds since the snapshot's ``updated_at`` (None when unparseable)."""
    raw = snapshot.get("updated_at")
    if not raw:
        return None
    try:
        stamp = dt.datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        # sqlite round-trips the reconciler's UTC stamp as a naive datetime.
        stamp = stamp.replace(tzinfo=dt.UTC)
    current = now if now is not None else dt.datetime.now(dt.UTC)
    return (current - stamp).total_seconds()


def is_snapshot_fresh(
    snapshot: Mapping[str, Any], *, now: dt.datetime | None = None
) -> bool:
    """Whether this node published recently enough to be considered reporting.

    An unparseable or missing ``updated_at`` counts as fresh: the stamp is
    always written by ``publish_node_snapshot``, and treating a formatting
    quirk as a dead node would unroute a healthy one.
    """
    age = snapshot_age_s(snapshot, now=now)
    return age is None or age <= snapshot_stale_after_s()


def stale_node_ids(
    snapshots: object, *, now: dt.datetime | None = None
) -> frozenset[str]:
    """The node ids that published before and have since stopped.

    Deliberately the STALE set rather than the live one: a caller filtering on
    "not in the live set" would also drop nodes that have never published,
    which are new rather than gone. Only provable death belongs here.
    """
    if not isinstance(snapshots, (list, tuple)):
        return frozenset()
    stale: set[str] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        node_id = str(snapshot.get("node_id") or "")
        if node_id and not is_snapshot_fresh(snapshot, now=now):
            stale.add(node_id)
    return frozenset(stale)


__all__ = [
    "SNAPSHOT_STALE_FLOOR_S",
    "SNAPSHOT_STALE_INTERVALS",
    "is_snapshot_fresh",
    "reconcile_interval_s",
    "snapshot_age_s",
    "snapshot_stale_after_s",
    "stale_node_ids",
]
