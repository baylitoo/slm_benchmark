"""Taking a node out of the rotation, and when its runtimes may stop (#536).

Drain is two moments, not one, and conflating them is what makes a naive
implementation drop requests. The operator's intent is "stop sending this
machine work and let me have it back":

1. **Stop admitting.** Immediate, central, and unconditional -- the router
   excludes the node's placements from the moment the flag is written. Nothing
   on the node has to agree, or even be reachable, for this half to work.
2. **Stop the runtimes.** Not immediate. A request admitted an instant before
   the drain was declared is still running, and killing llama-server takes its
   connection with it. So the node waits out a grace period measured from
   ``drained_at`` before it unloads anything.

The grace period comes from the deployments themselves. Each carries a
``timeout_seconds`` (#475), which is how long a request on it is allowed to
take -- so the longest of those is exactly how long an in-flight request could
still legitimately be running. A fixed sleep would be a guess that is either
too short (killing work) or too long (an operator waiting on a machine).

DECLARED, NOT INFERRED. This is the distinction from the staleness rule in
:mod:`docie_bench.serving.node_liveness`, which is deliberately only ever a
*preference*: a node that stopped publishing may still be serving perfectly,
so routing demotes it but will still use it rather than refuse the only
replica. A drain is the opposite kind of fact. An operator said to take this
machine out, so routing excludes it even when it holds the only replica --
otherwise draining the last node serving a model silently does nothing, which
is a worse outcome than the request failing with a reason.

Pure, for the reason :mod:`docie_bench.serving.convergence` is: the cases that
destroy running capacity when wrong are cheaper to be wrong about in a
table-driven test than on a two-node hardware session.
"""

from __future__ import annotations

import datetime as dt
import enum
from collections.abc import Iterable

#: Minimum grace, used when no deployment declares a timeout. Long enough for
#: an ordinary extraction to finish, short enough that an operator draining an
#: idle node is not left waiting on a machine with nothing running.
DRAIN_GRACE_FLOOR_S = 30.0

#: Cap on the computed grace. A deployment may declare a very long timeout for
#: a batch job; honouring it literally would leave a drain apparently stuck for
#: an hour with no way to tell that from a broken reconciler. Past this, the
#: operator's request wins over the longest-running request.
DRAIN_GRACE_CEILING_S = 600.0


class DrainPhase(enum.Enum):
    """Where a node is between admitting traffic and being free to stop."""

    ADMITTING = "admitting"
    """Not drained. Routes normally."""

    DRAINING = "draining"
    """Declared drained, inside the grace period.

    Receives no new requests, and its runtimes stay up so the requests it had
    can finish. This is the state a drain spends most of its visible life in,
    and the UI should name it rather than showing "drained" while the machine
    is still working.
    """

    DRAINED = "drained"
    """Grace elapsed. The node may stop its runtimes and report empty."""


def drain_grace_seconds(
    timeouts: Iterable[float | int | None],
    *,
    floor_s: float = DRAIN_GRACE_FLOOR_S,
    ceiling_s: float = DRAIN_GRACE_CEILING_S,
) -> float:
    """How long to wait before a drained node may stop its runtimes.

    ``timeouts`` is the per-deployment ``timeout_seconds`` of everything this
    node is running; ``None`` entries are deployments that declare no timeout
    and so contribute nothing but the floor.

    The longest declared timeout is the longest a request admitted just before
    the drain could still legitimately run. Clamped into
    ``[floor_s, ceiling_s]``.
    """
    longest = floor_s
    for timeout in timeouts:
        if timeout is None:
            continue
        try:
            value = float(timeout)
        except (TypeError, ValueError):
            continue
        # A non-positive timeout means "no limit" in the launch contract, which
        # cannot be waited out; it contributes the floor like an absent one.
        if value > longest:
            longest = value
    return min(longest, ceiling_s)


def drain_phase(
    *,
    drained_at: dt.datetime | None,
    now: dt.datetime,
    grace_s: float,
) -> DrainPhase:
    """Which phase a node is in, from its drain timestamp.

    A ``drained_at`` in the future -- clock skew between the central API that
    wrote it and the node reading it -- is treated as the drain having just
    started, never as already elapsed. Getting that backwards would unload the
    runtimes immediately and drop the in-flight requests the grace exists for,
    so the safe reading is the conservative one.
    """
    if drained_at is None:
        return DrainPhase.ADMITTING
    elapsed = (now - drained_at).total_seconds()
    if elapsed < 0:
        return DrainPhase.DRAINING
    return DrainPhase.DRAINED if elapsed >= grace_s else DrainPhase.DRAINING


def parse_drained_at(raw: object) -> dt.datetime | None:
    """Read a ``drained_at`` out of a catalog snapshot, tolerating its shapes.

    The value crosses process boundaries as an ISO string (``_node_view`` /
    ``desired_snapshot`` serialise it) but arrives as a ``datetime`` when read
    in-process. An unparseable value is read as NOT drained: inventing a drain
    from a malformed timestamp would unload a healthy node, while ignoring one
    leaves it serving, which is the direction #527 requires erring in.
    """
    if raw is None:
        return None
    if isinstance(raw, dt.datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=dt.UTC)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = dt.datetime.fromisoformat(raw)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    return None


__all__ = [
    "DRAIN_GRACE_CEILING_S",
    "DRAIN_GRACE_FLOOR_S",
    "DrainPhase",
    "drain_grace_seconds",
    "drain_phase",
    "parse_drained_at",
]
