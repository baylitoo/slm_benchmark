"""What this node has already converged on (#534).

Two facts, persisted together beside ``deployments.json``:

* ``generation`` — the last per-node desired generation applied. This is what
  makes the convergence decision possible at all: the catalog's answer is
  byte-identical whether nothing was ever written for this node or everything
  written was lost, and only the node's own history separates them (see
  ``convergence.decide_convergence``).
* ``names`` — which deployments this node created *from desired state*.

``names`` exists for safety, and it is the more important of the two.
Convergence has to be able to remove a deployment the catalog no longer wants,
but a node also holds deployments nobody asked the catalog about: `docie up`
creates one directly, and that is the documented way to deploy on a chosen
node while lifecycle events are not node-addressable. If removal were simply
"local records not in desired state", the first convergence pass would tear
those down. So only names this node previously applied FROM desired state are
ever removal candidates; anything else is left alone.

Durability matters as much as the content. This lives in ``serving_home``
alongside the desired-state snapshot and the node id, and shares their
lifecycle on purpose: a node that lost this file has also lost
``deployments.json``, so it has nothing to tear down and no history to
contradict. Keeping it anywhere less durable would let a node forget what it
applied while still running it — and then read a wiped catalog as a fresh
install.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from docie_bench.serving.paths import serving_home

logger = logging.getLogger("docie_bench.serving.applied_state")

APPLIED_STATE_FILENAME = "applied_desired_state.json"


@dataclass(frozen=True)
class AppliedState:
    """The generation this node applied, and what it created doing so."""

    generation: int = 0
    names: frozenset[str] = field(default_factory=frozenset)


def applied_state_path() -> Path:
    return serving_home() / APPLIED_STATE_FILENAME


def read_applied_state() -> AppliedState:
    """What this node last applied, or a zero state when it has never applied.

    A missing or unreadable file reads as "nothing applied", which is the safe
    direction: generation 0 means the convergence decision treats an empty
    catalog as "nothing written yet" rather than "everything was lost", and an
    empty name set means nothing is a removal candidate.
    """
    try:
        raw = applied_state_path().read_text(encoding="utf-8")
    except OSError:
        return AppliedState()
    try:
        decoded = json.loads(raw)
    except ValueError:
        logger.warning("applied-state file is not valid JSON; treating as empty")
        return AppliedState()
    if not isinstance(decoded, dict):
        return AppliedState()
    generation = decoded.get("generation")
    names = decoded.get("names")
    return AppliedState(
        generation=generation if isinstance(generation, int) and generation >= 0 else 0,
        names=frozenset(str(name) for name in names) if isinstance(names, list) else frozenset(),
    )


def write_applied_state(state: AppliedState) -> None:
    """Persist atomically: temp file then replace, like the other state files.

    A partially written file would be read as "nothing applied", which is safe
    but would make the node re-converge from scratch, so the rename matters.
    """
    path = applied_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    payload = {"generation": state.generation, "names": sorted(state.names)}
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "APPLIED_STATE_FILENAME",
    "AppliedState",
    "applied_state_path",
    "read_applied_state",
    "write_applied_state",
]
