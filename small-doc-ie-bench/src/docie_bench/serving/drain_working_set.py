"""What a drain stopped, so a resume can put it back (#634).

A drain is reversible by design: an operator takes a machine out, replaces its
container or reboots it, and admits it again. The acceptance criterion for that
second half is "resuming restores the deployments the drain unloaded, with no
redeploy" -- which needs the node to remember which ones those were.

**Why the record alone cannot answer it.** A drain stops its runtimes through
``supervisor.stop``, which preserves the record, its port reservation and its
placement row. So after a drain the node holds a set of stopped records, and
nothing in them separates:

* a deployment the drain stopped, which must come back;
* one an operator stopped before the maintenance, which must stay down;
* one the autoloader evicted for memory, which a request reloads on demand and
  which convergence must not pre-empt.

``activation`` distinguishes the third (``MANAGED``) but not the first two:
``stop`` sets ``MANUAL`` whoever called it. A node that restored every stopped
record would start deployments the operator had deliberately turned off.

So the working set is recorded at drain time, as its own fact, and the drain is
the only thing that writes it.

Lives in ``serving_home`` beside ``deployments.json`` and the applied-state
file, and shares their lifecycle for the same reason: a node that lost this
file has also lost the records it refers to, so there is nothing to restore and
no history to contradict. A replaced container with the same volume -- which is
the case this exists for -- keeps both.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from uuid import uuid4

from docie_bench.serving.paths import serving_home

logger = logging.getLogger("docie_bench.serving.drain_working_set")

DRAIN_WORKING_SET_FILENAME = "drain_working_set.json"


def drain_working_set_path() -> Path:
    return serving_home() / DRAIN_WORKING_SET_FILENAME


def read_drain_working_set() -> frozenset[str]:
    """Deployments a drain stopped and a resume has not yet restored.

    Empty when there is no pending restore, and empty when the file cannot be
    read. Erring towards empty means a resume starts nothing it cannot justify:
    the cost is a deployment left stopped that an operator can start by hand,
    against starting one somebody deliberately turned off.
    """
    try:
        raw = drain_working_set_path().read_text(encoding="utf-8")
    except OSError:
        return frozenset()
    try:
        decoded = json.loads(raw)
    except ValueError:
        logger.warning("drain working set is not valid JSON; treating as empty")
        return frozenset()
    if not isinstance(decoded, dict):
        return frozenset()
    names = decoded.get("names")
    if not isinstance(names, list):
        return frozenset()
    return frozenset(str(name) for name in names if str(name))


def write_drain_working_set(names: frozenset[str] | set[str]) -> None:
    """Record what to restore, atomically. An empty set removes the file."""
    path = drain_working_set_path()
    if not names:
        clear_drain_working_set()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps({"names": sorted(names)}, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def clear_drain_working_set() -> None:
    """Forget the pending restore: there is nothing left to put back."""
    drain_working_set_path().unlink(missing_ok=True)


__all__ = [
    "DRAIN_WORKING_SET_FILENAME",
    "clear_drain_working_set",
    "drain_working_set_path",
    "read_drain_working_set",
    "write_drain_working_set",
]
