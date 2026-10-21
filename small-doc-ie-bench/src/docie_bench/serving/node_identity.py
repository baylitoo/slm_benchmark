"""Stable identity for one serving node (#529, phase 1 of #527).

The id keys the node's ``serving_node`` capacity row, and later its placement
rows, so N nodes publishing into one Postgres stay distinguishable instead of
overwriting a single ``"node"`` row.

Generated once and persisted in ``serving_home()``, beside ``deployments.json``
— the node-local, single-writer desired state. That coupling is deliberate: the
id survives a container restart for the same reason desired state does (same
volume, same file), and it is lost exactly when it should be, since a node that
lost the volume also lost the deployments it owned. ``DOCIE_SERVING_NODE_ID``
overrides when an operator wants a readable id.

Hostname is NOT used: compose sets a container's hostname to the service name,
so every node running the project's compose file would call itself ``serving``
and two machines would agree they are the same node.

SERVING ROLE ONLY. ``serving_home()`` is read by all three roles, and the api
container shares the same path; an api-role process generating an id there would
mint a second identity for one machine. Resolution is therefore reached only
from the reconciler's publish path.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from docie_bench.serving.paths import serving_home

NODE_ID_FILENAME = "node_id"
NODE_ID_ENV_VAR = "DOCIE_SERVING_NODE_ID"

# The id is the serving_node primary key (String(32), which uuid4().hex fills
# exactly) and travels through logs and placement rows, so an operator-supplied
# value is restricted rather than sanitized -- a silently truncated id is a
# collision between two nodes, the bug this module exists to prevent.
_MAX_NODE_ID_LEN = 32
_NODE_ID_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class InvalidNodeIdError(ValueError):
    """An operator-supplied ``DOCIE_SERVING_NODE_ID`` is not usable as a key."""


def validate_node_id(value: str) -> str:
    """Return ``value`` if it is usable as the node key, else raise."""
    if not value or len(value) > _MAX_NODE_ID_LEN:
        raise InvalidNodeIdError(
            f"{NODE_ID_ENV_VAR}={value!r} must be 1-{_MAX_NODE_ID_LEN} characters "
            f"(it is the serving_node primary key)"
        )
    if not _NODE_ID_PATTERN.match(value):
        raise InvalidNodeIdError(
            f"{NODE_ID_ENV_VAR}={value!r} must start alphanumeric and contain only "
            f"letters, digits, '.', '_' or '-'"
        )
    return value


def node_id_path() -> Path:
    return serving_home() / NODE_ID_FILENAME


def _read_persisted(path: Path) -> str | None:
    try:
        stored = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return validate_node_id(stored)
    except InvalidNodeIdError:
        # A corrupted file is not a reason to refuse to serve; regenerate.
        return None


def _persist(path: Path, node_id: str) -> str:
    """Create-if-absent write; returns the id that is actually on disk.

    Write-temp-then-rename so a reader never sees a partial id, and the loser of
    a race between two serving processes overlapping on one volume (a replaced
    container within the lease window) adopts the winner's id rather than
    overwriting it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(node_id, encoding="utf-8")
        try:
            # O_EXCL semantics via link: fails if another process got there first.
            os.link(temporary, path)
        except FileExistsError:
            existing = _read_persisted(path)
            if existing is not None:
                return existing
            temporary.replace(path)
        except OSError:
            # Filesystems without hard links (some bind mounts): fall back to a
            # plain rename and let last-writer-wins, then re-read so both
            # processes agree on whatever landed.
            temporary.replace(path)
            return _read_persisted(path) or node_id
    finally:
        temporary.unlink(missing_ok=True)
    return _read_persisted(path) or node_id


@lru_cache(maxsize=1)
def resolve_node_id() -> str:
    """This node's stable id: env override, else persisted, else generated.

    Cached for the process: the file is read once, not every reconcile cycle.
    Call ``resolve_node_id.cache_clear()`` in tests that change the environment.
    """
    override = os.environ.get(NODE_ID_ENV_VAR, "").strip()
    if override:
        return validate_node_id(override)
    path = node_id_path()
    persisted = _read_persisted(path)
    if persisted is not None:
        return persisted
    return _persist(path, uuid4().hex)


__all__ = [
    "NODE_ID_ENV_VAR",
    "NODE_ID_FILENAME",
    "InvalidNodeIdError",
    "node_id_path",
    "resolve_node_id",
    "validate_node_id",
]
