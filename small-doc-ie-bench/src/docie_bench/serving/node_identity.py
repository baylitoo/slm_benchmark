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


class NodeIdUnreadableError(OSError):
    """``node_id`` is on disk but could not be read.

    Distinct from absent and from invalid, both of which mean "mint one". Every
    caller resolves the id inside a broad ``except Exception`` that logs and
    continues, so this surfaces as a skipped publish cycle (and as a doctor
    finding), never as a dead container.
    """


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
    """The id to adopt, or ``None`` when there is nothing valid to adopt.

    Three states, not two, for the same reason the fleet distinguishes
    ``missing`` from ``unreported``. ABSENT and INVALID both mean "mint one":
    a corrupted file is not a reason to refuse to serve. UNREADABLE must not
    collapse into them -- reading an I/O error as "no id here" makes the node
    re-mint its identity, orphaning the ``serving_node`` and placement rows it
    had already published and leaving a node in the Fleet page that never
    reports again. Raising costs publish cycles instead: ``resolve_node_id``'s
    cache does not hold exceptions, so the next cycle retries, and the real id
    is cached as soon as the read succeeds.
    """
    try:
        stored = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise NodeIdUnreadableError(
            f"{path} exists but could not be read ({exc}); refusing to mint a "
            f"second identity for this node"
        ) from exc
    try:
        return validate_node_id(stored)
    except InvalidNodeIdError:
        # A corrupted file is not a reason to refuse to serve; regenerate.
        return None


def _fsync_dir(path: Path) -> None:
    """Make a new directory entry durable. No-op where that is not possible.

    Syncing the temp file only syncs its CONTENTS; the link that gives it its
    name is a directory update of its own. Windows cannot open a directory for
    fsync, and the id is re-minted rather than lost there, so a failure here is
    not worth failing the write over.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _persist(path: Path, node_id: str) -> str:
    """Create-if-absent write; returns the id that is actually on disk.

    Write-temp-then-rename so a reader never sees a partial id, and the loser of
    a race between two serving processes overlapping on one volume (a replaced
    container within the lease window) adopts the winner's id rather than
    overwriting it.

    Synced before the link, because the id is published to Postgres as soon as
    it is minted: a power loss that takes the buffered write but not the
    placement rows would leave the node re-minting on boot, against rows the
    old id still owns.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(node_id)
            handle.flush()
            os.fsync(handle.fileno())
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
            # plain rename. Adopt an id that landed in the meantime rather than
            # clobbering it -- the same rule as the FileExistsError branch, which
            # this one used to skip.
            existing = _read_persisted(path)
            if existing is not None:
                return existing
            temporary.replace(path)
            _fsync_dir(path.parent)
            return _read_persisted(path) or node_id
        _fsync_dir(path.parent)
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
    "NodeIdUnreadableError",
    "node_id_path",
    "resolve_node_id",
    "validate_node_id",
]
