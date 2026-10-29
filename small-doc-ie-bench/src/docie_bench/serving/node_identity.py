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
import time
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
#: Bounded wait for a racing creator's 32-byte write to land.
_CREATE_RACE_ATTEMPTS = 20
_CREATE_RACE_SLEEP_S = 0.01
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


def _create_exclusively(path: Path, node_id: str) -> str:
    """Create ``path`` holding ``node_id``, or adopt whoever got there first.

    Writes the final name directly, which temp-then-rename exists to avoid,
    so the loser cannot assume a file it can see is complete. It waits for
    content that validates instead of minting a second id -- the same rule
    :func:`_read_persisted` applies to an unreadable file, for the same reason:
    a node that cannot read its identity must not invent a new one and orphan
    the rows the old one owns.
    """
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return _await_persisted(path, node_id)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(node_id)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        _fsync_dir(path.parent)
    return node_id


def _await_persisted(path: Path, fallback: str) -> str:
    """The winner's id once its write lands, else claim the corrupt file.

    The wait is bounded and tiny -- an id is 32 bytes -- and happens at most
    once per process, when two creators raced on a filesystem that refused the
    hardlink protocol.

    What follows the wait is the distinction between a racing writer and a dead
    one. A creator still mid-write finishes within the wait; a file that is
    present and STILL invalid afterwards was left by a process killed between
    the create and the write. Writing the final name directly is what makes
    that state reachable at all -- temp-then-rename never leaves a partial id
    -- so this path has to clean up after itself. Refusing instead would leave
    the node unable to resolve an identity at all, which contradicts the rule
    the module already states for a corrupt file: it is not a reason to refuse
    to serve.

    An UNREADABLE file still raises, from ``_read_persisted``. That is the one
    case where regenerating is wrong, because the id may be perfectly good and
    merely unreachable.
    """
    for _ in range(_CREATE_RACE_ATTEMPTS):
        existing = _read_persisted(path)
        if existing is not None:
            return existing
        time.sleep(_CREATE_RACE_SLEEP_S)
    return _replace_corrupt(path, fallback)


def _replace_corrupt(path: Path, node_id: str) -> str:
    """Overwrite a corrupt id file atomically, then re-read what landed.

    Temp-then-rename, so the replacement is complete-or-absent for anyone
    reading concurrently. Two processes both finding corruption both write, and
    the final re-read is what makes them agree on whichever rename landed --
    last-writer-wins, never two live identities.
    """
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(node_id)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return _read_persisted(path) or node_id


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
            # Filesystems without hard links (some bind mounts).
            #
            # O_EXCL rather than read-then-rename: reading, finding nothing and
            # then renaming lets two creators BOTH observe absence and both
            # write, so each returns its own id while the disk keeps only the
            # loser's. Because resolution is cached per process, the two then
            # disagree about who this node is for the rest of their lives --
            # which is the split identity this module exists to prevent.
            # Exclusive creation makes exactly one of them the winner.
            return _create_exclusively(path, node_id)
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
