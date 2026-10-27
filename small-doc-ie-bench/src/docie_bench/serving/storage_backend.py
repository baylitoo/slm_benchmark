"""Where model bytes durably live: local disk today, a bucket tomorrow (#595).

Stage 1 of #595, and deliberately the boring half. A serving node gets its
weights by hand today -- you shell into the machine and seed it -- which is N
manual operations for N nodes. The fix is a shared repository with a local
cache per node, and this module is the seam that makes that possible without
yet changing anything.

**The constraint that shapes the interface.** A bucket can never be where a
model is served from: ``llama-server`` mmaps a local GGUF, and the snapshot
families (``encoder``, ``transformers``, ``multi_vector``,
``dense_embedding``) load a local directory. Neither streams weights from
object storage. So a remote backend is a *distribution* tier and local disk
stays the *serving* tier -- which is why the read side of this protocol is
:meth:`StorageBackend.resolve_local_path`, "give me a real filesystem path",
rather than "give me bytes". A remote implementation materializes on the way.

**What this stage does NOT do.** No remote backend, no configuration, no
behaviour change. ``ModelStore`` delegates its transfers here and
:class:`LocalDiskBackend` performs them exactly as ``model_store`` did --
the hard-link-or-copy, the verified atomic publish, the tree staging, all
moved rather than rewritten. The interface is shaped so stage 2 can add
digest-addressed manifests, bounded materialization and concurrent-pull
dedup without redesigning it; none of that is here.

Lifted from the interface in #206, which had this right a year ago, and
applied to today's ``model_store.py`` rather than rebasing that branch (218
commits of drift, and the multi-node work moved the file under it).

**Errors.** This module raises :class:`StorageBackendError`. ``ModelStore``
translates that to its own ``ModelStoreError`` so its callers see the
contract they always saw -- a storage seam must not change what a store
failure looks like from the outside.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Protocol


class StorageBackendError(RuntimeError):
    """A transfer could not be completed, or completed with the wrong bytes."""


class StorageBackend(Protocol):
    """What ``ModelStore`` needs from wherever model bytes durably live.

    Keys are POSIX-relative paths beneath the store root -- ``"<name>/model.gguf"``,
    ``"<name>/mmproj.gguf"``, ``"<name>/snapshot"``. They were already the
    on-disk layout, and they map to object keys unchanged, which is the point.
    """

    def write_verified(
        self,
        source: Path,
        key: str,
        *,
        link: bool,
        expected_digest: str | None,
    ) -> None:
        """Durably store ``source`` under ``key``, verified by sha256.

        Must be atomic as observed by a reader: ``key`` either does not exist
        or holds fully-written, verified bytes. Never both, and never
        partially.
        """
        ...

    def write_tree(self, source_dir: Path, key_prefix: str, *, link: bool) -> None:
        """Durably store every file under ``source_dir`` beneath ``key_prefix``.

        Atomic as a whole: a half-transferred tree must never be visible under
        the canonical prefix. Snapshot families are multi-file checkpoints
        loaded by ``from_pretrained``, and a partial one loads and then answers
        badly, which is worse than failing.
        """
        ...

    def resolve_local_path(self, key: str) -> Path:
        """A real filesystem path holding ``key``'s bytes.

        THE method that makes a remote backend possible: a remote
        implementation materializes the object locally and returns where it
        put it, because the runtime needs a file it can mmap. Stage 2 gives
        this a bounded execution path and progress reporting -- it must not
        silently block a request or a reconcile cycle for minutes (#595).
        """
        ...

    def resolve_local_dir(self, key_prefix: str) -> Path:
        """Directory form of :meth:`resolve_local_path`, for a snapshot tree."""
        ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None:
        """Remove everything under ``key``, object or tree. Missing is not an error."""
        ...

    def list_prefix(self, key_prefix: str) -> list[str]: ...


def _sha256_file(path: Path) -> str:
    """Chunked sha256 of ``path`` as bare lowercase hex (no ``sha256:`` prefix).

    Mirrors ``extract.service.hash_file`` but keeps the serving layer
    self-contained (no cross-layer import) and returns bare hex so it compares
    directly against an Ollama manifest digest's hex tail.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hex(digest: str | None) -> str | None:
    """Hex tail of a canonical ``sha256:<hex>`` digest, else ``None``.

    ``None`` means "no canonical digest to verify against" -- the caller falls
    back to a best-effort copy-fidelity check. Guarding on the ``sha256:``
    prefix means a non-sha256 manifest digest (should Ollama ever emit one)
    skips the canonical comparison rather than false-failing an otherwise valid
    transfer.
    """
    prefix = "sha256:"
    if digest and digest.startswith(prefix):
        return digest[len(prefix) :]
    return None


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)  # hard link: zero extra disk, instant
    except OSError:
        shutil.copy2(source, destination)  # cross-device or unsupported FS


def _transfer(source: Path, destination: Path, *, link: bool) -> None:
    if link:
        _link_or_copy(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


class LocalDiskBackend:
    """Bytes directly under ``root`` -- today's ``ModelStore`` behaviour.

    The transfer logic here is ``model_store``'s, moved rather than rewritten:
    the hard-link-or-copy, the ``*.tmp``-then-verify-then-``os.replace``
    publish, and the tree staging. It is the most safety-critical code in the
    store (it is what stands between a corrupt download and a model that loads
    and answers badly), so stage 1 does not touch its behaviour.

    ``resolve_local_path`` is a pure path join: the bytes are already local, so
    there is nothing to materialize. That is the whole difference between this
    and a remote backend.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        """``root / key``, refusing a key that escapes the root.

        Keys derive from store names, which are attacker-influenced strings
        (``ModelStore`` guards them too). Containment is checked in both places
        on purpose: this one travels with the backend, so a future caller that
        forgets the store's guard still cannot write outside the root.
        """
        candidate = (self.root / key).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise StorageBackendError(
                f"refusing key {key!r}: resolved path escapes {root} "
                f"(possible path traversal)"
            )
        return candidate

    def write_verified(
        self,
        source: Path,
        key: str,
        *,
        link: bool,
        expected_digest: str | None,
    ) -> None:
        """Transfer ``source`` to ``key`` and refuse a mislabeled/corrupt result.

        Writes into a sibling ``*.tmp`` first, sha256s it, and ``os.replace``
        moves it onto the canonical name ONLY when the hash matches. So the
        canonical file never appears unless it is fully written and verified;
        on any mismatch or error the tmp is removed and
        :class:`StorageBackendError` is raised, leaving no partial or
        mislabeled file behind.

        Verification target:
          * ``expected_digest`` is a canonical ``sha256:<hex>`` (an Ollama
            manifest layer digest) -> the destination is checked against the
            CANONICAL content hash. This catches a wrong-sha even for hard
            links, i.e. a corrupt SOURCE blob, because a same-inode tmp still
            hashes to the source's real (wrong) bytes.
          * ``expected_digest`` is ``None`` (or non-sha256) -> best-effort
            copy-fidelity only (source hash == dest hash). For a hard link this
            is trivially true (same inode); for a copy it catches a truncated
            or altered copy. There is no canonical digest to assert here (the
            HF trust boundary).
        """
        destination = self._path(key)
        tmp = destination.with_name(destination.name + ".tmp")
        try:
            _transfer(source, tmp, link=link)
            got = _sha256_file(tmp)
            want = _canonical_hex(expected_digest)
            canonical = want is not None
            if want is None:
                want = _sha256_file(source)  # copy-fidelity fallback
            if got.lower() != want.lower():
                kind = "manifest digest" if canonical else "source copy"
                raise StorageBackendError(
                    f"blob integrity check failed for {destination.name}: "
                    f"got sha256:{got} != want sha256:{want} ({kind}; source={source})"
                )
            os.replace(tmp, destination)
        except BaseException:
            # Never leave a half-written or mislabeled tmp behind, even on
            # KeyboardInterrupt / a mid-transfer crash.
            tmp.unlink(missing_ok=True)
            raise

    def write_tree(self, source_dir: Path, key_prefix: str, *, link: bool) -> None:
        """Transfer a whole directory to ``key_prefix``, atomically.

        Staged in a ``*.tmp`` sibling and ``os.replace``d, so a half-copied
        snapshot never appears under the canonical name. The caller keeps its
        own rollback for anything beyond this prefix -- whether an orphaned
        parent directory should be removed depends on store state this backend
        does not see.
        """
        destination = self._path(key_prefix)
        staging = destination.with_name(destination.name + ".tmp")
        superseded = destination.with_name(destination.name + ".superseded")
        try:
            if staging.exists():
                shutil.rmtree(staging)
            if superseded.exists():
                shutil.rmtree(superseded)
            for file in sorted(p for p in source_dir.rglob("*") if p.is_file()):
                target = staging / file.relative_to(source_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                _transfer(file, target, link=link)

            # Move the existing tree ASIDE rather than deleting it.
            #
            # The first version removed the destination and then replaced it,
            # which left a window where a node had neither: if the replace
            # failed -- a cross-device rename, a permissions error, or on
            # Windows simply a running encoder holding the snapshot directory
            # open -- the working model was already gone, and the handler below
            # then removed the staged replacement too. Re-seeding a model a
            # node was serving could destroy it.
            #
            # A rename is atomic and reversible, so the old tree survives until
            # the new one is in place.
            replaced = False
            if destination.exists():
                os.replace(destination, superseded)
                replaced = True
            try:
                os.replace(staging, destination)
            except BaseException:
                if replaced:
                    os.replace(superseded, destination)
                raise
            shutil.rmtree(superseded, ignore_errors=True)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def resolve_local_path(self, key: str) -> Path:
        """Already local: a path join, with nothing to materialize."""
        return self._path(key)

    def resolve_local_dir(self, key_prefix: str) -> Path:
        return self._path(key_prefix)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        target = self._path(key)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)

    def list_prefix(self, key_prefix: str) -> list[str]:
        base = self._path(key_prefix)
        if not base.is_dir():
            return []
        root = self.root.resolve()
        return sorted(
            str(path.resolve().relative_to(root).as_posix())
            for path in base.rglob("*")
            if path.is_file()
        )


__all__ = [
    "LocalDiskBackend",
    "StorageBackend",
    "StorageBackendError",
]
