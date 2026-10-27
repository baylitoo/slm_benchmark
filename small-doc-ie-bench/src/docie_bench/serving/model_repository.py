"""The shared model repository: publish here, materialise there (#595 stage 2).

**Why this is not a StorageBackend.** Stage 1 (#597) gave ``ModelStore`` a
seam for where its bytes live locally. It is tempting to make S3 another
implementation of it, but the two are different tiers with different shapes.
A ``StorageBackend`` answers "where is key K on this node"; a repository
answers "what publications of model M exist, and which blobs does one need".
The local store stays the SERVING tier -- ``llama-server`` mmaps a file and
the snapshot families load a directory -- and the repository is the
DISTRIBUTION tier that fills it.

They meet at exactly one point, and stage 1 built it on purpose:
``StorageBackend.write_verified(source, key, expected_digest=...)`` is the
verified cache fill. A materialisation downloads to a temporary file and hands
it to that method with the digest from the manifest, so a corrupt or truncated
transfer can never appear under a canonical name.

Publication order
-----------------

The decided rule, and the reason the layout looks the way it does:

1. upload every blob that is not already present, verifying each,
2. upload the immutable manifest,
3. write ``current.json`` last.

Before step 3 the publication is invisible; after it, complete. An interrupted
publish leaves orphaned blobs and no deployable model -- **garbage, never
corruption** -- and nothing is deleted as rollback, because deleting objects
another publication may share is the genuinely dangerous move.

S3 and Postgres share no transaction, so the catalog upsert is a separate,
idempotent step the caller runs AFTER a successful publication, and its
failure is reported as its own outcome rather than failing the publish.

Read-only credentials are enough to serve: materialising touches ``blobs/``,
``models/<name>/<digest>.json`` and ``current.json`` with GET only. Publishing
is the sole operation needing write.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from docie_bench.serving import manifest as manifest_mod
from docie_bench.serving.manifest import (
    Manifest,
    ManifestFile,
    pointer_document,
    pointer_key,
)

logger = logging.getLogger(__name__)

#: Streamed rather than read whole: a GGUF is measured in gigabytes and the
#: publishing node is also serving.
_CHUNK = 8 * 1024 * 1024


class ModelRepositoryError(RuntimeError):
    """The repository could not be reached, read, or written."""


class ObjectStore(Protocol):
    """The handful of object operations a repository needs.

    Deliberately smaller than any S3 client. A protocol this narrow can be
    implemented by a local directory for tests, which is what lets the
    repository's rules be tested without a bucket -- and it keeps the S3
    specifics in one adapter instead of threading boto3 through the logic.
    """

    def get(self, key: str) -> bytes: ...

    def put(self, key: str, payload: bytes) -> None: ...

    def exists(self, key: str) -> bool: ...

    def download(self, key: str, destination: Path) -> None:
        """Stream ``key`` to ``destination``. Must not leave a partial file."""
        ...

    def upload(self, source: Path, key: str) -> None: ...

    def list_prefix(self, prefix: str) -> list[str]: ...


class ObjectNotFoundError(ModelRepositoryError):
    """A key that must exist does not."""


@dataclass(frozen=True)
class PublishResult:
    """What a publication did, reported per outcome.

    ``uploaded`` and ``skipped`` are separate because "published" with nothing
    uploaded is the normal case for a re-publication of unchanged weights, and
    an operator reading a log should be able to tell that from a no-op.
    """

    name: str
    manifest_digest: str
    manifest_key: str
    uploaded: tuple[str, ...]
    skipped: tuple[str, ...]
    #: Whether the pointer ALREADY named this manifest before the publish.
    was_already_published: bool = False

    @property
    def already_complete(self) -> bool:
        """Whether this publication changed nothing.

        Derived from the pointer rather than from the upload count. A retry
        after an interrupted publication skips every blob -- they were all
        uploaded before the crash -- yet it is the call that makes the model
        visible for the first time, which is the opposite of a no-op.
        """
        return self.was_already_published and not self.uploaded


def local_for(files: Sequence[tuple[str, str, Path]], relative: str) -> Path:
    """The local path a manifest entry came from.

    The publish loop hashes and validates everything first, so by upload time
    it is iterating manifest entries rather than the caller's tuples; this maps
    back. Validation has already rejected duplicate paths, so the first match
    is the only match.
    """
    for _role, path, local in files:
        if path == relative:
            return local
    raise ModelRepositoryError(f"no local file for manifest entry {relative!r}")


def sha256_file(path: Path, *, chunk: int = _CHUNK) -> tuple[str, int]:
    """``(digest, size)`` for ``path``, read once.

    Both in one pass because every caller needs both -- the digest names the
    blob, the size drives the headroom check -- and a multi-GB file should be
    read once rather than twice.
    """
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return f"sha256:{digest.hexdigest()}", size


class ModelRepository:
    """Publish complete models, and resolve what a node should materialise."""

    def __init__(self, store: ObjectStore, *, prefix: str = "") -> None:
        self._store = store
        self._prefix = prefix.strip("/")

    # ------------------------------------------------------------- keys
    def _key(self, suffix: str) -> str:
        return f"{self._prefix}/{suffix}" if self._prefix else suffix

    # ---------------------------------------------------------- publish
    def publish(
        self,
        *,
        name: str,
        family: str,
        kind: str,
        files: Sequence[tuple[str, str, Path]],
        source: str | None = None,
        published_by: str | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> PublishResult:
        """Publish one model. ``files`` is ``(role, relative_path, local_path)``.

        Blobs first, manifest next, pointer last -- see the module docstring
        for why that order is the whole safety property.

        A blob already present is not re-uploaded. That is what makes
        re-publishing a model whose projector did not change cheap, and it is
        safe precisely because blobs are content-addressed: a key that exists
        holds exactly those bytes.
        """
        # Hash and VALIDATE everything before a single byte is uploaded.
        # Building the manifest after the upload loop meant any validation
        # failure -- an unsafe path, a duplicate entry, a kind/role mismatch --
        # fired only once a multi-gigabyte transfer had finished, costing the
        # whole upload and leaving orphaned blobs behind.
        entries: list[ManifestFile] = []
        for role, relative, local in files:
            if not local.is_file():
                raise ModelRepositoryError(f"{local} is not a file")
            digest, size = sha256_file(local)
            entries.append(
                ManifestFile(role=role, path=relative, digest=digest, size_bytes=size)
            )

        manifest = manifest_mod.build(
            name=name,
            family=family,
            kind=kind,
            files=entries,
            source=source,
            published_by=published_by,
        )
        payload = manifest.to_json()
        pointer = self._key(pointer_key(name))

        # What the pointer names right now, so "already published" can be told
        # apart from "every blob happened to be present" -- a retry after an
        # interrupted publication skips every blob and is NOT a no-op.
        previously: str | None = None
        try:
            previously = manifest_mod.parse_pointer(self._store.get(pointer))
        except (ObjectNotFoundError, manifest_mod.InvalidManifestError):
            previously = None

        uploaded: list[str] = []
        skipped: list[str] = []
        for entry in manifest.files:
            key = self._key(entry.blob_key)
            if self._store.exists(key):
                skipped.append(entry.path)
                continue
            if progress is not None:
                progress(entry.path, 0, entry.size_bytes)
            self._store.upload(local_for(files, entry.path), key)
            uploaded.append(entry.path)
            if progress is not None:
                progress(entry.path, entry.size_bytes, entry.size_bytes)

        digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        key = self._key(manifest.manifest_key(digest))
        self._store.put(key, payload)

        # LAST. Everything above is inert until this exists.
        self._store.put(pointer, pointer_document(digest))

        return PublishResult(
            name=name,
            manifest_digest=digest,
            manifest_key=key,
            uploaded=tuple(uploaded),
            skipped=tuple(skipped),
            was_already_published=previously == digest,
        )

    # ---------------------------------------------------------- resolve
    def resolve(self, name: str) -> str:
        """The manifest digest ``name`` currently points at.

        Called ONCE, when a deployment's intent is accepted. The digest is
        then pinned in that intent, so materialisation and every retry use the
        same build and a later publication cannot change what a running node
        serves.
        """
        try:
            raw = self._store.get(self._key(pointer_key(name)))
        except ObjectNotFoundError:
            raise ObjectNotFoundError(
                f"model {name!r} is not published in this repository "
                f"(no {pointer_key(name)})"
            ) from None
        return manifest_mod.parse_pointer(raw)

    def manifest(self, name: str, digest: str) -> Manifest:
        """Fetch one immutable manifest by digest, verifying it is that one.

        The bytes are hashed and compared BEFORE parsing. The whole design
        rests on manifests being content-addressed -- it is what lets a
        deployment pin a build and know a republish cannot change it -- and
        that property only holds if the digest is checked on read.

        Blobs were verified from the start; manifests were not, and the gap
        was the dangerous half: ``fetch_blob`` verifies each blob against the
        manifest it was handed, so a substituted manifest passes every
        integrity check and the node serves a different model than the one its
        pinned digest named.
        """
        hex_tail = manifest_mod.digest_hex(digest)
        key = self._key(f"{manifest_mod.MODEL_PREFIX}/{name}/{hex_tail}.json")
        raw = self._store.get(key)
        got = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        if got != digest:
            raise ModelRepositoryError(
                f"manifest at {key} hashes to {got}, not the {digest} that was "
                f"requested -- the object does not hold the build it is named "
                f"for"
            )
        parsed = manifest_mod.parse(raw)
        if parsed.name != name:
            raise ModelRepositoryError(
                f"manifest at {key} names {parsed.name!r}, not {name!r}"
            )
        return parsed

    def published_names(self) -> list[str]:
        """Models with a pointer. A blob without one is not a model."""
        prefix = self._key(manifest_mod.MODEL_PREFIX) + "/"
        names = []
        for key in self._store.list_prefix(prefix):
            tail = key[len(prefix) :]
            if tail.endswith("/" + manifest_mod.POINTER_NAME):
                names.append(tail[: -len("/" + manifest_mod.POINTER_NAME)])
        return sorted(names)

    # ------------------------------------------------------- materialise
    def fetch_blob(self, entry: ManifestFile, destination: Path) -> None:
        """Download one artifact and verify it before it is usable.

        Streamed to a temporary file beside the destination and renamed only
        on a digest match, so a truncated or corrupted transfer never leaves
        something that looks valid. A corrupt object that a runtime mmaps
        fails like a bad model rather than a bad download, which is why this
        is not optional.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A UNIQUE temp name. A fixed ".partial" meant two concurrent pulls of
        # the same model on one node wrote to the same file: each would hash
        # bytes the other was still writing, both fail, and the first one's
        # cleanup would delete the second's transfer.
        handle, staged = tempfile.mkstemp(
            dir=destination.parent, prefix=destination.name + ".", suffix=".partial"
        )
        os.close(handle)
        tmp = Path(staged)
        try:
            self._store.download(self._key(entry.blob_key), tmp)
            got, size = sha256_file(tmp)
            if got != entry.digest:
                raise ModelRepositoryError(
                    f"integrity check failed for {entry.path}: got {got} != "
                    f"{entry.digest} (repository object "
                    f"{self._key(entry.blob_key)})"
                )
            if size != entry.size_bytes:
                raise ModelRepositoryError(
                    f"size mismatch for {entry.path}: got {size} bytes, "
                    f"manifest says {entry.size_bytes}"
                )
            os.replace(tmp, destination)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


# ---------------------------------------------------------------------------
# Object stores
# ---------------------------------------------------------------------------


class LocalObjectStore:
    """A directory pretending to be a bucket.

    Not a test double for its own sake: it makes every rule in
    ``ModelRepository`` -- publication order, dedup, verification, what counts
    as published -- testable without a network, so the MinIO run is left to
    prove the S3 adapter rather than the logic.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ModelRepositoryError(f"key {key!r} escapes {root}")
        return candidate

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(key)
        return path.read_bytes()

    def put(self, key: str, payload: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def download(self, key: str, destination: Path) -> None:
        source = self._path(key)
        if not source.is_file():
            raise ObjectNotFoundError(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def upload(self, source: Path, key: str) -> None:
        """Copy into place atomically.

        A bare ``copyfile`` to the canonical key meant a crash mid-copy left a
        TRUNCATED blob under a digest's name. Every later publish would then
        skip it as present, and every fetch would fail its integrity check --
        a permanently unfetchable model with no way to self-heal, because
        presence is taken as proof of content.
        """
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, staged = tempfile.mkstemp(dir=path.parent, suffix=".incoming")
        os.close(handle)
        try:
            shutil.copyfile(source, staged)
            os.replace(staged, path)
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise

    def list_prefix(self, prefix: str) -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.is_dir():
            return []
        root = self.root.resolve()
        return sorted(
            p.resolve().relative_to(root).as_posix()
            for p in base.rglob("*")
            if p.is_file()
        )


class S3ObjectStore:
    """An S3-compatible bucket (AWS, MinIO, R2, B2).

    ``boto3`` is imported lazily and is an optional dependency: a deployment
    that never configures a repository must not be forced to install it, and
    the import error should name the extra rather than surface as a traceback
    from a module nobody asked for.
    """

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self._client = client if client is not None else _s3_client(endpoint_url, region)

    def get(self, key: str) -> bytes:
        try:
            return bytes(self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read())
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _mapped(exc, key) from exc

    def put(self, key: str, payload: bytes) -> None:
        try:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=payload)
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _mapped(exc, key) from exc

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:  # noqa: BLE001 - absence is an answer, not a failure
            if _is_missing(exc):
                return False
            raise _mapped(exc, key) from exc

    def download(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Through a temporary file so an interrupted transfer never leaves a
        # short file at the destination the caller is about to hash.
        fd, staged = tempfile.mkstemp(dir=destination.parent, suffix=".s3")
        try:
            with os.fdopen(fd, "wb") as handle:
                self._client.download_fileobj(self.bucket, key, handle)
            os.replace(staged, destination)
        except Exception as exc:  # noqa: BLE001 - mapped below
            Path(staged).unlink(missing_ok=True)
            raise _mapped(exc, key) from exc
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise

    def upload(self, source: Path, key: str) -> None:
        try:
            self._client.upload_file(str(source), self.bucket, key)
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _mapped(exc, key) from exc

    def list_prefix(self, prefix: str) -> list[str]:
        keys: list[str] = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                keys.extend(item["Key"] for item in page.get("Contents", []))
        except Exception as exc:  # noqa: BLE001 - mapped below
            raise _mapped(exc, prefix) from exc
        return sorted(keys)


def _s3_client(endpoint_url: str | None, region: str | None) -> Any:
    try:
        import boto3  # noqa: PLC0415 - optional dependency, imported on use
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ModelRepositoryError(
            "the model repository needs boto3; install the 's3' extra "
            "(pip install -e '.[s3]') on any node that publishes or pulls"
        ) from exc
    return boto3.client("s3", endpoint_url=endpoint_url, region_name=region)


def _is_missing(exc: BaseException) -> bool:
    code = getattr(getattr(exc, "response", None), "get", lambda _k, _d=None: None)(
        "Error", {}
    )
    status = (code or {}).get("Code") if isinstance(code, dict) else None
    return str(status) in {"404", "NoSuchKey", "NotFound"}


def _mapped(exc: BaseException, key: str) -> ModelRepositoryError:
    """Turn a client error into something an operator can act on.

    Access denied and "no such key" are the two an operator will actually hit,
    and they call for opposite responses -- fix the credentials versus publish
    the model -- so they must not arrive as the same opaque failure.
    """
    if _is_missing(exc):
        return ObjectNotFoundError(f"{key} is not in the repository")
    message = str(exc)
    if "AccessDenied" in message or "403" in message:
        return ModelRepositoryError(
            f"access denied for {key}. Serving nodes need read access; "
            f"publishing additionally needs write."
        )
    return ModelRepositoryError(f"repository operation failed for {key}: {exc}")


def repository_from_settings() -> ModelRepository | None:
    """The configured repository, or ``None`` when there is not one.

    ``None`` is the ordinary answer, not a failure: a single-machine
    deployment has no shared repository and every path that would use one
    falls back to node-local behaviour. Returning ``None`` rather than raising
    is what keeps that fallback from being an error path.
    """
    from docie_bench.settings import get_settings

    settings = get_settings()
    bucket = getattr(settings, "model_store_s3_bucket", None)
    if not bucket:
        return None
    store = S3ObjectStore(
        bucket,
        endpoint_url=getattr(settings, "model_store_s3_endpoint_url", None),
        region=getattr(settings, "model_store_s3_region", None),
    )
    return ModelRepository(store, prefix=getattr(settings, "model_store_s3_prefix", ""))


def iter_store_files(entry: Any) -> Iterator[tuple[str, str, Path]]:
    """``(role, relative_path, local_path)`` for a local store entry.

    One place that knows how a local model maps onto manifest files, so the
    publish path and any later importer cannot disagree about it.
    """
    model_path = Path(entry.model_path)
    if model_path.is_dir():
        for file in sorted(p for p in model_path.rglob("*") if p.is_file()):
            yield "snapshot", file.relative_to(model_path).as_posix(), file
        return
    yield "model", "model.gguf", model_path
    mmproj = getattr(entry, "mmproj_path", None)
    if mmproj:
        yield "mmproj", "mmproj.gguf", Path(mmproj)


__all__ = [
    "LocalObjectStore",
    "ModelRepository",
    "ModelRepositoryError",
    "ObjectNotFoundError",
    "ObjectStore",
    "PublishResult",
    "S3ObjectStore",
    "iter_store_files",
    "local_for",
    "repository_from_settings",
    "sha256_file",
]
