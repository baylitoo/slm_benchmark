"""Pluggable storage backend for the model store — local disk today, S3-compatible tomorrow.

``ModelStore``'s blob transfer (model_store.py) is currently LOCAL-DISK-SPECIFIC:
hard link, sha256-verify-then-``os.replace`` so the canonical path only ever
appears fully written. A bucket (S3, MinIO, R2, B2 — anything speaking the S3
API) cannot be hard-linked into or served from directly: ``llama-server``
needs a real filesystem path. So a bucket backend is necessarily two-tier —
the bucket is the durable store, and reading a model MATERIALIZES the object
to a local cache path before anything can launch it. That split is the reason
this is a distinct module rather than a few ``if backend == "s3"`` branches
inside ``model_store.py``.

``StorageBackend`` is the seam. ``LocalDiskBackend`` is today's behavior
(the hard-link/atomic-verify transfer extracted verbatim from
``model_store.py``, not rewritten) — selected by default, so nothing about
existing seed/deploy paths changes. ``S3CompatibleBackend`` (boto3, a
configurable ``endpoint_url`` so it works against AWS S3, MinIO, Cloudflare
R2 or Backblaze B2 alike) is implemented and unit-tested against a mocked S3
(moto), but is opt-in via ``DOCIE_STORAGE_BACKEND=s3`` and has not been
exercised against a live bucket. Wiring it into ``ModelStore`` as a real
alternative to local disk, and the self-hosted-MinIO deployment story, are
follow-up work.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client


class StorageBackendError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StorageBackend(Protocol):
    """What ``ModelStore`` needs from wherever model bytes durably live.

    ``key`` is a store-relative path (e.g. ``"nuextract3/model.gguf"``,
    posix separators) — never an absolute filesystem path. Backends translate
    it into whatever addressing they use (a joined local path; a bucket
    object key).
    """

    def write_verified(
        self, source: Path, key: str, *, link: bool = True, expected_digest: str | None = None
    ) -> None:
        """Durably store ``source`` under ``key``, verified by sha256.

        ``expected_digest`` is a canonical ``sha256:<hex>`` to check the
        written bytes against (an Ollama manifest digest); ``None`` falls
        back to copy-fidelity (written bytes == ``source`` bytes) — the same
        two-tier verification ``model_store._transfer_verified`` already
        does. Must raise ``StorageBackendError`` and leave no partial/mislabeled
        object under ``key`` on any mismatch or error.
        """
        ...

    def write_tree(self, source_dir: Path, key_prefix: str, *, link: bool = True) -> None:
        """Durably store every file under ``source_dir`` beneath ``key_prefix``.

        Must be all-or-nothing: a failure partway through leaves nothing new
        reachable under ``key_prefix``.
        """
        ...

    def resolve_local_path(self, key: str) -> Path:
        """Return a real filesystem path holding ``key``'s bytes, materializing
        (downloading) it first if the backend isn't already local-disk-backed.

        This is what a caller hands to ``llama-server --model`` or
        ``from_pretrained(...)`` — never the bucket key itself.
        """
        ...

    def resolve_local_dir(self, key_prefix: str) -> Path:
        """Directory form of ``resolve_local_path`` for a snapshot tree."""
        ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None:
        """Remove everything stored under ``key`` (a single object or a tree
        prefix). Missing ``key`` is not an error."""
        ...

    def list_prefix(self, key_prefix: str) -> list[str]: ...


class LocalDiskBackend:
    """Store bytes directly under ``root`` — today's ``ModelStore`` behavior.

    Extracted verbatim from ``model_store._transfer`` /
    ``model_store._transfer_verified``: hard-link when possible (zero extra
    disk, instant), copy on cross-device/unsupported filesystems, and never
    let the canonical ``key`` path appear until it is written to a sibling
    ``*.tmp`` and sha256-verified.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def write_verified(
        self, source: Path, key: str, *, link: bool = True, expected_digest: str | None = None
    ) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + ".tmp")
        try:
            self._transfer(source, tmp, link=link)
            got = _sha256_file(tmp)
            want = _canonical_hex(expected_digest)
            canonical = want is not None
            if want is None:
                want = _sha256_file(source)  # copy-fidelity fallback
            if got.lower() != want.lower():
                kind = "manifest digest" if canonical else "source copy"
                raise StorageBackendError(
                    f"blob integrity check failed for {key}: "
                    f"got sha256:{got} != want sha256:{want} ({kind}; source={source})"
                )
            os.replace(tmp, destination)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def write_tree(self, source_dir: Path, key_prefix: str, *, link: bool = True) -> None:
        destination = self._path(key_prefix)
        staging = destination.with_name(destination.name + ".tmp")
        try:
            if staging.exists():
                shutil.rmtree(staging)
            for file in sorted(p for p in source_dir.rglob("*") if p.is_file()):
                target = staging / file.relative_to(source_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                self._transfer(file, target, link=link)
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def resolve_local_path(self, key: str) -> Path:
        return self._path(key)

    def resolve_local_dir(self, key_prefix: str) -> Path:
        return self._path(key_prefix)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    def list_prefix(self, key_prefix: str) -> list[str]:
        base = self._path(key_prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [key_prefix]
        return sorted(
            (base / f).relative_to(self.root).as_posix()
            for f in (p.relative_to(base) for p in base.rglob("*") if p.is_file())
        )

    @staticmethod
    def _transfer(source: Path, destination: Path, *, link: bool) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        if link:
            try:
                os.link(source, destination)  # hard link: zero extra disk, instant
                return
            except OSError:
                pass  # cross-device or unsupported FS: fall through to copy
        shutil.copy2(source, destination)


class S3CompatibleBackend:
    """Store bytes in an S3-API bucket — AWS S3, MinIO, Cloudflare R2, Backblaze B2.

    Any of those work through the same client because ``endpoint_url`` is the
    only thing that differs between them; AWS itself is just the case where
    ``endpoint_url`` is left unset. Credentials are resolved by boto3's normal
    chain (env vars, shared config, instance profile) — never passed as
    plaintext arguments here.

    Bucket objects have no native content hash we control (S3 ETag is not
    sha256, and is NOT even a whole-object hash for multipart uploads), so a
    sha256 is stored as object metadata (``x-amz-meta-sha256``) at write time
    and re-checked after every download.

    ``resolve_local_path``/``resolve_local_dir`` materialize into
    ``cache_dir`` and skip the download when a same-named file already
    verifies against the stored digest — the two-tier design this module's
    docstring describes. Requires the ``storage`` extra (``boto3``); not
    imported unless this class is actually instantiated.
    """

    def __init__(
        self,
        *,
        bucket: str,
        cache_dir: str | Path,
        prefix: str = "",
        endpoint_url: str | None = None,
        client: S3Client | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or _new_s3_client(endpoint_url)

    def _object_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key

    def write_verified(
        self, source: Path, key: str, *, link: bool = True, expected_digest: str | None = None
    ) -> None:
        want = _canonical_hex(expected_digest) or _sha256_file(source)
        got = _sha256_file(source)
        if got.lower() != want.lower():
            raise StorageBackendError(
                f"refusing to upload {key}: source sha256:{got} != want sha256:{want}"
            )
        self._client.upload_file(
            str(source),
            self.bucket,
            self._object_key(key),
            ExtraArgs={"Metadata": {"sha256": got}},
        )

    def write_tree(self, source_dir: Path, key_prefix: str, *, link: bool = True) -> None:
        files = sorted(p for p in source_dir.rglob("*") if p.is_file())
        uploaded: list[str] = []
        try:
            for file in files:
                rel = file.relative_to(source_dir).as_posix()
                key = f"{key_prefix}/{rel}"
                self.write_verified(file, key, link=link)
                uploaded.append(key)
        except BaseException:
            for key in uploaded:
                self._client.delete_object(Bucket=self.bucket, Key=self._object_key(key))
            raise

    def resolve_local_path(self, key: str) -> Path:
        cached = self._cache_path(key)
        head = self._client.head_object(Bucket=self.bucket, Key=self._object_key(key))
        want = head.get("Metadata", {}).get("sha256")
        if cached.is_file() and want and _sha256_file(cached).lower() == want.lower():
            return cached  # already materialized and verified — skip the download
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(cached.name + ".tmp")
        try:
            self._client.download_file(self.bucket, self._object_key(key), str(tmp))
            if want:
                got = _sha256_file(tmp)
                if got.lower() != want.lower():
                    raise StorageBackendError(
                        f"download integrity check failed for {key}: "
                        f"got sha256:{got} != want sha256:{want}"
                    )
            os.replace(tmp, cached)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return cached

    def resolve_local_dir(self, key_prefix: str) -> Path:
        for key in self.list_prefix(key_prefix):
            rel = key[len(key_prefix) + 1 :]
            self._materialize_into(key, self._cache_path(key_prefix) / rel)
        return self._cache_path(key_prefix)

    def _materialize_into(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        head = self._client.head_object(Bucket=self.bucket, Key=self._object_key(key))
        want = head.get("Metadata", {}).get("sha256")
        if destination.is_file() and want and _sha256_file(destination).lower() == want.lower():
            return
        tmp = destination.with_name(destination.name + ".tmp")
        self._client.download_file(self.bucket, self._object_key(key), str(tmp))
        os.replace(tmp, destination)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._object_key(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        for found in self.list_prefix(key):
            self._client.delete_object(Bucket=self.bucket, Key=self._object_key(found))
        if not self.list_prefix(key):
            self._client.delete_object(Bucket=self.bucket, Key=self._object_key(key))

    def list_prefix(self, key_prefix: str) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        object_prefix = self._object_key(key_prefix)
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=object_prefix):
            for obj in page.get("Contents", []):
                full_key = obj["Key"]
                keys.append(full_key[len(self.prefix) + 1 :] if self.prefix else full_key)
        return sorted(keys)


def _new_s3_client(endpoint_url: str | None) -> S3Client:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise StorageBackendError(
            "S3CompatibleBackend requires the 'storage' extra: pip install "
            "'small-doc-ie-bench[storage]'"
        ) from exc
    return boto3.client("s3", endpoint_url=endpoint_url)


def _canonical_hex(digest: str | None) -> str | None:
    """Hex tail of a canonical ``sha256:<hex>`` digest, else ``None``.

    Mirrors ``model_store._canonical_hex``: ``None`` means "no canonical
    digest to verify against", not "verification skipped".
    """
    prefix = "sha256:"
    if digest and digest.startswith(prefix):
        return digest[len(prefix) :]
    return None


def resolve_storage_backend(root: str | Path) -> StorageBackend:
    """Build the configured backend. Local disk unless ``DOCIE_STORAGE_BACKEND=s3``.

    ``root`` is always required: it's the local disk root (default backend,
    or path/naming compat) and doubles as the S3 backend's local cache dir
    (``DOCIE_STORAGE_CACHE_DIR`` overrides that if set).
    """
    kind = os.environ.get("DOCIE_STORAGE_BACKEND", "local").strip().lower()
    if kind == "local":
        return LocalDiskBackend(root)
    if kind == "s3":
        bucket = os.environ.get("DOCIE_S3_BUCKET")
        if not bucket:
            raise StorageBackendError("DOCIE_STORAGE_BACKEND=s3 requires DOCIE_S3_BUCKET")
        cache_dir = os.environ.get("DOCIE_STORAGE_CACHE_DIR") or root
        return S3CompatibleBackend(
            bucket=bucket,
            cache_dir=cache_dir,
            prefix=os.environ.get("DOCIE_S3_PREFIX", ""),
            endpoint_url=os.environ.get("DOCIE_S3_ENDPOINT_URL"),
        )
    raise StorageBackendError(
        f"Unknown DOCIE_STORAGE_BACKEND={kind!r}; expected 'local' or 's3'"
    )
