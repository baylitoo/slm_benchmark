from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_PREFIX = "sha256:"
_COPY_CHUNK_SIZE = 1024 * 1024


class RegistryError(RuntimeError):
    """Base error raised by the local model registry."""


class ModelNotFoundError(RegistryError):
    pass


class ModelConflictError(RegistryError):
    pass


class ArtifactVerificationError(RegistryError):
    pass


class UnsafePathError(RegistryError):
    pass


class ModelState(StrEnum):
    DOWNLOADING = "downloading"
    READY = "ready"
    SERVING = "serving"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class TrustPolicy(StrEnum):
    DENY_REMOTE_CODE = "deny_remote_code"
    ALLOW_REMOTE_CODE = "allow_remote_code"


class ArtifactKind(StrEnum):
    WEIGHTS = "weights"
    TOKENIZER = "tokenizer"
    CONFIG = "config"
    CHAT_TEMPLATE = "chat_template"
    OTHER = "other"


class ArtifactManifest(BaseModel):
    """A content-addressed artifact referenced by a model manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=255)
    digest: str
    size_bytes: int = Field(ge=0)
    kind: ArtifactKind = ArtifactKind.OTHER
    source: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("artifact name must be a plain file name")
        return value

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return normalize_digest(value)


class RuntimeCompatibilityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    compatible: bool
    reason: str
    checked_version: str | None = None


class BenchmarkRecord(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    runtime: str
    tokens_per_second: float | None = Field(default=None, ge=0)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    structured_output_validity: float | None = Field(default=None, ge=0, le=1)


class ModelManifest(BaseModel):
    """Portable metadata for one pinned model revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, max_length=512)
    source: str = Field(min_length=1, max_length=2048)
    revision: str = Field(min_length=1, max_length=256)
    license: str | None = Field(default=None, max_length=256)
    trust_policy: TrustPolicy = TrustPolicy.DENY_REMOTE_CODE
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    artifacts: tuple[ArtifactManifest, ...] = ()
    tokenizer: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    chat_template: str | None = None
    generation_defaults: dict[str, Any] = Field(default_factory=dict)
    quantization: str | None = None
    precision: str | None = None
    required_memory_gb: float | None = Field(default=None, ge=0)
    required_disk_gb: float | None = Field(default=None, ge=0)
    context_length: int | None = Field(default=None, ge=1)
    supported_tasks: tuple[str, ...] = ()
    runtime_compatibility: dict[str, RuntimeCompatibilityRecord] = Field(default_factory=dict)
    benchmark_history: tuple[BenchmarkRecord, ...] = ()
    state: ModelState = ModelState.READY
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def canonical_id(self) -> str:
        return self.model_id

    @field_validator("model_id", "source", "revision")
    @classmethod
    def validate_identity_text(cls, value: str) -> str:
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("identity values must be non-empty and may not contain NUL")
        return value

    @field_validator("aliases", "tags", "supported_tasks")
    @classmethod
    def validate_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value or "\x00" in value for value in cleaned):
            raise ValueError("aliases, tags, and tasks must be non-empty")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("aliases, tags, and tasks must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_artifacts(self) -> ModelManifest:
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique")
        return self


class ModelRegistry:
    """Durable local registry with content-addressed artifacts and atomic metadata."""

    def __init__(self, root: str | Path, *, lock_timeout_seconds: float = 10.0) -> None:
        self.root = Path(root).expanduser().resolve()
        self.lock_timeout_seconds = lock_timeout_seconds
        self._manifests_dir = self._safe_path("manifests")
        self._artifacts_dir = self._safe_path("artifacts", "sha256")
        self._locks_dir = self._safe_path("locks")
        self._index_path = self._safe_path("index.json")
        for directory in (self.root, self._manifests_dir, self._artifacts_dir, self._locks_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if not self._index_path.exists():
            with self._lock("index"):
                if not self._index_path.exists():
                    self._atomic_json_write(self._index_path, self._empty_index())

    def register(self, manifest: ModelManifest, *, verify: bool = True) -> ModelManifest:
        with self._lock("index"):
            if verify:
                for artifact in manifest.artifacts:
                    self.verify_artifact(artifact)
            index = self._read_index()
            aliases = index["aliases"]
            alias_owner = aliases.get(manifest.model_id)
            if alias_owner is not None and alias_owner != manifest.model_id:
                raise ModelConflictError(
                    f"Model ID {manifest.model_id!r} conflicts with an alias for {alias_owner!r}"
                )
            for alias in manifest.aliases:
                owner = aliases.get(alias)
                if owner is not None and owner != manifest.model_id:
                    raise ModelConflictError(f"Alias {alias!r} already refers to {owner!r}")
                if alias in index["models"] and alias != manifest.model_id:
                    raise ModelConflictError(f"Alias {alias!r} conflicts with a canonical model ID")

            previous = self._read_manifest_from_index(index, manifest.model_id)
            if previous is not None:
                for alias in previous.aliases:
                    aliases.pop(alias, None)
                for tag in previous.tags:
                    self._remove_tag(index, tag, previous.model_id)

            manifest_file = self._manifest_file(manifest.model_id)
            self._atomic_json_write(manifest_file, manifest.model_dump(mode="json"))
            index["models"][manifest.model_id] = manifest_file.name
            for alias in manifest.aliases:
                aliases[alias] = manifest.model_id
            for tag in manifest.tags:
                members = index["tags"].setdefault(tag, [])
                if manifest.model_id not in members:
                    members.append(manifest.model_id)
                    members.sort()
            self._atomic_json_write(self._index_path, index)
        return manifest

    def import_model(
        self,
        manifest: ModelManifest,
        artifacts: Mapping[str, str | Path],
        *,
        kinds: Mapping[str, ArtifactKind] | None = None,
    ) -> ModelManifest:
        expected = {artifact.name: artifact for artifact in manifest.artifacts}
        missing = set(expected) - set(artifacts)
        if missing:
            raise ArtifactVerificationError(f"Missing artifact sources: {sorted(missing)}")
        imported: list[ArtifactManifest] = []
        for name, source in artifacts.items():
            imported.append(
                self.import_artifact(
                    source,
                    name=name,
                    kind=(kinds or {}).get(
                        name,
                        expected[name].kind if name in expected else ArtifactKind.OTHER,
                    ),
                    expected_digest=expected[name].digest if name in expected else None,
                )
            )
        ready = manifest.model_copy(
            update={"artifacts": tuple(imported), "state": ModelState.READY}
        )
        return self.register(ready)

    def pull(
        self,
        manifest: ModelManifest,
        sources: Mapping[str, str | Path],
        *,
        kinds: Mapping[str, ArtifactKind] | None = None,
    ) -> ModelManifest:
        """Acquire local files or HTTP(S) URLs and register the verified model."""
        local_sources: dict[str, Path] = {}
        temporary: list[Path] = []
        try:
            for name, source in sources.items():
                source_text = str(source)
                if source_text.startswith(("http://", "https://")):
                    target = self._download(source_text)
                    local_sources[name] = target
                    temporary.append(target)
                else:
                    local_sources[name] = Path(source)
            return self.import_model(manifest, local_sources, kinds=kinds)
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)

    def import_artifact(
        self,
        source: str | Path | BinaryIO,
        *,
        name: str | None = None,
        kind: ArtifactKind = ArtifactKind.OTHER,
        expected_digest: str | None = None,
    ) -> ArtifactManifest:
        if hasattr(source, "read"):
            stream = cast(BinaryIO, source)
            artifact_name = name or "artifact.bin"
            close_stream = False
        else:
            source_path = Path(source)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            stream = source_path.open("rb")
            artifact_name = name or source_path.name
            close_stream = True

        temp = self._safe_path("artifacts", f".import-{uuid.uuid4().hex}.tmp")
        digest = hashlib.sha256()
        size = 0
        try:
            try:
                with temp.open("xb") as output:
                    while chunk := stream.read(_COPY_CHUNK_SIZE):
                        digest.update(chunk)
                        size += len(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        finally:
            if close_stream:
                stream.close()

        normalized = _SHA256_PREFIX + digest.hexdigest()
        if expected_digest is not None and normalized != normalize_digest(expected_digest):
            temp.unlink(missing_ok=True)
            raise ArtifactVerificationError(
                f"Artifact {artifact_name!r} checksum mismatch: expected "
                f"{normalize_digest(expected_digest)}, got {normalized}"
            )
        destination = self._artifact_path(normalized)
        with self._lock(normalized):
            if destination.exists():
                temp.unlink(missing_ok=True)
                self._verify_file(destination, normalized, size)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp, destination)
        return ArtifactManifest(
            name=artifact_name,
            digest=normalized,
            size_bytes=size,
            kind=kind,
            source=str(source) if not hasattr(source, "read") else None,
        )

    def get(self, reference: str, *, verify: bool = False) -> ModelManifest:
        index = self._read_index()
        model_id = reference if reference in index["models"] else index["aliases"].get(reference)
        if model_id is None:
            raise ModelNotFoundError(f"Unknown model or alias {reference!r}")
        manifest = self._read_manifest_file(index["models"][model_id])
        if verify:
            for artifact in manifest.artifacts:
                self.verify_artifact(artifact)
        return manifest

    def list_models(self, *, tag: str | None = None, verify: bool = False) -> list[ModelManifest]:
        index = self._read_index()
        model_ids = sorted(index["tags"].get(tag, [])) if tag else sorted(index["models"])
        return [self.get(model_id, verify=verify) for model_id in model_ids]

    def list(self, *, tag: str | None = None, verify: bool = False) -> list[ModelManifest]:
        return self.list_models(tag=tag, verify=verify)

    def remove(
        self, reference: str, *, remove_unreferenced_artifacts: bool = True
    ) -> ModelManifest:
        with self._lock("index"):
            index = self._read_index()
            model_id = (
                reference if reference in index["models"] else index["aliases"].get(reference)
            )
            if model_id is None:
                raise ModelNotFoundError(f"Unknown model or alias {reference!r}")
            manifest = self._read_manifest_file(index["models"].pop(model_id))
            for alias in manifest.aliases:
                index["aliases"].pop(alias, None)
            for tag in manifest.tags:
                self._remove_tag(index, tag, model_id)
            self._atomic_json_write(self._index_path, index)
            self._manifest_file(model_id).unlink(missing_ok=True)
            if remove_unreferenced_artifacts:
                referenced = {
                    artifact.digest
                    for other_id in index["models"]
                    for artifact in self._read_manifest_file(index["models"][other_id]).artifacts
                }
                for artifact in manifest.artifacts:
                    if artifact.digest not in referenced:
                        self._artifact_path(artifact.digest).unlink(missing_ok=True)
        return manifest

    def add_alias(self, reference: str, alias: str) -> ModelManifest:
        manifest = self.get(reference)
        aliases = tuple(dict.fromkeys((*manifest.aliases, alias)))
        return self.register(manifest.model_copy(update={"aliases": aliases}))

    def add_tag(self, reference: str, tag: str) -> ModelManifest:
        manifest = self.get(reference)
        tags = tuple(dict.fromkeys((*manifest.tags, tag)))
        return self.register(manifest.model_copy(update={"tags": tags}))

    def remove_alias(self, reference: str, alias: str) -> ModelManifest:
        manifest = self.get(reference)
        aliases = tuple(value for value in manifest.aliases if value != alias)
        return self.register(manifest.model_copy(update={"aliases": aliases}))

    def remove_tag(self, reference: str, tag: str) -> ModelManifest:
        manifest = self.get(reference)
        tags = tuple(value for value in manifest.tags if value != tag)
        return self.register(manifest.model_copy(update={"tags": tags}))

    def artifact_path(self, digest: str, *, verify: bool = True) -> Path:
        path = self._artifact_path(digest)
        if not path.is_file():
            raise ArtifactVerificationError(f"Artifact {normalize_digest(digest)} is missing")
        if verify:
            self._verify_file(path, normalize_digest(digest))
        return path

    def verify_artifact(self, artifact: ArtifactManifest) -> Path:
        path = self.artifact_path(artifact.digest, verify=False)
        self._verify_file(path, artifact.digest, artifact.size_bytes)
        return path

    def _manifest_file(self, model_id: str) -> Path:
        key = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:32]
        return self._safe_path("manifests", f"{key}.json")

    def _artifact_path(self, digest: str) -> Path:
        hexdigest = normalize_digest(digest).removeprefix(_SHA256_PREFIX)
        return self._safe_path("artifacts", "sha256", hexdigest[:2], hexdigest)

    _DOWNLOAD_TIMEOUT_S = 3600

    def _download(self, url: str) -> Path:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        target = self._safe_path("downloads", f"{key}.part")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(f"download:{url}"):
            existing_size = target.stat().st_size if target.exists() else 0
            request = urllib.request.Request(url)  # noqa: S310
            if existing_size:
                request.add_header("Range", f"bytes={existing_size}-")
            with urllib.request.urlopen(request, timeout=self._DOWNLOAD_TIMEOUT_S) as response:  # noqa: S310
                append = existing_size > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with target.open(mode) as output:
                    while chunk := response.read(_COPY_CHUNK_SIZE):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
        return target

    def _safe_path(self, *parts: str) -> Path:
        candidate = self.root.joinpath(*parts).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise UnsafePathError(f"Path escapes registry root: {candidate}")
        return candidate

    def _read_manifest_from_index(
        self, index: dict[str, Any], model_id: str
    ) -> ModelManifest | None:
        filename = index["models"].get(model_id)
        return self._read_manifest_file(filename) if filename else None

    def _read_manifest_file(self, filename: str) -> ModelManifest:
        path = self._safe_path("manifests", filename)
        return ModelManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def _read_index(self) -> dict[str, Any]:
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._empty_index()
        if not isinstance(data, dict) or set(data) != {"models", "aliases", "tags"}:
            raise RegistryError("Registry index is invalid")
        return data

    @staticmethod
    def _empty_index() -> dict[str, dict[str, Any]]:
        return {"models": {}, "aliases": {}, "tags": {}}

    @staticmethod
    def _remove_tag(index: dict[str, Any], tag: str, model_id: str) -> None:
        members = index["tags"].get(tag, [])
        if model_id in members:
            members.remove(model_id)
        if not members:
            index["tags"].pop(tag, None)

    def _atomic_json_write(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{uuid.uuid4().hex[:8]}.tmp")
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        try:
            with temp.open("x", encoding="utf-8", newline="\n") as output:
                output.write(payload)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _verify_file(path: Path, digest: str, size_bytes: int | None = None) -> None:
        if size_bytes is not None and path.stat().st_size != size_bytes:
            raise ArtifactVerificationError(
                f"Artifact {normalize_digest(digest)} checksum mismatch: expected size "
                f"{size_bytes}, got {path.stat().st_size}"
            )
        actual = sha256_file(path)
        if actual != normalize_digest(digest):
            raise ArtifactVerificationError(
                f"Artifact {normalize_digest(digest)} checksum mismatch: got {actual}"
            )

    @contextmanager
    def _lock(self, key: str) -> Iterator[None]:
        lock_name = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".lock"
        path = self._safe_path("locks", lock_name)
        deadline = time.monotonic() + self.lock_timeout_seconds
        pid_bytes = str(os.getpid()).encode()
        while True:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(descriptor, pid_bytes)
                os.close(descriptor)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise RegistryError(f"Timed out waiting for registry lock {key!r}") from None
                try:
                    import psutil

                    holder_pid = int(path.read_text(encoding="utf-8").strip())
                    if not psutil.pid_exists(holder_pid):
                        path.unlink(missing_ok=True)
                        continue
                except (OSError, ValueError):
                    pass
                time.sleep(0.01)
        try:
            yield
        finally:
            path.unlink(missing_ok=True)


def normalize_digest(value: str) -> str:
    normalized = value.lower()
    if not normalized.startswith(_SHA256_PREFIX):
        normalized = _SHA256_PREFIX + normalized
    hexdigest = normalized.removeprefix(_SHA256_PREFIX)
    if len(hexdigest) != 64 or any(char not in "0123456789abcdef" for char in hexdigest):
        raise ValueError("digest must be a SHA-256 hexadecimal digest")
    return normalized


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_SIZE):
            digest.update(chunk)
    return _SHA256_PREFIX + digest.hexdigest()


Artifact = ArtifactManifest
