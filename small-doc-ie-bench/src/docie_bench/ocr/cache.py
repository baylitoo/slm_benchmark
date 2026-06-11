from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from docie_bench.ocr.artifact import ARTIFACT_FORMAT_VERSION, OCRArtifact


class OCRCache:
    """Content-addressed OCR artifact cache with atomic writes and corruption checks."""

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = 2 * 1024 * 1024 * 1024,
        lock_timeout_seconds: float = 60.0,
        stale_lock_seconds: float = 3600.0,
    ) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self.lock_timeout_seconds = lock_timeout_seconds
        self.stale_lock_seconds = stale_lock_seconds

    @staticmethod
    def key(
        *,
        document_hash: str,
        backend: str,
        language: str | None,
        backend_version: str,
        configuration: dict[str, Any],
    ) -> str:
        canonical = json.dumps(
            {
                "document_hash": document_hash,
                "backend": backend,
                "language": language,
                "backend_version": backend_version,
                "configuration": configuration,
                "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def get(self, key: str) -> OCRArtifact | None:
        path = self._entry_path(key)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope["cache_key"] != key:
                raise ValueError("cache key mismatch")
            payload = envelope["artifact"]
            expected = envelope["sha256"]
            if expected != self._payload_hash(payload):
                raise ValueError("artifact checksum mismatch")
            artifact = OCRArtifact.model_validate(payload)
            os.utime(path, None)
            return artifact
        except FileNotFoundError:
            return None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, artifact: OCRArtifact) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._entry_path(key)
        payload = artifact.model_dump(mode="json")
        envelope = {
            "cache_key": key,
            "sha256": self._payload_hash(payload),
            "artifact": payload,
        }
        nonce = uuid.uuid4().hex[:8]
        temporary = self.root / f".{key}.{os.getpid()}.{nonce}.tmp"
        try:
            temporary.write_text(
                json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.evict()
        return path

    def evict(self) -> int:
        if self.max_bytes < 0 or not self.root.exists():
            return 0
        entries: list[tuple[Path, int, float]] = []
        for path in self.root.glob("*.json"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            entries.append((path, stat.st_size, stat.st_mtime))
        entries.sort(key=lambda entry: entry[2])
        total = sum(entry[1] for entry in entries)
        removed = 0
        for path, size, _mtime in entries:
            if total <= self.max_bytes:
                break
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            total -= size
            removed += 1
        return removed

    @contextmanager
    def lock(self, key: str) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / f"{key}.lock"
        token = f"{os.getpid()}:{uuid.uuid4().hex}"
        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(descriptor, token.encode("ascii"))
                finally:
                    os.close(descriptor)
                break
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > self.stale_lock_seconds:
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for OCR cache lock {key}") from None
                time.sleep(0.02)
        try:
            yield
        finally:
            try:
                if lock_path.read_text(encoding="ascii") == token:
                    lock_path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass

    def _entry_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    @staticmethod
    def _payload_hash(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
