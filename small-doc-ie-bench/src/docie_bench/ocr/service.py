from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docie_bench.ocr.artifact import OCRArtifact, quality_signals
from docie_bench.ocr.base import OCRBackend
from docie_bench.ocr.cache import OCRCache
from docie_bench.ocr.factory import get_ocr_backend


@dataclass(frozen=True)
class OCRResult:
    artifact: OCRArtifact
    cache_hit: bool
    cache_key: str


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class OCRProcessor:
    def __init__(self, cache: OCRCache | None = None) -> None:
        self.cache = cache

    def process(
        self,
        path: Path,
        *,
        backend_name: str,
        language: str | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> OCRResult:
        backend = get_ocr_backend(backend_name, language=language)
        merged_configuration = {**backend.configuration(), **(configuration or {})}
        document_hash = hash_file(path)
        key = OCRCache.key(
            document_hash=document_hash,
            backend=backend.name,
            language=language,
            backend_version=backend.version(),
            configuration=merged_configuration,
        )
        if self.cache is None:
            return OCRResult(
                self._extract(path, document_hash, backend, language, merged_configuration),
                False,
                key,
            )
        cached = self.cache.get(key)
        if cached is not None:
            return OCRResult(cached, True, key)
        with self.cache.lock(key):
            cached = self.cache.get(key)
            if cached is not None:
                return OCRResult(cached, True, key)
            artifact = self._extract(path, document_hash, backend, language, merged_configuration)
            self.cache.put(key, artifact)
            return OCRResult(artifact, False, key)

    @staticmethod
    def _extract(
        path: Path,
        document_hash: str,
        backend: OCRBackend,
        language: str | None,
        configuration: dict[str, Any],
    ) -> OCRArtifact:
        started = time.perf_counter()
        blocks = backend.extract(path)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return OCRArtifact(
            document_hash=document_hash,
            backend=backend.name,
            backend_version=backend.version(),
            language=language,
            configuration=configuration,
            blocks=blocks,
            quality=quality_signals(blocks),
            latency_ms=latency_ms,
        )


def processor_from_settings(settings: Any) -> OCRProcessor:
    if not getattr(settings, "ocr_cache_enabled", False):
        return OCRProcessor()
    max_mb = int(getattr(settings, "ocr_cache_max_mb", 2048))
    return OCRProcessor(OCRCache(Path(settings.ocr_cache_dir), max_bytes=max_mb * 1024 * 1024))
