from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from docie_bench.ocr.artifact import OCRArtifact, quality_signals
from docie_bench.ocr.cache import OCRCache
from docie_bench.ocr.service import OCRProcessor
from docie_bench.schemas.common import OCRBlock


def _artifact(text: str = "hello", *, backend_version: str = "1") -> OCRArtifact:
    blocks = [OCRBlock(id="b1", text=text, source="manual")]
    return OCRArtifact(
        document_hash="sha256:document",
        backend="fake",
        backend_version=backend_version,
        blocks=blocks,
        quality=quality_signals(blocks),
        latency_ms=3,
    )


def test_cache_key_invalidates_for_version_language_and_configuration() -> None:
    base = {
        "document_hash": "sha256:document",
        "backend": "fake",
        "language": "en",
        "backend_version": "1",
        "configuration": {"dpi": 150, "options": {"deskew": True}},
    }

    first = OCRCache.key(**base)
    reordered = OCRCache.key(
        **{**base, "configuration": {"options": {"deskew": True}, "dpi": 150}}
    )

    assert first == reordered
    assert first != OCRCache.key(**{**base, "backend_version": "2"})
    assert first != OCRCache.key(**{**base, "language": "fr"})
    assert first != OCRCache.key(**{**base, "configuration": {"dpi": 300}})


def test_cache_detects_corrupt_and_incomplete_entries(tmp_path: Path) -> None:
    cache = OCRCache(tmp_path)
    cache.put("corrupt", _artifact())
    path = tmp_path / "corrupt.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["artifact"]["blocks"][0]["text"] = "tampered"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    assert cache.get("corrupt") is None
    assert not path.exists()

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text('{"artifact": {}}', encoding="utf-8")
    assert cache.get("incomplete") is None
    assert not incomplete.exists()


def test_cache_evicts_least_recently_used_entries(tmp_path: Path) -> None:
    unlimited = OCRCache(tmp_path)
    first = unlimited.put("first", _artifact("first"))
    second = unlimited.put("second", _artifact("second"))
    old = time.time() - 100
    os.utime(second, (old, old))
    os.utime(first, None)
    size = first.stat().st_size

    evicting = OCRCache(tmp_path, max_bytes=size)
    removed = evicting.evict()

    assert removed == 1
    assert first.exists()
    assert not second.exists()


def test_concurrent_processors_only_extract_once(monkeypatch, tmp_path: Path) -> None:
    document = tmp_path / "document.txt"
    document.write_text("hello", encoding="utf-8")
    calls = 0
    calls_lock = threading.Lock()

    class FakeBackend:
        name = "fake"

        def version(self):
            return "1"

        def configuration(self):
            return {}

        def extract(self, path):
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            return [OCRBlock(id="b1", text=path.read_text(encoding="utf-8"), source="manual")]

    monkeypatch.setattr(
        "docie_bench.ocr.service.get_ocr_backend", lambda *args, **kwargs: FakeBackend()
    )
    processor = OCRProcessor(OCRCache(tmp_path / "cache"))
    results = []

    threads = [
        threading.Thread(
            target=lambda: results.append(processor.process(document, backend_name="fake"))
        )
        for _ in range(5)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len(results) == 5
    assert sum(result.cache_hit for result in results) == 4


def test_processor_rebuilds_corrupt_entry(monkeypatch, tmp_path: Path) -> None:
    document = tmp_path / "document.txt"
    document.write_text("hello", encoding="utf-8")
    calls = 0

    class FakeBackend:
        name = "fake"

        def version(self):
            return "1"

        def configuration(self):
            return {}

        def extract(self, path):
            nonlocal calls
            calls += 1
            return [OCRBlock(id="b1", text=path.read_text(encoding="utf-8"), source="manual")]

    monkeypatch.setattr(
        "docie_bench.ocr.service.get_ocr_backend", lambda *args, **kwargs: FakeBackend()
    )
    cache = OCRCache(tmp_path / "cache")
    processor = OCRProcessor(cache)
    first = processor.process(document, backend_name="fake")
    entry = cache.root / f"{first.cache_key}.json"
    entry.write_text("not json", encoding="utf-8")

    rebuilt = processor.process(document, backend_name="fake")

    assert calls == 2
    assert rebuilt.cache_hit is False
    assert cache.get(rebuilt.cache_key) is not None
