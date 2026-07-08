"""Seed-integrity edge cases for the serving seed path (PR feat/seed-integrity).

Three regressions this suite pins down:

1. No-DB (dev/local) mode must NOT fail every seed. When the catalog is
   *genuinely unavailable* (CatalogUnavailableError, no DATABASE_URL), the seed
   succeeds store-only and keeps the on-disk entry; only a *configured* catalog
   whose write fails is fatal and compensates.
2. A crash in ``ModelStore._write_entry`` after a verified blob transfer must not
   leave an orphan ``root/name`` dir with no index key (fresh seed) — while a
   re-seed of an existing name must NOT delete the still-referenced dir.
3. A post-success ``TOPIC_RESULT`` publish failure must not fail an
   already-succeeded seed run.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from docie_bench.inngest import functions
from docie_bench.serving.catalog import CatalogUnavailableError
from docie_bench.serving.model_store import ModelStore, StoreEntry


# --------------------------------------------------------------------------- #
# Finding 1: no-DB seed degrades gracefully; configured-but-failing is fatal.
# --------------------------------------------------------------------------- #
class _FakeEntry:
    def __init__(self, name: str, model_path: Path) -> None:
        self.name = name
        self.family = "openai_chat"
        self.model_path = model_path
        self.mmproj_path: Path | None = None
        self.source = "ollama:ref"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "model_path": self.model_path.as_posix(),
            "mmproj_path": None,
            "source": self.source,
        }


class _FakeStore:
    """Stands in for ModelStore: no real Ollama/blob I/O, records compensation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.removed: list[str] = []
        self._entry: _FakeEntry | None = None

    def bind(self, entry: _FakeEntry) -> None:
        self._entry = entry

    def seed_from_ollama(
        self, reference: str, *, name: str, family: str, mmproj_source: Any
    ) -> Any:
        assert self._entry is not None
        return self._entry

    def remove_entry(self, name: str) -> None:
        self.removed.append(name)


def _install_fake_store(monkeypatch: pytest.MonkeyPatch, entry: _FakeEntry) -> _FakeStore:
    holder: dict[str, _FakeStore] = {}

    def _factory(root: str | Path) -> _FakeStore:
        store = _FakeStore(root)
        store.bind(entry)
        holder["store"] = store
        return store

    monkeypatch.setattr("docie_bench.serving.model_store.ModelStore", _factory)
    return holder  # type: ignore[return-value]


def test_seed_succeeds_store_only_when_catalog_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No catalog configured -> seed succeeds, on-disk entry KEPT, warning logged."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"weights")
    entry = _FakeEntry("dev-model", model_file)
    holder = _install_fake_store(monkeypatch, entry)

    class _UnavailableCatalog:
        def upsert(self, entry: Any, *, size_bytes: int | None = None) -> dict[str, Any]:
            raise CatalogUnavailableError("DATABASE_URL is not configured")

    monkeypatch.setattr("docie_bench.serving.catalog.ModelCatalog", _UnavailableCatalog)

    with caplog.at_level(logging.WARNING, logger="docie_bench.inngest.functions"):
        result = asyncio.run(
            functions._run_seed_ollama({"reference": "qwen2.5:1.5b", "name": "dev-model"})
        )

    # Seed succeeded store-only and is honestly flagged as not catalog-registered.
    assert result["name"] == "dev-model"
    assert result["catalog_registered"] is False
    assert result["size_bytes"] == len(b"weights")
    # On-disk entry KEPT: no compensation removal ran.
    assert holder["store"].removed == []
    # A clear warning was logged about the store-only outcome.
    assert any(
        "NOT catalog-registered" in rec.getMessage() for rec in caplog.records
    ), "expected a store-only warning"


def test_seed_is_fatal_and_compensates_when_configured_catalog_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catalog IS configured but the WRITE fails -> fatal + on-disk rollback."""
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"weights")
    entry = _FakeEntry("prod-model", model_file)
    holder = _install_fake_store(monkeypatch, entry)

    class _FailingCatalog:
        def upsert(self, entry: Any, *, size_bytes: int | None = None) -> dict[str, Any]:
            raise RuntimeError("unique violation / connection reset")

    monkeypatch.setattr("docie_bench.serving.catalog.ModelCatalog", _FailingCatalog)

    with pytest.raises(RuntimeError, match="unique violation"):
        asyncio.run(functions._run_seed_ollama({"reference": "ref", "name": "prod-model"}))

    # Compensation ran: the just-written on-disk entry was rolled back.
    assert holder["store"].removed == ["prod-model"]


# --------------------------------------------------------------------------- #
# Finding 2: orphan dir cleanup on _write_entry failure (fresh seed only).
# --------------------------------------------------------------------------- #
def test_write_entry_cleans_orphan_dir_on_index_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh-seed _write_entry crash must leave NO orphan root/name dir."""
    store = ModelStore(tmp_path / "models")
    name = "orphan-me"
    destination = store.root / name
    destination.mkdir(parents=True)
    (destination / "model.gguf").write_bytes(b"weights")  # simulate a done transfer

    def _boom(index: dict[str, Any]) -> None:
        raise OSError("disk full while writing index.json")

    monkeypatch.setattr(store, "_write_index", _boom)

    entry = StoreEntry(name=name, family="openai_chat", model_path=destination / "model.gguf")
    with pytest.raises(OSError, match="disk full"):
        store._write_entry(entry)

    # No orphan dir with no index key: cleaned up.
    assert not destination.exists()


def test_write_entry_reseed_failure_keeps_still_referenced_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-seed of an EXISTING name that fails the index write must NOT delete
    the dir the current (unchanged, atomic) index still references."""
    store = ModelStore(tmp_path / "models")
    name = "keep-me"
    destination = store.root / name
    destination.mkdir(parents=True)
    (destination / "model.gguf").write_bytes(b"old-weights")

    # Seed once successfully so the index references root/name.
    entry = StoreEntry(name=name, family="openai_chat", model_path=destination / "model.gguf")
    store._write_entry(entry)
    assert name in store._read_index()

    # Now a re-seed whose index write fails must leave the referenced dir intact.
    def _boom(index: dict[str, Any]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_index", _boom)
    with pytest.raises(OSError, match="disk full"):
        store._write_entry(entry)

    assert destination.exists(), "re-seed failure must not orphan the referenced dir"


# --------------------------------------------------------------------------- #
# Finding 3: post-success result-publish failure must not fail the seed run.
# --------------------------------------------------------------------------- #
class _FakeStep:
    async def run(self, _name: str, fn: Any) -> Any:
        result = fn()
        if asyncio.iscoroutine(result):
            result = await result
        return result


class _FakeEvent:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.id = "evt-seed-1"


class _FakeCtx:
    def __init__(self, data: dict[str, Any]) -> None:
        self.event = _FakeEvent(data)
        self.step = _FakeStep()


def test_seed_run_survives_result_publish_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A publish failure AFTER a successful seed must not fail the run."""
    seed_result = {"name": "m", "catalog_registered": True}

    async def _fake_seed(data: dict[str, Any]) -> dict[str, Any]:
        return seed_result

    monkeypatch.setattr(functions, "_run_seed_ollama", _fake_seed)

    published: list[str] = []

    async def _fake_publish(channel: str, topic: str, data: Any) -> None:
        published.append(topic)
        if topic == functions.TOPIC_RESULT:
            raise RuntimeError("realtime backend down")

    monkeypatch.setattr(functions, "publish", _fake_publish)

    ctx = _FakeCtx({"reference": "ref", "name": "m"})
    # seed_ollama_job is wrapped by @create_function into an inngest Function;
    # ._handler is the original coroutine the worker invokes.
    result = asyncio.run(functions.seed_ollama_job._handler(ctx))  # type: ignore[attr-defined]

    # The run succeeded and returned the seed result despite the publish blowing up.
    assert result == seed_result
    assert functions.TOPIC_RESULT in published
