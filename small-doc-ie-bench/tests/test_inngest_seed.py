"""Unit tests for the Ollama seed job's integrity + honest-error contract.

Covers PR-A's inngest-side guarantees, all against STUBBED collaborators (a fake
Ollama home built from real content-addressed blobs, a fake catalog, and a fake
Inngest context) — no live stack, no model pulls:

* (c) a CONFIGURED catalog whose upsert fails (generic write error) re-raises AND
      compensates (the just-seeded on-disk entry is removed), so a seed is
      all-or-nothing. (A genuinely unavailable catalog degrades store-only instead;
      that path is covered by test_seed_integrity.py.)
* (d) a seed-step failure logs a full traceback to worker logs AND publishes the
      error topic AND re-raises the original error.
* (e) an error-publish that itself raises does not mask the original failure.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import pytest

from docie_bench.inngest import functions
from docie_bench.inngest.functions import seed_ollama_job
from docie_bench.inngest.realtime import TOPIC_ERROR
from docie_bench.serving.model_store import ModelStore

# The decorated job is an inngest ``Function`` wrapper; call its underlying async
# handler directly with a fake context. Cast through ``Any`` so accessing the
# private ``_handler`` stays clean under ruff/mypy --strict.
_seed_handler = cast(
    Callable[[Any], Awaitable[dict[str, Any]]], cast(Any, seed_ollama_job)._handler
)


def _build_ollama_home(tmp_path: Path) -> Path:
    """Minimal Ollama models dir with one library model whose blob is faithful.

    The blob is content-addressed by its real sha256 and the manifest advertises
    that same digest, so the store's post-transfer integrity check passes.
    """
    home = tmp_path / "ollama"
    blobs = home / "blobs"
    blobs.mkdir(parents=True)
    content = b"GGUF-model-weights"
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    (blobs / digest.replace(":", "-")).write_bytes(content)
    layers = [{"mediaType": "application/vnd.ollama.image.model", "digest": digest}]
    manifest_dir = home.joinpath("manifests", "registry.ollama.ai", "library", "m")
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "latest").write_text(json.dumps({"layers": layers}), encoding="utf-8")
    return home


class _FakeStep:
    async def run(self, step_id: str, fn: Callable[[], Any], *args: Any, **kwargs: Any) -> Any:
        result = fn()
        if inspect.isawaitable(result):
            return await result
        return result


class _FakeEvent:
    def __init__(self, data: dict[str, Any], event_id: str) -> None:
        self.data = data
        self.id = event_id


class _FakeCtx:
    def __init__(self, data: dict[str, Any], event_id: str = "evt-1") -> None:
        self.event = _FakeEvent(data, event_id)
        self.step = _FakeStep()


# --------------------------------------------------------------------------- (c)
async def test_run_seed_ollama_compensates_when_catalog_upsert_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _build_ollama_home(tmp_path)
    serving = tmp_path / "serving"
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(serving))
    monkeypatch.setenv("OLLAMA_MODELS", str(home))

    class _FailingCatalog:
        """A CONFIGURED catalog (DB reachable) whose WRITE fails — the fatal path."""

        def __init__(self) -> None:
            pass

        def upsert(self, entry: object, *, size_bytes: int | None = None) -> dict[str, Any]:
            raise RuntimeError("unique violation / connection reset")

    monkeypatch.setattr("docie_bench.serving.catalog.ModelCatalog", _FailingCatalog)

    data = {"reference": "m:latest", "name": "seeded", "family": "openai_chat"}
    # A configured catalog whose write fails is FATAL: the generic error re-raises.
    with pytest.raises(RuntimeError, match="unique violation"):
        await functions._run_seed_ollama(data)

    # Compensation: the on-disk entry the seed just wrote is rolled back through the
    # REAL ModelStore.remove_entry, so the seed left NEITHER an index row NOR a
    # store dir (all-or-nothing).
    store = ModelStore(serving / "models")
    assert store.list() == []
    assert not (serving / "models" / "seeded").exists()


async def test_run_seed_ollama_compensation_failure_does_not_mask_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _build_ollama_home(tmp_path)
    serving = tmp_path / "serving"
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(serving))
    monkeypatch.setenv("OLLAMA_MODELS", str(home))

    class _FailingCatalog:
        """A CONFIGURED catalog whose WRITE fails with a distinct error type."""

        def __init__(self) -> None:
            pass

        def upsert(self, entry: object, *, size_bytes: int | None = None) -> dict[str, Any]:
            raise ValueError("catalog write failed")

    monkeypatch.setattr("docie_bench.serving.catalog.ModelCatalog", _FailingCatalog)

    def _boom(self: ModelStore, name: str) -> None:
        raise RuntimeError("compensation blew up")

    monkeypatch.setattr(ModelStore, "remove_entry", _boom)

    data = {"reference": "m:latest", "name": "seeded", "family": "openai_chat"}
    # The ORIGINAL catalog error (ValueError) propagates, NOT the compensation's
    # RuntimeError — a failed rollback must never mask the real failure.
    with pytest.raises(ValueError, match="catalog write failed"):
        await functions._run_seed_ollama(data)


# --------------------------------------------------------------------------- (d)
async def test_seed_job_logs_traceback_and_publishes_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    published: list[tuple[str, str, Any]] = []

    async def _fake_publish(channel: str, topic: str, data: Any) -> None:
        published.append((channel, topic, data))

    async def _boom(data: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("seed exploded")

    monkeypatch.setattr(functions, "publish", _fake_publish)
    monkeypatch.setattr(functions, "_run_seed_ollama", _boom)

    ctx = _FakeCtx({"reference": "m:latest", "name": "x", "channel": "chan-d"})
    with (
        caplog.at_level(logging.ERROR, logger="docie_bench.inngest.functions"),
        pytest.raises(RuntimeError, match="seed exploded"),
    ):
        await _seed_handler(ctx)

    # A full traceback landed in worker logs (exc_info set on an ERROR record).
    assert any(rec.levelno == logging.ERROR and rec.exc_info for rec in caplog.records)
    # The error topic carried the failure message.
    errors = [payload for (_c, topic, payload) in published if topic == TOPIC_ERROR]
    assert errors
    assert errors[0]["message"] == "seed exploded"


# --------------------------------------------------------------------------- (e)
async def test_seed_job_publish_failure_does_not_mask_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _selective_publish(channel: str, topic: str, data: Any) -> None:
        # Status publish succeeds; the error-topic publish blows up mid-handling.
        if topic == TOPIC_ERROR:
            raise RuntimeError("realtime is down")

    async def _boom(data: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("original seed error")

    monkeypatch.setattr(functions, "publish", _selective_publish)
    monkeypatch.setattr(functions, "_run_seed_ollama", _boom)

    ctx = _FakeCtx({"reference": "m:latest", "name": "x", "channel": "chan-e"})
    # The original ValueError re-raises; the publish RuntimeError does not win.
    with pytest.raises(ValueError, match="original seed error"):
        await _seed_handler(ctx)
