"""seed_ollama_job / seed_hf_job's SeedRun persistence wiring: claim before
work, complete/fail after, and the redelivery short-circuit that skips
re-downloading an already-completed seed. All against a real sqlite-backed
seed_store (init_engine), a stubbed download step, and a fake Inngest
context -- no live Ollama/HF stack."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import pytest

from docie_bench.inngest import functions
from docie_bench.inngest.functions import seed_hf_job, seed_ollama_job
from docie_bench.storage.db import dispose_engine, init_engine
from docie_bench.studio.seed_store import claim_seed_run, list_seed_runs


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
    def __init__(self, data: dict[str, Any], event_id: str) -> None:
        self.event = _FakeEvent(data, event_id)
        self.step = _FakeStep()


_ollama_handler = cast(
    Callable[[Any], Awaitable[dict[str, Any]]], cast(Any, seed_ollama_job)._handler
)
_hf_handler = cast(Callable[[Any], Awaitable[dict[str, Any]]], cast(Any, seed_hf_job)._handler)


async def _fake_publish(channel: str, topic: str, data: Any) -> None:
    pass


async def _fake_publish_error_safely(channel: str, message: str) -> None:
    pass


def _fake_clear_progress(channel: str) -> None:
    pass


@pytest.fixture(autouse=True)
def seed_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'seeds.db'}")
    yield
    dispose_engine()


async def test_ollama_seed_success_persists_a_completed_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_run(data: dict[str, Any]) -> dict[str, Any]:
        return {"name": "qwen", "family": "openai_chat", "size_bytes": 42}

    monkeypatch.setattr(functions, "_run_seed_ollama", _fake_run)
    monkeypatch.setattr(functions, "publish", _fake_publish)

    ctx = _FakeCtx(
        {"reference": "qwen2.5:1.5b", "name": "qwen", "channel": "seed:1", "tenant_id": "t1"},
        event_id="ev1",
    )
    result = await _ollama_handler(ctx)

    assert result == {"name": "qwen", "family": "openai_chat", "size_bytes": 42}
    rows = list_seed_runs(tenant_id="t1")
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["kind"] == "ollama"
    assert rows[0]["reference"] == "qwen2.5:1.5b"
    assert rows[0]["result"]["size_bytes"] == 42


async def test_ollama_seed_failure_persists_a_failed_row_with_error_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(data: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("disk full")

    monkeypatch.setattr(functions, "_run_seed_ollama", _boom)
    monkeypatch.setattr(functions, "publish", _fake_publish)
    monkeypatch.setattr(functions, "_publish_error_safely", _fake_publish_error_safely)

    ctx = _FakeCtx(
        {"reference": "qwen2.5:1.5b", "name": "qwen", "channel": "seed:1", "tenant_id": "t1"},
        event_id="ev1",
    )
    with pytest.raises(RuntimeError, match="disk full"):
        await _ollama_handler(ctx)

    rows = list_seed_runs(tenant_id="t1")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "disk full"


async def test_ollama_seed_redelivery_of_completed_run_skips_redownload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    async def _fake_run(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        return {"name": "qwen"}

    monkeypatch.setattr(functions, "_run_seed_ollama", _fake_run)
    monkeypatch.setattr(functions, "publish", _fake_publish)

    data = {"reference": "qwen2.5:1.5b", "name": "qwen", "channel": "seed:1", "tenant_id": "t1"}
    first = await _ollama_handler(_FakeCtx(dict(data), event_id="ev1"))
    second = await _ollama_handler(_FakeCtx(dict(data), event_id="ev1"))

    assert len(calls) == 1  # the redelivery short-circuits before the download step
    assert first == second == {"name": "qwen"}


async def test_hf_seed_success_persists_a_completed_row_with_resolved_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_run(data: dict[str, Any], channel: str) -> dict[str, Any]:
        # HF seeds may have no explicit `name` at trigger time -- derived here.
        return {"name": "lfm25-350m-instruct", "family": "openai_chat"}

    monkeypatch.setattr(functions, "_run_seed_hf", _fake_run)
    monkeypatch.setattr(functions, "publish", _fake_publish)
    monkeypatch.setattr(functions, "clear_progress", _fake_clear_progress)

    ctx = _FakeCtx(
        {"repo": "LiquidAI/LFM2.5-350M-Instruct-GGUF", "channel": "seed:2", "tenant_id": "t1"},
        event_id="ev2",
    )
    result = await _hf_handler(ctx)

    assert result["name"] == "lfm25-350m-instruct"
    rows = list_seed_runs(tenant_id="t1")
    assert len(rows) == 1
    assert rows[0]["kind"] == "hf"
    assert rows[0]["reference"] == "LiquidAI/LFM2.5-350M-Instruct-GGUF"
    # Row's own `name` was empty at claim time; complete_seed_run resolves it.
    assert rows[0]["name"] == "lfm25-350m-instruct"


async def test_hf_seed_failure_persists_a_failed_row(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(data: dict[str, Any], channel: str) -> dict[str, Any]:
        raise RuntimeError("404 repo not found")

    monkeypatch.setattr(functions, "_run_seed_hf", _boom)
    monkeypatch.setattr(functions, "publish", _fake_publish)
    monkeypatch.setattr(functions, "_publish_error_safely", _fake_publish_error_safely)
    monkeypatch.setattr(functions, "clear_progress", _fake_clear_progress)

    ctx = _FakeCtx(
        {"repo": "nobody/nothing", "channel": "seed:3", "tenant_id": "t1"}, event_id="ev3"
    )
    with pytest.raises(RuntimeError, match="404 repo not found"):
        await _hf_handler(ctx)

    rows = list_seed_runs(tenant_id="t1")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "404 repo not found"


async def test_seed_runs_are_tenant_scoped() -> None:
    claim_seed_run(
        event_id="ev1", channel="seed:a", tenant_id="tenant-a", kind="ollama", name="a"
    )
    claim_seed_run(
        event_id="ev2", channel="seed:b", tenant_id="tenant-b", kind="ollama", name="b"
    )

    assert [r["event_id"] for r in list_seed_runs(tenant_id="tenant-a")] == ["ev1"]
    assert [r["event_id"] for r in list_seed_runs(tenant_id="tenant-b")] == ["ev2"]
