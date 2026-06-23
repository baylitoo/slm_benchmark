"""Headless smoke tests for the Textual dashboards.

A TUI can't be exercised in a normal shell, so these drive the apps through
``App.run_test()`` and assert on the rendered widgets. They cover the pure
rendering paths (event polling, deployment rendering) without launching a real
benchmark or touching the host control plane.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("textual")

from textual.widgets import DataTable, RichLog  # noqa: E402

from docie_bench.dashboard import BenchmarkDashboard  # noqa: E402
from docie_bench.serving.dashboard import ServingDashboard  # noqa: E402


async def test_benchmark_dashboard_polls_event_stream(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"doc_id": "a"}\n{"doc_id": "b"}\n', encoding="utf-8")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    events = [
        {"model_profile": "p1", "doc_id": "a", "state": "queued", "at": "2026-06-23T10:00:00Z"},
        {"model_profile": "p1", "doc_id": "a", "state": "running", "at": "2026-06-23T10:00:01Z"},
        {"model_profile": "p1", "doc_id": "a", "state": "completed", "at": "2026-06-23T10:00:02Z"},
    ]
    (run_dir / "task-events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )

    app = BenchmarkDashboard(
        dataset_path=manifest,
        config_path=tmp_path / "models.yaml",
        profiles=["p1"],
        concurrency=1,
        repeat=1,
        run_dir=run_dir,
        auto_start=False,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app._poll()
        await pilot.pause()

        table = app.query_one("#results", DataTable)
        assert table.get_cell("p1", "done") == "1/2"  # one of two docs completed
        # 1 of 2 done → still in progress, so the status reads "running".
        assert "running" in str(table.get_cell("p1", "status"))

        log = app.query_one("#log", RichLog)
        assert log.lines  # events were streamed into the log pane


async def test_serving_dashboard_renders_deployments() -> None:
    data = {
        "deployments": [
            {
                "spec": {
                    "name": "invoice",
                    "launch": {"model": "org/x", "runtime": "llamacpp"},
                    "desired_state": "running",
                },
                "state": "ready",
                "endpoint": "http://127.0.0.1:8001",
                "restart_count": 0,
                "consecutive_health_failures": 0,
            },
            {
                "spec": {
                    "name": "receipts",
                    "launch": {"model": "org/y", "runtime": "ollama"},
                    "desired_state": "running",
                },
                "state": "failed",
                "endpoint": None,
                "restart_count": 2,
                "consecutive_health_failures": 3,
            },
        ],
        "models": [{"name": "org/x"}],
        "runtimes": [{"name": "llamacpp"}],
    }

    class _FakePlane:
        async def list_deployments(self):
            return data["deployments"]

        async def list_models(self):
            return data["models"]

        async def list_runtimes(self):
            return data["runtimes"]

    # High interval + injected plane: the on-mount refresh hits the fake, not the host.
    app = ServingDashboard(refresh_interval=3600.0, plane_factory=_FakePlane)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._render(data)  # deterministic, independent of worker-thread timing
        await pilot.pause()

        table = app.query_one("#deployments", DataTable)
        assert table.row_count == 2
        first = table.get_row_at(0)  # sorted by name → "invoice"
        assert first[0] == "invoice"
        assert "ready" in str(first[4])  # state cell is colour-coded Text
        assert first[6] == "http://127.0.0.1:8001"
