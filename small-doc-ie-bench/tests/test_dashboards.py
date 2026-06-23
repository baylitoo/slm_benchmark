"""Headless smoke tests for the Textual dashboards.

A TUI can't be exercised in a normal shell, so these drive the apps through
``App.run_test()`` and assert on the rendered widgets — covering the live
event-polling path, the in-app setup screen's validation, deployment
rendering, and the subset-config helper, without launching a real benchmark
or touching the host control plane.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("textual")

import yaml  # noqa: E402
from textual.widgets import DataTable, RichLog, SelectionList, Static  # noqa: E402

from docie_bench.dashboard import (  # noqa: E402
    DocieBenchApp,
    RunScreen,
    SetupScreen,
    _effective_config,
)
from docie_bench.serving.dashboard import ServingDashboard  # noqa: E402


def test_effective_config_filters_to_subset(tmp_path) -> None:
    config = tmp_path / "models.yaml"
    profiles = {"a": {"model": "a"}, "b": {"model": "b"}, "c": {"model": "c"}}
    config.write_text(yaml.safe_dump({"profiles": profiles}), encoding="utf-8")
    all_names = ["a", "b", "c"]

    # Whole set selected → original file reused untouched.
    assert _effective_config(config, all_names, all_names) == config

    # Strict subset → a new temp config with only those profiles.
    out = _effective_config(config, ["a", "c"], all_names)
    assert out != config
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert sorted(data["profiles"]) == ["a", "c"]


async def test_run_screen_polls_event_stream(tmp_path) -> None:
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

    screen = RunScreen(
        dataset_path=manifest,
        config_path=tmp_path / "models.yaml",
        profiles=["p1"],
        concurrency=1,
        repeat=1,
        run_dir=run_dir,
        auto_start=False,  # no real benchmark — we feed events by hand
    )
    app = DocieBenchApp(config_path=tmp_path / "models.yaml", initial_screen=screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen._poll()
        await pilot.pause()

        table = screen.query_one("#results", DataTable)
        assert table.get_cell("p1", "done") == "1/2"  # one of two docs completed
        # 1 of 2 done → still in progress, so the status reads "running".
        assert "running" in str(table.get_cell("p1", "status"))
        assert screen.query_one("#log", RichLog).lines  # events streamed to the log pane


async def test_setup_screen_requires_a_profile(tmp_path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump({"profiles": {"a": {"model": "a"}, "b": {"model": "b"}}}), encoding="utf-8"
    )
    dataset = tmp_path / "manifest.jsonl"
    dataset.write_text('{"doc_id": "x"}\n', encoding="utf-8")

    screen = SetupScreen(config_path=config, datasets=[dataset])
    app = DocieBenchApp(config_path=config, initial_screen=screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        # The profile list offers both profiles, none selected by default.
        assert list(screen.query_one("#profiles", SelectionList).selected) == []

        # Clicking Run with nothing selected must not navigate; it shows a hint.
        await pilot.click("#run")
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen)  # stayed on setup
        assert "profile" in str(screen.query_one("#hint", Static).render()).lower()


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
