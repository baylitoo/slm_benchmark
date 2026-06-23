"""Live benchmark dashboard — watch every model profile race in one terminal view.

Launch with:  docie-bench-dash

This is a thin, *non-invasive* viewer over the existing benchmark runner. It runs a
single ``run_benchmark`` pass with ``model_profile=None`` (so every profile competes
under one shared concurrency budget — the numbers stay comparable to a normal
``docie-bench`` run) and subscribes to the run's live signal: the ``task-events.jsonl``
file the runner already appends to on every ``queued → running → completed/failed``
transition. The runner is untouched; the dashboard only reads what it writes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, ProgressBar, RichLog, Static
from textual.worker import Worker, WorkerState

# Reuse the wizard helpers the interactive UI already ships — single source of truth.
from docie_bench.cli2 import (
    _ask_dataset,
    _ask_options,
    _ask_profiles,
    _count_docs,
    _fmt_ms,
    _load_profile_names,
)

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent

# task-events.jsonl "state" values, mapped to a status label + Rich colour.
_STATE_STYLE = {
    "queued": ("queued", "dim"),
    "running": ("running", "yellow"),
    "completed": ("done", "green"),
    "failed": ("failed", "red"),
}
_TERMINAL_STATES = {"completed", "failed"}


class BenchmarkDashboard(App):
    """A live, multi-profile benchmark view built on the runner's event stream."""

    TITLE = "docie-bench"
    SUB_TITLE = "live benchmark dashboard"

    CSS = """
    Screen { layout: vertical; }
    #config { padding: 0 1; color: $text-muted; }
    #overall { height: 1; margin: 0 1; }
    #results { height: auto; max-height: 60%; margin: 1 1 0 1; }
    #log { height: 1fr; border: round $primary; margin: 1; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    # Columns rendered in the results table, in order: (key, label).
    _COLUMNS = [
        ("profile", "Model Profile"),
        ("status", "Status"),
        ("done", "Done"),
        ("ok", "OK"),
        ("acc", "Field Acc."),
        ("p50", "p50"),
        ("p95", "p95"),
        ("thru", "Throughput"),
    ]

    def __init__(
        self,
        *,
        dataset_path: Path,
        config_path: Path,
        profiles: list[str],
        concurrency: int,
        repeat: int,
        log_level: str = "INFO",
        run_dir: Path | None = None,
        auto_start: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.profiles = profiles
        self.concurrency = concurrency
        self.repeat = repeat
        self.log_level = log_level
        self.auto_start = auto_start

        self._docs = _count_docs(dataset_path)
        self._per_profile_total = self._docs * repeat
        self._grand_total = self._per_profile_total * max(len(profiles), 1)

        # Set late (or injected by tests). events_path derives from it.
        self._run_dir: Path | None = run_dir
        self._row_keys: dict[str, object] = {}
        self._emitted = 0  # lines of the events file already streamed to the log
        self._finished = False
        self._poll_timer = None

    # ── layout ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static(self._config_line(), id="config")
            yield ProgressBar(total=self._grand_total or None, show_eta=False, id="overall")
            yield DataTable(id="results", zebra_stripes=True, cursor_type="row")
            yield RichLog(id="log", highlight=True, markup=True, wrap=False)
        yield Footer()

    def _config_line(self) -> str:
        docs = f"{self._docs}" + (f" × {self.repeat}" if self.repeat > 1 else "")
        rel = self.dataset_path
        with contextlib.suppress(ValueError):
            rel = self.dataset_path.relative_to(_PROJECT_ROOT)
        return (
            f"dataset [b]{rel}[/]   docs [b]{docs}[/]   "
            f"profiles [b]{len(self.profiles)}[/]   concurrency [b]{self.concurrency}[/]"
        )

    def on_mount(self) -> None:
        table = self.query_one("#results", DataTable)
        for key, label in self._COLUMNS:
            table.add_column(label, key=key)
        for name in self.profiles:
            blanks = ["—"] * 5
            row = [name, _status_text("queued"), f"0/{self._per_profile_total}", *blanks]
            self._row_keys[name] = table.add_row(*row, key=name)

        log = self.query_one("#log", RichLog)
        log.write("[dim]Waiting for the runner to start…[/]")

        if self.auto_start:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")  # noqa: DTZ005 (display id only)
            from docie_bench.settings import get_settings

            self._run_dir = get_settings().runs_dir / f"dashboard-{stamp}"
            self.run_benchmark_worker()

        # Poll the event stream regardless of who started the run (tests start it manually).
        self._poll_timer = self.set_interval(0.25, self._poll)

    # ── benchmark execution (off the UI thread) ─────────────────────────────

    @work(thread=True, exit_on_error=False, name="bench")
    def run_benchmark_worker(self):
        """Run the whole sweep in a worker thread so sync work never stalls the UI."""
        from docie_bench.benchmark.runner import run_benchmark

        assert self._run_dir is not None
        self._run_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self._run_dir / "dashboard.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(self.log_level)
        try:
            return asyncio.run(
                run_benchmark(
                    dataset_path=self.dataset_path,
                    models_config_path=self.config_path,
                    model_profile=None,
                    output_dir=self._run_dir,
                    concurrency=self.concurrency,
                    repeat=self.repeat,
                )
            )
        finally:
            root.removeHandler(handler)
            handler.close()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "bench":
            return
        if event.state == WorkerState.SUCCESS:
            self._poll()  # flush any final events before the summary overwrites the rows
            self._finalize(event.worker.result)
        elif event.state == WorkerState.ERROR:
            self._finished = True
            log = self.query_one("#log", RichLog)
            log.write(f"[red]Benchmark failed:[/] {event.worker.error!r}")

    # ── live polling of task-events.jsonl ───────────────────────────────────

    def _read_events(self) -> list[dict]:
        if self._run_dir is None:
            return []
        events_path = self._run_dir / "task-events.jsonl"
        if not events_path.exists():
            return []
        events: list[dict] = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn final line during an in-flight append; it'll be whole next tick.
                continue
        return events

    def _poll(self) -> None:
        events = self._read_events()
        if not events:
            return

        # Stream only events we haven't shown yet.
        log = self.query_one("#log", RichLog)
        for ev in events[self._emitted :]:
            state = ev.get("state", "?")
            _, colour = _STATE_STYLE.get(state, (state, "white"))
            at = str(ev.get("at", ""))[11:19]  # HH:MM:SS slice of the ISO timestamp
            reason = ev.get("reason")
            tail = f"  [red]{reason}[/]" if reason else ""
            log.write(
                f"[dim]{at}[/]  {ev.get('model_profile', '?')} · "
                f"{ev.get('doc_id', '?')} → [{colour}]{state}[/]{tail}"
            )
        self._emitted = len(events)

        # Recompute per-profile counts from scratch (files are tiny).
        counts: dict[str, dict[str, int]] = {}
        for ev in events:
            profile = ev.get("model_profile")
            if profile is None:
                continue
            counts.setdefault(profile, {})
            state = ev.get("state", "")
            counts[profile][state] = counts[profile].get(state, 0) + 1

        table = self.query_one("#results", DataTable)
        done_total = 0
        for profile, row_key in self._row_keys.items():
            if self._finished:
                continue  # final summary owns the rows now
            c = counts.get(profile, {})
            done = c.get("completed", 0) + c.get("failed", 0)
            done_total += done
            running = c.get("running", 0) - done  # running events fire before terminal ones
            if done >= self._per_profile_total and self._per_profile_total:
                status = "done"
            elif running > 0 or done > 0:
                status = "running"
            else:
                status = "queued"
            table.update_cell(row_key, "status", _status_text(status))
            table.update_cell(row_key, "done", f"{done}/{self._per_profile_total}")

        if not self._finished:
            self.query_one("#overall", ProgressBar).update(progress=done_total)

    # ── final summary from metrics.json ─────────────────────────────────────

    def _finalize(self, result) -> None:
        self._finished = True
        if self._poll_timer is not None:
            self._poll_timer.stop()
        table = self.query_one("#results", DataTable)
        log = self.query_one("#log", RichLog)

        metrics_path = getattr(result, "metrics_path", None) or (self._run_dir / "metrics.json")
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        for s in metrics.get("summary", []):
            name = s.get("model_profile")
            row_key = self._row_keys.get(name)
            if row_key is None:
                continue
            acc = f"{s['field_accuracy']:.1%}" if s.get("field_accuracy") is not None else "—"
            sim = s.get("avg_similarity")
            thru = (
                f"{s['throughput_docs_per_min']:.1f}/min"
                if s.get("throughput_docs_per_min")
                else "—"
            )
            table.update_cell(row_key, "status", _status_text("done"))
            table.update_cell(row_key, "ok", f"{s.get('ok_rate', 0):.0%}")
            table.update_cell(row_key, "acc", acc)
            table.update_cell(row_key, "p50", _fmt_ms(s.get("p50_latency_ms")))
            table.update_cell(row_key, "p95", _fmt_ms(s.get("p95_latency_ms")))
            table.update_cell(row_key, "thru", thru)
            _ = sim  # similarity available in metrics.json; omitted to keep the table compact

        self.query_one("#overall", ProgressBar).update(progress=self._grand_total)
        report = getattr(result, "report_path", None)
        if report is not None:
            log.write(f"\n[green]✓ Done.[/] Report: [b]{report}[/]")
        log.write("[dim]Press q to quit.[/]")


def _status_text(status: str) -> Text:
    label, colour = next(
        ((lbl, col) for lbl, col in _STATE_STYLE.values() if lbl == status),
        (status, "white"),
    )
    return Text(label, style=colour)


# ── entry point ─────────────────────────────────────────────────────────────


def _effective_config(config_path: Path, selected: list[str], all_names: list[str]) -> Path:
    """Return a models config restricted to ``selected`` profiles.

    The runner is all-or-one (a single ``model_profile`` or every profile), so to race
    an arbitrary subset under one shared concurrency budget we write a temporary config
    containing only the chosen profiles and run it with ``model_profile=None``. When the
    whole set is selected we reuse the original file untouched.
    """
    if set(selected) == set(all_names):
        return config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    profiles = data.get("profiles", {})
    data["profiles"] = {name: profiles[name] for name in selected if name in profiles}
    fd, tmp = tempfile.mkstemp(prefix="docie-dash-models-", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return Path(tmp)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docie-bench-dash",
        description="Live dashboard: watch model profiles race on a dataset.",
    )
    parser.add_argument("--models-config", type=Path, default=None, help="Path to a models.yaml.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to a manifest.jsonl.")
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        metavar="NAME",
        help="Profile to include (repeatable). Default: prompt, then all.",
    )
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel requests.")
    parser.add_argument("--repeat", type=int, default=1, help="Passes over the dataset.")
    args = parser.parse_args()

    config_path = args.models_config or (_PROJECT_ROOT / "configs" / "models.yaml")
    if not config_path.exists():
        print(f"models config not found at {config_path}")  # noqa: T201
        return
    all_names = _load_profile_names(config_path)
    if not all_names:
        print(f"No model profiles defined in {config_path}")  # noqa: T201
        return

    dataset_path = args.dataset or _ask_dataset()
    if not dataset_path:
        return
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")  # noqa: T201
        return

    if args.profiles is not None:
        selected = args.profiles
    else:
        selected = _ask_profiles(config_path) or all_names
    unknown = [name for name in selected if name not in all_names]
    if unknown:
        print(f"Unknown profile(s): {', '.join(unknown)}. Available: {', '.join(all_names)}")  # noqa: T201
        return

    if args.concurrency is not None:
        concurrency, repeat, log_level = args.concurrency, args.repeat, "INFO"
    else:
        concurrency, repeat, log_level = _ask_options()

    BenchmarkDashboard(
        dataset_path=dataset_path,
        config_path=_effective_config(config_path, selected, all_names),
        profiles=selected,
        concurrency=concurrency,
        repeat=repeat,
        log_level=log_level,
    ).run()


if __name__ == "__main__":
    main()
