"""Live serving dashboard — watch local model deployments and their health.

Launch with:  docie-serve-dash

The serving control plane is the *other half* of this framework: it acquires,
plans, and operates local model deployments. This is a read-only, auto-refreshing
view over ``ControlPlane`` — it polls ``list_deployments`` and renders one colour-coded
row per deployment (desired vs. actual state, health, endpoint). It never mutates
state; use the ``docie`` CLI for ``serve`` / ``start`` / ``stop``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static
from textual.worker import Worker, WorkerState

# LifecycleState value → Rich colour for the State cell.
_STATE_COLOUR = {
    "ready": "green",
    "starting": "yellow",
    "degraded": "dark_orange",
    "failed": "red",
    "stopped": "dim",
}


class ServingDashboard(App):
    """A live, read-only view of the serving control plane's deployments."""

    TITLE = "docie-serving"
    SUB_TITLE = "live deployment dashboard"

    CSS = """
    Screen { layout: vertical; }
    #summary { padding: 0 1; color: $text-muted; }
    #deployments { height: 1fr; margin: 1; }
    #hint { padding: 0 1; color: $text-muted; }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh now"),
        ("q", "quit", "Quit"),
    ]

    _COLUMNS = [
        ("name", "Deployment"),
        ("model", "Model"),
        ("runtime", "Runtime"),
        ("desired", "Desired"),
        ("state", "State"),
        ("health", "Health"),
        ("endpoint", "Endpoint"),
        ("restarts", "Restarts"),
    ]

    def __init__(self, *, refresh_interval: float = 2.0, plane_factory=None) -> None:
        super().__init__()
        self._refresh_interval = refresh_interval
        # Injectable for tests; defaults to the real local control plane.
        self._plane_factory = plane_factory
        self._last_error: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static("Loading control plane…", id="summary")
            yield DataTable(id="deployments", zebra_stripes=True, cursor_type="row")
            yield Static("", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#deployments", DataTable)
        for key, label in self._COLUMNS:
            table.add_column(label, key=key)
        self.query_one("#hint", Static).update(
            "[dim]read-only · use the [b]docie[/b] CLI to serve / start / stop[/]"
        )
        self.action_refresh()
        self.set_interval(self._refresh_interval, self.action_refresh)

    def action_refresh(self) -> None:
        self.refresh_worker()

    # ── data fetch (off the UI thread) ──────────────────────────────────────

    @work(thread=True, exclusive=True, name="refresh", exit_on_error=False)
    def refresh_worker(self) -> dict[str, Any]:
        from docie_bench.serving.control_plane import ControlPlane

        plane = (self._plane_factory or ControlPlane.from_defaults)()

        async def _gather() -> dict[str, Any]:
            return {
                "deployments": await plane.list_deployments(),
                "models": await plane.list_models(),
                "runtimes": await plane.list_runtimes(),
            }

        return asyncio.run(_gather())

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "refresh":
            return
        if event.state == WorkerState.SUCCESS:
            self._render(event.worker.result)
        elif event.state == WorkerState.ERROR:
            self.query_one("#summary", Static).update(
                f"[red]Control plane error:[/] {event.worker.error!r}"
            )

    # ── rendering ───────────────────────────────────────────────────────────

    def _render(self, data: dict[str, Any]) -> None:
        deployments = _as_list(data.get("deployments"))
        models = _as_list(data.get("models"))
        runtimes = _as_list(data.get("runtimes"))
        now = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005 (display only)
        running = sum(1 for d in deployments if _field(d, "state") == "ready")
        self.query_one("#summary", Static).update(
            f"deployments [b]{len(deployments)}[/] ([green]{running} ready[/])   "
            f"models [b]{len(models)}[/]   runtimes [b]{len(runtimes)}[/]   "
            f"[dim]refreshed {now}[/]"
        )

        table = self.query_one("#deployments", DataTable)
        table.clear()
        for d in sorted(deployments, key=lambda r: str(_deployment_name(r))):
            table.add_row(*self._row(d))

    def _row(self, d: dict[str, Any]) -> list[Any]:
        spec = d.get("spec") or {}
        launch = (spec.get("launch") if isinstance(spec, dict) else {}) or {}
        state = str(_field(d, "state") or "—")
        fails = int(d.get("consecutive_health_failures", 0) or 0)
        if state == "ready" and not fails:
            health = Text("ok", style="green")
        elif fails:
            health = Text(f"{fails} fail(s)", style="red" if state == "failed" else "yellow")
        else:
            health = Text("—", style="dim")
        return [
            _deployment_name(d),
            launch.get("model", "—"),
            launch.get("runtime", "—"),
            str(spec.get("desired_state", "—")),
            Text(state, style=_STATE_COLOUR.get(state, "white")),
            health,
            d.get("endpoint") or "—",
            str(d.get("restart_count", 0)),
        ]


def _deployment_name(d: dict[str, Any]) -> str:
    spec = d.get("spec") or {}
    if isinstance(spec, dict) and spec.get("name"):
        return str(spec["name"])
    return str(d.get("name", "?"))


def _field(d: dict[str, Any], key: str) -> Any:
    return d.get(key) if isinstance(d, dict) else None


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def main() -> None:
    ServingDashboard().run()


if __name__ == "__main__":
    main()
