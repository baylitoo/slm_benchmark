"""Interactive benchmark runner.

Launch with:  docie-bench-ui
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import questionary
import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_datasets() -> list[Path]:
    data_dir = _PROJECT_ROOT / "data"
    return sorted(data_dir.rglob("manifest.jsonl")) if data_dir.exists() else []


def _count_docs(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def _load_profile_names(config_path: Path) -> list[str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return list(data.get("profiles", {}).keys())


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "—"
    if ms >= 60_000:
        return f"{ms / 60_000:.1f} min"
    if ms >= 1_000:
        return f"{ms / 1_000:.1f} s"
    return f"{ms:,.0f} ms"


# ── UI sections ───────────────────────────────────────────────────────────────

def _print_header() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]docie-bench[/bold cyan]  [dim]structured document extraction benchmark[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


def _ask_dataset() -> Path | None:
    datasets = _find_datasets()
    choices: list[questionary.Choice] = []
    for p in datasets:
        n = _count_docs(p)
        label = f"{p.relative_to(_PROJECT_ROOT)}  ({n} doc{'s' if n != 1 else ''})"
        choices.append(questionary.Choice(title=label, value=p))
    choices.append(questionary.Choice(title="Enter path manually…", value="__manual__"))

    choice = questionary.select("Select dataset:", choices=choices).ask()
    if choice is None:
        return None
    if choice == "__manual__":
        raw = questionary.path("Dataset path (manifest.jsonl):").ask()
        if not raw:
            return None
        choice = Path(raw)

    if not choice.exists():
        console.print(f"[red]Not found: {choice}[/red]")
        return None
    return choice


def _ask_profiles(config_path: Path) -> list[str]:
    names = _load_profile_names(config_path)
    choices = [questionary.Choice(title=n, value=n) for n in names]
    selected: list[str] | None = questionary.checkbox(
        "Select model profile(s):",
        choices=choices,
        instruction="(space to toggle, enter to confirm)",
    ).ask()
    return selected or []


def _ask_options() -> tuple[int, int, str]:
    concurrency_str: str = questionary.select(
        "Concurrency (parallel requests):",
        choices=["1", "2", "4", "8"],
        default="1",
    ).ask()

    def _validate_repeat(v: str) -> bool | str:
        return True if (v.isdigit() and int(v) >= 1) else "Enter a positive integer"

    repeat_str: str = questionary.text(
        "Repeat dataset N times (1 = single pass):",
        default="1",
        validate=_validate_repeat,
    ).ask()

    log_level: str = questionary.select(
        "Log level:",
        choices=["INFO", "DEBUG"],
        default="INFO",
    ).ask()

    return int(concurrency_str), int(repeat_str), log_level


def _print_config_panel(dataset_path: Path, profiles: list[str], concurrency: int, repeat: int) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold dim", min_width=14)
    grid.add_column()
    n = _count_docs(dataset_path)
    docs_label = str(n) + (f" × {repeat} = {n * repeat}" if repeat > 1 else "")
    grid.add_row("Dataset", str(dataset_path.relative_to(_PROJECT_ROOT)))
    grid.add_row("Docs", docs_label)
    grid.add_row("Models", f"{len(profiles)}  —  " + ", ".join(profiles))
    grid.add_row("Concurrency", str(concurrency))
    console.print()
    console.print(Panel(grid, title="[bold]Run configuration[/bold]", border_style="green"))
    console.print()


# ── benchmark execution ───────────────────────────────────────────────────────

async def _run_one_profile(
    profile_name: str,
    dataset_path: Path,
    config_path: Path,
    concurrency: int,
    repeat: int,
    log_level: str,
) -> dict:
    from docie_bench.benchmark.runner import run_benchmark
    from docie_bench.logging_config import configure_logging

    configure_logging(log_level)
    result = await run_benchmark(
        dataset_path=dataset_path,
        models_config_path=config_path,
        model_profile=profile_name,
        output_dir=None,
        concurrency=concurrency,
        repeat=repeat,
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    summary_row = next(
        (s for s in metrics["summary"] if s["model_profile"] == profile_name), {}
    )
    return {"profile": profile_name, "result": result, "summary_row": summary_row}


def _print_results_table(results: list[dict]) -> None:
    table = Table(
        title="Results",
        box=box.ROUNDED,
        header_style="bold",
        show_lines=False,
    )
    table.add_column("Model Profile", style="cyan", no_wrap=True)
    table.add_column("OK", justify="right")
    table.add_column("Exact Acc.", justify="right")
    table.add_column("Soft Sim.", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Report", style="dim")

    for r in results:
        s = r["summary_row"]
        ok_str = f"{s.get('ok_rate', 0):.0%}"
        acc_str = f"{s['field_accuracy']:.1%}" if s.get("field_accuracy") is not None else "—"
        sim_str = f"{s['avg_similarity']:.1%}" if s.get("avg_similarity") is not None else "—"
        thru_str = f"{s['throughput_docs_per_min']:.1f}/min" if s.get("throughput_docs_per_min") else "—"
        table.add_row(
            s.get("model_profile", r["profile"]),
            ok_str,
            acc_str,
            sim_str,
            _fmt_ms(s.get("p50_latency_ms")),
            _fmt_ms(s.get("p95_latency_ms")),
            thru_str,
            str(r["result"].report_path),
        )

    console.print()
    console.print(table)
    console.print()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _print_header()

    config_path = _PROJECT_ROOT / "configs" / "models.yaml"
    if not config_path.exists():
        console.print(f"[red]configs/models.yaml not found at {config_path}[/red]")
        return

    dataset_path = _ask_dataset()
    if not dataset_path:
        return

    profiles = _ask_profiles(config_path)
    if not profiles:
        console.print("[yellow]No profiles selected. Exiting.[/yellow]")
        return

    concurrency, repeat, log_level = _ask_options()
    _print_config_panel(dataset_path, profiles, concurrency, repeat)

    if not questionary.confirm("Start benchmark?", default=True).ask():
        console.print("[yellow]Aborted.[/yellow]")
        return

    all_results: list[dict] = []
    for i, profile_name in enumerate(profiles, 1):
        console.print(
            f"\n[bold cyan]▶ [{i}/{len(profiles)}][/bold cyan] "
            f"Running [cyan]{profile_name}[/cyan]…"
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"{profile_name}…", total=None)
            try:
                run_result = asyncio.run(
                    _run_one_profile(
                        profile_name, dataset_path, config_path,
                        concurrency=concurrency, repeat=repeat, log_level=log_level,
                    )
                )
                all_results.append(run_result)
                s = run_result["summary_row"]
                acc = f"{s['field_accuracy']:.1%}" if s.get("field_accuracy") is not None else "—"
                console.print(
                    f"  [green]✓[/green]  "
                    f"ok=[bold]{s.get('ok_rate', 0):.0%}[/bold]  "
                    f"field_acc=[bold]{acc}[/bold]  "
                    f"p50={_fmt_ms(s.get('p50_latency_ms'))}  "
                    f"p95={_fmt_ms(s.get('p95_latency_ms'))}"
                )
            except Exception as exc:
                console.print(f"  [red]✗  {exc}[/red]")

    if len(all_results) > 1:
        _print_results_table(all_results)
    elif all_results:
        console.print(f"\n[green]Report:[/green] {all_results[0]['result'].report_path}")

    console.print("[dim]Done.[/dim]\n")


if __name__ == "__main__":
    main()
