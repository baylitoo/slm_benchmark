from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich import print

from docie_bench.benchmark.comparison import (
    compare_runs,
    list_baselines,
    promote_baseline,
    resolve_run,
)
from docie_bench.benchmark.judge import EvaluationMode
from docie_bench.benchmark.runner import run_benchmark
from docie_bench.logging_config import configure_logging
from docie_bench.schemas.extraction import SCHEMA_REGISTRY, schema_json

app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True)
baseline_app = typer.Typer(no_args_is_help=True)
schema_app = typer.Typer(no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark")
benchmark_app.add_typer(baseline_app, name="baseline")
app.add_typer(schema_app, name="schema")


@benchmark_app.command("run")
def benchmark_run(
    dataset: Path | None = typer.Option(None, exists=True, readable=True),
    document: Path | None = typer.Option(None, exists=True, readable=True),
    schema_name: str = typer.Option("invoice", help="Schema used with --document"),
    language: str | None = typer.Option(None, help="Document language used with --document"),
    models_config: Path = typer.Option(Path("configs/models.yaml"), exists=True, readable=True),
    model_profile: str | None = typer.Option(None),
    eval_mode: EvaluationMode = typer.Option(EvaluationMode.GROUND_TRUTH),
    judge_profile: str | None = typer.Option(
        None, help="Judge profile; defaults to judge.profile in models config"
    ),
    output_dir: Path | None = typer.Option(None),
    concurrency: int = typer.Option(1, min=1, max=32),
    repeat: int = typer.Option(
        1, min=1, help="Repeat the dataset N times (useful for stress testing)"
    ),
    log_level: str = typer.Option(
        "INFO", help="Logging level (DEBUG shows full prompts and LLM output)"
    ),
) -> None:
    if (dataset is None) == (document is None):
        raise typer.BadParameter("Provide exactly one of --dataset or --document")
    if document is not None and not eval_mode.uses_judge:
        raise typer.BadParameter("--document requires --eval-mode llm_judge or both")
    configure_logging(log_level)
    result = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models_config,
            model_profile=model_profile,
            output_dir=output_dir,
            concurrency=concurrency,
            repeat=repeat,
            eval_mode=eval_mode,
            judge_profile=judge_profile,
            document_path=document,
            schema_name=schema_name,
            language=language,
        )
    )
    print(f"[green]Benchmark complete[/green]: {result.run_dir}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Report: {result.report_path}")


@benchmark_app.command("compare")
def benchmark_compare(
    baseline: str = typer.Argument(
        ..., help="Run path or named baseline (optionally name@version)"
    ),
    candidate: str = typer.Argument(..., help="Candidate run path"),
    budgets: Path | None = typer.Option(None, exists=True, readable=True),
    output_dir: Path = typer.Option(Path("comparison")),
    registry_dir: Path = typer.Option(Path(".benchmarks/baselines")),
) -> None:
    try:
        result = compare_runs(
            resolve_run(baseline, registry_dir=registry_dir),
            resolve_run(candidate, registry_dir=registry_dir),
            output_dir=output_dir,
            budgets_path=budgets,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    color = "green" if result.verdict == "pass" else "red"
    print(f"[{color}]Comparison verdict: {result.verdict.upper()}[/{color}]")
    print(f"Verdict: {result.verdict_path}")
    print(f"Report: {result.report_path}")
    if result.exit_code:
        raise typer.Exit(result.exit_code)


@baseline_app.command("promote")
def baseline_promote(
    run: Path = typer.Argument(..., exists=True, readable=True),
    name: str = typer.Argument(...),
    registry_dir: Path = typer.Option(Path(".benchmarks/baselines")),
) -> None:
    try:
        entry = promote_baseline(run, name, registry_dir=registry_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print(json.dumps(entry, indent=2))


@baseline_app.command("list")
def baseline_list(registry_dir: Path = typer.Option(Path(".benchmarks/baselines"))) -> None:
    print(json.dumps(list_baselines(registry_dir), indent=2))


@schema_app.command("list")
def list_schemas() -> None:
    for name in sorted(SCHEMA_REGISTRY):
        print(name)


@schema_app.command("show")
def show_schema(name: str) -> None:
    print(json.dumps(schema_json(name), indent=2, ensure_ascii=False))
