from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich import print

from docie_bench.benchmark.runner import run_benchmark
from docie_bench.logging_config import configure_logging
from docie_bench.schemas.extraction import SCHEMA_REGISTRY, schema_json

app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True)
schema_app = typer.Typer(no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(schema_app, name="schema")


@benchmark_app.command("run")
def benchmark_run(
    dataset: Path = typer.Option(..., exists=True, readable=True),
    models_config: Path = typer.Option(Path("configs/models.yaml"), exists=True, readable=True),
    model_profile: str | None = typer.Option(None),
    output_dir: Path | None = typer.Option(None),
    concurrency: int = typer.Option(1, min=1, max=32),
    repeat: int = typer.Option(1, min=1, help="Repeat the dataset N times (useful for stress testing)"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG shows full prompts and LLM output)"),
) -> None:
    configure_logging(log_level)
    result = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models_config,
            model_profile=model_profile,
            output_dir=output_dir,
            concurrency=concurrency,
            repeat=repeat,
        )
    )
    print(f"[green]Benchmark complete[/green]: {result.run_dir}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Report: {result.report_path}")


@schema_app.command("list")
def list_schemas() -> None:
    for name in sorted(SCHEMA_REGISTRY):
        print(name)


@schema_app.command("show")
def show_schema(name: str) -> None:
    import json

    print(json.dumps(schema_json(name), indent=2, ensure_ascii=False))
