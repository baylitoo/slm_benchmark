from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from docie_bench.benchmark.judge import EvaluationMode
from docie_bench.benchmark.runner import run_benchmark
from docie_bench.logging_config import configure_logging
from docie_bench.ocr.runner import run_ocr_benchmark
from docie_bench.schemas.extraction import SCHEMA_REGISTRY, schema_json

app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True)
ocr_app = typer.Typer(no_args_is_help=True)
schema_app = typer.Typer(no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark")
benchmark_app.add_typer(ocr_app, name="ocr")
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
    repeat: int = typer.Option(1, min=1, help="Repeat the dataset N times (useful for stress testing)"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG shows full prompts and LLM output)"),
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


@ocr_app.command("run")
def ocr_benchmark_run(
    dataset: Annotated[Path, typer.Option(exists=True, readable=True)],
    backend: Annotated[
        list[str], typer.Option("--backend", help="Repeat to compare multiple OCR backends")
    ] = ["pdf_text"],
    output_dir: Annotated[Path | None, typer.Option()] = None,
    cache_dir: Annotated[Path | None, typer.Option()] = None,
    cache_max_mb: Annotated[int | None, typer.Option(min=0)] = None,
    extraction_metrics: Annotated[
        Path | None, typer.Option(exists=True, readable=True)
    ] = None,
) -> None:
    result = run_ocr_benchmark(
        dataset_path=dataset,
        backends=backend,
        output_dir=output_dir,
        cache_dir=cache_dir,
        cache_max_bytes=cache_max_mb * 1024 * 1024 if cache_max_mb is not None else None,
        extraction_metrics_path=extraction_metrics,
    )
    print(f"[green]OCR benchmark complete[/green]: {result.run_dir}")
    print(f"Artifacts: {result.artifacts_path}")
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
