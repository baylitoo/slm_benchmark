from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich import print

from docie_bench.benchmark.judge import EvaluationMode
from docie_bench.benchmark.registry import (
    DEFAULT_REGISTRY_PATH,
    detect_leakage,
    migrate_manifest,
    register_dataset_version,
    resolve_dataset,
    validate_dataset,
)
from docie_bench.benchmark.runner import run_benchmark
from docie_bench.logging_config import configure_logging
from docie_bench.schemas.extraction import SCHEMA_REGISTRY, schema_json

app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True)
schema_app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(schema_app, name="schema")
app.add_typer(dataset_app, name="dataset")


@benchmark_app.command("run")
def benchmark_run(
    dataset: str | None = typer.Option(None, help="Manifest path or registry reference name@version"),
    dataset_registry: Path = typer.Option(
        DEFAULT_REGISTRY_PATH, help="Versioned dataset registry YAML"
    ),
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
    split: str | None = typer.Option(None, help="Run only this dataset split"),
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
            dataset_registry_path=dataset_registry,
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
            split=split,
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
    print(json.dumps(schema_json(name), indent=2, ensure_ascii=False))


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


@dataset_app.command("validate")
def dataset_validate(
    source: str = typer.Argument(..., help="Manifest path or registry reference name@version"),
    registry: Path = typer.Option(DEFAULT_REGISTRY_PATH),
    near_duplicate_threshold: float = typer.Option(0.92, min=0, max=1),
) -> None:
    try:
        direct_path = Path(source)
        if direct_path.is_file():
            report = validate_dataset(
                direct_path,
                near_duplicate_threshold=near_duplicate_threshold,
            )
            _print_json(report)
            if not report["valid"]:
                raise typer.Exit(code=1)
            return
        resolved = resolve_dataset(source, registry_path=registry)
        report = validate_dataset(
            resolved.manifest_path,
            near_duplicate_threshold=near_duplicate_threshold,
            expected_hash=resolved.dataset_hash if resolved.version else None,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_json(report)
    if not report["valid"]:
        raise typer.Exit(code=1)


@dataset_app.command("inspect")
def dataset_inspect(
    source: str = typer.Argument(..., help="Manifest path or registry reference name@version"),
    registry: Path = typer.Option(DEFAULT_REGISTRY_PATH),
) -> None:
    try:
        direct_path = Path(source)
        if direct_path.is_file():
            _print_json({"manifest_path": direct_path.resolve(), **validate_dataset(direct_path)})
            return
        resolved = resolve_dataset(source, registry_path=registry)
        report = validate_dataset(resolved.manifest_path)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_json(
        {
            "reference": resolved.reference,
            "version": resolved.version,
            "manifest_path": resolved.manifest_path,
            "dataset_hash": resolved.dataset_hash,
            **report,
        }
    )


@dataset_app.command("leakage")
def dataset_leakage(
    source: str = typer.Argument(..., help="Manifest path or registry reference name@version"),
    registry: Path = typer.Option(DEFAULT_REGISTRY_PATH),
    near_duplicate_threshold: float = typer.Option(0.92, min=0, max=1),
) -> None:
    try:
        resolved = resolve_dataset(source, registry_path=registry)
        report = detect_leakage(resolved.items, near_duplicate_threshold)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_json(report)
    if report["leakage_pairs"]:
        raise typer.Exit(code=1)


@dataset_app.command("version")
def dataset_version(
    name: str = typer.Argument(...),
    version: str = typer.Argument(...),
    manifest: Path = typer.Option(..., exists=True, readable=True),
    registry: Path = typer.Option(DEFAULT_REGISTRY_PATH),
    description: str | None = typer.Option(None),
) -> None:
    try:
        entry = register_dataset_version(
            registry_path=registry,
            name=name,
            version=version,
            manifest_path=manifest,
            description=description,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_json({"reference": f"{name}@{version}", **entry.model_dump(mode="json")})


@dataset_app.command("migrate")
def dataset_migrate(
    source: Path = typer.Argument(..., exists=True, readable=True),
    destination: Path = typer.Argument(...),
    default_split: str = typer.Option("test"),
    split_map: Path | None = typer.Option(None, exists=True, readable=True),
) -> None:
    try:
        mapping = (
            json.loads(split_map.read_text(encoding="utf-8")) if split_map is not None else None
        )
        migrated = migrate_manifest(
            source,
            destination,
            default_split=default_split,
            split_map=mapping,
        )
        report = validate_dataset(migrated)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _print_json({"manifest_path": migrated, **report})
