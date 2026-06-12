from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from docie_bench.benchmark.comparison import (
    compare_runs,
    list_baselines,
    promote_baseline,
    resolve_run,
)
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
from docie_bench.ocr.runner import run_ocr_benchmark
from docie_bench.schemas.extraction import SCHEMA_REGISTRY, schema_json

app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True)
baseline_app = typer.Typer(no_args_is_help=True)
ocr_app = typer.Typer(no_args_is_help=True)
schema_app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
app.add_typer(benchmark_app, name="benchmark")
benchmark_app.add_typer(baseline_app, name="baseline")
benchmark_app.add_typer(ocr_app, name="ocr")
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
    routing_policy: Path | None = typer.Option(
        None,
        exists=True,
        readable=True,
        help="Declarative routing policy (YAML/JSON). Runs each document through the "
        "multi-stage router; stage names must match profiles in the models config.",
    ),
    eval_mode: EvaluationMode = typer.Option(EvaluationMode.GROUND_TRUTH),
    judge_profile: str | None = typer.Option(
        None, help="Judge profile; defaults to judge.profile in models config"
    ),
    output_dir: Path | None = typer.Option(None),
    concurrency: int = typer.Option(1, min=1, max=32),
    repeat: int = typer.Option(1, min=1, help="Repeat the dataset N times (useful for stress testing)"),
    split: str | None = typer.Option(None, help="Run only this dataset split"),
    resume: bool = typer.Option(False, help="Resume an interrupted run (requires --output-dir)"),
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG shows full prompts and LLM output)"),
) -> None:
    if (dataset is None) == (document is None):
        raise typer.BadParameter("Provide exactly one of --dataset or --document")
    if document is not None and not eval_mode.uses_judge:
        raise typer.BadParameter("--document requires --eval-mode llm_judge or both")
    if resume and output_dir is None:
        raise typer.BadParameter("--resume requires --output-dir")
    if routing_policy is not None and model_profile is not None:
        raise typer.BadParameter("--routing-policy cannot be combined with --model-profile")
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
            resume=resume,
            routing_policy_path=routing_policy,
        )
    )
    print(f"[green]Benchmark complete[/green]: {result.run_dir}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Report: {result.report_path}")
    print(f"Manifest: {result.manifest_path}")


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
def show_schema(
    names: list[str] = typer.Argument(..., help="One or more schema names to show"),
) -> None:
    for name in names:
        if len(names) > 1:
            print(f"# {name}")
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
