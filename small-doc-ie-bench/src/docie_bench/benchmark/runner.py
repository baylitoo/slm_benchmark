from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docie_bench.benchmark.dataset import DatasetItem
from docie_bench.benchmark.judge import EvaluationMode, judge_extraction
from docie_bench.benchmark.metrics import score_evidence, score_prediction
from docie_bench.benchmark.registry import DEFAULT_REGISTRY_PATH, resolve_dataset
from docie_bench.benchmark.report import write_report
from docie_bench.benchmark.reproducibility import (
    MANIFEST_VERSION,
    TERMINAL_TASK_STATES,
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    dependency_snapshot,
    git_snapshot,
    hash_file,
    load_jsonl_recover,
    profile_snapshot,
    stable_hash,
    system_snapshot,
    utc_now,
    validate_resume_manifest,
    write_manifest,
)
from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile, load_judge_profile, load_model_profiles
from docie_bench.ocr.factory import get_ocr_backend
from docie_bench.ocr.service import processor_from_settings
from docie_bench.settings import get_settings

try:
    import psutil as _psutil

    _HAS_PSUTIL = True
except ImportError:
    _psutil = None
    _HAS_PSUTIL = False


class CpuSampler:
    """Samples system-wide CPU% every `interval` seconds in a background thread."""

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = 0.0

    def __enter__(self) -> CpuSampler:
        if not _HAS_PSUTIL:
            return self
        self._t0 = time.time()
        _psutil.cpu_percent()  # prime — first call always returns 0.0
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    @property
    def samples(self) -> list[tuple[float, float]]:
        return list(self._samples)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            if _psutil is not None:
                self._samples.append((time.time() - self._t0, _psutil.cpu_percent()))


@dataclass(frozen=True)
class BenchmarkResult:
    run_dir: Path
    predictions_path: Path
    metrics_path: Path
    report_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    profile: ModelProfile
    item: DatasetItem
    source_doc_id: str
    repetition: int
    document_hash: str


async def run_benchmark(
    *,
    dataset_path: str | Path | None,
    dataset_registry_path: Path = DEFAULT_REGISTRY_PATH,
    models_config_path: Path,
    model_profile: str | None = None,
    output_dir: Path | None = None,
    concurrency: int = 1,
    repeat: int = 1,
    eval_mode: EvaluationMode = EvaluationMode.GROUND_TRUTH,
    judge_profile: str | None = None,
    document_path: Path | None = None,
    schema_name: str = "invoice",
    language: str | None = None,
    split: str | None = None,
    resume: bool = False,
) -> BenchmarkResult:
    settings = get_settings()
    if resume and output_dir is None:
        raise ValueError("Resume requires an explicit output_dir")
    if (dataset_path is None) == (document_path is None):
        raise ValueError("Provide exactly one of dataset_path or document_path")
    dataset_metadata: dict[str, Any] | None = None
    if dataset_path is not None:
        resolved_dataset = resolve_dataset(dataset_path, registry_path=dataset_registry_path)
        base_items = resolved_dataset.items
        dataset_metadata = {
            "reference": resolved_dataset.reference,
            "version": resolved_dataset.version,
            "manifest_path": str(resolved_dataset.manifest_path),
            "dataset_hash": resolved_dataset.dataset_hash,
            "selected_split": split,
        }
        if split is not None:
            base_items = [item for item in base_items if item.split == split]
            if not base_items:
                raise ValueError(
                    f"Dataset {resolved_dataset.reference} has no documents in split {split!r}"
                )
    else:
        assert document_path is not None
        base_items = [
            DatasetItem(
                doc_id=document_path.stem,
                file_path=str(document_path.resolve()),
                schema_name=schema_name,
                language=language,
            )
        ]
    profiles = load_model_profiles(models_config_path)
    selected_judge = (
        load_judge_profile(models_config_path, judge_profile) if eval_mode.uses_judge else None
    )
    selected_profiles = [profiles[model_profile]] if model_profile else list(profiles.values())
    if selected_judge is not None and model_profile is None:
        selected_profiles = [
            profile for profile in selected_profiles if profile.name != selected_judge.name
        ]
    if not selected_profiles:
        raise ValueError("No extraction model profiles selected")

    document_hashes = {item.file_path: hash_file(Path(item.file_path)) for item in base_items}
    task_config = {
        "eval_mode": eval_mode.value,
        "judge_profile": profile_snapshot(selected_judge) if selected_judge else None,
        "ocr_backend": settings.default_ocr_backend,
    }
    tasks: list[BenchmarkTask] = []
    for profile in selected_profiles:
        for repetition in range(repeat):
            for item in base_items:
                repeated_item = (
                    item.model_copy(update={"doc_id": f"{item.doc_id}_r{repetition}"})
                    if repeat > 1
                    else item
                )
                identity = {
                    "source_doc_id": item.doc_id,
                    "repetition": repetition,
                    "document_hash": document_hashes[item.file_path],
                    "dataset_item": item.model_dump(exclude={"file_path"}),
                    "model_profile": profile_snapshot(profile),
                    "task_config": task_config,
                }
                tasks.append(
                    BenchmarkTask(
                        task_id=stable_hash(identity),
                        profile=profile,
                        item=repeated_item,
                        source_doc_id=item.doc_id,
                        repetition=repetition,
                        document_hash=document_hashes[item.file_path],
                    )
                )
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(
            "Benchmark task identity collision; dataset rows must be uniquely identified"
        )

    git = git_snapshot(Path(__file__).resolve().parents[3])
    inputs = {
        "git": git,
        "models_config_hash": hash_file(models_config_path),
        "selected_profiles": [profile_snapshot(profile) for profile in selected_profiles],
        "judge_profile": profile_snapshot(selected_judge) if selected_judge else None,
        "dataset": {
            "kind": "dataset" if dataset_path is not None else "document",
            "source_hash": hash_file(dataset_path) if dataset_path is not None else None,
            "items": [
                {
                    **item.model_dump(exclude={"file_path"}),
                    "document_hash": document_hashes[item.file_path],
                }
                for item in base_items
            ],
        },
        "task_config": task_config,
        "repeat": repeat,
    }
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": utc_now(),
        "input_fingerprint": stable_hash(inputs),
        "inputs": inputs,
        "invocation": {
            "dataset_path": str(dataset_path.resolve()) if dataset_path else None,
            "document_path": str(document_path.resolve()) if document_path else None,
            "models_config_path": str(models_config_path.resolve()),
            "model_profile": model_profile,
            "output_dir": str(output_dir.resolve()) if output_dir else None,
            "concurrency": concurrency,
            "repeat": repeat,
            "eval_mode": eval_mode.value,
            "judge_profile": judge_profile,
            "schema_name": schema_name,
            "language": language,
        },
        "environment": {
            "dependencies": dependency_snapshot(),
            "system": system_snapshot(),
        },
        "task_ids": task_ids,
    }

    default_name = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    run_dir = output_dir or settings.runs_dir / default_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    predictions_path = run_dir / "predictions.jsonl"
    events_path = run_dir / "task-events.jsonl"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "report.html"

    if resume:
        if not manifest_path.exists():
            raise ValueError(f"Cannot resume run without manifest: {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resume_warnings = validate_resume_manifest(
            existing_manifest, manifest, (task.task_id for task in tasks)
        )
    else:
        resume_warnings = []
        existing_files = [path for path in run_dir.iterdir() if path.is_file()]
        if existing_files:
            raise FileExistsError(
                f"Refusing to start a new run in non-empty directory {run_dir}; use resume=True"
            )
        write_manifest(manifest_path, manifest)

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    load_jsonl_recover(events_path)
    prediction_rows = load_jsonl_recover(predictions_path)
    task_rows: dict[str, dict[str, Any]] = {}
    for row in prediction_rows:
        task_id = row.get("task_id")
        if not task_id:
            raise ValueError(f"Prediction row is missing task_id in {predictions_path}")
        if task_id in task_rows:
            raise ValueError(f"Duplicate completed task_id {task_id} in {predictions_path}")
        task_rows[task_id] = row
    unknown_task_ids = set(task_rows) - set(task_ids)
    if unknown_task_ids:
        raise ValueError(
            f"Predictions contain {len(unknown_task_ids)} task(s) not present in the run manifest"
        )
    completed_task_ids = {
        task_id
        for task_id, row in task_rows.items()
        if row.get("task_state") in TERMINAL_TASK_STATES
    }
    pending_tasks = [task for task in tasks if task.task_id not in completed_task_ids]
    if resume and not pending_tasks and metrics_path.exists() and report_path.exists():
        return BenchmarkResult(run_dir, predictions_path, metrics_path, report_path, manifest_path)

    async def record_event(task: BenchmarkTask, state: str, **details: Any) -> None:
        async with write_lock:
            append_jsonl(
                events_path,
                {
                    "task_id": task.task_id,
                    "state": state,
                    "at": utc_now(),
                    "model_profile": task.profile.name,
                    "doc_id": task.item.doc_id,
                    **details,
                },
            )

    async def run_one(task: BenchmarkTask) -> dict[str, Any]:
        async with sem:
            await record_event(task, "running")
            profile = task.profile
            item = task.item
            service = ExtractionService(profile)
            started = time.perf_counter()
            try:
                response = await service.extract_from_file(
                    path=Path(item.file_path),
                    ocr_backend_name=settings.default_ocr_backend,
                    schema_name=item.schema_name,
                    schema_mode=item.schema_mode,
                    dynamic_schema=item.dynamic_schema,
                    language=item.language,
                    metadata={"doc_id": item.doc_id, **item.metadata},
                )
                row = {
                    "task_id": task.task_id,
                    "task_state": "completed",
                    "doc_id": item.doc_id,
                    "split": item.split,
                    "schema_name": response.schema_name,
                    "language": item.language,
                    "dynamic_schema": response.dynamic_schema,
                    "model_profile": profile.name,
                    "ingestion_path": (
                        "vision" if profile.vision else f"ocr:{settings.default_ocr_backend}"
                    ),
                    "ok": True,
                    "latency_ms": response.latency_ms,
                    "validation": response.validation.model_dump(),
                    "prediction": response.result,
                    "routing": getattr(response, "routing", None),
                    "ground_truth": item.ground_truth,
                    "score": score_evidence(response.result),
                }
                if eval_mode.uses_ground_truth:
                    row["score"] = score_prediction(item.ground_truth, response.result)
                if selected_judge is not None:
                    try:
                        if hasattr(settings, "ocr_cache_enabled"):
                            ocr_result = processor_from_settings(settings).process(
                                Path(item.file_path),
                                backend_name=settings.default_ocr_backend,
                                language=item.language,
                            )
                            blocks = ocr_result.artifact.blocks
                        else:
                            backend = get_ocr_backend(
                                settings.default_ocr_backend, language=item.language
                            )
                            blocks = backend.extract(Path(item.file_path))
                        document_text = "\n".join(block.text for block in blocks)
                        row["judge"] = await judge_extraction(
                            profile=selected_judge,
                            document_text=document_text,
                            extraction=response.result,
                        )
                    except Exception as exc:
                        row["judge_error"] = repr(exc)
                return row
            except Exception as exc:
                # A benchmark continues unless the caller chooses fail-fast later.
                return {
                    "task_id": task.task_id,
                    "task_state": "failed",
                    "doc_id": item.doc_id,
                    "split": item.split,
                    "schema_name": item.schema_name,
                    "language": item.language,
                    "dynamic_schema": item.dynamic_schema,
                    "model_profile": profile.name,
                    "ingestion_path": (
                        "vision" if profile.vision else f"ocr:{settings.default_ocr_backend}"
                    ),
                    "ok": False,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "error": repr(exc),
                    "ground_truth": item.ground_truth,
                }

    coroutines = [run_one(task) for task in pending_tasks]
    tasks_by_id = {task.task_id: task for task in tasks}

    wall_start = time.perf_counter()
    with CpuSampler() as sampler:
        for task in pending_tasks:
            await record_event(task, "queued")
        for coro in asyncio.as_completed(coroutines):
            row = await coro
            prediction_rows.append(row)
            async with write_lock:
                append_jsonl(predictions_path, row)
            event_details = {"reason": row["error"]} if row["task_state"] == "failed" else {}
            await record_event(tasks_by_id[row["task_id"]], row["task_state"], **event_details)
    wall_seconds = time.perf_counter() - wall_start
    task_order = {task.task_id: index for index, task in enumerate(tasks)}
    prediction_rows.sort(key=lambda row: task_order[row["task_id"]])
    atomic_write_text(
        predictions_path,
        "".join(f"{canonical_json(row)}\n" for row in prediction_rows),
    )

    metrics = summarize(
        prediction_rows,
        sampler.samples,
        wall_seconds=wall_seconds,
        concurrency=concurrency,
        eval_mode=eval_mode,
    )
    metrics["dataset"] = dataset_metadata
    metrics["reproducibility"] = {
        "resumed": resume,
        "tasks_skipped": len(completed_task_ids),
        "tasks_executed": len(pending_tasks),
        "task_count": len(tasks),
        "input_fingerprint": manifest.get("input_fingerprint"),
        "manifest_path": str(manifest_path),
        "warnings": resume_warnings,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, default=str), encoding="utf-8")
    report_path = write_report(run_dir, metrics)
    return BenchmarkResult(run_dir, predictions_path, metrics_path, report_path, manifest_path)


def summarize(
    rows: list[dict[str, Any]],
    cpu_samples: list[tuple[float, float]] | None = None,
    wall_seconds: float = 0.0,
    concurrency: int = 1,
    eval_mode: EvaluationMode = EvaluationMode.GROUND_TRUTH,
) -> dict[str, Any]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_profile.setdefault(row["model_profile"], []).append(row)
    summary = []
    for profile, profile_rows in by_profile.items():
        ok_rows = [r for r in profile_rows if r.get("ok")]
        valid_rows = [r for r in ok_rows if r.get("validation", {}).get("valid")]
        field_total = sum(r.get("score", {}).get("field_total", 0) for r in ok_rows)
        field_correct = sum(r.get("score", {}).get("field_correct", 0) for r in ok_rows)
        evidence_total = sum(
            r.get("score", {}).get("evidence_field_total", 0) for r in ok_rows
        )
        evidence_grounded = sum(
            r.get("score", {}).get("evidence_grounded", 0) for r in ok_rows
        )
        row_expected = sum(r.get("score", {}).get("row_expected", 0) for r in ok_rows)
        row_predicted = sum(r.get("score", {}).get("row_predicted", 0) for r in ok_rows)
        row_correct = sum(r.get("score", {}).get("row_correct", 0) for r in ok_rows)
        evidence_row_total = sum(r.get("score", {}).get("evidence_row_total", 0) for r in ok_rows)
        evidence_rows_grounded = sum(
            r.get("score", {}).get("evidence_rows_grounded", 0) for r in ok_rows
        )
        sim_values = [
            r["score"]["avg_similarity"]
            for r in ok_rows
            if r.get("score", {}).get("avg_similarity") is not None
        ]
        judge_rows = [r["judge"] for r in ok_rows if r.get("judge")]
        faithfulness = [r["overall_faithfulness"] for r in judge_rows]
        completeness = [r["overall_completeness"] for r in judge_rows]
        field_accuracy = field_correct / field_total if field_total else None
        row_precision = row_correct / row_predicted if row_predicted else None
        row_recall = row_correct / row_expected if row_expected else None
        judge_faithfulness = (
            round(sum(faithfulness) / len(faithfulness), 4) if faithfulness else None
        )
        n = len(profile_rows)
        routed_rows = [r["routing"] for r in profile_rows if r.get("routing")]
        routed_n = len(routed_rows)
        routed_stages = [stage for route in routed_rows for stage in route.get("stages", [])]
        summary.append(
            {
                "model_profile": profile,
                "ingestion_path": profile_rows[0].get("ingestion_path", "unknown"),
                "docs": n,
                "concurrency": concurrency,
                "wall_seconds": round(wall_seconds, 1),
                "throughput_docs_per_min": (
                    round(n / wall_seconds * 60, 2) if wall_seconds else None
                ),
                "ok_rate": len(ok_rows) / n if n else 0,
                "valid_rate": len(valid_rows) / n if n else 0,
                "field_accuracy": field_accuracy,
                "row_precision": row_precision,
                "row_recall": row_recall,
                "row_f1": (
                    2 * row_precision * row_recall / (row_precision + row_recall)
                    if row_precision is not None
                    and row_recall is not None
                    and row_precision + row_recall
                    else None
                ),
                "evidence_coverage": evidence_grounded / evidence_total if evidence_total else None,
                "evidence_row_coverage": (
                    evidence_rows_grounded / evidence_row_total if evidence_row_total else None
                ),
                "hallucination_rate": (
                    (evidence_total - evidence_grounded) / evidence_total
                    if evidence_total
                    else None
                ),
                "avg_similarity": (
                    round(sum(sim_values) / len(sim_values), 4) if sim_values else None
                ),
                "judge_faithfulness": judge_faithfulness,
                "judge_completeness": (
                    round(sum(completeness) / len(completeness), 4) if completeness else None
                ),
                "judge_field_accuracy_delta": (
                    round(judge_faithfulness - field_accuracy, 4)
                    if eval_mode is EvaluationMode.BOTH
                    and judge_faithfulness is not None
                    and field_accuracy is not None
                    else None
                ),
                "judge_model": judge_rows[0]["judge_model"] if judge_rows else None,
                "avg_latency_ms": sum(r.get("latency_ms", 0) for r in profile_rows) / n if n else 0,
                "p50_latency_ms": _percentile([r.get("latency_ms", 0) for r in profile_rows], 50),
                "p95_latency_ms": _percentile([r.get("latency_ms", 0) for r in profile_rows], 95),
                "routing_accept_rate": (
                    sum(r.get("terminal_decision") == "accept" for r in routed_rows) / routed_n
                    if routed_n
                    else None
                ),
                "routing_escalation_rate": (
                    sum(r.get("terminal_decision") == "escalate" for r in routed_rows) / routed_n
                    if routed_n
                    else None
                ),
                "routing_fallback_rate": (
                    sum(r.get("fallback_count", 0) > 0 for r in routed_rows) / routed_n
                    if routed_n
                    else None
                ),
                "routing_budget_exhaustion_rate": (
                    sum(bool(r.get("budget_exhausted")) for r in routed_rows) / routed_n
                    if routed_n
                    else None
                ),
                "avg_routing_attempts": (
                    sum(r.get("attempts", 0) for r in routed_rows) / routed_n
                    if routed_n
                    else None
                ),
                "avg_routing_latency_ms": (
                    sum(r.get("latency_ms", 0) for r in routed_rows) / routed_n
                    if routed_n
                    else None
                ),
                "avg_routing_tokens": (
                    sum(r.get("total_tokens", 0) for r in routed_rows) / routed_n
                    if routed_n
                    else None
                ),
                "avg_routing_cost_units": (
                    sum(r.get("cost_units", 0) for r in routed_rows) / routed_n
                    if routed_n
                    else None
                ),
                "routing_stage_failure_rate": (
                    sum(stage.get("status") == "error" for stage in routed_stages)
                    / len(routed_stages)
                    if routed_stages
                    else None
                ),
            }
        )
    return {
        "summary": summary,
        "rows": rows,
        "cpu_samples": cpu_samples or [],
        "wall_seconds": round(wall_seconds, 1),
        "concurrency": concurrency,
        "eval_mode": eval_mode.value,
    }


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)
