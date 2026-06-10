from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docie_bench.benchmark.dataset import DatasetItem, load_dataset
from docie_bench.benchmark.judge import EvaluationMode, judge_extraction
from docie_bench.benchmark.metrics import score_evidence, score_prediction
from docie_bench.benchmark.report import write_report
from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile, load_judge_profile, load_model_profiles
from docie_bench.ocr.factory import get_ocr_backend
from docie_bench.settings import get_settings

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


class CpuSampler:
    """Samples system-wide CPU% every `interval` seconds in a background thread."""

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = 0.0

    def __enter__(self) -> "CpuSampler":
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


async def run_benchmark(
    *,
    dataset_path: Path | None,
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
) -> BenchmarkResult:
    settings = get_settings()
    if (dataset_path is None) == (document_path is None):
        raise ValueError("Provide exactly one of dataset_path or document_path")
    if dataset_path is not None:
        base_items = load_dataset(dataset_path)
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
    items = [
        item.model_copy(update={"doc_id": f"{item.doc_id}_r{i}"}) if repeat > 1 else item
        for i in range(repeat)
        for item in base_items
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
    run_dir = output_dir or settings.runs_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"

    sem = asyncio.Semaphore(concurrency)
    prediction_rows: list[dict[str, Any]] = []

    async def run_one(profile: ModelProfile, item: DatasetItem) -> dict[str, Any]:
        async with sem:
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
                    "doc_id": item.doc_id,
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
                    "ground_truth": item.ground_truth,
                    "score": score_evidence(response.result),
                }
                if eval_mode.uses_ground_truth:
                    row["score"] = score_prediction(item.ground_truth, response.result)
                if selected_judge is not None:
                    try:
                        backend = get_ocr_backend(
                            settings.default_ocr_backend,
                            language=item.language,
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
            except Exception as exc:  # benchmark must continue unless caller chooses fail-fast later
                return {
                    "doc_id": item.doc_id,
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

    tasks = [run_one(profile, item) for profile in selected_profiles for item in items]

    wall_start = time.perf_counter()
    with CpuSampler() as sampler:
        for coro in asyncio.as_completed(tasks):
            row = await coro
            prediction_rows.append(row)
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    wall_seconds = time.perf_counter() - wall_start

    metrics = summarize(
        prediction_rows,
        sampler.samples,
        wall_seconds=wall_seconds,
        concurrency=concurrency,
        eval_mode=eval_mode,
    )
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, default=str), encoding="utf-8")
    report_path = write_report(run_dir, metrics)
    return BenchmarkResult(run_dir, predictions_path, metrics_path, report_path)


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
        sim_values = [
            r["score"]["avg_similarity"]
            for r in ok_rows
            if r.get("score", {}).get("avg_similarity") is not None
        ]
        judge_rows = [r["judge"] for r in ok_rows if r.get("judge")]
        faithfulness = [r["overall_faithfulness"] for r in judge_rows]
        completeness = [r["overall_completeness"] for r in judge_rows]
        field_accuracy = field_correct / field_total if field_total else None
        judge_faithfulness = (
            round(sum(faithfulness) / len(faithfulness), 4) if faithfulness else None
        )
        n = len(profile_rows)
        summary.append(
            {
                "model_profile": profile,
                "ingestion_path": profile_rows[0].get("ingestion_path", "unknown"),
                "docs": n,
                "concurrency": concurrency,
                "wall_seconds": round(wall_seconds, 1),
                "throughput_docs_per_min": round(n / wall_seconds * 60, 2) if wall_seconds else None,
                "ok_rate": len(ok_rows) / n if n else 0,
                "valid_rate": len(valid_rows) / n if n else 0,
                "field_accuracy": field_accuracy,
                "evidence_coverage": evidence_grounded / evidence_total if evidence_total else None,
                "hallucination_rate": (
                    (evidence_total - evidence_grounded) / evidence_total
                    if evidence_total
                    else None
                ),
                "avg_similarity": round(sum(sim_values) / len(sim_values), 4) if sim_values else None,
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
