from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docie_bench.benchmark.dataset import DatasetItem, load_dataset
from docie_bench.ocr.cache import OCRCache
from docie_bench.ocr.metrics import score_ocr
from docie_bench.ocr.service import OCRProcessor
from docie_bench.settings import get_settings


@dataclass(frozen=True)
class OCRBenchmarkResult:
    run_dir: Path
    artifacts_path: Path
    metrics_path: Path
    report_path: Path


def run_ocr_benchmark(
    *,
    dataset_path: Path,
    backends: list[str],
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    cache_max_bytes: int | None = None,
    extraction_metrics_path: Path | None = None,
) -> OCRBenchmarkResult:
    settings = get_settings()
    run_dir = output_dir or settings.runs_dir / f"ocr-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_path = run_dir / "ocr-artifacts.jsonl"
    cache = OCRCache(
        cache_dir or settings.ocr_cache_dir,
        max_bytes=cache_max_bytes
        if cache_max_bytes is not None
        else settings.ocr_cache_max_mb * 1024 * 1024,
    )
    processor = OCRProcessor(cache)
    rows: list[dict[str, Any]] = []
    for item in load_dataset(dataset_path):
        reference_text = _reference_text(item)
        for backend in backends:
            started = time.perf_counter()
            try:
                result = processor.process(
                    Path(item.file_path), backend_name=backend, language=item.language
                )
                row = {
                    "doc_id": item.doc_id,
                    "backend": result.artifact.backend,
                    "backend_version": result.artifact.backend_version,
                    "cache_key": result.cache_key,
                    "cache_hit": result.cache_hit,
                    "ok": True,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "ocr_latency_ms": result.artifact.latency_ms,
                    "quality": result.artifact.quality.model_dump(mode="json"),
                    "artifact": result.artifact.model_dump(mode="json"),
                    "score": (
                        score_ocr(reference_text, result.artifact.blocks, item.ocr_reference_blocks)
                        if reference_text is not None
                        else None
                    ),
                }
            except Exception as exc:
                row = {
                    "doc_id": item.doc_id,
                    "backend": backend,
                    "ok": False,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "error": repr(exc),
                }
            rows.append(row)
            with artifacts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    metrics = summarize_ocr(rows, extraction_metrics_path)
    metrics_path = run_dir / "ocr-metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = _write_report(run_dir, metrics)
    return OCRBenchmarkResult(run_dir, artifacts_path, metrics_path, report_path)


def summarize_ocr(
    rows: list[dict[str, Any]], extraction_metrics_path: Path | None = None
) -> dict[str, Any]:
    summary = []
    for backend in sorted({row["backend"] for row in rows}):
        backend_rows = [row for row in rows if row["backend"] == backend]
        ok_rows = [row for row in backend_rows if row.get("ok")]
        scored = [row for row in ok_rows if row.get("score")]
        summary.append(
            {
                "backend": backend,
                "docs": len(backend_rows),
                "ok_rate": len(ok_rows) / len(backend_rows) if backend_rows else 0.0,
                "cache_hit_rate": (
                    sum(bool(row.get("cache_hit")) for row in ok_rows) / len(ok_rows)
                    if ok_rows
                    else 0.0
                ),
                "low_quality_rate": (
                    sum(bool(row["quality"]["low_quality"]) for row in ok_rows) / len(ok_rows)
                    if ok_rows
                    else None
                ),
                "avg_latency_ms": _mean([row["latency_ms"] for row in ok_rows]),
                "character_error_rate": _mean(
                    [row["score"]["character_error_rate"] for row in scored]
                ),
                "word_error_rate": _mean([row["score"]["word_error_rate"] for row in scored]),
                "layout_preservation": _mean(
                    [row["score"]["layout_preservation"] for row in scored]
                ),
            }
        )
    return {
        "summary": summary,
        "rows": rows,
        "correlations": _correlations(rows, extraction_metrics_path),
    }


def _reference_text(item: DatasetItem) -> str | None:
    if item.ocr_reference_text is not None:
        return item.ocr_reference_text
    if item.ocr_reference_path is not None:
        return Path(item.ocr_reference_path).read_text(encoding="utf-8")
    if item.ocr_reference_blocks is not None:
        return "\n".join(block.text for block in item.ocr_reference_blocks)
    return None


def _mean(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(sum(present) / len(present), 6) if present else None


def _correlations(rows: list[dict[str, Any]], path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    extraction = json.loads(path.read_text(encoding="utf-8"))
    accuracy_by_doc = {
        row["doc_id"]: row.get("score", {}).get("field_accuracy")
        for row in extraction.get("rows", [])
        if row.get("score", {}).get("field_accuracy") is not None
    }
    correlations = []
    for backend in sorted({row["backend"] for row in rows}):
        pairs = [
            (row["score"]["character_accuracy"], accuracy_by_doc[row["doc_id"]])
            for row in rows
            if row["backend"] == backend
            and row.get("score", {}).get("character_accuracy") is not None
            and row["doc_id"] in accuracy_by_doc
        ]
        correlations.append(
            {
                "backend": backend,
                "documents": len(pairs),
                "ocr_character_accuracy_vs_field_accuracy": _pearson(pairs),
            }
        )
    return correlations


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs, strict=True)
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return round(numerator / denominator, 6) if denominator else None


def _write_report(run_dir: Path, metrics: dict[str, Any]) -> Path:
    rows = "\n".join(
        "<tr>"
        f"<td>{row['backend']}</td><td>{row['docs']}</td><td>{row['ok_rate']:.1%}</td>"
        f"<td>{row['cache_hit_rate']:.1%}</td><td>{_display(row['character_error_rate'])}</td>"
        f"<td>{_display(row['word_error_rate'])}</td><td>{_display(row['layout_preservation'])}</td>"
        f"<td>{_display(row['avg_latency_ms'])}</td></tr>"
        for row in metrics["summary"]
    )
    path = run_dir / "ocr-report.html"
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>OCR Benchmark</title>"
        "<h1>OCR Benchmark</h1><table border='1' cellpadding='6'><thead><tr>"
        "<th>Backend</th><th>Docs</th><th>OK</th><th>Cache hits</th><th>CER</th>"
        "<th>WER</th><th>Layout</th><th>Avg latency ms</th></tr></thead>"
        f"<tbody>{rows}</tbody></table><h2>Correlations</h2><pre>"
        f"{json.dumps(metrics['correlations'], indent=2)}</pre>",
        encoding="utf-8",
    )
    return path


def _display(value: Any) -> str:
    return "N/A" if value is None else f"{value:.4f}"
