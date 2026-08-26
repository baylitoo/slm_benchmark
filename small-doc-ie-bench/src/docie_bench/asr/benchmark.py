from __future__ import annotations

import html
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docie_bench.asr.metrics import TranscriptScore, score_transcript
from docie_bench.asr.models import ASRBackend, TranscriptionOptions
from docie_bench.benchmark.reproducibility import (
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    dependency_snapshot,
    git_snapshot,
    hash_file,
    stable_hash,
    system_snapshot,
    utc_now,
    write_manifest,
)


@dataclass(frozen=True)
class ASRBenchmarkItem:
    id: str
    audio_path: Path
    reference: str
    language: str | None = None
    prompt: str | None = None


@dataclass(frozen=True)
class ASRBenchmarkResult:
    run_dir: Path
    predictions_path: Path
    metrics_path: Path
    manifest_path: Path
    report_path: Path
    metrics: dict[str, Any]


def load_asr_manifest(path: Path) -> list[ASRBenchmarkItem]:
    manifest_path = path.resolve()
    items: list[ASRBenchmarkItem] = []
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {manifest_path}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"ASR manifest line {line_number} must be a JSON object")
            audio_value = row.get("audio")
            reference = row.get("reference")
            if not isinstance(audio_value, str) or not audio_value.strip():
                raise ValueError(f"ASR manifest line {line_number} requires non-empty 'audio'")
            if not isinstance(reference, str):
                raise ValueError(f"ASR manifest line {line_number} requires string 'reference'")
            audio_path = Path(audio_value)
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            audio_path = audio_path.resolve()
            if not audio_path.is_file():
                raise ValueError(
                    f"ASR manifest line {line_number} audio does not exist: {audio_path}"
                )
            item_id = str(row.get("id") or audio_path.stem)
            if item_id in seen_ids:
                raise ValueError(f"Duplicate ASR manifest id {item_id!r} on line {line_number}")
            seen_ids.add(item_id)
            language = row.get("language")
            prompt = row.get("prompt")
            if language is not None and not isinstance(language, str):
                raise ValueError(f"ASR manifest line {line_number} 'language' must be a string")
            if prompt is not None and not isinstance(prompt, str):
                raise ValueError(f"ASR manifest line {line_number} 'prompt' must be a string")
            items.append(
                ASRBenchmarkItem(
                    id=item_id,
                    audio_path=audio_path,
                    reference=reference,
                    language=language,
                    prompt=prompt,
                )
            )
    if not items:
        raise ValueError(f"ASR manifest is empty: {manifest_path}")
    return items


def run_asr_benchmark(
    *,
    manifest_path: Path,
    backend: ASRBackend,
    output_dir: Path | None = None,
    temperature: float = 0.0,
) -> ASRBenchmarkResult:
    items = load_asr_manifest(manifest_path)
    run_dir = output_dir or _default_run_dir()
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    benchmark_manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "report.html"
    for target in (predictions_path, metrics_path, benchmark_manifest_path, report_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite ASR benchmark artifact: {target}")
    run_dir.mkdir(parents=True, exist_ok=True)

    input_records = [
        {
            "id": item.id,
            "audio": str(item.audio_path),
            "sha256": hash_file(item.audio_path),
            "reference": item.reference,
            "language": item.language,
            "prompt": item.prompt,
        }
        for item in items
    ]
    benchmark_manifest = {
        "manifest_version": 1,
        "kind": "asr",
        "created_at": utc_now(),
        "model": {"backend": backend.backend_name, "name": backend.model_name},
        "invocation": {"temperature": temperature},
        "inputs": {
            "dataset": str(manifest_path.resolve()),
            "dataset_sha256": hash_file(manifest_path.resolve()),
            "items": input_records,
        },
        "input_fingerprint": stable_hash(input_records),
        "environment": {
            "git": git_snapshot(Path.cwd()),
            "system": system_snapshot(),
            "dependencies": dependency_snapshot(),
        },
    }
    write_manifest(benchmark_manifest_path, benchmark_manifest)

    predictions: list[dict[str, Any]] = []
    total_word_errors = 0
    total_reference_words = 0
    total_character_errors = 0
    total_reference_characters = 0
    total_audio_seconds = 0.0
    total_processing_seconds = 0.0
    for item in items:
        started = time.perf_counter()
        result = backend.transcribe(
            item.audio_path,
            TranscriptionOptions(
                language=item.language,
                prompt=item.prompt,
                temperature=temperature,
            ),
        )
        wall_seconds = time.perf_counter() - started
        processing_seconds = result.processing_seconds or wall_seconds
        score = score_transcript(item.reference, result.text)
        total_word_errors += score.word_errors
        total_reference_words += score.reference_words
        total_character_errors += score.character_errors
        total_reference_characters += score.reference_characters
        total_audio_seconds += result.duration
        total_processing_seconds += processing_seconds
        predictions.append(
            {
                "id": item.id,
                "audio": str(item.audio_path),
                "reference": item.reference,
                "text": result.text,
                "requested_language": item.language,
                "detected_language": result.language,
                "duration_seconds": result.duration,
                "processing_seconds": processing_seconds,
                "real_time_factor": (
                    processing_seconds / result.duration if result.duration > 0 else None
                ),
                "score": asdict(score),
            }
        )

    metrics: dict[str, Any] = {
        "items": len(predictions),
        "word_errors": total_word_errors,
        "reference_words": total_reference_words,
        "wer": _corpus_rate(total_word_errors, total_reference_words),
        "character_errors": total_character_errors,
        "reference_characters": total_reference_characters,
        "cer": _corpus_rate(total_character_errors, total_reference_characters),
        "audio_seconds": total_audio_seconds,
        "processing_seconds": total_processing_seconds,
        "real_time_factor": (
            total_processing_seconds / total_audio_seconds if total_audio_seconds > 0 else None
        ),
        "model": backend.model_name,
        "backend": backend.backend_name,
    }
    atomic_write_text(
        predictions_path,
        "".join(f"{canonical_json(prediction)}\n" for prediction in predictions),
    )
    atomic_write_json(metrics_path, metrics, indent=2)
    atomic_write_text(report_path, _render_report(metrics, predictions))
    return ASRBenchmarkResult(
        run_dir=run_dir,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        manifest_path=benchmark_manifest_path,
        report_path=report_path,
        metrics=metrics,
    )


def _default_run_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("runs") / "asr" / timestamp


def _corpus_rate(errors: int, reference_units: int) -> float:
    if reference_units:
        return errors / reference_units
    return 0.0 if errors == 0 else 1.0


def _render_report(metrics: dict[str, Any], predictions: list[dict[str, Any]]) -> str:
    rows = []
    for prediction in predictions:
        score: TranscriptScore | dict[str, Any] = prediction["score"]
        score_data = asdict(score) if isinstance(score, TranscriptScore) else score
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(prediction['id']))}</td>"
            f"<td>{score_data['wer']:.4f}</td>"
            f"<td>{score_data['cer']:.4f}</td>"
            f"<td>{html.escape(str(prediction['reference']))}</td>"
            f"<td>{html.escape(str(prediction['text']))}</td>"
            "</tr>"
        )
    rtf = metrics["real_time_factor"]
    rtf_text = "n/a" if rtf is None else f"{rtf:.4f}"
    model = html.escape(str(metrics["model"]))
    backend = html.escape(str(metrics["backend"]))
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            "<title>ASR benchmark report</title>",
            "<style>body{font-family:system-ui;margin:2rem;max-width:1200px}"
            "table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #ccc;padding:.5rem;text-align:left;vertical-align:top}"
            "th{background:#f5f5f5}</style>",
            "</head><body><h1>ASR benchmark</h1>",
            f"<p><strong>Model:</strong> {model} ({backend})</p>",
            f"<ul><li>Corpus WER: {metrics['wer']:.4f}</li>",
            f"<li>Corpus CER: {metrics['cer']:.4f}</li>",
            f"<li>Real-time factor: {rtf_text}</li>",
            f"<li>Items: {metrics['items']}</li></ul>",
            "<table><thead><tr><th>ID</th><th>WER</th><th>CER</th>"
            "<th>Reference</th><th>Transcript</th></tr></thead>",
            f"<tbody>{''.join(rows)}</tbody></table>",
            "</body></html>",
            "",
        ]
    )
