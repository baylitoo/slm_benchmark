"""Durable execution for ASR transcription jobs.

This module deliberately contains no Inngest decorator.  ``inngest.functions``
owns registration and calls :func:`process_transcription_job`, while the core
execution stays directly testable with an immediate fake step runner.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from docie_bench.asr.formats import TranscriptionFormat, render_transcription
from docie_bench.asr.job_store import ASRJobStore, default_asr_job_store
from docie_bench.asr.metrics import score_transcript
from docie_bench.asr.models import TranscriptionResult, TranscriptionSegment
from docie_bench.asr.routing import ASRRoute, resolve_asr_route
from docie_bench.inngest.realtime import TOPIC_PROGRESS, TOPIC_RESULT, TOPIC_STATUS, publish
from docie_bench.serving.recency import stamp
from docie_bench.settings import get_settings
from docie_bench.studio.usage_store import record_usage
from docie_bench.telemetry import (
    ASR_JOB_ITEM_LATENCY,
    ASR_JOB_ITEMS,
    ASR_JOB_REAL_TIME_FACTOR,
)

logger = logging.getLogger("docie_bench.asr.jobs")

_ITEM_ATTEMPTS = 3
_RETRY_DELAYS = (0.0, 1.0, 4.0)
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


class ASRJobItemError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


async def process_transcription_job(
    data: dict[str, Any],
    *,
    event_id: str,
    step: Any,
    store: ASRJobStore | None = None,
) -> dict[str, Any]:
    """Transcribe every recording as an independent memoized step.

    Item failures are values, not raised whole-job failures.  This is what
    lets a corrupt third recording produce a failed item while recordings 1,
    2, and 4 still complete and publish artifacts.
    """
    store = store or default_asr_job_store()
    current = store.start(event_id)
    if current is None:
        raise KeyError(f"ASR job {event_id!r} was not claimed before execution")
    if current["status"] in {"completed", "completed_with_errors", "failed", "cancelled"}:
        return current

    channel = str(data.get("channel") or f"asr:{event_id}")
    deployment = str(data["deployment"])
    route = resolve_asr_route(deployment)
    raw_items = data.get("items")
    items = (
        [dict(item) for item in raw_items if isinstance(item, dict)]
        if isinstance(raw_items, list)
        else []
    )
    await publish(
        channel,
        TOPIC_STATUS,
        {"state": "running", "job_id": event_id, "total": len(items), "deployment": deployment},
    )

    outcomes: list[dict[str, Any]] = []
    for position, item in enumerate(items):
        if store.is_cancel_requested(event_id):
            break

        async def _one(item: dict[str, Any] = item, position: int = position) -> dict[str, Any]:
            return await _process_item(
                store,
                event_id=event_id,
                position=position,
                item=item,
                route=route,
                tenant_id=str(data.get("tenant_id") or "anonymous"),
                options=dict(data.get("options") or {}),
            )

        outcome = await step.run(f"transcribe-{position}", _one)
        outcomes.append(dict(outcome))
        await publish(
            channel,
            TOPIC_PROGRESS,
            {
                "job_id": event_id,
                "total": len(items),
                "processed": len(outcomes),
                "completed": sum(o.get("status") == "completed" for o in outcomes),
                "failed": sum(o.get("status") == "failed" for o in outcomes),
                "current": item.get("filename"),
                "percent": round(100 * len(outcomes) / max(len(items), 1), 1),
            },
        )

    metrics = _aggregate_metrics(outcomes)

    async def _manifest() -> None:
        snapshot = store.get_internal(event_id) or {"event_id": event_id}
        predicted_status = (
            "cancelled"
            if store.is_cancel_requested(event_id)
            else "completed_with_errors"
            if any(outcome.get("status") == "failed" for outcome in outcomes)
            else "completed"
        )
        body = json.dumps(
            {
                "kind": "asr_transcription_job",
                "event_id": event_id,
                "status": predicted_status,
                "deployment": deployment,
                "model": data.get("model"),
                "raw_audio_retention": snapshot.get("raw_retention"),
                "metrics": metrics,
                "items": snapshot.get("items", []),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        blob = store.blobs.put(
            name="manifest.json", content=body, media_type="application/json"
        )
        store.add_job_artifact(event_id, name="manifest.json", kind="manifest", blob=blob)

    await step.run("write-manifest", _manifest)
    result = store.settle(event_id, metrics=metrics)
    await publish(channel, TOPIC_RESULT, result)
    return result


async def _process_item(
    store: ASRJobStore,
    *,
    event_id: str,
    position: int,
    item: dict[str, Any],
    route: ASRRoute,
    tenant_id: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    last_error = "transcription failed"
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        if store.is_cancel_requested(event_id):
            return {"position": position, "status": "cancelled", "attempts": attempt - 1}
        if not store.begin_item(event_id, position, attempt=attempt):
            replay = _existing_item_outcome(store, event_id=event_id, position=position)
            if replay is not None:
                return replay
            if store.is_cancel_requested(event_id):
                return {"position": position, "status": "cancelled", "attempts": attempt - 1}
            raise ASRJobItemError(f"ASR job item {event_id}:{position} is unavailable")
        if delay:
            await asyncio.sleep(delay)
        if store.is_cancel_requested(event_id):
            return {"position": position, "status": "cancelled", "attempts": attempt - 1}
        started = asyncio.get_running_loop().time()
        try:
            payload = await _request_verbose_transcription(
                route, item=item, options=options, store=store
            )
            elapsed = max(0.0, asyncio.get_running_loop().time() - started)
            result = _result_from_payload(payload, route=route, elapsed=elapsed)
            metrics = None
            reference = item.get("reference")
            if reference is not None:
                metrics = dataclasses.asdict(score_transcript(str(reference), result.text))
            artifacts = _store_outputs(
                store,
                position=position,
                filename=str(item["filename"]),
                result=result,
            )
            verbose, _ = render_transcription(result, TranscriptionFormat.VERBOSE_JSON)
            assert isinstance(verbose, dict)
            store.complete_item(
                event_id,
                position,
                result=verbose,
                metrics=metrics,
                artifacts=artifacts,
                attempts=attempt,
            )
            stamp(route.deployment)
            record_usage(
                deployment=route.deployment,
                surface="asr",
                tenant_id=tenant_id,
                latency_ms=round(result.processing_seconds * 1000),
                status="ok",
            )
            ASR_JOB_ITEMS.labels(deployment=route.deployment, outcome="completed").inc()
            ASR_JOB_ITEM_LATENCY.labels(deployment=route.deployment).observe(
                result.processing_seconds
            )
            if result.real_time_factor is not None:
                ASR_JOB_REAL_TIME_FACTOR.labels(deployment=route.deployment).observe(
                    result.real_time_factor
                )
            return {
                "position": position,
                "status": "completed",
                "attempts": attempt,
                "duration_seconds": result.duration,
                "processing_seconds": result.processing_seconds,
                "metrics": metrics,
            }
        except Exception as exc:  # noqa: BLE001 - per-item failure is isolated
            last_error = str(exc)
            if attempt < _ITEM_ATTEMPTS and _retryable(exc):
                logger.warning(
                    "ASR job %s item %d attempt %d/%d failed; retrying: %s",
                    event_id,
                    position,
                    attempt,
                    _ITEM_ATTEMPTS,
                    exc,
                )
                continue
            break

    store.fail_item(event_id, position, error=last_error, attempts=attempt)
    record_usage(
        deployment=route.deployment,
        surface="asr",
        tenant_id=tenant_id,
        latency_ms=0,
        status="error",
    )
    ASR_JOB_ITEMS.labels(deployment=route.deployment, outcome="failed").inc()
    return {"position": position, "status": "failed", "attempts": attempt, "error": last_error}


async def _request_verbose_transcription(
    route: ASRRoute,
    *,
    item: dict[str, Any],
    options: dict[str, Any],
    store: ASRJobStore,
) -> dict[str, Any]:
    relkey = str(item["relkey"])
    path = store.blobs.path_for(relkey)
    if not path.is_file():
        raise ASRJobItemError(f"input blob is missing for {item.get('filename')!r}")
    data = {
        "model": route.model,
        "response_format": TranscriptionFormat.VERBOSE_JSON.value,
        "temperature": str(float(options.get("temperature", 0.0))),
    }
    if options.get("language"):
        data["language"] = str(options["language"])
    if options.get("prompt"):
        data["prompt"] = str(options["prompt"])
    try:
        with path.open("rb") as audio:
            async with httpx.AsyncClient(timeout=get_settings().asr_timeout_seconds) as client:
                response = await client.post(
                    f"{route.base_url}/audio/transcriptions",
                    data=data,
                    files={
                        "file": (
                            str(item["filename"]),
                            audio,
                            str(item.get("mime_type") or "application/octet-stream"),
                        )
                    },
                )
    except httpx.RequestError as exc:
        raise ASRJobItemError(
            f"ASR deployment {route.deployment!r} is unreachable: {exc}"
        ) from exc
    if response.status_code >= 400:
        detail: Any
        try:
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text
        raise ASRJobItemError(
            f"ASR deployment {route.deployment!r} returned HTTP {response.status_code}: {detail}",
            status_code=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ASRJobItemError("ASR runtime returned invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise ASRJobItemError("ASR runtime returned an invalid verbose transcription")
    return dict(payload)


def _result_from_payload(
    payload: dict[str, Any], *, route: ASRRoute, elapsed: float
) -> TranscriptionResult:
    segments: list[TranscriptionSegment] = []
    for index, raw in enumerate(payload.get("segments") or []):
        if not isinstance(raw, dict):
            continue
        segments.append(
            TranscriptionSegment(
                id=int(raw.get("id", index)),
                seek=int(raw.get("seek", 0)),
                start=float(raw.get("start", 0.0)),
                end=float(raw.get("end", 0.0)),
                text=str(raw.get("text", "")),
                tokens=tuple(int(token) for token in (raw.get("tokens") or [])),
                temperature=float(raw.get("temperature", 0.0)),
                avg_logprob=_optional_float(raw.get("avg_logprob")),
                compression_ratio=_optional_float(raw.get("compression_ratio")),
                no_speech_prob=_optional_float(raw.get("no_speech_prob")),
            )
        )
    processing = _optional_float(payload.get("processing_seconds"))
    return TranscriptionResult(
        text=str(payload["text"]),
        language=str(payload["language"]) if payload.get("language") is not None else None,
        duration=float(payload.get("duration") or 0.0),
        segments=tuple(segments),
        processing_seconds=processing if processing is not None else elapsed,
        model=str(payload.get("model") or route.model),
        backend=str(payload.get("backend") or "managed-asr"),
    )


def _store_outputs(
    store: ASRJobStore,
    *,
    position: int,
    filename: str,
    result: TranscriptionResult,
) -> list[tuple[str, str, Any]]:
    stem = _SAFE_STEM.sub("_", Path(filename).stem).strip("._") or "audio"
    prefix = f"{position:04d}-{stem[:120]}"
    specs = (
        (TranscriptionFormat.TEXT, f"{prefix}.txt", "text"),
        (TranscriptionFormat.VERBOSE_JSON, f"{prefix}.json", "verbose_json"),
        (TranscriptionFormat.SRT, f"{prefix}.srt", "srt"),
        (TranscriptionFormat.VTT, f"{prefix}.vtt", "vtt"),
    )
    stored: list[tuple[str, str, Any]] = []
    for response_format, name, kind in specs:
        rendered, media_type = render_transcription(result, response_format)
        content = (
            json.dumps(rendered, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
            if isinstance(rendered, dict)
            else rendered.encode("utf-8")
        )
        stored.append(
            (
                name,
                kind,
                store.blobs.put(name=name, content=content, media_type=media_type),
            )
        )
    return stored


def _aggregate_metrics(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in outcomes if item.get("status") == "completed"]
    failed = [item for item in outcomes if item.get("status") == "failed"]
    scored = [item["metrics"] for item in completed if isinstance(item.get("metrics"), dict)]
    word_errors = sum(int(score.get("word_errors", 0)) for score in scored)
    reference_words = sum(int(score.get("reference_words", 0)) for score in scored)
    character_errors = sum(int(score.get("character_errors", 0)) for score in scored)
    reference_characters = sum(int(score.get("reference_characters", 0)) for score in scored)
    audio_seconds = sum(float(item.get("duration_seconds") or 0.0) for item in completed)
    processing_seconds = sum(float(item.get("processing_seconds") or 0.0) for item in completed)
    return {
        "completed_items": len(completed),
        "failed_items": len(failed),
        "scored_items": len(scored),
        "audio_seconds": audio_seconds,
        "processing_seconds": processing_seconds,
        "real_time_factor": processing_seconds / audio_seconds if audio_seconds > 0 else None,
        "word_errors": word_errors,
        "reference_words": reference_words,
        "wer": (
            word_errors / reference_words
            if reference_words
            else (0.0 if not word_errors else 1.0)
        ),
        "character_errors": character_errors,
        "reference_characters": reference_characters,
        "cer": (
            character_errors / reference_characters
            if reference_characters
            else (0.0 if not character_errors else 1.0)
        ),
    }


def _existing_item_outcome(
    store: ASRJobStore, *, event_id: str, position: int
) -> dict[str, Any] | None:
    """Recover a DB-committed item when the Inngest step result was not saved.

    A worker can crash after ``complete_item`` commits but before Inngest
    records the step output. On replay, returning the durable row avoids a
    second upstream transcription, duplicate usage, or duplicate artifacts.
    """
    job = store.get_internal(event_id)
    if job is None:
        return None
    item = next(
        (candidate for candidate in job.get("items", []) if candidate["position"] == position),
        None,
    )
    if item is None or item.get("status") not in {"completed", "failed", "cancelled"}:
        return None
    return {
        "position": position,
        "status": item["status"],
        "attempts": item.get("attempts", 0),
        "duration_seconds": item.get("duration_seconds"),
        "processing_seconds": item.get("processing_seconds"),
        "metrics": item.get("metrics"),
        "error": item.get("error"),
    }


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status is None or status == 429 or int(status) >= 500


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


__all__ = ["ASRJobItemError", "process_transcription_job"]
