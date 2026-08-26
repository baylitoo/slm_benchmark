from __future__ import annotations

from enum import StrEnum
from typing import Any

from docie_bench.asr.models import TranscriptionResult, TranscriptionSegment


class TranscriptionFormat(StrEnum):
    JSON = "json"
    VERBOSE_JSON = "verbose_json"
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"


def render_transcription(
    result: TranscriptionResult,
    response_format: TranscriptionFormat,
) -> tuple[dict[str, Any] | str, str]:
    if response_format == TranscriptionFormat.JSON:
        return {"text": result.text}, "application/json"
    if response_format == TranscriptionFormat.VERBOSE_JSON:
        payload: dict[str, Any] = {
            "task": "transcribe",
            "language": result.language,
            "duration": result.duration,
            "text": result.text,
            "segments": [_segment_payload(segment) for segment in result.segments],
            "processing_seconds": result.processing_seconds,
            "real_time_factor": result.real_time_factor,
            "model": result.model,
            "backend": result.backend,
        }
        return payload, "application/json"
    if response_format == TranscriptionFormat.TEXT:
        return _trailing_newline(result.text), "text/plain; charset=utf-8"
    if response_format == TranscriptionFormat.SRT:
        return render_srt(result), "application/x-subrip; charset=utf-8"
    return render_vtt(result), "text/vtt; charset=utf-8"


def render_srt(result: TranscriptionResult) -> str:
    blocks = [
        f"{index}\n{_timestamp(segment.start, decimal=',')} --> "
        f"{_timestamp(segment.end, decimal=',')}\n{segment.text.strip()}"
        for index, segment in enumerate(_subtitle_segments(result), start=1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(result: TranscriptionResult) -> str:
    blocks = [
        f"{_timestamp(segment.start, decimal='.')} --> "
        f"{_timestamp(segment.end, decimal='.')}\n{segment.text.strip()}"
        for segment in _subtitle_segments(result)
    ]
    return "WEBVTT\n\n" + "\n\n".join(blocks) + ("\n" if blocks else "")


def _subtitle_segments(result: TranscriptionResult) -> tuple[TranscriptionSegment, ...]:
    populated = tuple(segment for segment in result.segments if segment.text.strip())
    if populated or not result.text:
        return populated
    return (
        TranscriptionSegment(id=0, start=0.0, end=max(result.duration, 0.0), text=result.text),
    )


def _timestamp(seconds: float, *, decimal: str) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{decimal}{millis:03d}"


def _segment_payload(segment: TranscriptionSegment) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": segment.id,
        "seek": segment.seek,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "tokens": list(segment.tokens),
        "temperature": segment.temperature,
    }
    for key in ("avg_logprob", "compression_ratio", "no_speech_prob"):
        value = getattr(segment, key)
        if value is not None:
            payload[key] = value
    return payload


def _trailing_newline(value: str) -> str:
    return value.rstrip("\r\n") + "\n" if value else ""
