"""HTTP control plane for durable single- and batch-transcription jobs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import inngest
from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from docie_bench.asr.job_store import (
    ASRJobStoreUnavailableError,
    default_asr_job_store,
)
from docie_bench.asr.routing import ASRRoutingError, resolve_asr_route
from docie_bench.asr.uploads import AUDIO_MIME_BY_SUFFIX, detect_audio_mime_type
from docie_bench.inngest.client import inngest_client, send_or_503
from docie_bench.inngest.realtime import (
    TOPIC_ERROR,
    TOPIC_PROGRESS,
    TOPIC_RESULT,
    TOPIC_STATUS,
)
from docie_bench.security import TenantDependency
from docie_bench.settings import get_settings

router = APIRouter()

TRANSCRIPTION_JOB_EVENT = "asr/transcription.requested"
_MAX_JOB_ITEMS = 100
_TOPICS = [TOPIC_STATUS, TOPIC_PROGRESS, TOPIC_RESULT, TOPIC_ERROR]


class RawAudioRetention(StrEnum):
    DELETE_AFTER_COMPLETION = "delete_after_completion"
    RETAIN_7D = "retain_7d"
    RETAIN_30D = "retain_30d"


class TranscriptionJobRecording(BaseModel):
    filename: str = Field(min_length=1, max_length=300)
    content_b64: str = Field(min_length=1)
    reference: str | None = Field(default=None, max_length=1_000_000)


class TranscriptionJobRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    recordings: list[TranscriptionJobRecording] = Field(min_length=1, max_length=_MAX_JOB_ITEMS)
    language: str | None = Field(default=None, min_length=2, max_length=16)
    prompt: str | None = Field(default=None, max_length=4_000)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_audio_retention: RawAudioRetention = RawAudioRetention.DELETE_AFTER_COMPLETION
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class TranscriptionJobTriggerResponse(BaseModel):
    job_id: str
    status: str
    channel: str
    topics: list[str]
    status_uri: str
    deduplicated: bool = False


@router.post("/transcription-jobs", response_model=TranscriptionJobTriggerResponse)
async def create_transcription_job(
    payload: TranscriptionJobRequest,
    tenant: TenantDependency,
    idempotency_header: Annotated[
        str | None, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ] = None,
) -> TranscriptionJobTriggerResponse:
    """Queue one durable job containing one or more validated audio files.

    Audio bytes are committed to the shared blob store before enqueue; the
    event carries only content-addressed keys, keeping the queue payload small
    and making retries consume the exact same input bytes.
    """
    settings = get_settings()
    store = default_asr_job_store()
    if not store.enabled:
        raise HTTPException(
            status_code=503,
            detail="Durable ASR jobs require DATABASE_URL and the shared artifact store",
        )
    try:
        route = resolve_asr_route(payload.model)
    except ASRRoutingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    prepared: list[dict[str, Any]] = []
    blobs = store.blobs
    for index, recording in enumerate(payload.recordings):
        raw, filename, mime_type = _decode_recording(
            recording,
            index=index,
            max_bytes=settings.asr_max_upload_bytes,
            allowed_mime_types=settings.allowed_asr_mime_types,
        )
        stored = blobs.put(name=filename, content=raw, media_type=mime_type)
        prepared.append(
            {
                "filename": filename,
                "relkey": stored.relkey,
                "sha256": stored.sha256,
                "size_bytes": stored.size_bytes,
                "mime_type": mime_type,
                "reference": recording.reference,
            }
        )

    options: dict[str, Any] = {"temperature": payload.temperature}
    if payload.language is not None:
        options["language"] = payload.language
    if payload.prompt is not None:
        options["prompt"] = payload.prompt
    key = idempotency_header or payload.idempotency_key or _derived_key(
        tenant_id=tenant.tenant_id,
        deployment=route.deployment,
        model=route.model,
        options=options,
        retention=payload.raw_audio_retention.value,
        items=prepared,
    )
    event_id = f"asr-{uuid.uuid4().hex}"
    channel = f"asr:{event_id}"
    try:
        outcome, record = store.claim(
            event_id=event_id,
            tenant_id=tenant.tenant_id,
            idempotency_key=key,
            channel=channel,
            deployment=route.deployment,
            model=route.model,
            options=options,
            raw_retention=payload.raw_audio_retention.value,
            raw_expires_at=None,
            items=prepared,
        )
    except ASRJobStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if outcome == "exists":
        return _trigger_response(record, deduplicated=True)

    event_data: dict[str, Any] = {
        "job_id": event_id,
        "channel": channel,
        "tenant_id": tenant.tenant_id,
        "idempotency_key": key,
        "deployment": route.deployment,
        "model": route.model,
        "options": options,
        "items": prepared,
    }
    try:
        await send_or_503(
            inngest_client,
            inngest.Event(id=event_id, name=TRANSCRIPTION_JOB_EVENT, data=event_data),
        )
    except HTTPException:
        # The queue never accepted the event. Remove only the queued row so a
        # caller retrying the same key can actually enqueue; unreferenced input
        # blobs are safely reclaimed by the shared grace-gated sweep.
        store.discard_queued(event_id)
        raise
    return _trigger_response(record, deduplicated=False)


@router.get("/transcription-jobs")
async def list_transcription_jobs(
    tenant: TenantDependency,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[dict[str, Any]]:
    try:
        return default_asr_job_store().list(tenant_id=tenant.tenant_id, limit=limit)
    except ASRJobStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/transcription-jobs/{job_id}")
async def get_transcription_job(job_id: str, tenant: TenantDependency) -> dict[str, Any]:
    try:
        record = default_asr_job_store().get(job_id, tenant_id=tenant.tenant_id)
    except ASRJobStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"transcription job {job_id!r} not found")
    return record


@router.post("/transcription-jobs/{job_id}/cancel")
async def cancel_transcription_job(job_id: str, tenant: TenantDependency) -> dict[str, Any]:
    try:
        record = default_asr_job_store().request_cancel(job_id, tenant_id=tenant.tenant_id)
    except ASRJobStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"transcription job {job_id!r} not found")
    return record


@router.get("/transcription-jobs/{job_id}/artifacts/{artifact_id}")
async def download_transcription_artifact(
    job_id: str, artifact_id: str, tenant: TenantDependency
) -> Response:
    store = default_asr_job_store()
    try:
        job = store.get(job_id, tenant_id=tenant.tenant_id)
        opened = store.open_artifact(artifact_id, tenant_id=tenant.tenant_id)
    except ASRJobStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if job is None or opened is None:
        raise HTTPException(status_code=404, detail="transcription artifact not found")
    meta, content = opened
    # Bind artifact -> requested job too; an authenticated tenant must not use
    # job A's URL as an alias for one of its own artifacts from job B.
    expected = f"/v1/audio/transcription-jobs/{job_id}/artifacts/{artifact_id}"
    if str(meta.get("uri")) != expected:
        raise HTTPException(status_code=404, detail="transcription artifact not found")
    return Response(
        content=content,
        media_type=str(meta["media_type"]),
        headers={"Content-Disposition": f'attachment; filename="{meta["name"]}"'},
    )


def _decode_recording(
    recording: TranscriptionJobRecording,
    *,
    index: int,
    max_bytes: int,
    allowed_mime_types: set[str],
) -> tuple[bytes, str, str]:
    filename = Path(recording.filename).name
    if filename != recording.filename or not filename:
        raise HTTPException(
            status_code=422,
            detail=f"recordings[{index}].filename must be a plain file name",
        )
    suffix = Path(filename).suffix.lower()
    expected = AUDIO_MIME_BY_SUFFIX.get(suffix)
    if expected is None:
        raise HTTPException(status_code=415, detail=f"Unsupported audio suffix: {suffix}")
    if expected not in allowed_mime_types:
        raise HTTPException(status_code=415, detail=f"Audio type is disabled: {expected}")
    try:
        raw = base64.b64decode(recording.content_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"recordings[{index}].content_b64 is not valid base64: {exc}",
        ) from exc
    if not raw:
        raise HTTPException(status_code=400, detail=f"recordings[{index}] is empty")
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"recordings[{index}] exceeds the {max_bytes}-byte per-file limit",
        )
    detected = detect_audio_mime_type(raw[:64])
    if detected != expected:
        raise HTTPException(
            status_code=415,
            detail=(
                f"recordings[{index}] content does not match {suffix}; "
                f"detected {detected or 'unknown'}"
            ),
        )
    return raw, filename, detected


def _derived_key(
    *,
    tenant_id: str,
    deployment: str,
    model: str,
    options: dict[str, Any],
    retention: str,
    items: list[dict[str, Any]],
) -> str:
    material = {
        "tenant": tenant_id,
        "deployment": deployment,
        "model": model,
        "options": options,
        "retention": retention,
        "items": [
            {"sha256": item["sha256"], "reference": item.get("reference")} for item in items
        ],
    }
    digest = hashlib.sha256(
        json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"asr-{digest[:48]}"


def _trigger_response(
    record: dict[str, Any], *, deduplicated: bool
) -> TranscriptionJobTriggerResponse:
    event_id = str(record["event_id"])
    return TranscriptionJobTriggerResponse(
        job_id=event_id,
        status=str(record["status"]),
        channel=str(record["channel"]),
        topics=list(_TOPICS),
        status_uri=f"/v1/audio/transcription-jobs/{event_id}",
        deduplicated=deduplicated,
    )


__all__ = [
    "RawAudioRetention",
    "TRANSCRIPTION_JOB_EVENT",
    "TranscriptionJobRecording",
    "TranscriptionJobRequest",
    "TranscriptionJobTriggerResponse",
    "router",
]
