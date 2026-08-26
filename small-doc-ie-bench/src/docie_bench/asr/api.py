from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from docie_bench.asr.formats import TranscriptionFormat
from docie_bench.asr.routing import ASRRoutingError, resolve_asr_route
from docie_bench.asr.uploads import store_validated_audio_upload
from docie_bench.security import TenantDependency
from docie_bench.serving.recency import stamp
from docie_bench.settings import get_settings

router = APIRouter(prefix="/v1/audio", tags=["audio"])


@router.post("/transcriptions")
async def create_transcription(
    file: Annotated[UploadFile, File(description="Audio file to transcribe")],
    model: Annotated[str, Form(min_length=1, max_length=200)],
    _tenant: TenantDependency,
    language: Annotated[str | None, Form(min_length=2, max_length=16)] = None,
    prompt: Annotated[str | None, Form(max_length=4_000)] = None,
    response_format: Annotated[TranscriptionFormat, Form()] = TranscriptionFormat.JSON,
    temperature: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
) -> Response:
    """Transcribe audio using the OpenAI-compatible multipart request shape."""

    settings = get_settings()
    try:
        route = resolve_asr_route(model)
    except ASRRoutingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    upload = await store_validated_audio_upload(
        file,
        max_bytes=settings.asr_max_upload_bytes,
        allowed_mime_types=settings.allowed_asr_mime_types,
    )
    try:
        data = {
            "model": route.model,
            "response_format": response_format.value,
            "temperature": str(temperature),
        }
        if language is not None:
            data["language"] = language
        if prompt is not None:
            data["prompt"] = prompt
        try:
            with upload.path.open("rb") as audio:
                async with httpx.AsyncClient(timeout=settings.asr_timeout_seconds) as client:
                    upstream = await client.post(
                        f"{route.base_url}/audio/transcriptions",
                        data=data,
                        files={"file": (f"upload{upload.suffix}", audio, upload.mime_type)},
                    )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"ASR deployment {route.deployment!r} is unreachable: {exc}",
            ) from exc
        if upstream.status_code < 500:
            stamp(route.deployment)
    finally:
        upload.path.unlink(missing_ok=True)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )


# Kept as a sub-router so the synchronous OpenAI-compatible endpoint and the
# durable job control plane share the same authenticated /v1/audio namespace.
from docie_bench.asr.jobs_api import router as jobs_router  # noqa: E402

router.include_router(jobs_router)
