from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from docie_bench.asr.backend import (
    ASRDependencyUnavailableError,
    ASRModelLoadError,
    ASRTranscriptionError,
    get_backend,
)
from docie_bench.asr.formats import TranscriptionFormat, render_transcription
from docie_bench.asr.models import TranscriptionOptions
from docie_bench.asr.uploads import store_validated_audio_upload
from docie_bench.security import TenantDependency
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
    if model not in {settings.asr_model_alias, settings.asr_model}:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown ASR model {model!r}; use the configured alias "
                f"{settings.asr_model_alias!r}"
            ),
        )
    upload = await store_validated_audio_upload(
        file,
        max_bytes=settings.asr_max_upload_bytes,
        allowed_mime_types=settings.allowed_asr_mime_types,
    )
    try:
        backend = get_backend(
            model=settings.asr_model,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
            cpu_threads=settings.asr_cpu_threads,
            num_workers=settings.asr_num_workers,
            beam_size=settings.asr_beam_size,
            vad_filter=settings.asr_vad_filter,
        )
        result = await asyncio.to_thread(
            backend.transcribe,
            upload.path,
            TranscriptionOptions(
                language=language,
                prompt=prompt,
                temperature=temperature,
            ),
        )
    except (ASRDependencyUnavailableError, ASRModelLoadError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ASRTranscriptionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        upload.path.unlink(missing_ok=True)

    payload, media_type = render_transcription(result, response_format)
    if isinstance(payload, dict):
        return JSONResponse(payload)
    return Response(content=payload, media_type=media_type)
