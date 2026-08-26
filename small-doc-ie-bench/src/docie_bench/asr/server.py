"""Managed OpenAI-compatible ASR runtime.

This app runs inside the single serving node. The public API validates tenant
access and uploads, then proxies to this private runtime. Loading at startup is
intentional: the control plane must not publish readiness before weights exist.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from docie_bench.asr.backend import (
    ASRDependencyUnavailableError,
    ASRModelLoadError,
    ASRTranscriptionError,
    FasterWhisperBackend,
)
from docie_bench.asr.formats import TranscriptionFormat, render_transcription
from docie_bench.asr.models import ASRBackend, TranscriptionOptions
from docie_bench.asr.uploads import store_validated_audio_upload
from docie_bench.settings import get_settings


def create_asr_app(
    *,
    model_id: str,
    alias: str,
    backend: ASRBackend | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    cpu_threads: int = 0,
    num_workers: int = 1,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> FastAPI:
    """Create one model-pinned ASR runtime app."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.backend is None:
            loaded = FasterWhisperBackend(
                model=model_id,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
            await asyncio.to_thread(loaded.load)
            app.state.backend = loaded
        yield

    app = FastAPI(title="docie ASR runtime", lifespan=lifespan)
    app.state.backend = backend
    app.state.model_id = model_id
    app.state.alias = alias

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"status": "ok", "model": alias, "kind": "asr"}

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": alias, "object": "model", "owned_by": "docie"}],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: Annotated[UploadFile, File(description="Audio file to transcribe")],
        model: Annotated[str, Form(min_length=1, max_length=200)],
        language: Annotated[str | None, Form(min_length=2, max_length=16)] = None,
        prompt: Annotated[str | None, Form(max_length=4_000)] = None,
        response_format: Annotated[TranscriptionFormat, Form()] = TranscriptionFormat.JSON,
        temperature: Annotated[float, Form(ge=0.0, le=1.0)] = 0.0,
    ) -> Response:
        if model != alias:
            raise HTTPException(status_code=404, detail=f"Unknown ASR model {model!r}")
        settings = get_settings()
        upload = await store_validated_audio_upload(
            file,
            max_bytes=settings.asr_max_upload_bytes,
            allowed_mime_types=settings.allowed_asr_mime_types,
        )
        try:
            result = await asyncio.to_thread(
                app.state.backend.transcribe,
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

    return app
