"""``configs/models.yaml`` profile listing + kind=pipeline/ocr authoring routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import _shared

router = APIRouter()


@router.get("/model-profiles")
async def list_model_profiles() -> list[dict[str, Any]]:
    """Profiles from ``configs/models.yaml`` -- what the Benchmark tab's "Model
    profile" field, and a new pipeline profile's extractor/ocr_model pickers, can
    actually reference instead of a free-text guess. Missing file degrades to an
    empty list, same contract as ``GET /datasets``.
    """
    from docie_bench.llm.model_profiles import load_model_profiles

    if not _shared.MODELS_CONFIG_PATH.exists():
        return []
    profiles = load_model_profiles(_shared.MODELS_CONFIG_PATH)
    return [
        {"name": p.name, "kind": p.kind, "vision": p.vision, "model": p.model}
        for p in sorted(profiles.values(), key=lambda p: p.name)
    ]


class PipelineProfileRequest(BaseModel):
    name: str
    extractor: str
    ocr_backend: str | None = None
    ocr_model: str | None = None
    language: str | None = None


@router.post("/model-profiles/pipeline", status_code=201)
async def create_pipeline_profile(payload: PipelineProfileRequest) -> dict[str, Any]:
    """Author a ``kind: pipeline`` (OCR->LLM) profile into ``configs/models.yaml``.

    The missing counterpart to #180-183's read-side wiring: the benchmark runner and
    gateway could already RUN a pipeline profile, and #183 let the Benchmark tab
    REFERENCE one by name, but nothing let an operator CREATE one without hand-editing
    the file on the server's filesystem. Create-only -- 409 on an existing name,
    mirroring ``AgentRegistry.create``'s conflict behavior; see
    ``model_profiles.add_pipeline_profile`` for why an in-place update is out of scope
    for this slice.
    """
    from docie_bench.llm.model_profiles import (
        ProfileConflictError,
        ProfileWriteError,
        add_pipeline_profile,
    )

    try:
        profile = add_pipeline_profile(
            _shared.MODELS_CONFIG_PATH,
            name=payload.name.strip(),
            extractor=payload.extractor.strip(),
            ocr_backend=payload.ocr_backend.strip() if payload.ocr_backend else None,
            ocr_model=payload.ocr_model.strip() if payload.ocr_model else None,
            language=payload.language.strip() if payload.language else None,
        )
    except ProfileConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ProfileWriteError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"name": profile.name, "kind": profile.kind, "options": dict(profile.options)}


class OcrProfileRequest(BaseModel):
    name: str
    backend: str
    language: str | None = None


@router.post("/model-profiles/ocr", status_code=201)
async def create_ocr_profile(payload: OcrProfileRequest) -> dict[str, Any]:
    """Author a ``kind: ocr`` (OCR-only, no LLM stage) profile into ``configs/models.yaml``.

    The gap #188 explicitly flagged and deferred: that PR let an operator author a
    ``kind: pipeline`` (OCR->LLM) profile; a ``kind: ocr`` profile is narrower --
    ``serving.solutions.OcrSolution`` runs only an OCR backend and returns its raw
    transcribed text as the completion, no extractor. Not a schema-scored benchmark
    model (see ``model_profiles.add_ocr_profile`` for why); reachable directly
    through the gateway by name, but no Studio UI surface invokes one yet. Create-only,
    same 409/422 contract as the pipeline route above.
    """
    from docie_bench.llm.model_profiles import (
        ProfileConflictError,
        ProfileWriteError,
        add_ocr_profile,
    )

    try:
        profile = add_ocr_profile(
            _shared.MODELS_CONFIG_PATH,
            name=payload.name.strip(),
            backend=payload.backend.strip(),
            language=payload.language.strip() if payload.language else None,
        )
    except ProfileConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ProfileWriteError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"name": profile.name, "kind": profile.kind, "options": dict(profile.options)}


@router.delete("/model-profiles/{name}")
async def delete_pipeline_profile_route(name: str) -> dict[str, Any]:
    """Remove a ``kind: pipeline``/``kind: ocr`` profile from ``configs/models.yaml``.

    The missing counterpart to ``create_pipeline_profile``: a profile authored (or
    hand-added) as pipeline/ocr can now be retired the same way it was made, without
    hand-editing the file on the server's filesystem. Scoped to those two kinds
    only -- never ``passthrough`` (same restriction ``delete_pipeline_profile`` itself
    enforces): a live deployment, or another pipeline profile's `extractor`/
    `ocr_model`, can reference a passthrough profile by name, and neither this route
    nor ``model_profiles`` has a way to check for that at delete time. An in-place
    UPDATE (change an existing profile's extractor/ocr_backend without delete +
    recreate) is a still harder text-splice problem than this DELETE and stays out of
    scope -- see ``model_profiles.add_pipeline_profile``'s own docstring for why.
    """
    from docie_bench.llm.model_profiles import (
        ProfileNotFoundError,
        ProfileWriteError,
        delete_pipeline_profile,
    )

    try:
        delete_pipeline_profile(_shared.MODELS_CONFIG_PATH, name=name)
    except ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc), headers=_shared._DOMAIN_404) from exc
    except ProfileWriteError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"deleted": name}
