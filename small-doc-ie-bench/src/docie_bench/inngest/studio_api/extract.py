"""Extraction trigger + document rasterization routes."""

from __future__ import annotations

import uuid
from typing import Any

import inngest
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from docie_bench.inngest.client import inngest_client, send_or_503
from docie_bench.security import TenantDependency

from . import _shared

router = APIRouter()


class ExtractRequest(BaseModel):
    text: str | None = None
    content_b64: str | None = None
    filename: str | None = None
    schema_name: str = "invoice"
    # Name of a schema saved via POST /schemas/dynamic. Wins over schema_name
    # when present -- ExtractionService already accepts schema_mode="dynamic" +
    # dynamic_schema inline (extract/service.py), this just resolves the spec
    # by name server-side instead of requiring the full field list every call.
    dynamic_schema_name: str | None = None
    # Explicit live-deployment selector (a DeploymentRecord ``spec.name``). Wins
    # over ``model_profile``; forwarded verbatim into the event data (auto-included
    # by ``model_dump(exclude_none=True)`` below) for the worker's resolver.
    deployment: str | None = None
    model_profile: str | None = None
    ocr_backend: str | None = None
    language: str | None = None


@router.post("/extract", response_model=_shared.TriggerResponse)
async def trigger_extract(
    payload: ExtractRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    if not payload.text and not payload.content_b64:
        raise HTTPException(status_code=422, detail="Provide either 'text' or 'content_b64'")
    channel = f"extract:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    # Bind provenance to the authenticated principal (mirrors trigger_benchmark) so
    # the worker's audit row is tenant-scoped rather than anonymous.
    data["tenant_id"] = tenant.tenant_id
    ids = await send_or_503(
        inngest_client, inngest.Event(name=_shared.EXTRACT_EVENT, data=data)
    )
    # Record ownership so the run-status proxy is tenant-scoped: an extraction run
    # has no durable StudioRun row, so this is its only ownership signal.
    _shared._record_event_owners(list(ids), tenant.tenant_id)
    return _shared.TriggerResponse(
        event_ids=list(ids), channel=channel, topics=_shared.DEFAULT_TOPICS
    )


class RenderDocumentRequest(BaseModel):
    """Rasterize an uploaded document to page images for a vision model."""

    content_b64: str
    filename: str = "document.pdf"
    max_pages: int = 8
    # Rasterization DPI. 200 (not liteparse's 150 default) sharpens dense
    # document text noticeably for small vision models, at a larger payload;
    # clamped below to keep a page from exploding into millions of pixels.
    dpi: int = 200


@router.post("/render-document")
async def render_document(
    payload: RenderDocumentRequest, tenant: TenantDependency
) -> dict[str, Any]:
    """Turn an uploaded PDF (or image) into PNG page images as data URLs.

    Vision models take images, not PDFs — llama-server rejects a
    ``data:application/pdf`` URL. The Playground's Vision panel posts a PDF here
    first and forwards the returned image data URLs as ``image_url`` parts (one
    per page). Reuses the same PDFium rasterizer the benchmark's vision path uses
    (``vision.load_document_images``); images pass through normalized to PNG.
    """
    del tenant  # authenticated principal required; no per-tenant scoping
    import asyncio
    import base64
    import binascii
    from pathlib import Path
    from tempfile import NamedTemporaryFile

    from docie_bench.vision import load_document_images

    try:
        raw = base64.b64decode(payload.content_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64 content: {exc}") from exc
    max_pages = max(1, min(int(payload.max_pages), 20))
    dpi = max(72, min(int(payload.dpi), 400))
    suffix = Path(payload.filename).suffix or ".pdf"

    def _render() -> list[str]:
        with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(raw)
            tmp = Path(handle.name)
        try:
            return [
                img.data_url()
                for img in load_document_images(tmp, max_pages=max_pages, pdf_dpi=dpi)
            ]
        finally:
            tmp.unlink(missing_ok=True)

    try:
        images = await asyncio.to_thread(_render)
    except ValueError as exc:
        # Unsupported type / too many pages / empty PDF — a client error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"images": images, "pages": len(images)}
