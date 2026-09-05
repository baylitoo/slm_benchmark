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
    # A saved routing policy (POST /routing-policies) to run this extraction
    # through INSTEAD of a single model: first stage, escalate on the policy's
    # own confidence/validity rules and budgets. Mutually exclusive with
    # deployment/model_profile (a policy names its profiles per stage). The
    # result's ``routing`` carries the audit. Forwarded into the event data
    # like every other field; the worker builds the router.
    routing_policy: str | None = None
    ocr_backend: str | None = None
    language: str | None = None


@router.post("/extract", response_model=_shared.TriggerResponse)
async def trigger_extract(
    payload: ExtractRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    if not payload.text and not payload.content_b64:
        raise HTTPException(status_code=422, detail="Provide either 'text' or 'content_b64'")
    if payload.routing_policy:
        # Fail fast at the API edge: a bad selector should be a 4xx NOW, not a
        # failed Inngest run the caller only discovers by polling. The worker
        # re-checks (it is the source of truth), this just saves the round-trip.
        if payload.model_profile or payload.deployment:
            raise HTTPException(
                status_code=400,
                detail="'routing_policy' is mutually exclusive with 'model_profile'/"
                "'deployment': a policy names its model profiles per stage",
            )
        from docie_bench.studio.routing_policies import (
            RoutingPolicyUnavailableError,
            get_routing_policy,
        )

        try:
            if get_routing_policy(payload.routing_policy) is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"routing policy {payload.routing_policy!r} not found -- save "
                    "one via POST /v1/studio/routing-policies (or pick it in the Studio)",
                )
        except RoutingPolicyUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
    # Explicit 1-indexed page numbers to rasterize (e.g. a page-1 thumbnail
    # preview, or a click-to-enlarge fetch of a specific range). When set,
    # ONLY those pages are rendered and `max_pages` is not enforced -- the
    # caller told us exactly what it wants. `None` (default) keeps the
    # existing vision-send contract: rasterize everything, reject if it's
    # more than `max_pages`.
    pages: list[int] | None = None


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
    pages = [int(p) for p in payload.pages] if payload.pages else None
    suffix = Path(payload.filename).suffix or ".pdf"
    is_pdf = suffix.lower() == ".pdf"

    def _render() -> tuple[list[str], int]:
        with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(raw)
            tmp = Path(handle.name)
        try:
            images = [
                img.data_url()
                for img in load_document_images(
                    tmp, max_pages=max_pages, pdf_dpi=dpi, pages=pages
                )
            ]
            # The true page count, computed cheaply (no rasterization) via
            # pdf_inspector -- `len(images)` is wrong/misleading once a caller
            # can request a subset of pages. A non-PDF upload is just the one
            # image.
            total_pages = len(images)
            if is_pdf:
                try:
                    import pdf_inspector

                    total_pages = pdf_inspector.classify_pdf(str(tmp)).page_count
                except Exception:  # noqa: BLE001, S110 - best-effort; fall back to len(images)
                    pass
            return images, total_pages
        finally:
            tmp.unlink(missing_ok=True)

    try:
        images, total_pages = await asyncio.to_thread(_render)
    except ValueError as exc:
        # Unsupported type / too many pages / empty PDF — a client error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"images": images, "pages": len(images), "total_pages": total_pages}


class UploadSessionDocumentRequest(BaseModel):
    """Add a document to a session-scoped directory docs-search can search
    for THIS conversation only (#296)."""

    content_b64: str
    filename: str
    # None starts a new session (a fresh id is minted and returned); an id
    # this endpoint already returned adds another document to that SAME
    # session. Any other value is rejected — never a client-invented id.
    session_id: str | None = None


class UploadSessionDocumentResponse(BaseModel):
    session_id: str
    stored_name: str


@router.post("/session-documents", response_model=UploadSessionDocumentResponse)
async def upload_session_document(
    payload: UploadSessionDocumentRequest, tenant: TenantDependency
) -> UploadSessionDocumentResponse:
    """Upload a file docs-search can see during this conversation only.

    The Playground calls this alongside render-document when docs-search is
    selected: docs-search's real corpus is a separate, operator-controlled
    directory an attachment otherwise never reaches (render-document only
    ever produces page images for vision, nothing docs-search can read).
    See ``mcp_session_documents`` and ``chat_api._chat_with_mcp_tools``'s
    per-request env override that points docs-search at this session's
    directory instead of the shared one.
    """
    del tenant  # authenticated principal required; session id IS the scope
    import base64
    import binascii

    from docie_bench.doc_summarization import spawn_summarize_document
    from docie_bench.mcp_session_documents import (
        SessionDocumentError,
        save_document,
        session_documents_dir,
    )

    try:
        raw = base64.b64decode(payload.content_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64 content: {exc}") from exc
    try:
        session_id, stored_name = save_document(payload.session_id, payload.filename, raw)
    except SessionDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Fire-and-forget: a quick description in list_files is enrichment, not
    # something this upload response should block on (see doc_summarization
    # module docstring). No-ops immediately if doc_summary_model is unset.
    document_path = session_documents_dir(session_id) / stored_name
    spawn_summarize_document(document_path)
    return UploadSessionDocumentResponse(session_id=session_id, stored_name=stored_name)
