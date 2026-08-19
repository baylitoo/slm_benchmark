"""Batch extraction: N documents through one schema + model, as one durable
job with per-document state.

The whole platform was one document per request; the actual job -- "extract
from MY invoices" -- is plural. This is the plural surface, built on the async
substrate that already exists (Inngest jobs, the shared ArtifactBlobStore,
the seed/benchmark durable-run patterns) rather than a new one.

Design points that matter:

* Documents are NOT embedded in the Inngest event. A batch of PDFs is tens of
  MB; event payloads are small. The trigger route writes each document into
  the shared ``ArtifactBlobStore`` (the api/worker-shared volume the
  benchmark artifacts already ride) and the event carries only blob keys.
  A retry re-reads the same durable bytes.
* Each document is its own Inngest step (``functions.batch_extract_job``),
  so a crash on document 150 of 200 resumes at 150 -- durable per-document
  progress WITHOUT touching ``run_benchmark``'s loop -- and one bad PDF
  records its error and the batch continues.
* Results are served through the batch's OWN tenant-scoped route
  (``/batches/{id}/results.jsonl``), not ``/artifacts/{id}``: that route
  resolves tenant via ``StudioRunArtifact -> StudioRun``, and a batch is
  deliberately not a StudioRun (see ``studio.models.BatchRun``).
* Requires DATABASE_URL: item results are the product, so this answers 503
  rather than run an unrecoverable job (unlike seed tracking, which degrades).
"""

from __future__ import annotations

import base64
import binascii
import io
import uuid
import zipfile
from pathlib import PurePosixPath
from typing import Any

import inngest
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from docie_bench.inngest.client import inngest_client, send_or_503
from docie_bench.security import TenantDependency
from docie_bench.studio.batch_store import (
    BatchStoreUnavailableError,
    get_batch_run,
    list_batch_runs,
)
from docie_bench.studio.store import default_blob_store

from . import _shared

router = APIRouter()

BATCH_EVENT = "doc/batch.requested"

# Documents a batch will accept out of a zip (or as inline items). Anything
# else in the archive (macOS resource forks, READMEs, nested manifests) is
# skipped, not fatal -- a real user zip is rarely pristine.
_DOC_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt"}
_MAX_DOCUMENTS = 500


class BatchDocument(BaseModel):
    filename: str = Field(min_length=1, max_length=300)
    content_b64: str


class BatchExtractRequest(BaseModel):
    """Either ``zip_b64`` (an archive of documents) or ``documents`` (inline
    items) -- one of the two, not both. Same model/schema selectors as a
    single extraction; they apply to every document."""

    name: str | None = Field(default=None, max_length=200)
    zip_b64: str | None = None
    documents: list[BatchDocument] | None = None
    schema_name: str = "invoice"
    dynamic_schema_name: str | None = None
    deployment: str | None = None
    model_profile: str | None = None
    ocr_backend: str | None = None
    language: str | None = None


def _decode_b64(payload: str, *, what: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{what} is not valid base64: {exc}") from exc


def _documents_from_zip(raw: bytes) -> list[tuple[str, bytes]]:
    """(filename, bytes) for every servable document in the archive, in
    archive order. Skips directories, hidden/system files, and unsupported
    types. Rejects an empty or non-zip payload with a 422."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=422, detail=f"zip_b64 is not a valid zip archive: {exc}"
        ) from exc
    out: list[tuple[str, bytes]] = []
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename)
            if any(part.startswith(".") or part == "__MACOSX" for part in name.parts):
                continue
            if name.suffix.lower() not in _DOC_SUFFIXES:
                continue
            out.append((name.name, archive.read(info)))
    if not out:
        raise HTTPException(
            status_code=422,
            detail=(
                "the archive contains no documents of a supported type "
                f"({', '.join(sorted(_DOC_SUFFIXES))})"
            ),
        )
    return out


def _collect_documents(payload: BatchExtractRequest) -> list[tuple[str, bytes]]:
    if bool(payload.zip_b64) == bool(payload.documents):
        raise HTTPException(
            status_code=422, detail="provide exactly one of 'zip_b64' or 'documents'"
        )
    if payload.zip_b64:
        docs = _documents_from_zip(_decode_b64(payload.zip_b64, what="zip_b64"))
    else:
        docs = [
            (PurePosixPath(d.filename).name, _decode_b64(d.content_b64, what=f"documents[{i}]"))
            for i, d in enumerate(payload.documents or [])
        ]
        if not docs:
            raise HTTPException(status_code=422, detail="'documents' is empty")
    if len(docs) > _MAX_DOCUMENTS:
        raise HTTPException(
            status_code=413,
            detail=f"a batch is capped at {_MAX_DOCUMENTS} documents ({len(docs)} given)",
        )
    return docs


@router.post("/extract/batch", response_model=_shared.TriggerResponse)
async def trigger_batch_extract(
    payload: BatchExtractRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    if payload.deployment and payload.model_profile:
        raise HTTPException(
            status_code=400, detail="'deployment' and 'model_profile' are mutually exclusive"
        )
    docs = _collect_documents(payload)

    # Fail-fast on persistence: a batch without its item store is unrecoverable
    # (there is nothing to hand back), so refuse now rather than start a job
    # that can only end in "completed, results lost". The worker re-checks.
    try:
        list_batch_runs(tenant_id=tenant.tenant_id, limit=1)
    except BatchStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Stash every document in the shared blob store NOW; the event carries only
    # keys. Content-addressed, so a re-submitted identical document dedups.
    blobs = default_blob_store()
    inputs: list[dict[str, Any]] = []
    for filename, content in docs:
        stored = blobs.put(
            name=filename, content=content, media_type="application/octet-stream"
        )
        inputs.append(
            {"filename": filename, "relkey": stored.relkey, "size_bytes": stored.size_bytes}
        )

    channel = f"batch:{uuid.uuid4().hex}"
    data: dict[str, Any] = {
        "channel": channel,
        "tenant_id": tenant.tenant_id,
        "name": payload.name or f"batch of {len(docs)}",
        "schema_name": payload.schema_name,
        "inputs": inputs,
    }
    for key in ("dynamic_schema_name", "deployment", "model_profile", "ocr_backend", "language"):
        value = getattr(payload, key)
        if value:
            data[key] = value
    ids = await send_or_503(inngest_client, inngest.Event(name=BATCH_EVENT, data=data))
    _shared._record_event_owners(list(ids), tenant.tenant_id)
    return _shared.TriggerResponse(
        event_ids=list(ids), channel=channel, topics=_shared.DEFAULT_TOPICS
    )


@router.get("/batches")
async def list_batches(tenant: TenantDependency) -> list[dict[str, Any]]:
    """This tenant's recent batches (running, completed, failed), newest first."""
    try:
        return list_batch_runs(tenant_id=tenant.tenant_id)
    except BatchStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/batches/{event_id}")
async def get_batch(event_id: str, tenant: TenantDependency) -> dict[str, Any]:
    """One batch with every item's status/result/error. A foreign tenant's id
    is a 404, never a 403 that confirms existence."""
    try:
        run = get_batch_run(event_id, tenant_id=tenant.tenant_id)
    except BatchStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail=f"batch {event_id!r} not found")
    return run


def _results_artifact(run: dict[str, Any], suffix: str) -> dict[str, Any] | None:
    return next((a for a in run.get("artifacts") or [] if a.get("name", "").endswith(suffix)), None)


@router.get("/batches/{event_id}/results.{fmt}")
async def download_batch_results(event_id: str, fmt: str, tenant: TenantDependency) -> Response:
    """The batch's results file (``jsonl`` or ``csv``), served through the
    batch's own tenant scope. Blobs live in the shared ArtifactBlobStore
    but are authorized HERE, not via /artifacts/{id} (that route resolves
    tenant through StudioRun, and a batch is not one)."""
    if fmt not in ("jsonl", "csv"):
        raise HTTPException(status_code=404, detail="results are available as .jsonl or .csv")
    try:
        run = get_batch_run(event_id, tenant_id=tenant.tenant_id)
    except BatchStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail=f"batch {event_id!r} not found")
    artifact = _results_artifact(run, f".{fmt}")
    if artifact is None:
        raise HTTPException(
            status_code=409,
            detail=f"batch {event_id!r} has no {fmt} results yet (status={run.get('status')})",
        )
    blobs = default_blob_store()
    try:
        content = blobs.read(str(artifact["relkey"]))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=410, detail=f"results blob is gone: {exc}") from exc
    media = "application/x-ndjson" if fmt == "jsonl" else "text/csv"
    return Response(
        content=content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{run.get("name", "batch")}.{fmt}"'},
    )


__all__ = ["BATCH_EVENT", "BatchExtractRequest", "router"]
