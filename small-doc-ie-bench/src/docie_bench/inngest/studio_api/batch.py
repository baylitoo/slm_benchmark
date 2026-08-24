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
    # A saved routing policy (POST /v1/studio/routing-policies): every document
    # runs through the policy's escalation ladder instead of a single model.
    # Mutually exclusive with deployment/model_profile.
    routing_policy: str | None = None
    ocr_backend: str | None = None
    language: str | None = None
    # Optional completion webhook: when the batch settles, the worker POSTs the
    # run summary (status, counts, result-download URIs) to this URL. With
    # ``callback_secret`` set, the body is HMAC-SHA256-signed into an
    # ``X-DocIE-Signature`` header so the receiver can authenticate the caller.
    # http(s) only; delivered best-effort with retries -- a dead receiver never
    # fails the batch.
    callback_url: str | None = Field(default=None, max_length=1000)
    callback_secret: str | None = Field(default=None, max_length=200)


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
    selectors = [payload.deployment, payload.model_profile, payload.routing_policy]
    if sum(1 for sel in selectors if sel) > 1:
        raise HTTPException(
            status_code=400,
            detail="'deployment', 'model_profile' and 'routing_policy' are mutually "
            "exclusive: pick exactly one (a policy names its profiles per stage)",
        )
    if payload.routing_policy:
        # Fail fast at the edge (mirrors /extract): a bad policy name is a 4xx
        # NOW, not a failed Inngest run discovered by polling.
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
    if payload.callback_url and not payload.callback_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=422, detail="'callback_url' must be an http(s) URL"
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
    for key in (
        "dynamic_schema_name",
        "deployment",
        "model_profile",
        "routing_policy",
        "ocr_backend",
        "language",
        "callback_url",
        "callback_secret",
    ):
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


class RetryFailedRequest(BaseModel):
    """Optional model override for the retry -- e.g. re-run the failures with a
    STRONGER model or a routing policy than the original batch used. Empty
    body = the original selectors."""

    deployment: str | None = None
    model_profile: str | None = None
    routing_policy: str | None = None


@router.post("/batches/{event_id}/retry-failed", response_model=_shared.TriggerResponse)
async def retry_failed_items(
    event_id: str, payload: RetryFailedRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    """Re-run ONLY a settled batch's failed documents, as a new batch.

    The per-item state makes this cheap: failed items are identified by
    position, their documents re-read from the blob store via the
    ``input_relkey`` persisted at claim time -- no re-upload. A NEW batch keeps
    the semantics simple (the original stays as the durable record of what
    happened; the retry is its own run, named after it). Selectors default to
    the original submission's (``selectors_json``) and can be overridden --
    the classic move being "retry the failures with the stronger model".
    """
    override = [payload.deployment, payload.model_profile, payload.routing_policy]
    if sum(1 for sel in override if sel) > 1:
        raise HTTPException(
            status_code=400,
            detail="'deployment', 'model_profile' and 'routing_policy' are mutually exclusive",
        )
    try:
        run = get_batch_run(event_id, tenant_id=tenant.tenant_id)
    except BatchStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail=f"batch {event_id!r} not found")
    if run["status"] == "running":
        raise HTTPException(
            status_code=409, detail="batch is still running -- retry once it settles"
        )
    failed = [item for item in run.get("items", []) if item["status"] == "failed"]
    if not failed:
        raise HTTPException(status_code=400, detail="batch has no failed items to retry")
    missing = [item["filename"] for item in failed if not item.get("input_relkey")]
    if missing:
        raise HTTPException(
            status_code=409,
            detail=(
                "these failed documents predate durable input storage and cannot be "
                f"retried without a re-upload: {', '.join(missing[:5])}"
            ),
        )
    blobs = default_blob_store()
    gone = [item["filename"] for item in failed if not blobs.exists(str(item["input_relkey"]))]
    if gone:
        raise HTTPException(
            status_code=410,
            detail=f"input documents no longer in the store: {', '.join(gone[:5])}",
        )

    selectors = dict(run.get("selectors") or {})
    if any(override):
        for key in ("deployment", "model_profile", "routing_policy"):
            selectors.pop(key, None)
        for key, value in (
            ("deployment", payload.deployment),
            ("model_profile", payload.model_profile),
            ("routing_policy", payload.routing_policy),
        ):
            if value:
                selectors[key] = value

    channel = f"batch:{uuid.uuid4().hex}"
    data: dict[str, Any] = {
        "channel": channel,
        "tenant_id": tenant.tenant_id,
        "name": f"retry: {run['name']}",
        "schema_name": run["schema_name"],
        "retry_of": event_id,
        "inputs": [
            {"filename": item["filename"], "relkey": item["input_relkey"]} for item in failed
        ],
        **{key: value for key, value in selectors.items() if value},
    }
    ids = await send_or_503(inngest_client, inngest.Event(name=BATCH_EVENT, data=data))
    _shared._record_event_owners(list(ids), tenant.tenant_id)
    return _shared.TriggerResponse(
        event_ids=list(ids), channel=channel, topics=_shared.DEFAULT_TOPICS
    )


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
