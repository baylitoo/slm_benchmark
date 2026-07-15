from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from docie_bench.agents.api import router as agents_router
from docie_bench.extract.service import ExtractionService, hash_bytes
from docie_bench.inngest.serving_api import router as serving_router
from docie_bench.inngest.studio_api import router as studio_router
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.logging_config import configure_logging
from docie_bench.orchestrator.api import configure_orchestrator
from docie_bench.orchestrator.api import router as orchestrator_router
from docie_bench.orchestrator.service import OrchestratorService
from docie_bench.review import (
    ReviewConflictError,
    ReviewNotFoundError,
    ReviewValidationError,
    claim_review,
    correct_review,
    decide_review,
    enqueue_review,
    export_annotations,
    get_review,
    list_reviews,
    release_review,
    review_metrics,
)
from docie_bench.schemas.api import BenchmarkRunRequest, ExtractTextRequest
from docie_bench.schemas.common import ExtractionResponse
from docie_bench.schemas.extraction import SCHEMA_REGISTRY, schema_json
from docie_bench.schemas.review import (
    AnnotationExportRequest,
    AnnotationExportView,
    ClaimRequest,
    CorrectionRequest,
    DecisionRequest,
    ReleaseRequest,
    ReviewMetricsView,
    ReviewStatus,
    ReviewTaskCreate,
    ReviewTaskView,
)
from docie_bench.security import (
    TenantDependency,
    read_validated_upload,
    redact_fields,
    tenant_guard,
)
from docie_bench.serving import recency
from docie_bench.serving.placement_resolver import (
    STORE_PROFILE_PREFIX,
    PlacementNotFoundError,
    PlacementNotReadyError,
    endpoint_is_loopback,
)
from docie_bench.serving.profile_resolver import (
    ProfileResolutionError,
    resolve_extraction_profile,
)
from docie_bench.settings import get_settings
from docie_bench.storage.audit import record_extraction
from docie_bench.storage.db import get_session_factory, init_engine
from docie_bench.telemetry import (
    REVIEW_ACTIONS,
    REVIEW_QUEUE_DEPTH,
)

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Small Document IE Benchmark API",
    version="0.1.0",
)
# Privileged/destructive routers (worker lease ops + experiment control, studio
# job triggers, serving control plane) are gated by tenant_guard at include time,
# so every route they expose requires a valid API key (B1). The guard is cached
# per request, so worker routes that also declare `tenant` acquire quota once.
app.include_router(orchestrator_router, dependencies=[Depends(tenant_guard)])
app.include_router(studio_router, dependencies=[Depends(tenant_guard)])
app.include_router(serving_router, dependencies=[Depends(tenant_guard)])
# The agents router carries its own guard (agents_tenant_guard): same manager,
# but the OpenAI surface also accepts `Authorization: Bearer` so stock OpenAI
# SDK clients can consume agents without custom headers.
app.include_router(agents_router)

# Allow the DocIE Studio frontend (separate origin) to call the API from the
# browser. Defaults to the local Studio UI origins; override via
# STUDIO_CORS_ORIGINS (comma-separated). Set "*" explicitly to allow any origin.
_DEFAULT_CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def parse_cors_origins(raw: str | None) -> list[str]:
    """Parse STUDIO_CORS_ORIGINS into an allow-origins list.

    Empty/unset falls back to the explicit localhost Studio origins rather than
    a wildcard, so a networked deployment does not default to allowing any
    origin. Users can still opt into "*" explicitly.
    """
    if raw is None:
        return list(_DEFAULT_CORS_ORIGINS)
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or list(_DEFAULT_CORS_ORIGINS)


_cors_origins = parse_cors_origins(os.getenv("STUDIO_CORS_ORIGINS"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_request_content_length(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        if size < 0:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        if size > settings.max_request_body_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request body too large"})
    return await call_next(request)


def resolve_profile(profile_name: str | None) -> ModelProfile:
    """Resolve a request's ``model_profile`` via the shared resolver.

    ``None`` deterministically resolves ``settings.default_model_profile``
    (``studio_default``) FROM ``configs/models.yaml`` — the honest label, not the
    old env-synthesized profile. An unknown name is a 400 (unchanged surface).
    ``store:<name>`` refs resolve via the Postgres placement recorded at deploy
    time — mapped to 404 (never deployed) / 409 (not ready). Note: these direct
    endpoints run in the api container, so a name that resolves to a worker-local
    deployment endpoint is unreachable here (deployment routing is supported via
    the worker ``/v1/studio/extract`` path — see profile_resolver).
    """
    try:
        profile = resolve_extraction_profile(model_profile=profile_name)
    except PlacementNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PlacementNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ProfileResolutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if (
        profile_name
        and profile_name.startswith(STORE_PROFILE_PREFIX)
        and endpoint_is_loopback(profile.base_url)
    ):
        # The deploy runtime records its endpoint from the WORKER's point of
        # view; in the documented api/worker compose topology a loopback
        # endpoint is unreachable from this process. Fail fast (worker-only
        # for now) instead of burning timeout_seconds x retries on doomed
        # connects. Deployments recorded with an advertised (non-loopback)
        # endpoint pass this guard untouched.
        raise HTTPException(
            status_code=501,
            detail=(
                f"{profile_name} resolved to {profile.base_url}, which is "
                "loopback on the worker that deployed it and not reachable "
                "from the API. Run the extraction through the worker "
                "(POST /v1/studio/extract) or record a non-loopback "
                "advertised endpoint at deploy time."
            ),
        )
    return profile


def validate_text_request(payload: ExtractTextRequest) -> None:
    if payload.text is not None and len(payload.text) > settings.max_text_chars:
        raise HTTPException(status_code=413, detail="Text content exceeds configured limit")
    blocks = payload.ocr_blocks or []
    if len(blocks) > settings.max_ocr_blocks:
        raise HTTPException(status_code=413, detail="OCR block count exceeds configured limit")
    if any(len(block.text) > settings.max_ocr_block_chars for block in blocks):
        raise HTTPException(status_code=413, detail="An OCR block exceeds configured limit")
    if sum(len(block.text) for block in blocks) > settings.max_text_chars:
        raise HTTPException(status_code=413, detail="OCR text exceeds configured limit")
    if len(payload.metadata) > settings.max_metadata_entries:
        raise HTTPException(status_code=413, detail="Metadata entry count exceeds configured limit")
    if any(len(key) > 128 or len(value) > 2_000 for key, value in payload.metadata.items()):
        raise HTTPException(
            status_code=413, detail="Metadata key or value exceeds configured limit"
        )


def finalize_response(response: ExtractionResponse, *, tenant_id: str) -> ExtractionResponse:
    record_extraction(response, tenant_id=tenant_id)
    # PR-4 recency (review fix): the direct API extract endpoints serve
    # traffic too, so they must stamp last_served like the worker path — or a
    # deployment driven only through this surface reads as idle forever and
    # becomes the first idle-TTL/LRU eviction victim mid-use. Best-effort,
    # sidecar-only (never deployments.json); the api mounts the shared
    # serving-state volume. `model_profile` is the resolved profile's honest
    # name (a deployment name, or `store:<name>` — the helper strips it).
    recency.stamp_served_profile(response.model_profile)
    if not settings.response_redaction_fields:
        return response
    return response.model_copy(
        update={"result": redact_fields(response.result, settings.response_redaction_fields)}
    )


@app.on_event("startup")
def startup() -> None:
    init_engine()
    sessions = get_session_factory()
    configure_orchestrator(OrchestratorService(sessions) if sessions is not None else None)
    settings.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.annotation_export_dir.mkdir(parents=True, exist_ok=True)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/schemas")
def list_schemas(_tenant: TenantDependency) -> dict[str, list[str]]:
    return {"schemas": sorted(SCHEMA_REGISTRY)}


@app.get("/v1/schemas/{schema_name}")
def get_schema(schema_name: str, _tenant: TenantDependency) -> dict[str, Any]:
    try:
        return schema_json(schema_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/extract/text", response_model=ExtractionResponse)
async def extract_text(
    payload: ExtractTextRequest,
    tenant: TenantDependency,
) -> ExtractionResponse:
    validate_text_request(payload)
    profile = resolve_profile(payload.model_profile)
    proposer_profile = (
        resolve_profile(payload.schema_proposer_profile)
        if payload.schema_proposer_profile
        else None
    )
    service = ExtractionService(profile, proposer_profile=proposer_profile)
    response = await service.extract_from_text(
        text=payload.text,
        ocr_blocks=payload.ocr_blocks,
        schema_name=payload.schema_name,
        schema_mode=payload.schema_mode,
        dynamic_schema=payload.dynamic_schema,
        language=payload.language,
        document_hash=payload.document_hash
        or (hash_bytes(payload.text.encode("utf-8")) if payload.text else None),
        metadata=payload.metadata,
    )
    return finalize_response(response, tenant_id=tenant.tenant_id)


@app.post("/v1/extract/file", response_model=ExtractionResponse)
async def extract_file(
    request: Request,
    file: Annotated[UploadFile, File()],
    tenant: TenantDependency,
    schema_name: Annotated[str, Form()] = "invoice",
    model_profile: Annotated[str | None, Form()] = None,
    ocr_backend: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
) -> ExtractionResponse:
    body, suffix, detected_mime = await read_validated_upload(
        file,
        max_bytes=settings.max_upload_bytes,
        allowed_mime_types=settings.allowed_mime_types,
    )

    profile = resolve_profile(model_profile)
    service = ExtractionService(profile)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)
    try:
        response = await service.extract_from_file(
            path=tmp_path,
            ocr_backend_name=ocr_backend or settings.default_ocr_backend,
            schema_name=schema_name,
            language=language,
            metadata={
                "filename": file.filename or "unknown",
                "client": request.client.host if request.client else "unknown",
                "content_type": detected_mime,
            },
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return finalize_response(response, tenant_id=tenant.tenant_id)


@app.post("/v1/benchmarks/run")
async def run_benchmark_endpoint(
    payload: BenchmarkRunRequest,
    _tenant: TenantDependency,
) -> dict[str, str]:
    if not settings.enable_benchmark_api:
        raise HTTPException(status_code=404, detail="Benchmark API is disabled")
    # Production benchmark runs should normally use the CLI.
    from docie_bench.benchmark.runner import run_benchmark

    result = await run_benchmark(
        dataset_path=payload.dataset,
        models_config_path=Path(payload.models_config),
        model_profile=payload.model_profile,
        output_dir=Path(payload.output_dir) if payload.output_dir else None,
        concurrency=payload.concurrency,
        split=payload.split,
    )
    return {"run_dir": str(result.run_dir), "metrics_path": str(result.metrics_path)}


def _review_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ReviewNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ReviewConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ReviewValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    raise exc


@app.post("/v1/reviews", response_model=ReviewTaskView | None)
def create_review(
    payload: ReviewTaskCreate, _tenant: TenantDependency, force: bool = Query(default=False)
) -> ReviewTaskView | None:
    try:
        task = enqueue_review(payload, force=force)
        if task:
            REVIEW_ACTIONS.labels("enqueued").inc()
        return task
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.get("/v1/reviews", response_model=list[ReviewTaskView])
def review_queue(
    _tenant: TenantDependency,
    status: ReviewStatus | None = None,
    reviewer_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[ReviewTaskView]:
    try:
        return list_reviews(status=status, reviewer_id=reviewer_id, limit=limit)
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.get("/v1/reviews/metrics", response_model=ReviewMetricsView)
def get_review_metrics(_tenant: TenantDependency) -> ReviewMetricsView:
    try:
        result = review_metrics()
        for review_status, count in result.queue_depth.items():
            REVIEW_QUEUE_DEPTH.labels(review_status).set(count)
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/exports", response_model=AnnotationExportView)
def export_review_annotations(
    payload: AnnotationExportRequest, _tenant: TenantDependency
) -> AnnotationExportView:
    try:
        result = export_annotations(
            version=payload.version,
            split=payload.split,
            output_root=settings.annotation_export_dir,
            task_ids=payload.task_ids,
        )
        REVIEW_ACTIONS.labels("exported").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.get("/v1/reviews/{task_id}", response_model=ReviewTaskView)
def review_detail(task_id: int, _tenant: TenantDependency) -> ReviewTaskView:
    try:
        return get_review(task_id)
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/{task_id}/claim", response_model=ReviewTaskView)
def claim_review_task(
    task_id: int, payload: ClaimRequest, tenant: TenantDependency
) -> ReviewTaskView:
    try:
        result = claim_review(
            task_id,
            reviewer_id=tenant.tenant_id,  # B2: provenance = authenticated principal
            expected_version=payload.expected_version,
            lease_seconds=payload.lease_seconds or settings.review_claim_lease_seconds,
        )
        REVIEW_ACTIONS.labels("claimed").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/{task_id}/release", response_model=ReviewTaskView)
def release_review_task(
    task_id: int, payload: ReleaseRequest, tenant: TenantDependency
) -> ReviewTaskView:
    try:
        result = release_review(
            task_id,
            reviewer_id=tenant.tenant_id,  # B2
            expected_version=payload.expected_version,
            comment=payload.comment,
        )
        REVIEW_ACTIONS.labels("released").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/{task_id}/correct", response_model=ReviewTaskView)
def correct_review_task(
    task_id: int, payload: CorrectionRequest, tenant: TenantDependency
) -> ReviewTaskView:
    try:
        result = correct_review(
            task_id,
            reviewer_id=tenant.tenant_id,  # B2
            expected_version=payload.expected_version,
            corrections=payload.corrections,
            comment=payload.comment,
        )
        REVIEW_ACTIONS.labels("corrected").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/{task_id}/approve", response_model=ReviewTaskView)
def approve_review_task(
    task_id: int, payload: DecisionRequest, tenant: TenantDependency
) -> ReviewTaskView:
    try:
        result = decide_review(
            task_id,
            reviewer_id=tenant.tenant_id,  # B2
            expected_version=payload.expected_version,
            decision=ReviewStatus.APPROVED,
            comment=payload.comment,
        )
        REVIEW_ACTIONS.labels("approved").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/{task_id}/reject", response_model=ReviewTaskView)
def reject_review_task(
    task_id: int, payload: DecisionRequest, tenant: TenantDependency
) -> ReviewTaskView:
    try:
        result = decide_review(
            task_id,
            reviewer_id=tenant.tenant_id,  # B2
            expected_version=payload.expected_version,
            decision=ReviewStatus.REJECTED,
            comment=payload.comment,
        )
        REVIEW_ACTIONS.labels("rejected").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc
