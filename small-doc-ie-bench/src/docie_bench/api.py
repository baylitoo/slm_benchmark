from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from docie_bench.extract.service import ExtractionService, hash_bytes
from docie_bench.llm.model_profiles import ModelProfile, load_model_profiles
from docie_bench.logging_config import configure_logging
from docie_bench.orchestrator.api import configure_orchestrator, router as orchestrator_router
from docie_bench.orchestrator.service import OrchestratorService
from docie_bench.schemas.api import BenchmarkRunRequest, ExtractTextRequest
from docie_bench.schemas.common import ExtractionResponse
from docie_bench.schemas.extraction import SCHEMA_REGISTRY, schema_json
from docie_bench.security import (
    TenantContext,
    TenantQuotaManager,
    parse_api_keys,
    read_validated_upload,
    redact_fields,
)
from docie_bench.settings import get_settings
from docie_bench.storage.audit import save_extraction_audit
from docie_bench.storage.db import get_session_factory, init_engine
from docie_bench.telemetry import EXTRACTION_LATENCY, EXTRACTION_REQUESTS

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)
quota_manager = TenantQuotaManager(
    api_keys=parse_api_keys(settings.api_keys.get_secret_value()),
    auth_required=settings.auth_required,
    requests_per_window=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
    max_concurrent=settings.tenant_max_concurrent_requests,
)

app = FastAPI(
    title="Small Document IE Benchmark API",
    version="0.1.0",
)
app.include_router(orchestrator_router)


@app.middleware("http")
async def enforce_request_content_length(request: Request, call_next):
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


def default_profile() -> ModelProfile:
    return ModelProfile(
        name=settings.default_model_profile,
        model=settings.openai_compat_model,
        base_url=settings.openai_compat_base_url.rstrip("/"),
        api_key=settings.openai_compat_api_key.get_secret_value(),
        response_format_style=settings.openai_compat_response_format_style,
        timeout_seconds=settings.openai_compat_timeout_seconds,
    )


def resolve_profile(profile_name: str | None) -> ModelProfile:
    if profile_name is None:
        return default_profile()
    config_path = Path("configs/models.yaml")
    if config_path.exists():
        profiles = load_model_profiles(config_path)
        if profile_name in profiles:
            return profiles[profile_name]
    if profile_name == settings.default_model_profile:
        return default_profile()
    raise HTTPException(status_code=400, detail=f"Unknown model_profile={profile_name!r}")


async def tenant_guard(
    x_api_key: Annotated[str | None, Header()] = None,
) -> AsyncIterator[TenantContext]:
    context = quota_manager.authenticate(x_api_key)
    quota_manager.acquire(context)
    try:
        yield context
    finally:
        quota_manager.release(context)


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
    save_extraction_audit(response, tenant_id=tenant_id)
    if not settings.response_redaction_fields:
        return response
    return response.model_copy(
        update={"result": redact_fields(response.result, settings.response_redaction_fields)}
    )


TenantDependency = Annotated[TenantContext, Depends(tenant_guard)]


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
def get_schema(schema_name: str, _tenant: TenantDependency) -> dict:
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
    EXTRACTION_REQUESTS.labels(
        response.schema_name, response.model_profile, str(response.validation.valid).lower()
    ).inc()
    EXTRACTION_LATENCY.labels(response.schema_name, response.model_profile).observe(
        response.latency_ms / 1000
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
    EXTRACTION_REQUESTS.labels(
        response.schema_name, response.model_profile, str(response.validation.valid).lower()
    ).inc()
    EXTRACTION_LATENCY.labels(response.schema_name, response.model_profile).observe(
        response.latency_ms / 1000
    )
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
    payload: ReviewTaskCreate, force: bool = Query(default=False)
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
    status: ReviewStatus | None = None,
    reviewer_id: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[ReviewTaskView]:
    try:
        return list_reviews(status=status, reviewer_id=reviewer_id, limit=limit)
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.get("/v1/reviews/metrics", response_model=ReviewMetricsView)
def get_review_metrics() -> ReviewMetricsView:
    try:
        result = review_metrics()
        for review_status, count in result.queue_depth.items():
            REVIEW_QUEUE_DEPTH.labels(review_status).set(count)
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/exports", response_model=AnnotationExportView)
def export_review_annotations(payload: AnnotationExportRequest) -> AnnotationExportView:
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
def review_detail(task_id: int) -> ReviewTaskView:
    try:
        return get_review(task_id)
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/{task_id}/claim", response_model=ReviewTaskView)
def claim_review_task(task_id: int, payload: ClaimRequest) -> ReviewTaskView:
    try:
        result = claim_review(
            task_id,
            reviewer_id=payload.reviewer_id,
            expected_version=payload.expected_version,
            lease_seconds=payload.lease_seconds or settings.review_claim_lease_seconds,
        )
        REVIEW_ACTIONS.labels("claimed").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/{task_id}/release", response_model=ReviewTaskView)
def release_review_task(task_id: int, payload: ReleaseRequest) -> ReviewTaskView:
    try:
        result = release_review(
            task_id,
            reviewer_id=payload.reviewer_id,
            expected_version=payload.expected_version,
            comment=payload.comment,
        )
        REVIEW_ACTIONS.labels("released").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/{task_id}/correct", response_model=ReviewTaskView)
def correct_review_task(task_id: int, payload: CorrectionRequest) -> ReviewTaskView:
    try:
        result = correct_review(
            task_id,
            reviewer_id=payload.reviewer_id,
            expected_version=payload.expected_version,
            corrections=payload.corrections,
            comment=payload.comment,
        )
        REVIEW_ACTIONS.labels("corrected").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/{task_id}/approve", response_model=ReviewTaskView)
def approve_review_task(task_id: int, payload: DecisionRequest) -> ReviewTaskView:
    try:
        result = decide_review(
            task_id,
            reviewer_id=payload.reviewer_id,
            expected_version=payload.expected_version,
            decision=ReviewStatus.APPROVED,
            comment=payload.comment,
        )
        REVIEW_ACTIONS.labels("approved").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc


@app.post("/v1/reviews/{task_id}/reject", response_model=ReviewTaskView)
def reject_review_task(task_id: int, payload: DecisionRequest) -> ReviewTaskView:
    try:
        result = decide_review(
            task_id,
            reviewer_id=payload.reviewer_id,
            expected_version=payload.expected_version,
            decision=ReviewStatus.REJECTED,
            comment=payload.comment,
        )
        REVIEW_ACTIONS.labels("rejected").inc()
        return result
    except Exception as exc:
        raise _review_http_error(exc) from exc
