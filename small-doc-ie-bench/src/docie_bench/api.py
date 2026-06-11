from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from docie_bench.extract.service import ExtractionService, hash_bytes
from docie_bench.llm.model_profiles import ModelProfile, load_model_profiles
from docie_bench.logging_config import configure_logging
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
from docie_bench.settings import get_settings
from docie_bench.storage.audit import save_extraction_audit
from docie_bench.storage.db import init_engine
from docie_bench.telemetry import (
    EXTRACTION_LATENCY,
    EXTRACTION_REQUESTS,
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


@app.on_event("startup")
def startup() -> None:
    init_engine()
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
def list_schemas() -> dict[str, list[str]]:
    return {"schemas": sorted(SCHEMA_REGISTRY)}


@app.get("/v1/schemas/{schema_name}")
def get_schema(schema_name: str) -> dict[str, Any]:
    try:
        return schema_json(schema_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/extract/text", response_model=ExtractionResponse)
async def extract_text(payload: ExtractTextRequest) -> ExtractionResponse:
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
    save_extraction_audit(response)
    EXTRACTION_REQUESTS.labels(
        response.schema_name, response.model_profile, str(response.validation.valid).lower()
    ).inc()
    EXTRACTION_LATENCY.labels(response.schema_name, response.model_profile).observe(
        response.latency_ms / 1000
    )
    return response


@app.post("/v1/extract/file", response_model=ExtractionResponse)
async def extract_file(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    schema_name: str = Form(default="invoice"),
    model_profile: str | None = Form(default=None),
    ocr_backend: str | None = Form(default=None),
    language: str | None = Form(default=None),
) -> ExtractionResponse:
    body = await file.read()
    if len(body) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413, detail=f"File too large. Max {settings.max_upload_mb} MB"
        )
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    if suffix not in {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise HTTPException(status_code=415, detail=f"Unsupported file suffix: {suffix}")

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
            },
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    save_extraction_audit(response)
    EXTRACTION_REQUESTS.labels(
        response.schema_name, response.model_profile, str(response.validation.valid).lower()
    ).inc()
    EXTRACTION_LATENCY.labels(response.schema_name, response.model_profile).observe(
        response.latency_ms / 1000
    )
    return response


@app.post("/v1/benchmarks/run")
async def run_benchmark_endpoint(payload: BenchmarkRunRequest) -> dict[str, str]:
    # Production benchmark runs should normally use the CLI; this endpoint supports orchestration.
    from docie_bench.benchmark.runner import run_benchmark

    result = await run_benchmark(
        dataset_path=Path(payload.dataset),
        models_config_path=Path(payload.models_config),
        model_profile=payload.model_profile,
        output_dir=Path(payload.output_dir) if payload.output_dir else None,
        concurrency=payload.concurrency,
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
