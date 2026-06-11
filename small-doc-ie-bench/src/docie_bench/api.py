from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from docie_bench.extract.service import ExtractionService, hash_bytes
from docie_bench.llm.model_profiles import ModelProfile, load_model_profiles
from docie_bench.logging_config import configure_logging
from docie_bench.orchestrator.api import configure_orchestrator, router as orchestrator_router
from docie_bench.orchestrator.service import OrchestratorService
from docie_bench.schemas.api import BenchmarkRunRequest, ExtractTextRequest
from docie_bench.schemas.common import ExtractionResponse
from docie_bench.schemas.extraction import SCHEMA_REGISTRY, schema_json
from docie_bench.settings import get_settings
from docie_bench.storage.audit import save_extraction_audit
from docie_bench.storage.db import get_session_factory, init_engine
from docie_bench.telemetry import EXTRACTION_LATENCY, EXTRACTION_REQUESTS

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Small Document IE Benchmark API",
    version="0.1.0",
)
app.include_router(orchestrator_router)


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
    sessions = get_session_factory()
    configure_orchestrator(OrchestratorService(sessions) if sessions is not None else None)
    settings.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)


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
def get_schema(schema_name: str) -> dict:
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
        document_hash=payload.document_hash or (hash_bytes(payload.text.encode("utf-8")) if payload.text else None),
        metadata=payload.metadata,
    )
    save_extraction_audit(response)
    EXTRACTION_REQUESTS.labels(
        response.schema_name, response.model_profile, str(response.validation.valid).lower()
    ).inc()
    EXTRACTION_LATENCY.labels(response.schema_name, response.model_profile).observe(response.latency_ms / 1000)
    return response


@app.post("/v1/extract/file", response_model=ExtractionResponse)
async def extract_file(
    request: Request,
    file: UploadFile = File(...),
    schema_name: str = Form(default="invoice"),
    model_profile: str | None = Form(default=None),
    ocr_backend: str | None = Form(default=None),
    language: str | None = Form(default=None),
) -> ExtractionResponse:
    body = await file.read()
    if len(body) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.max_upload_mb} MB")
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
            metadata={"filename": file.filename or "unknown", "client": request.client.host if request.client else "unknown"},
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    save_extraction_audit(response)
    EXTRACTION_REQUESTS.labels(
        response.schema_name, response.model_profile, str(response.validation.valid).lower()
    ).inc()
    EXTRACTION_LATENCY.labels(response.schema_name, response.model_profile).observe(response.latency_ms / 1000)
    return response


@app.post("/v1/benchmarks/run")
async def run_benchmark_endpoint(payload: BenchmarkRunRequest) -> dict[str, str]:
    # The API exposes this for orchestration, but production benchmark runs should normally use the CLI.
    from docie_bench.benchmark.runner import run_benchmark

    result = await run_benchmark(
        dataset_path=Path(payload.dataset),
        models_config_path=Path(payload.models_config),
        model_profile=payload.model_profile,
        output_dir=Path(payload.output_dir) if payload.output_dir else None,
        concurrency=payload.concurrency,
    )
    return {"run_dir": str(result.run_dir), "metrics_path": str(result.metrics_path)}
