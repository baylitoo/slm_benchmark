from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from docie_bench.agents.api import router as agents_router
from docie_bench.benchmark.routing_config import build_extraction_router
from docie_bench.chat_api import _resolve_or_error, _sse_event
from docie_bench.chat_api import router as chat_router
from docie_bench.extract.routing import (
    ExtractionRouter,
    RoutingPolicy,
    RoutingResult,
    live_routing_audit,
)
from docie_bench.extract.service import ExtractionService, hash_bytes
from docie_bench.inngest.serving_api import router as serving_router
from docie_bench.inngest.serving_api import trigger_deployment_load
from docie_bench.inngest.studio_api import router as studio_router
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.logging_config import configure_logging
from docie_bench.mcp_api import router as mcp_router
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
    get_review_evidence,
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
    ReviewEvidenceView,
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
from docie_bench.storage.db import dispose_engine, get_session_factory, init_engine
from docie_bench.studio import usage_store
from docie_bench.studio.routing_policies import (
    RoutingPolicyUnavailableError,
    get_routing_policy,
)
from docie_bench.telemetry import (
    CONTENT_TYPE_LATEST,
    REVIEW_ACTIONS,
    REVIEW_QUEUE_DEPTH,
    generate_metrics,
)

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_engine()
    sessions = get_session_factory()
    configure_orchestrator(OrchestratorService(sessions) if sessions is not None else None)
    settings.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.annotation_export_dir.mkdir(parents=True, exist_ok=True)
    yield
    dispose_engine()


app = FastAPI(
    title="Small Document IE Benchmark API",
    version="0.1.0",
    lifespan=lifespan,
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
# Generic OpenAI chat over the serving stack (Playground "Chat" + platform
# surface): POST /v1/chat/completions with model = deployment/profile.
app.include_router(chat_router)
# MCP catalog + registry management (browse/enable/disable/test).
app.include_router(mcp_router)

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


async def resolve_profile(profile_name: str | None) -> ModelProfile:
    """Resolve a request's ``model_profile`` via the shared resolver.

    ``None`` deterministically resolves ``settings.default_model_profile``
    (``studio_default``) FROM ``configs/models.yaml`` — the honest label, not the
    old env-synthesized profile. An unknown name is a 400 (unchanged surface).
    ``store:<name>`` refs resolve via the Postgres placement recorded at deploy
    time. A store model that ISN'T live yet (never deployed, or deployed then
    evicted) doesn't just 404/409 here: it fires the load/deploy itself and
    answers 202 with an ETA, so a caller picking any catalog model gets an
    honest "starting, retry in ~Ns" instead of an error for something one API
    call away from working — the whole point being nobody has to know in
    advance whether a model happens to be warm. A name that isn't in the
    catalog at all still 404s; there's nothing to start. Note: these direct
    endpoints run in the api container, so a name that resolves to a worker-local
    deployment endpoint is unreachable here (deployment routing is supported via
    the worker ``/v1/studio/extract`` path — see profile_resolver).
    """
    try:
        profile = resolve_extraction_profile(model_profile=profile_name)
    except (PlacementNotFoundError, PlacementNotReadyError) as exc:
        store_name = (
            profile_name[len(STORE_PROFILE_PREFIX) :]
            if profile_name and profile_name.startswith(STORE_PROFILE_PREFIX)
            else None
        )
        triggered = await trigger_deployment_load(store_name) if store_name else None
        if triggered is not None:
            name, eta = triggered
            raise HTTPException(
                status_code=202,
                detail={
                    "status": "loading",
                    "deployment": name,
                    "eta_seconds": round(eta, 1),
                    "message": (
                        f"Model {name!r} is starting — this can take a moment "
                        f"the first time. Retry in ~{int(eta)}s."
                    ),
                },
            ) from exc
        status_code = 404 if isinstance(exc, PlacementNotFoundError) else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
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


async def resolve_extraction_executor(
    *,
    model_profile: str | None,
    routing_policy: str | None,
    proposer_profile: ModelProfile | None = None,
) -> ExtractionService | ExtractionRouter:
    """The thing that runs this extraction: a single-model service, or a router
    over a saved routing policy's stages.

    ``routing_policy`` names a policy saved via POST /v1/studio/routing-policies
    (the same registry the Benchmark tab picks from -- this is what makes those
    policies USABLE on a real document instead of only evaluable in a
    benchmark). Each stage's profile resolves through the SAME
    ``resolve_profile`` a single-model request uses, so ``store:`` refs,
    load-on-demand 202s and the worker-loopback guard all apply per stage.
    Every stage is resolved UP FRONT: a policy whose escalation target is cold
    fails fast at request time (or fires its load and answers 202), rather than
    mid-route on the document that finally needed it.

    Both return types expose the same ``extract_from_text`` /
    ``extract_from_file`` names, so call sites don't branch on which they got.
    """
    if routing_policy and model_profile:
        raise HTTPException(
            status_code=400,
            detail="'routing_policy' and 'model_profile' are mutually exclusive: a "
            "policy names its model profiles per stage",
        )
    if not routing_policy:
        return ExtractionService(
            await resolve_profile(model_profile), proposer_profile=proposer_profile
        )
    try:
        record = get_routing_policy(routing_policy)
    except RoutingPolicyUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"routing policy {routing_policy!r} not found -- save one via "
            "POST /v1/studio/routing-policies (or pick it in the Studio)",
        )
    policy = RoutingPolicy.model_validate(record["policy"])
    profiles = {
        stage.name: await resolve_profile(stage.name) for stage in policy.stages
    }
    return build_extraction_router(policy, profiles)


def _finalize_outcome(
    outcome: ExtractionResponse | RoutingResult, *, routing_policy: str | None
) -> ExtractionResponse:
    """A route's outcome as an ExtractionResponse. A RoutingResult can only
    have come from a router, which is only built when ``routing_policy`` was
    given -- so the policy name is guaranteed set on that branch; assert the
    invariant rather than let it be a silent ``None`` in the audit."""
    if not isinstance(outcome, RoutingResult):
        return outcome
    assert routing_policy is not None, "a RoutingResult implies a routing_policy"
    return _finalize_routed(outcome, routing_policy=routing_policy)


def _finalize_routed(
    result: RoutingResult, *, routing_policy: str
) -> ExtractionResponse:
    """Unwrap a router's result into the ExtractionResponse the live routes
    return, carrying the audit in the response's ``routing`` field.

    A router can legitimately produce NO response (every stage errored, or the
    budget ran out before any stage answered) -- that's a 502 with the
    terminal reason, not a 500. The audit is shaped by ``live_routing_audit``
    (per-stage ``output`` stripped, policy name added) -- the same helper the
    Studio worker path uses, so both surfaces return the identical shape.
    """
    if result.response is None:
        raise HTTPException(
            status_code=502,
            detail=(
                f"routing policy {routing_policy!r} produced no extraction: "
                f"{result.audit.terminal_reason} "
                f"(decision={result.audit.terminal_decision.value}, "
                f"attempts={result.audit.attempts})"
            ),
        )
    return result.response.model_copy(
        update={"routing": live_routing_audit(result, policy_name=routing_policy)}
    )


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
    # Usage ledger (GET /v1/studio/usage): the one seam both extract routes
    # funnel through with the full response in hand — model latency, token
    # usage, and the RESOLVED profile name. Success-only by construction (an
    # errored extraction raises before reaching here); best-effort, never
    # raises. Chat/embed/rerank record their own rows in chat_api.py.
    usage_store.record_usage(
        deployment=response.model_profile,
        surface="extract",
        tenant_id=tenant_id,
        latency_ms=response.latency_ms,
        prompt_tokens=response.usage.prompt_tokens if response.usage else None,
        completion_tokens=response.usage.completion_tokens if response.usage else None,
        status="ok",
    )
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


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_metrics(), media_type=CONTENT_TYPE_LATEST)


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
    proposer_profile = (
        await resolve_profile(payload.schema_proposer_profile)
        if payload.schema_proposer_profile
        else None
    )
    executor = await resolve_extraction_executor(
        model_profile=payload.model_profile,
        routing_policy=payload.routing_policy,
        proposer_profile=proposer_profile,
    )
    outcome = await executor.extract_from_text(
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
    response = _finalize_outcome(outcome, routing_policy=payload.routing_policy)
    return finalize_response(response, tenant_id=tenant.tenant_id)


@app.post("/v1/extract/file", response_model=ExtractionResponse)
async def extract_file(
    request: Request,
    file: Annotated[UploadFile, File()],
    tenant: TenantDependency,
    schema_name: Annotated[str, Form()] = "invoice",
    model_profile: Annotated[str | None, Form()] = None,
    routing_policy: Annotated[str | None, Form()] = None,
    ocr_backend: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
) -> ExtractionResponse:
    body, suffix, detected_mime = await read_validated_upload(
        file,
        max_bytes=settings.max_upload_bytes,
        allowed_mime_types=settings.allowed_mime_types,
    )

    executor = await resolve_extraction_executor(
        model_profile=model_profile, routing_policy=routing_policy
    )
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)
    try:
        outcome = await executor.extract_from_file(
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
    response = _finalize_outcome(outcome, routing_policy=routing_policy)
    return finalize_response(response, tenant_id=tenant.tenant_id)


class ExtractStreamRequest(BaseModel):
    """Body of ``POST /v1/extract/stream`` (#397) — same fields the
    Playground already builds for ``POST /v1/studio/extract`` (Inngest), so
    the frontend swaps one URL for the other with no payload change."""

    text: str | None = None
    content_b64: str | None = None
    filename: str | None = None
    schema_name: str = "invoice"
    dynamic_schema_name: str | None = None
    deployment: str | None = None
    model_profile: str | None = None
    routing_policy: str | None = None
    ocr_backend: str | None = None
    language: str | None = None


@app.post("/v1/extract/stream")
async def extract_stream(payload: ExtractStreamRequest, tenant: TenantDependency) -> Response:
    """SSE counterpart to ``/v1/extract/text`` + ``/v1/extract/file`` for the
    Playground ONLY (#397) — resolves and runs a direct (non-Inngest)
    extraction exactly like those two routes, except the model call itself
    streams: ``{"type": "delta", "text": ...}`` frames arrive as the raw
    model output streams in (RAW, pre-normalization text — a live preview,
    never something to parse as the result), a ``{"type": "reset"}`` frame
    clears the buffer when ``chat_json`` abandons an attempt and retries
    (response-format downgrade or a gateway-level transient retry), and
    exactly one ``{"type": "result", "result": <ExtractionResponse>}`` frame
    carries the same post-processed, redacted, audited response
    ``/v1/extract/text``/``/v1/extract/file`` return — NuExtract
    normalization, schema rehydration, and grounding all still run after the
    stream ends; the frontend must render that event, never its own
    accumulated deltas, as the actual result. A routed extraction
    (``routing_policy``) has no per-model streaming hook to thread through a
    multi-stage router, so it runs blocking like the sync routes and skips
    straight to the ``result`` frame — still additive, just without a live
    preview for that one path. Terminated by ``data: [DONE]\\n\\n``, the
    same convention every other SSE surface here uses.

    A cold/evicted deployment auto-reloads and answers a plain HTTP 202
    (``_resolve_or_error``, same behavior ``/v1/chat/completions`` already
    gives the Playground's Chat/Vision pickers for a bare deployment name) —
    resolved BEFORE the stream starts, so that's an ordinary HTTP response,
    not an SSE frame.
    """
    if not payload.text and not payload.content_b64:
        raise HTTPException(status_code=422, detail="Provide either 'text' or 'content_b64'")
    if payload.text is not None and len(payload.text) > settings.max_text_chars:
        raise HTTPException(status_code=413, detail="Text content exceeds configured limit")
    if payload.routing_policy and (payload.model_profile or payload.deployment):
        raise HTTPException(
            status_code=400,
            detail="'routing_policy' is mutually exclusive with 'model_profile'/"
            "'deployment': a policy names its model profiles per stage",
        )

    schema_name = payload.schema_name
    schema_mode = "static"
    dynamic_schema: dict[str, Any] | None = None
    if payload.dynamic_schema_name:
        from docie_bench.studio.dynamic_schemas import get_dynamic_schema

        saved = get_dynamic_schema(payload.dynamic_schema_name)
        if saved is None:
            raise HTTPException(
                status_code=404,
                detail=f"dynamic schema {payload.dynamic_schema_name!r} not found",
            )
        schema_mode = "dynamic"
        dynamic_schema = saved["spec"]
        schema_name = payload.dynamic_schema_name

    content: bytes | None = None
    suffix = ".pdf"
    if payload.content_b64:
        import base64
        import binascii

        try:
            content = base64.b64decode(payload.content_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid base64 content: {exc}") from exc
        suffix = Path(payload.filename or "document.pdf").suffix or ".pdf"

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def on_delta(text: str) -> None:
        queue.put_nowait({"type": "delta", "text": text})

    def on_reset() -> None:
        queue.put_nowait({"type": "reset"})

    if payload.routing_policy:
        executor: ExtractionService | ExtractionRouter = await resolve_extraction_executor(
            model_profile=None, routing_policy=payload.routing_policy
        )
    else:
        resolved = await _resolve_or_error(payload.deployment or payload.model_profile or "")
        if isinstance(resolved, JSONResponse):
            return resolved
        executor = ExtractionService(resolved, on_delta=on_delta, on_reset=on_reset)

    async def drive() -> None:
        error: dict[str, Any] | None = None
        try:
            queue.put_nowait({"type": "phase", "phase": "processing"})
            if payload.text is not None:
                outcome = await executor.extract_from_text(
                    text=payload.text,
                    ocr_blocks=None,
                    schema_name=schema_name,
                    schema_mode=schema_mode,
                    dynamic_schema=dynamic_schema,
                    language=payload.language,
                    document_hash=hash_bytes(payload.text.encode("utf-8")),
                    metadata={"source": "playground_stream"},
                )
            else:
                assert content is not None
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)
                try:
                    outcome = await executor.extract_from_file(
                        path=tmp_path,
                        ocr_backend_name=payload.ocr_backend or settings.default_ocr_backend,
                        schema_name=schema_name,
                        schema_mode=schema_mode,
                        dynamic_schema=dynamic_schema,
                        language=payload.language,
                        metadata={
                            "source": "playground_stream",
                            "filename": payload.filename or "document",
                        },
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)
            response = _finalize_outcome(outcome, routing_policy=payload.routing_policy)
            response = finalize_response(response, tenant_id=tenant.tenant_id)
            queue.put_nowait({"type": "result", "result": response.model_dump(mode="json")})
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            error = {"message": detail, "type": "extraction_error", "code": str(exc.status_code)}
        except Exception as exc:  # noqa: BLE001 - must reach the client as an error frame
            error = {"message": str(exc), "type": "internal_error", "code": "internal_error"}
        finally:
            if error is not None:
                queue.put_nowait({"type": "error", "error": error})
            queue.put_nowait(None)

    async def body_iterator() -> AsyncIterator[bytes]:
        task = asyncio.create_task(drive())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse_event(item)
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        yield b"data: [DONE]\n\n"

    return StreamingResponse(body_iterator(), media_type="text/event-stream")


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
    blocks = payload.ocr_blocks or []
    if len(blocks) > settings.max_ocr_blocks:
        raise HTTPException(status_code=413, detail="OCR block count exceeds configured limit")
    if any(len(block.text) > settings.max_ocr_block_chars for block in blocks):
        raise HTTPException(status_code=413, detail="An OCR block exceeds configured limit")
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


@app.get("/v1/reviews/{task_id}/evidence", response_model=ReviewEvidenceView)
def review_evidence(task_id: int, _tenant: TenantDependency) -> ReviewEvidenceView:
    try:
        return get_review_evidence(task_id)
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
