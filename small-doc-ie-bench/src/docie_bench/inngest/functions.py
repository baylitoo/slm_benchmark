"""Inngest functions: the framework's operations as durable jobs.

Each function is triggered by an event and (best-effort) streams progress to a
realtime channel. The channel defaults to ``run:{event_id}`` but a caller may
pass its own ``channel`` in the event data so a frontend can subscribe *before*
firing the event.

Phase 1 ships ``doc/extract.requested``. Benchmark and model-deploy functions
are added in Phase 2.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import inngest

from docie_bench.extract.service import ExtractionService, hash_bytes
from docie_bench.inngest.client import inngest_client
from docie_bench.inngest.realtime import (
    TOPIC_ERROR,
    TOPIC_RESULT,
    TOPIC_STATUS,
    publish,
)
from docie_bench.llm.model_profiles import ModelProfile, load_model_profiles
from docie_bench.settings import get_settings

logger = logging.getLogger("docie_bench.inngest.functions")

MODELS_CONFIG_PATH = Path("configs/models.yaml")


def _resolve_profile(name: str | None) -> ModelProfile:
    """Resolve a model profile by name, falling back to the env default.

    Mirrors ``docie_bench.api.resolve_profile`` but never raises (jobs prefer a
    sensible default over a hard failure); an unknown name is logged.
    """
    settings = get_settings()
    if name and MODELS_CONFIG_PATH.exists():
        profiles = load_model_profiles(MODELS_CONFIG_PATH)
        if name in profiles:
            return profiles[name]
        if name != settings.default_model_profile:
            logger.warning("unknown model_profile=%r; using env default", name)
    return ModelProfile(
        name=name or settings.default_model_profile,
        model=settings.openai_compat_model,
        base_url=settings.openai_compat_base_url.rstrip("/"),
        api_key=settings.openai_compat_api_key.get_secret_value(),
        response_format_style=settings.openai_compat_response_format_style,
        timeout_seconds=settings.openai_compat_timeout_seconds,
    )


async def _run_extraction(data: dict[str, Any]) -> dict[str, Any]:
    """Run one extraction from event data; returns a JSON-serializable result."""
    schema_name = data.get("schema_name", "invoice")
    language = data.get("language")
    profile = _resolve_profile(data.get("model_profile"))
    service = ExtractionService(profile)

    text = data.get("text")
    if text is not None:
        response = await service.extract_from_text(
            text=text,
            ocr_blocks=None,
            schema_name=schema_name,
            language=language,
            document_hash=hash_bytes(text.encode("utf-8")),
            metadata={"source": "inngest"},
        )
        return response.model_dump(mode="json")

    content_b64 = data.get("content_b64")
    if not content_b64:
        raise ValueError("extract event must include either 'text' or 'content_b64'")
    try:
        raw = base64.b64decode(content_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"content_b64 is not valid base64: {exc}") from exc

    filename = data.get("filename") or "document"
    suffix = Path(filename).suffix or ".pdf"
    settings = get_settings()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        response = await service.extract_from_file(
            path=tmp_path,
            ocr_backend_name=data.get("ocr_backend") or settings.default_ocr_backend,
            schema_name=schema_name,
            language=language,
            metadata={"source": "inngest", "filename": filename},
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return response.model_dump(mode="json")


@inngest_client.create_function(
    fn_id="doc-extract",
    trigger=inngest.TriggerEvent(event="doc/extract.requested"),
)
async def extract_document(ctx: inngest.Context) -> dict[str, Any]:
    """Extract structured data from one document.

    Event ``data``:
      - ``text`` (str)            -- raw document text, OR
      - ``content_b64`` (str)     -- base64-encoded document bytes (PDF/image)
      - ``filename`` (str)        -- used to infer file type for ``content_b64``
      - ``schema_name`` (str)     -- target schema (default ``"invoice"``)
      - ``model_profile`` (str?)  -- profile from configs/models.yaml
      - ``ocr_backend`` (str?)    -- OCR backend for ``content_b64``
      - ``language`` (str?)
      - ``channel`` (str?)        -- realtime channel to publish to
    """
    data = dict(ctx.event.data or {})
    channel = data.get("channel") or f"run:{ctx.event.id}"

    await publish(channel, TOPIC_STATUS, {"state": "started", "schema": data.get("schema_name")})
    try:
        result = await ctx.step.run("extract", lambda: _run_extraction(data))
    except Exception as exc:  # noqa: BLE001 - surface error to the channel then re-raise
        await publish(channel, TOPIC_ERROR, {"message": str(exc)})
        raise
    await publish(channel, TOPIC_RESULT, result)
    return result


# Artifacts persisted to the durable store, in (filename, media_type) order. The
# large predictions.jsonl only ever lands in the blob store, never Postgres.
_BENCHMARK_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("metrics.json", "application/json"),
    ("report.html", "text/html; charset=utf-8"),
    ("predictions.jsonl", "application/x-ndjson"),
)


def benchmark_idempotency_key(data: dict[str, Any]) -> str:
    """Stable dedup key for a benchmark request.

    A caller may pass an explicit ``idempotency_key`` (e.g. to force a fresh run
    with a nonce); otherwise it is derived from the run-defining fields so a
    double-fire of the *same* request resolves to the existing run. Bookkeeping
    like ``channel`` and ``tenant_id`` is excluded so it does not perturb the key.
    """
    import hashlib
    import json

    provided = data.get("idempotency_key")
    if provided:
        return str(provided)
    fields = (
        "dataset",
        "split",
        "model_profile",
        "schema_name",
        "concurrency",
        "repeat",
        "language",
    )
    material = {key: data.get(key) for key in fields}
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return f"bench-{digest[:32]}"


async def _run_benchmark_job(data: dict[str, Any], *, event_id: str) -> dict[str, Any]:
    """Run a benchmark and persist its artifacts to the durable store.

    Returns a path-independent record ``{event_id, status, metrics, artifacts}``
    where each artifact carries an addressable ``uri`` (``/v1/studio/artifacts/{id}``)
    reachable from the api/web replicas — never a worker-local filesystem path.
    """
    import json

    from docie_bench.benchmark.runner import run_benchmark
    from docie_bench.studio.store import default_run_store

    dataset = data.get("dataset")
    if not dataset:
        raise ValueError("benchmark event must include a 'dataset' reference or path")

    store = default_run_store()
    idempotency_key = benchmark_idempotency_key(data)
    tenant_id = str(data.get("tenant_id") or "anonymous")

    # Idempotency: claim the run row BEFORE doing work. A redelivery (same
    # event id) or a duplicate trigger (same idempotency key) short-circuits to
    # the existing record instead of running the benchmark a second time.
    if store.enabled:
        outcome, record = store.claim(
            event_id=event_id,
            idempotency_key=idempotency_key,
            tenant_id=tenant_id,
            dataset=str(dataset),
            model_profile=data.get("model_profile"),
            schema_name=data.get("schema_name", "invoice"),
        )
        if outcome == "exists":
            logger.info("benchmark run %s deduplicated (key=%s)", event_id, idempotency_key)
            return record
    else:
        logger.warning("no DATABASE_URL: benchmark artifacts are not durably indexed")

    try:
        result = await run_benchmark(
            dataset_path=dataset,
            models_config_path=MODELS_CONFIG_PATH,
            model_profile=data.get("model_profile"),
            concurrency=int(data.get("concurrency", 1)),
            repeat=int(data.get("repeat", 1)),
            schema_name=data.get("schema_name", "invoice"),
            language=data.get("language"),
            split=data.get("split"),
        )
    except Exception as exc:  # noqa: BLE001 - record failure so the run can retry
        if store.enabled:
            store.fail(event_id=event_id, error=str(exc))
        raise

    metrics: dict[str, Any] = {}
    try:
        metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("could not read metrics at %s", result.metrics_path)

    if not store.enabled:
        # Degraded mode: no durable index. Return metrics inline (still deliverable)
        # but no addressable artifacts.
        return {"event_id": event_id, "status": "completed", "metrics": metrics, "artifacts": []}

    paths = {
        "metrics.json": result.metrics_path,
        "report.html": result.report_path,
        "predictions.jsonl": result.predictions_path,
    }
    stored: list[tuple[str, Any]] = []
    for name, media_type in _BENCHMARK_ARTIFACTS:
        source = Path(paths[name])
        if not source.exists():
            continue
        blob = store.blobs.put(name=name, content=source.read_bytes(), media_type=media_type)
        stored.append((name, blob))
    return store.complete(event_id=event_id, metrics=metrics, artifacts=stored)


@inngest_client.create_function(
    fn_id="benchmark-run",
    trigger=inngest.TriggerEvent(event="benchmark/run.requested"),
    # Platform-level dedup over 24h; complements the durable DB claim below so a
    # double-fire cannot double-run even before the worker records anything.
    idempotency="event.data.idempotency_key",
)
async def run_benchmark_job(ctx: inngest.Context) -> dict[str, Any]:
    """Run a full benchmark over a dataset and persist addressable artifacts.

    Event ``data``: ``dataset`` (required), ``split``, ``model_profile``,
    ``schema_name``, ``concurrency``, ``repeat``, ``language``, ``channel``,
    ``tenant_id`` (bound at trigger time), ``idempotency_key`` (optional).
    """
    data = dict(ctx.event.data or {})
    channel = data.get("channel") or f"run:{ctx.event.id}"
    event_id = ctx.event.id

    await publish(channel, TOPIC_STATUS, {"state": "started", "dataset": data.get("dataset")})
    try:
        result = await ctx.step.run(
            "benchmark", lambda: _run_benchmark_job(data, event_id=event_id)
        )
    except Exception as exc:  # noqa: BLE001 - surface error then re-raise
        await publish(channel, TOPIC_ERROR, {"message": str(exc)})
        raise
    await publish(channel, TOPIC_RESULT, result)
    return result


@lru_cache(maxsize=1)
def _serving_control_plane() -> Any:
    from docie_bench.serving.control_plane import ControlPlane

    return ControlPlane.from_defaults()


async def _run_deploy(data: dict[str, Any]) -> Any:
    """Deploy a model via the control plane (in-worker subprocess runtime).

    Bare ``model`` (a store-entry name) => ``up`` (GGUF/CPU store model);
    an explicit ``runtime`` => ``serve``. Returns the JSON-safe deployment record.
    """
    model = data.get("model")
    if not model:
        raise ValueError("deploy event must include 'model'")
    cp = _serving_control_plane()
    runtime = data.get("runtime")
    if runtime:
        return await cp.serve(
            model,
            name=data.get("name"),
            runtime=runtime,
            replicas=int(data.get("replicas", 1)),
        )
    return await cp.up(
        model,
        port=int(data.get("port", 8088)),
        context_length=int(data.get("context_length", 8192)),
    )


@inngest_client.create_function(
    fn_id="serving-deploy",
    trigger=inngest.TriggerEvent(event="serving/deploy.requested"),
)
async def deploy_model_job(ctx: inngest.Context) -> Any:
    """Deploy a model so it can serve the gateway/benchmark.

    Event ``data``: ``model`` (required, store-entry name or model id),
    ``runtime``, ``name``, ``port``, ``context_length``, ``replicas``, ``channel``.

    NOTE: requires worker ``scale=1`` (the runtime binds worker-local
    ``127.0.0.1`` and Inngest may route to any replica) and the ``llama-server``
    binary on PATH + a seeded model store; without them deploys fail cleanly on
    the ``error`` topic.
    """
    data = dict(ctx.event.data or {})
    channel = data.get("channel") or f"run:{ctx.event.id}"

    await publish(channel, TOPIC_STATUS, {"state": "started", "model": data.get("model")})
    try:
        result = await ctx.step.run("deploy", lambda: _run_deploy(data))
    except Exception as exc:  # noqa: BLE001 - surface error then re-raise
        await publish(channel, TOPIC_ERROR, {"message": str(exc)})
        raise
    await publish(channel, TOPIC_RESULT, result)
    return result


def _serving_home() -> Path:
    return Path(
        os.environ.get(
            "DOCIE_SERVING_HOME", Path.home() / ".local" / "share" / "docie-bench" / "serving"
        )
    )


async def _run_seed_ollama(data: dict[str, Any]) -> dict[str, Any]:
    """Seed a GGUF from the host's Ollama into the store + record it in the catalog."""
    from docie_bench.serving.catalog import ModelCatalog
    from docie_bench.serving.model_store import ModelStore

    reference = data.get("reference")
    name = data.get("name")
    family = data.get("family", "openai_chat")
    if not reference or not name:
        raise ValueError("seed event must include 'reference' and 'name'")

    store = ModelStore(_serving_home() / "models")
    # Blocking file I/O (hard-link or copy of multi-GB blobs) -> off the loop.
    entry = await asyncio.to_thread(
        store.seed_from_ollama, reference, name=name, family=family
    )
    size = entry.model_path.stat().st_size if entry.model_path.exists() else None
    return ModelCatalog().upsert(entry, size_bytes=size)


@inngest_client.create_function(
    fn_id="serving-seed-ollama",
    trigger=inngest.TriggerEvent(event="serving/seed.requested"),
)
async def seed_ollama_job(ctx: inngest.Context) -> dict[str, Any]:
    """Seed a model from the host Ollama into the GGUF store.

    Event ``data``: ``reference`` (e.g. "qwen2.5:1.5b"), ``name`` (store name),
    ``family`` (default "openai_chat"), ``channel``.
    """
    data = dict(ctx.event.data or {})
    channel = data.get("channel") or f"run:{ctx.event.id}"

    await publish(channel, TOPIC_STATUS, {"state": "seeding", "reference": data.get("reference")})
    try:
        result = await ctx.step.run("seed-ollama", lambda: _run_seed_ollama(data))
    except Exception as exc:  # noqa: BLE001 - surface error then re-raise
        await publish(channel, TOPIC_ERROR, {"message": str(exc)})
        raise
    await publish(channel, TOPIC_RESULT, result)
    return result


def _gc_studio_runs_sync() -> dict[str, int]:
    """Apply the retention policy to the durable Studio run index (blocking)."""
    from docie_bench.studio.store import default_run_store

    store = default_run_store()
    if not store.enabled:
        logger.info("studio run GC skipped: no DATABASE_URL")
        return {"deleted_runs": 0, "deleted_blobs": 0, "retained_runs": 0}
    settings = get_settings()
    summary = store.gc(
        max_age_days=settings.studio_run_retention_days,
        max_runs=settings.studio_run_retention_max,
        orphan_grace_hours=settings.studio_orphan_grace_hours,
    )
    logger.info("studio run GC: %s", summary)
    return summary


async def _gc_studio_runs() -> dict[str, int]:
    # Blocking DB + filesystem work -> off the event loop (mirrors seed_ollama_job).
    return await asyncio.to_thread(_gc_studio_runs_sync)


@inngest_client.create_function(
    fn_id="studio-runs-gc",
    trigger=inngest.TriggerCron(cron="0 3 * * *"),
)
async def gc_studio_runs_job(ctx: inngest.Context) -> dict[str, int]:
    """Nightly retention sweep for the Studio run index (rows + orphan blobs).

    Bounds unbounded run accumulation: deletes runs older than
    ``STUDIO_RUN_RETENTION_DAYS`` or beyond the newest ``STUDIO_RUN_RETENTION_MAX``,
    and prunes any blob no surviving run still references. Idempotent — a second
    sweep with nothing to collect is a no-op.
    """
    return await ctx.step.run("gc-studio-runs", _gc_studio_runs)


functions = [
    extract_document,
    run_benchmark_job,
    deploy_model_job,
    seed_ollama_job,
    gc_studio_runs_job,
]

__all__ = [
    "functions",
    "extract_document",
    "run_benchmark_job",
    "deploy_model_job",
    "seed_ollama_job",
    "gc_studio_runs_job",
    "benchmark_idempotency_key",
]
