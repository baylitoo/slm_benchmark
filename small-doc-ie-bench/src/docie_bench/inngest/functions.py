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


async def _run_benchmark_job(data: dict[str, Any]) -> dict[str, Any]:
    """Run a benchmark over a dataset; returns paths + parsed metrics."""
    import json

    from docie_bench.benchmark.runner import run_benchmark

    dataset = data.get("dataset")
    if not dataset:
        raise ValueError("benchmark event must include a 'dataset' reference or path")
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
    metrics: dict[str, Any] = {}
    try:
        metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("could not read metrics at %s", result.metrics_path)
    return {
        "run_dir": str(result.run_dir),
        "metrics_path": str(result.metrics_path),
        "report_path": str(result.report_path),
        "predictions_path": str(result.predictions_path),
        "metrics": metrics,
    }


@inngest_client.create_function(
    fn_id="benchmark-run",
    trigger=inngest.TriggerEvent(event="benchmark/run.requested"),
)
async def run_benchmark_job(ctx: inngest.Context) -> dict[str, Any]:
    """Run a full benchmark over a dataset.

    Event ``data``: ``dataset`` (required), ``split``, ``model_profile``,
    ``schema_name``, ``concurrency``, ``repeat``, ``language``, ``channel``.
    """
    data = dict(ctx.event.data or {})
    channel = data.get("channel") or f"run:{ctx.event.id}"

    await publish(channel, TOPIC_STATUS, {"state": "started", "dataset": data.get("dataset")})
    try:
        result = await ctx.step.run("benchmark", lambda: _run_benchmark_job(data))
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


functions = [extract_document, run_benchmark_job, deploy_model_job, seed_ollama_job]

__all__ = [
    "functions",
    "extract_document",
    "run_benchmark_job",
    "deploy_model_job",
    "seed_ollama_job",
]
