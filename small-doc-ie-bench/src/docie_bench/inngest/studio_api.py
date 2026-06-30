"""FastAPI glue for DocIE Studio: trigger jobs + consume results.

Mounted on the main API (``app.include_router(studio_router)``). Three routes
give the frontend everything it needs:

  - ``POST /v1/studio/extract``        -- fire ``doc/extract.requested``; returns
                                          the event id(s) and the realtime channel.
  - ``GET  /v1/studio/realtime-token`` -- mint a subscription token (the "hook").
  - ``GET  /v1/studio/runs/{id}``      -- polling fallback: proxy the Inngest
                                          server's run status for an event.

Realtime is best-effort; if the experimental API is unavailable the token route
returns 501 and the frontend falls back to polling ``/runs``.
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated, Any

import httpx
import inngest
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from docie_bench.inngest.client import inngest_client
from docie_bench.inngest.realtime import (
    TOPIC_ERROR,
    TOPIC_PROGRESS,
    TOPIC_RESULT,
    TOPIC_STATUS,
    subscription_token,
)

router = APIRouter(prefix="/v1/studio", tags=["studio"])

DEFAULT_TOPICS = [TOPIC_STATUS, TOPIC_PROGRESS, TOPIC_RESULT, TOPIC_ERROR]
EXTRACT_EVENT = "doc/extract.requested"
BENCHMARK_EVENT = "benchmark/run.requested"
DEPLOY_EVENT = "serving/deploy.requested"
SEED_EVENT = "serving/seed.requested"


class ExtractRequest(BaseModel):
    text: str | None = None
    content_b64: str | None = None
    filename: str | None = None
    schema_name: str = "invoice"
    model_profile: str | None = None
    ocr_backend: str | None = None
    language: str | None = None


class TriggerResponse(BaseModel):
    event_ids: list[str]
    channel: str
    topics: list[str]


@router.post("/extract", response_model=TriggerResponse)
async def trigger_extract(payload: ExtractRequest) -> TriggerResponse:
    if not payload.text and not payload.content_b64:
        raise HTTPException(status_code=422, detail="Provide either 'text' or 'content_b64'")
    channel = f"extract:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    ids = await inngest_client.send(inngest.Event(name=EXTRACT_EVENT, data=data))
    return TriggerResponse(event_ids=list(ids), channel=channel, topics=DEFAULT_TOPICS)


class BenchmarkRequest(BaseModel):
    dataset: str
    split: str | None = None
    model_profile: str | None = None
    schema_name: str = "invoice"
    concurrency: int = 1
    repeat: int = 1
    language: str | None = None


@router.post("/benchmark", response_model=TriggerResponse)
async def trigger_benchmark(payload: BenchmarkRequest) -> TriggerResponse:
    channel = f"benchmark:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    ids = await inngest_client.send(inngest.Event(name=BENCHMARK_EVENT, data=data))
    return TriggerResponse(event_ids=list(ids), channel=channel, topics=DEFAULT_TOPICS)


class DeployRequest(BaseModel):
    model: str
    name: str | None = None
    runtime: str | None = None
    port: int = 8088
    context_length: int = 8192
    replicas: int = 1


@router.post("/deploy", response_model=TriggerResponse)
async def trigger_deploy(payload: DeployRequest) -> TriggerResponse:
    channel = f"deploy:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    ids = await inngest_client.send(inngest.Event(name=DEPLOY_EVENT, data=data))
    return TriggerResponse(event_ids=list(ids), channel=channel, topics=DEFAULT_TOPICS)


class SeedOllamaRequest(BaseModel):
    reference: str  # e.g. "qwen2.5:1.5b" or "hf.co/numind/NuExtract3-GGUF:Q4_K_M"
    name: str  # store entry name
    family: str = "openai_chat"


@router.post("/seed-ollama", response_model=TriggerResponse)
async def trigger_seed_ollama(payload: SeedOllamaRequest) -> TriggerResponse:
    channel = f"seed:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    ids = await inngest_client.send(inngest.Event(name=SEED_EVENT, data=data))
    return TriggerResponse(event_ids=list(ids), channel=channel, topics=DEFAULT_TOPICS)


@router.get("/realtime-token")
async def realtime_token(
    channel: Annotated[str, Query(min_length=1)],
    topics: Annotated[list[str] | None, Query()] = None,
) -> Any:
    try:
        return await subscription_token(channel, topics or DEFAULT_TOPICS)
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.get("/runs/{event_id}")
async def event_runs(event_id: str) -> Any:
    """Polling fallback: proxy the Inngest server's run status for an event."""
    base = os.getenv("INNGEST_BASE_URL", "http://localhost:8288").rstrip("/")
    headers = {}
    signing_key = os.getenv("INNGEST_SIGNING_KEY")
    if signing_key:
        headers["Authorization"] = f"Bearer {signing_key}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{base}/v1/events/{event_id}/runs", headers=headers)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


__all__ = ["router"]
