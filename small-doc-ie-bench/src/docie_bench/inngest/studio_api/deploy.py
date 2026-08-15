"""Deploy trigger, HF-Hub search/inspect/seed, and Ollama seed routes."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

import httpx
import inngest
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from docie_bench.inngest.client import inngest_client, send_or_503
from docie_bench.security import TenantDependency
from docie_bench.serving.resources import DEFAULT_DEPLOY_CONTEXT_LENGTH

from . import _shared

router = APIRouter()


class DeployRequest(BaseModel):
    model: str
    name: str | None = None
    runtime: str | None = None
    # None => the control plane auto-allocates a free port at deploy time; the
    # UI sends no port unless the operator explicitly overrides it. model_dump(
    # exclude_none=True) at trigger time drops a None so the worker sees no port.
    port: int | None = None
    # The shared deploy-default context (resources.DEFAULT_DEPLOY_CONTEXT_LENGTH)
    # — the SAME constant the sizing engine prices uncalibrated fits at, so the
    # fit table and a default deploy consume the same KV budget.
    context_length: int = DEFAULT_DEPLOY_CONTEXT_LENGTH
    replicas: int = 1


@router.post("/deploy", response_model=_shared.TriggerResponse)
async def trigger_deploy(
    payload: DeployRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    channel = f"deploy:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    ids = await send_or_503(inngest_client, inngest.Event(name=_shared.DEPLOY_EVENT, data=data))
    # Deploy has no durable StudioRun row; record ownership so the triggering
    # principal can poll its status via /runs and a cross-tenant id is 404 (never
    # proxied). Every event-producing trigger records an owner for parity — an
    # unregistered one would 404 its own status polling (the Deploy.tsx fallback).
    _shared._record_event_owners(list(ids), tenant.tenant_id)
    return _shared.TriggerResponse(
        event_ids=list(ids), channel=channel, topics=_shared.DEFAULT_TOPICS
    )


class SeedHfRequest(BaseModel):
    """Seed a GGUF DIRECTLY from the Hugging Face Hub (preferred path)."""

    repo: str  # e.g. "LiquidAI/LFM2.5-350M-Instruct-GGUF"
    quant: str | None = None  # e.g. "Q4_K_M"; None = best available default
    # True (collection/batch): quant is a PREFERENCE — a repo lacking it falls
    # back to best-available instead of failing. False: quant is an explicit
    # pick and an unavailable one errors.
    quant_prefer: bool = False
    name: str | None = None  # store name; None = derived from the repo
    family: str = "openai_chat"


@router.post("/seed-hf", response_model=_shared.TriggerResponse)
async def trigger_seed_hf(
    payload: SeedHfRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    channel = f"seed:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    ids = await send_or_503(
        inngest_client, inngest.Event(name="serving/seed-hf.requested", data=data)
    )
    _shared._record_event_owners(list(ids), tenant.tenant_id)
    return _shared.TriggerResponse(
        event_ids=list(ids), channel=channel, topics=_shared.DEFAULT_TOPICS
    )


@router.get("/hf/repo")
async def hf_repo_ggufs(repo: Annotated[str, Query(min_length=3)]) -> dict[str, Any]:
    """Live GGUF listing of a Hub repo — backs the Studio's quant picker.

    Proxied server-side so the browser needs no HF credentials (HF_TOKEN on the
    api service covers gated repos) and no CORS exceptions.
    """
    from docie_bench.serving.hf_hub import HfHubError, default_store_name, list_repo_ggufs

    try:
        async with httpx.AsyncClient() as client:
            files = await list_repo_ggufs(repo, client=client)
    except HfHubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "repo": repo,
        "suggested_name": default_store_name(repo),
        "ggufs": [
            {
                "filename": f.filename,
                "size_bytes": f.size_bytes,
                "quant": f.quant,
                "is_mmproj": f.is_mmproj,
                "is_multipart": f.is_multipart,
            }
            for f in files
        ],
    }


def _node_available_bytes() -> int | None:
    """Deploy budget for a NEW model on this node: ``free - safety_margin``.

    The exact quantity the reconciler's fit-check withholds a restart against
    ("free minus the safety margin leaves N available"), so a catalog "fits this
    node" badge can't disagree with what actually gets denied at deploy. Best-
    effort + DB-optional: any gap (no snapshot, DB down) returns None → the badge
    reads "unknown", never a false "fits". The projector/loading refinements are
    a live-state concern the per-repo inspect + deploy path handle; this is a
    browse-time signal against a coarse size estimate."""
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog
    from docie_bench.serving.sizing import safety_margin_bytes
    from docie_bench.settings import get_settings

    try:
        snapshot = ModelCatalog().get_node_snapshot()
    except CatalogUnavailableError:
        return None
    except Exception:  # noqa: BLE001 - a DB hiccup must not fail the search
        return None
    if not snapshot:
        return None
    free = snapshot.get("free_bytes")
    total = snapshot.get("total_bytes")
    if not isinstance(free, int) or not isinstance(total, int):
        return None
    margin = safety_margin_bytes(total, get_settings().serving_sizing_margin_fraction)
    return max(free - margin, 0)


@router.get("/hf/search")
async def hf_search(
    query: Annotated[str, Query()] = "",
    limit: Annotated[int, Query(ge=1, le=50)] = 25,
    gguf_only: bool = True,
    sort: Annotated[str, Query(pattern="^(trending|downloads|likes|recent)$")] = "downloads",
    pipeline_tag: Annotated[str, Query()] = "",
    author: Annotated[str, Query()] = "",
) -> list[dict[str, Any]]:
    """Search the Hub for deployable models (server-side proxy).

    Enriched, catalog-oriented cards: each carries an inline PRELIMINARY support
    verdict (architecture -> family, no download). ``query`` may be empty — with
    ``sort=trending`` (or ``recent``/``likes``) that returns a discovery feed.
    ``pipeline_tag``/``author`` are Hub-side facets. ``/hf/inspect`` remains the
    authoritative per-repo check (it reads the projector the list omits)."""
    from docie_bench.serving.hf_hub import HfHubError, search_models

    try:
        async with httpx.AsyncClient() as client:
            cards = await search_models(
                query,
                client=client,
                limit=limit,
                gguf_only=gguf_only,
                sort=sort,
                pipeline_tag=pipeline_tag or None,
                author=author or None,
            )
    except HfHubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Annotate "fits this node": compare each card's coarse size estimate to the
    # node's deploy budget (free - safety margin). None on either side → unknown.
    return _shared._annotate_fits(cards, _node_available_bytes())


@router.get("/hf/inspect")
async def hf_inspect(repo: Annotated[str, Query(min_length=3)]) -> dict[str, Any]:
    """Pre-flight support verdict for a repo — detects the architecture from the
    Hub's metadata (no download) and resolves it to a family + verdict
    (supported / needs_family / unsupported). Backs a Deploy button that
    already knows whether the platform can serve the model."""
    from docie_bench.serving.hf_hub import HfHubError, inspect_repo

    try:
        async with httpx.AsyncClient() as client:
            return await inspect_repo(repo, client=client)
    except HfHubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/hf/collection")
async def hf_collection(slug: Annotated[str, Query(min_length=3)]) -> dict[str, Any]:
    """A provider-curated HF collection's model repos (seed-a-collection picker)."""
    from docie_bench.serving.hf_hub import HfHubError, list_collection

    try:
        async with httpx.AsyncClient() as client:
            return await list_collection(slug, client=client)
    except HfHubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


class SeedOllamaRequest(BaseModel):
    reference: str  # e.g. "qwen2.5:1.5b" or "hf.co/numind/NuExtract3-GGUF:Q4_K_M"
    name: str  # store entry name
    family: str = "openai_chat"
    # Optional on-disk vision projector (GGUF) for needs_mmproj families whose
    # pulled model ships no projector layer (e.g. a separately-downloaded
    # NuExtract3 mmproj). Path must be reachable inside the serving container.
    mmproj: str | None = None


@router.post("/seed-ollama", response_model=_shared.TriggerResponse)
async def trigger_seed_ollama(
    payload: SeedOllamaRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    channel = f"seed:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    ids = await send_or_503(inngest_client, inngest.Event(name=_shared.SEED_EVENT, data=data))
    # Seed has no durable StudioRun row; record ownership so the triggering
    # principal can poll its status via /runs (no 404 regression) while a
    # cross-tenant id stays 404 rather than leaking through the Inngest proxy.
    _shared._record_event_owners(list(ids), tenant.tenant_id)
    return _shared.TriggerResponse(
        event_ids=list(ids), channel=channel, topics=_shared.DEFAULT_TOPICS
    )
