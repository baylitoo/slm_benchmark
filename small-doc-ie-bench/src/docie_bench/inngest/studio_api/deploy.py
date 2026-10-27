"""Deploy trigger, HF-Hub search/inspect/seed, and Ollama seed routes."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

import httpx
import inngest
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from docie_bench.inngest.client import inngest_client, send_or_503
from docie_bench.security import TenantDependency
from docie_bench.serving.control_plane import mint_deployment_name
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
    # Default output budget inherited by Playground/Agents through the live
    # deployment profile. It does not change the runtime context window.
    max_tokens: int | None = Field(default=None, ge=1, le=131_072)
    # TARGET total instances of this store model (idempotent with the scale
    # endpoint's semantics): >1 fans out one deploy per missing replica, each
    # on its own auto-allocated port, load-balanced behind the model id.
    replicas: int = Field(default=1, ge=1, le=16)
    # Deploy to THIS node instead of wherever the lifecycle event happens to
    # land (#535). Omitted, the behaviour is exactly as before: an event on the
    # shared `docie-serving` app, delivered to whichever serving node Inngest
    # picks -- which is fine for one node and is why this is opt-in rather than
    # required. Given, the deploy is recorded as desired state owned by that
    # node and its own reconciler converges on it.
    node_id: str | None = None
    # Let the platform choose the node(s) by fit instead of naming one. With
    # replicas > 1 this spreads them over DISTINCT nodes, which is the only
    # arrangement that survives losing a machine -- a separate field from
    # node_id rather than a magic node_id value, so a node that happens to be
    # called "auto" is still addressable.
    auto_place: bool = False
    # llama-server request slots (#248/#321). 32 is an operator sanity ceiling,
    # not a llama.cpp hard limit. Meaningless for non-llamacpp runtimes; the
    # build_command for those simply never reads it.
    n_parallel: int = Field(default=1, ge=1, le=32)
    cache_reuse: int | None = Field(default=None, ge=1)
    # llama-server only (#387): overrides the GGUF's own embedded
    # chat_template with an operator-supplied Jinja file path (must be
    # reachable inside the serving container). Existence is the caller's
    # responsibility, same as `model` itself.
    chat_template_file: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    ubatch_size: int | None = Field(default=None, ge=1)
    numa: str | None = None
    cpu_threads: int | None = Field(default=None, ge=1)
    cpu_threads_batch: int | None = Field(default=None, ge=1)
    # llama-server only (#382): quantized KV cache, trading a small accuracy
    # cost for a large RAM saving on long-context deploys. Runtime.py already
    # validates these against llama-server's own allowed --cache-type-k/-v
    # values and auto-adds --flash-attn (required for a non-f16 cache type).
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    # llama-server only (#402): --reasoning-budget caps how many tokens a
    # reasoning model spends thinking before being forced to answer.
    # None/unset lets it think unbounded (or not at all, if the model has no
    # reasoning channel). -1 means unbounded-but-tracked; 0 disables thinking.
    reasoning_budget: int | None = Field(default=None, ge=-1)
    # Text injected right before the end-of-thinking tag when reasoning_budget
    # forces a cutoff. Requires reasoning_budget to be set (runtime.py enforces
    # this at validation, not here — the API layer doesn't duplicate it).
    reasoning_budget_message: str | None = None
    flash_attn: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)


async def _trigger_replicated_deploy(
    payload: DeployRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    """Fan out a ``replicas > 1`` deploy: one ordinary deploy event per missing
    replica (``<model>``, ``<model>-2``, …), sharing one progress channel.

    ``replicas`` is the TARGET total for the store model — the same idempotent
    semantics as ``POST /v1/serving/store/{name}/scale``, and the same RAM
    admission gate (N x per-instance footprint against the live sizing
    budget). Store-entry (Auto) deploys only: the explicit-runtime supervisor
    runs exactly one instance per deployment, and an explicit name/port cannot
    apply to N auto-named, auto-ported instances.
    """
    from docie_bench.inngest.serving_api import (
        admit_replica_ram,
        existing_deployment_records,
    )
    from docie_bench.serving.control_plane import replica_names_to_add

    if payload.runtime:
        raise HTTPException(
            status_code=422,
            detail=(
                "replicas > 1 requires a store-entry (Auto) deploy — an "
                "explicit-runtime deployment runs exactly one instance; deploy "
                "it several times under distinct names instead"
            ),
        )
    if payload.name or payload.port is not None:
        raise HTTPException(
            status_code=422,
            detail=(
                "replicas > 1 auto-names each instance (<model>, <model>-2, …) "
                "on auto-allocated ports — drop the explicit deployment "
                "name/port to deploy replicas"
            ),
        )
    existing = await existing_deployment_records()
    to_add = replica_names_to_add(payload.model, existing, payload.replicas)
    admit_replica_ram(
        payload.model, len(to_add), payload.context_length, n_parallel=payload.n_parallel
    )
    channel = f"deploy:{uuid.uuid4().hex}"
    event_ids: list[str] = []
    for deployment_name in to_add:
        data: dict[str, Any] = {
            "model": payload.model,
            "deployment_name": deployment_name,
            "context_length": payload.context_length,
            "channel": channel,
        }
        if payload.max_tokens is not None:
            data["max_tokens"] = payload.max_tokens
        if payload.n_parallel > 1:
            data["n_parallel"] = payload.n_parallel
        if payload.cache_reuse is not None:
            data["cache_reuse"] = payload.cache_reuse
        if payload.chat_template_file is not None:
            data["chat_template_file"] = payload.chat_template_file
        if payload.batch_size is not None:
            data["batch_size"] = payload.batch_size
        if payload.ubatch_size is not None:
            data["ubatch_size"] = payload.ubatch_size
        if payload.numa is not None:
            data["numa"] = payload.numa
        if payload.cpu_threads is not None:
            data["cpu_threads"] = payload.cpu_threads
        if payload.cpu_threads_batch is not None:
            data["cpu_threads_batch"] = payload.cpu_threads_batch
        if payload.cache_type_k is not None:
            data["cache_type_k"] = payload.cache_type_k
        if payload.cache_type_v is not None:
            data["cache_type_v"] = payload.cache_type_v
        if payload.reasoning_budget is not None:
            data["reasoning_budget"] = payload.reasoning_budget
        if payload.reasoning_budget_message is not None:
            data["reasoning_budget_message"] = payload.reasoning_budget_message
        if payload.flash_attn is not None:
            data["flash_attn"] = payload.flash_attn
        if payload.timeout_seconds is not None:
            data["timeout_seconds"] = payload.timeout_seconds
        ids = await send_or_503(
            inngest_client, inngest.Event(name=_shared.DEPLOY_EVENT, data=data)
        )
        event_ids.extend(ids)
    _shared._record_event_owners(event_ids, tenant.tenant_id)
    # Already at/above the target: empty event_ids, nothing spawned (idempotent).
    return _shared.TriggerResponse(
        event_ids=event_ids, channel=channel, topics=_shared.DEFAULT_TOPICS
    )



def _target_node_deploy(
    payload: DeployRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    """Record a deploy as desired state owned by ``payload.node_id`` (#535).

    This is the half that makes a deploy addressable. Until now a deploy was an
    event on the shared `docie-serving` app, so it landed on whichever serving
    node Inngest picked and a specific node could only be reached by running
    the CLI on it (#576). The desired row is owned by one node, and only that
    node's reconciler reads it.

    Validated BEFORE writing, because the two failure modes an operator will
    actually hit are both silent otherwise: targeting a node the catalog has
    never heard of writes a row nothing will ever read, and targeting a node
    without the model's blob produces a deploy that fails on that node for a
    reason the Studio does not show (the catalog knows the model globally --
    see #573).
    """
    from docie_bench.serving.catalog import (
        CatalogUnavailableError,
        DesiredStateConflictError,
        ModelCatalog,
    )
    from docie_bench.serving.node_liveness import stale_node_ids

    if payload.replicas > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "node_id targets ONE node, so it cannot be combined with "
                "replicas > 1 — issue one targeted deploy per node instead, "
                "which is also what makes each replica's placement distinct"
            ),
        )
    try:
        catalog = ModelCatalog()
        snapshots = list(catalog.list_node_snapshots())
    except CatalogUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "targeting a node needs the model catalog, which is not "
                "configured (DATABASE_URL); deploy without node_id to use the "
                "untargeted path"
            ),
        ) from exc

    known = {str(row.get("node_id") or ""): row for row in snapshots}
    node = known.get(payload.node_id or "")
    if node is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no serving node {payload.node_id!r} has reported; known nodes: "
                f"{sorted(name for name in known if name) or 'none'}"
            ),
        )
    if str(payload.node_id) in stale_node_ids(snapshots):
        raise HTTPException(
            status_code=409,
            detail=(
                f"serving node {payload.node_id!r} has stopped reporting, so it "
                f"would not converge on this deploy; check that node before "
                f"targeting it"
            ),
        )
    if node.get("drained_at"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"serving node {payload.node_id!r} is drained, so it is out of "
                f"the routing rotation and its reconciler stops what it is "
                f"given. Resume it first, or target a node that is admitting"
            ),
        )
    held = node.get("store_models")
    if isinstance(held, list) and payload.model not in held:
        raise HTTPException(
            status_code=409,
            detail=(
                f"serving node {payload.node_id!r} does not hold {payload.model!r}. "
                f"The catalog knows the model globally but the blob store is "
                f"node-local, so seed it there first "
                f"(`docie-serving store seed-ollama ...` on that node)"
            ),
        )

    deployment_name = payload.name or mint_deployment_name(payload.model)
    try:
        written = catalog.set_desired_deployment(
            str(payload.node_id),
            deployment_name,
            desired_state="running",
            spec=_desired_spec(payload),
        )
    except DesiredStateConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"desired state for {deployment_name!r} changed under us: {exc}",
        ) from exc
    return _shared.TriggerResponse(
        event_ids=[],
        channel=None,
        topics=[],
        detail=(
            f"recorded as desired state for node {payload.node_id!r} "
            f"(generation {written['node_desired_generation']}); that node's "
            f"reconciler converges within one cycle"
        ),
    )


def _desired_spec(payload: DeployRequest) -> dict[str, Any]:
    """The spec a desired-state row carries for a deploy.

    Deliberately thin. The owning node resolves the store entry, allocates the
    port and builds the launch command itself, so a desired row carries intent
    rather than a frozen command line -- which is also what lets a node that
    joins later converge on a row written before it existed.
    """
    spec: dict[str, Any] = {
        "model": payload.model,
        "context_length": payload.context_length,
    }
    if payload.port is not None:
        spec["port"] = payload.port
    return spec


def _auto_place_deploy(
    payload: DeployRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    """Choose the node(s) by fit and record the deploy as desired state (#536).

    The half of #536 that makes a fleet worth having. Naming a node is fine
    when there are two; by the time there are six, an operator asking "which
    one has room for this" is doing arithmetic the sizing engine already does
    per node.

    For ``replicas > 1`` the nodes are DISTINCT. Two replicas on one machine
    share its memory, its cgroup limit and its failure, so stacking them is not
    what asking for two replicas means -- and a partial placement is refused
    rather than silently delivered, because "I asked for 3 and got 2" is a
    capacity decision only the operator can make.

    Refuses with the per-node reasons rather than a bare "no node fits": which
    node failed which gate is what tells an operator whether to seed a blob,
    resume a drained node or add a machine.
    """
    from docie_bench.serving.catalog import (
        CatalogUnavailableError,
        DesiredStateConflictError,
        ModelCatalog,
    )
    from docie_bench.serving.node_liveness import stale_node_ids
    from docie_bench.serving.node_selection import select_nodes

    if payload.name and payload.replicas > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "replicas auto-name each instance so their records stay "
                "distinct across nodes, so an explicit name cannot be combined "
                "with replicas > 1"
            ),
        )
    try:
        catalog = ModelCatalog()
        models = catalog.list()
        snapshots = list(catalog.list_node_snapshots())
        placements = list(catalog.list_placements())
    except CatalogUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "choosing a node needs the model catalog, which is not "
                "configured (DATABASE_URL); deploy without auto_place to use "
                "the untargeted path"
            ),
        ) from exc

    if not any(str(row.get("node_id") or "") for row in snapshots):
        raise HTTPException(
            status_code=409,
            detail=(
                "no serving node has reported, so there is nothing to choose "
                "between; deploy without auto_place"
            ),
        )

    chosen, candidates = select_nodes(
        payload.model,
        count=payload.replicas,
        models=models,
        snapshots=snapshots,
        placements=placements,
        stale=stale_node_ids(snapshots),
        context_length=payload.context_length,
        n_parallel=payload.n_parallel,
    )
    if len(chosen) < payload.replicas:
        rejected = [
            f"{candidate.node_id}: {candidate.reason}"
            for candidate in candidates
            if not candidate.eligible
        ]
        raise HTTPException(
            status_code=409,
            detail=(
                f"{payload.replicas} replica(s) of {payload.model!r} need "
                f"{payload.replicas} distinct nodes and only {len(chosen)} can "
                f"take it. Replicas are never stacked on one node, since two on "
                f"one machine share its failure. "
                + (f"Rejected -- {'; '.join(rejected)}" if rejected else "")
            ).strip(),
        )

    placed: list[dict[str, Any]] = []
    for node_id in chosen:
        deployment_name = payload.name or mint_deployment_name(payload.model)
        try:
            written = catalog.set_desired_deployment(
                node_id,
                deployment_name,
                desired_state="running",
                spec=_desired_spec(payload),
            )
        except DesiredStateConflictError as exc:
            # A minted name collided, or a concurrent write touched this row.
            # Reported with what HAS been placed: silently rolling back would
            # mean undoing desired state another writer may already be acting
            # on, and reporting nothing would hide rows that now exist.
            raise HTTPException(
                status_code=409,
                detail=(
                    f"desired state for {deployment_name!r} on {node_id!r} "
                    f"changed under us: {exc}. Already placed: "
                    f"{placed or 'nothing'}"
                ),
            ) from exc
        placed.append(
            {
                "node_id": node_id,
                "name": deployment_name,
                "generation": written["node_desired_generation"],
            }
        )

    where = ", ".join(f"{row['name']} on {row['node_id']}" for row in placed)
    return _shared.TriggerResponse(
        event_ids=[],
        channel=None,
        topics=[],
        detail=(
            f"placed by fit as desired state: {where}. Each node's reconciler "
            f"converges within one cycle"
        ),
    )


@router.post("/deploy", response_model=_shared.TriggerResponse)
async def trigger_deploy(
    payload: DeployRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    if payload.node_id and payload.auto_place:
        raise HTTPException(
            status_code=422,
            detail=(
                "node_id names a node and auto_place asks the platform to "
                "choose one; send one or the other"
            ),
        )
    if payload.auto_place:
        return _auto_place_deploy(payload, tenant)
    if payload.node_id:
        return _target_node_deploy(payload, tenant)
    if payload.replicas > 1:
        return await _trigger_replicated_deploy(payload, tenant)
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
    # The store name becomes a directory under the serving store, the catalog
    # primary key, and the id a request names as `store:<name>`. Constrained
    # here so a name that cannot work is a 422 on the request rather than a
    # failure inside a job that has already downloaded several gigabytes --
    # the job validates it too, since events also arrive from other callers.
    name: str | None = Field(
        default=None,
        # ^...$ rather than \A...\Z: pydantic compiles this with the Rust
        # regex crate, which rejects those anchors. Both ends are still
        # absolute here -- `$` is end-of-haystack without multiline mode.
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        max_length=200,
        description="Store name; omitted = derived from the repo.",
    )
    family: str = "openai_chat"


@router.post("/seed-hf", response_model=_shared.TriggerResponse)
async def trigger_seed_hf(
    payload: SeedHfRequest, tenant: TenantDependency
) -> _shared.TriggerResponse:
    channel = f"seed:{uuid.uuid4().hex}"
    data: dict[str, Any] = payload.model_dump(exclude_none=True)
    data["channel"] = channel
    # Needed by seed_hf_job to scope its SeedRun row (see studio.seed_store) --
    # every other trigger route already does this (extract/benchmark); the
    # seed routes were the exception until the Downloads tab needed it too.
    data["tenant_id"] = tenant.tenant_id
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
    """Deploy budget for a new model after margin and in-flight reservations.

    Uses the same sizing engine as the deployment fit gate, including RAM still
    owed to models that are loading. A missing, stale, or unreachable snapshot
    returns None, so preflight says "unknown" rather than manufacturing a fit.
    """
    from docie_bench.inngest.serving_api import _gate_snapshot_staleness
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog
    from docie_bench.serving.resources import FootprintStore
    from docie_bench.serving.sizing import compute_sizing
    from docie_bench.settings import get_settings

    try:
        catalog = ModelCatalog()
        snapshot, _detail = _gate_snapshot_staleness(catalog.get_node_snapshot())
        if snapshot is None:
            return None
        report = compute_sizing(
            catalog.list(),
            snapshot,
            catalog.list_placements(),
            footprints=FootprintStore(),
            margin_fraction=get_settings().serving_sizing_margin_fraction,
        )
    except CatalogUnavailableError:
        return None
    except Exception:  # noqa: BLE001 - a DB hiccup must not fail the search
        return None
    available = report.free_effective_bytes
    return max(available, 0) if available is not None else None


@router.get("/hf/search")
async def hf_search(
    query: Annotated[str, Query()] = "",
    limit: Annotated[int, Query(ge=1, le=50)] = 25,
    gguf_only: bool = True,
    sort: Annotated[str, Query(pattern="^(trending|downloads|likes|recent)$")] = "downloads",
    pipeline_tag: Annotated[str, Query()] = "",
    author: Annotated[str, Query()] = "",
    num_parameters: Annotated[
        str,
        Query(
            pattern=(
                r"^(?:|min:\d+(?:\.\d+)?[KMBT](?:,max:\d+(?:\.\d+)?[KMBT])?"
                r"|max:\d+(?:\.\d+)?[KMBT])$"
            )
        ),
    ] = "",
) -> list[dict[str, Any]]:
    """Search the Hub for deployable models (server-side proxy).

    Enriched, catalog-oriented cards: each carries an inline PRELIMINARY support
    verdict (architecture -> family, no download). ``query`` may be empty — with
    ``sort=trending`` (or ``recent``/``likes``) that returns a discovery feed.
    ``pipeline_tag``/``author`` are Hub-side facets. ``num_parameters`` accepts
    the Hub's native range syntax (for example ``min:1B,max:3B``).
    ``/hf/inspect`` remains the authoritative per-repo check (it reads the
    projector the list omits)."""
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
                num_parameters=num_parameters or None,
            )
    except HfHubError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Annotate "fits this node": compare each card's coarse size estimate to the
    # node's deploy budget (free - safety margin). None on either side → unknown.
    return _shared._annotate_fits(cards, _node_available_bytes())


@router.get("/hf/inspect")
async def hf_inspect(
    repo: Annotated[str, Query(min_length=3)],
    context_length: Annotated[int, Query(ge=128, le=1_048_576)] = DEFAULT_DEPLOY_CONTEXT_LENGTH,
) -> dict[str, Any]:
    """Pre-flight support verdict for a repo — detects the architecture from the
    Hub's metadata (no download) and resolves it to a family + verdict
    (supported / needs_family / unsupported). Backs a Deploy button that
    already knows whether the platform can serve the model."""
    from docie_bench.serving.hf_hub import HfHubError, inspect_repo

    try:
        async with httpx.AsyncClient() as client:
            return await inspect_repo(
                repo,
                client=client,
                context_length=context_length,
                node_available_bytes=_node_available_bytes(),
            )
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
    # Needed by seed_ollama_job to scope its SeedRun row (see studio.seed_store).
    data["tenant_id"] = tenant.tenant_id
    ids = await send_or_503(inngest_client, inngest.Event(name=_shared.SEED_EVENT, data=data))
    # Seed has no StudioRun row (that table is benchmark-specific); record
    # ownership so the triggering principal can poll its status via /runs (no
    # 404 regression) while a cross-tenant id stays 404 rather than leaking
    # through the Inngest proxy.
    _shared._record_event_owners(list(ids), tenant.tenant_id)
    return _shared.TriggerResponse(
        event_ids=list(ids), channel=channel, topics=_shared.DEFAULT_TOPICS
    )
