"""Serving endpoints for the DocIE Studio Deploy tab.

Thin wrappers over ``ControlPlane`` (the same facade the ``docie`` CLI drives),
reading the shared serving home (``DOCIE_SERVING_HOME``, a named volume mounted
on ``api``, ``serving`` and ``worker``). Quick reads live here as plain HTTP;
mutations are Inngest events handled by the single-replica ``serving`` service
(*deploy* via ``studio_api.py``; *delete* via ``DELETE /deployments/{name}``
below, which fires ``serving/delete.requested``).

Liveness (PR-1): the api process still cannot see the serving container's PID
namespace, but it no longer needs to — ``/deployments`` overlays the OBSERVED
state (phase/rss/health) the in-``serving`` reconciler publishes to Postgres
every cycle, degrading to the (reconciler-refreshed) ``deployments.json`` view
when the database is down.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import inngest
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docie_bench.inngest.client import inngest_client
from docie_bench.security import TenantDependency
from docie_bench.serving.control_plane import ControlPlane
from docie_bench.settings import get_settings

router = APIRouter(prefix="/v1/serving", tags=["serving"])

DELETE_EVENT = "serving/delete.requested"


def _control_plane() -> ControlPlane:
    # NOT cached: deployment/registry state is owned and written by the *worker*
    # (deploy jobs); this API is a read-only viewer over the shared on-disk state
    # (DOCIE_SERVING_HOME). A cached ControlPlane holds a PersistentSupervisor that
    # loads deployments.json once at construction and never reloads, so the Deploy
    # tab would show a stale snapshot from the API's first read until it restarts.
    # from_defaults() only reads state (no _save), so rebuilding per request is a
    # cheap, always-fresh view.
    return ControlPlane.from_defaults()


@router.get("/models")
async def list_models() -> Any:
    return await _control_plane().list_models()


@router.get("/runtimes")
async def list_runtimes() -> Any:
    return await _control_plane().list_runtimes()


def _observed_placements() -> dict[str, dict[str, Any]] | None:
    """The reconciler-published observed rows, keyed by name (None = DB down).

    Best-effort: with no DATABASE_URL (or a DB hiccup) the Board degrades to
    the fresh-but-lean ``deployments.json`` view — which the reconciler also
    keeps de-staled via its per-cycle ``_save()`` — so Postgres is NOT required
    to kill liveness staleness (design doc fix #8), only for RSS/phase.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        return {row["name"]: row for row in ModelCatalog().list_placements()}
    except CatalogUnavailableError:
        return None
    except Exception:  # noqa: BLE001 - a DB hiccup must not 500 the Board
        return None


@router.get("/deployments")
async def list_deployments() -> Any:
    """Deployment records overlaid with the reconciler's OBSERVED state (PR-1).

    Each record gains an ``observed`` object (phase / pid / rss_bytes /
    health_ok / last_probe_at / last_error / endpoint, from the Postgres
    surface the reconciler UPDATEs every cycle) — ``None`` per record when the
    reconciler has not published it, and ``observed_available: false`` on all
    records when the database is unreachable (worker-local desired state only).
    """
    records = await _control_plane().list_deployments()
    observed = _observed_placements()
    if not isinstance(records, list):
        return records
    for record in records:
        if not isinstance(record, dict):
            continue
        spec = record.get("spec") or {}
        name = spec.get("name")
        record["observed_available"] = observed is not None
        record["observed"] = observed.get(name) if observed and name else None
    return records


@router.get("/resources")
async def serving_resources() -> dict[str, Any]:
    """Node RAM snapshot + per-deployment RSS (PR-2, read-only observed surface).

    Serves the single ``serving_node`` row the in-``serving`` reconciler
    publishes every cycle — measured inside the serving container
    (cgroup-v2-first; ``source: "cgroup" | "vm"`` flags a soft VM fallback so
    the UI can badge it). The api NEVER measures here: a psutil call in this
    process would describe the api container's cgroup, not the serving node's
    (design doc §2).

    Honest degradation: ``observed_available: false`` + a ``detail`` reason
    when the database is unreachable OR the reconciler has never published a
    snapshot — never a stale or locally-measured number. Auth matches the
    sibling serving reads (unauthenticated ops view; mutations stay evented).
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    node: dict[str, Any] | None = None
    deployments: list[dict[str, Any]] = []
    detail: str | None = None
    try:
        catalog = ModelCatalog()
        node = catalog.get_node_snapshot()
        if node is None:
            detail = (
                "no node snapshot published yet — is the serving service's "
                "reconciler running?"
            )
        deployments = [
            {
                "name": placement["name"],
                "rss_bytes": placement["rss_bytes"],
                "phase": placement["phase"],
            }
            for placement in catalog.list_placements()
        ]
    except CatalogUnavailableError:
        detail = "observed state unavailable: DATABASE_URL is not configured"
    except Exception:  # noqa: BLE001 - a DB hiccup must not 500 the Board
        detail = "observed state unavailable: database error"
    return {
        "observed_available": node is not None,
        "source": node["source"] if node is not None else None,
        "node": node,
        "deployments": deployments,
        "detail": detail,
    }


class WhatIfPlanItem(BaseModel):
    """One staged line of a hypothetical deployment mix."""

    model: str
    instances: int = Field(default=1, ge=1, le=1000)
    context_length: int | None = Field(default=None, ge=1, le=1_048_576)


class WhatIfRequest(BaseModel):
    plan: list[WhatIfPlanItem] = Field(min_length=1, max_length=100)


def _sizing_inputs() -> tuple[
    list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]], str | None
]:
    """(models, snapshot, placements, degradation detail) for the sizing engine.

    All three inputs come from the observed Postgres surface the reconciler
    publishes (design doc §3) — the api never measures locally. A missing
    database degrades to empty inputs + a reason; a missing snapshot keeps the
    store list (footprints still price) and lets the engine mark fits unknown.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        catalog = ModelCatalog()
        models = catalog.list()
        snapshot = catalog.get_node_snapshot()
        placements = list(catalog.list_placements())
    except CatalogUnavailableError:
        return [], None, [], "observed state unavailable: DATABASE_URL is not configured"
    except Exception:  # noqa: BLE001 - a DB hiccup must not 500 the Sizing tab
        return [], None, [], "observed state unavailable: database error"
    detail = None
    if snapshot is None:
        detail = (
            "no node snapshot published yet — is the serving service's reconciler running?"
        )
    return models, snapshot, placements, detail


@router.get("/sizing")
async def serving_sizing() -> dict[str, Any]:
    """Per-model fit table: how many MORE instances fit right now (PR-3, §3).

    Pure read over the observed surface: footprint per candidate instance is
    the PR-2 tracker's ``max(calibrated steady RSS, predicted)`` (mmproj-aware,
    calibration sidecars on the shared serving volume), free RAM is the
    reconciler-published node snapshot VERBATIM (running deployments' RSS is
    already inside "used" — the engine never subtracts them again; see the
    double-count guard in ``serving.sizing``), and the safety margin is the
    explicit, configurable ``serving_sizing_margin_fraction`` slice of total.

    Honest degradation mirrors ``/resources``: ``observed_available: false`` +
    a ``detail`` reason when the database is down or the snapshot was never
    published — footprints still price, ``fits_now`` stays null. Auth parity
    with the sibling serving reads (unauthenticated ops view).
    """
    from docie_bench.serving.resources import FootprintStore
    from docie_bench.serving.sizing import compute_sizing

    models, snapshot, placements, detail = _sizing_inputs()
    report = compute_sizing(
        models,
        snapshot,
        placements,
        footprints=FootprintStore(),
        margin_fraction=get_settings().serving_sizing_margin_fraction,
    )
    payload = report.as_dict()
    payload["node"] = snapshot  # full snapshot view: capacity bar input
    if detail is not None:
        payload["detail"] = detail
    return payload


@router.post("/sizing/whatif")
async def serving_sizing_whatif(request: WhatIfRequest) -> dict[str, Any]:
    """Price a hypothetical deployment mix → fits or an explicit deficit (§3).

    Same engine and same footprint math as ``/sizing`` — the what-if readout
    matches what the deploy path's fit gate would decide, by construction. A
    pure computation (nothing deploys, nothing mutates), so auth parity is
    with the sibling reads, not the evented mutations. 404 for a model not in
    the store; 422 when a staged model cannot be priced at all. With no node
    snapshot the plan still prices (``total_predicted_bytes``) but ``ok`` /
    ``remaining_bytes`` stay null — never a verdict against a made-up number.
    """
    from docie_bench.serving.resources import FootprintStore
    from docie_bench.serving.sizing import (
        UnknownModelError,
        UnpriceableModelError,
        compute_whatif,
    )

    models, snapshot, placements, detail = _sizing_inputs()
    del placements  # the plan prices prospective instances only
    if not models:
        # No store to resolve plan models against: the DB-down degrade path.
        raise HTTPException(
            status_code=503,
            detail=detail or "model store unavailable: cannot resolve plan models",
        )
    try:
        report = compute_whatif(
            models,
            snapshot,
            [item.model_dump() for item in request.plan],
            footprints=FootprintStore(),
            margin_fraction=get_settings().serving_sizing_margin_fraction,
        )
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnpriceableModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload = report.as_dict()
    if detail is not None:
        payload["detail"] = detail
    return payload


@router.get("/ports")
async def serving_ports() -> dict[str, Any]:
    """Record-derived view of the serving port window for the Deploy admin table.

    Approximate by design: like the rest of this module it reads the shared
    on-disk deployment state from the *api* netns and CANNOT socket-probe the
    worker's binds, so used/free/recommended are derived purely from the records.
    ``recommended_next`` is an explicit HINT computed by the SAME
    ``PortAllocator.recommend`` the worker uses, so the UI and the worker agree in
    logic; the worker re-derives and socket-probes authoritatively at deploy time
    and may legitimately pick a different port. Never a reservation.
    """
    from docie_bench.serving.control_plane import PortAllocator

    settings = get_settings()
    start = settings.serving_port_range_start
    end = settings.serving_port_range_end
    bind_host = settings.serving_bind_host

    records = await _control_plane().list_deployments()
    deployments: list[dict[str, Any]] = []
    used: set[int] = set()
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            spec = record.get("spec") or {}
            launch = spec.get("launch") or {}
            port = launch.get("port")
            if not isinstance(port, int):
                continue
            deployments.append(
                {
                    "name": spec.get("name"),
                    "port": port,
                    "state": record.get("state"),
                }
            )
            used.add(port)

    allocator = PortAllocator(range_start=start, range_end=end)
    try:
        recommended_next: int | None = allocator.recommend(bind_host=bind_host, reserved=used)
    except RuntimeError:
        recommended_next = None  # range exhausted -> no hint, not a 500

    free_sample = [port for port in range(start, end + 1) if port not in used][:10]

    return {
        "range": {"start": start, "end": end},
        "deployments": sorted(deployments, key=lambda item: item["port"]),
        "used": sorted(used),
        "free_sample": free_sample,
        "recommended_next": recommended_next,
    }


@router.get("/deployments/{name}")
async def deployment_status(name: str) -> Any:
    try:
        return await _control_plane().deployment_status(name)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/deployments/{name}")
async def delete_deployment(name: str, tenant: TenantDependency) -> dict[str, Any]:
    """Fire the real-teardown event (PR-1): a delete that actually deletes.

    The api cannot kill the runtime itself (different PID namespace), so this
    fires ``serving/delete.requested`` at the single-replica ``serving``
    service — the only process holding the Popen handle — which kills the
    process, drops the record (freeing its port), and DELETEs the placement
    row. Returns the event id(s) to poll; 404 for an unknown deployment so a
    typo does not queue a no-op job.
    """
    del tenant  # authenticated principal required; no per-tenant scoping (ops surface)
    try:
        await _control_plane().deployment_status(name)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    channel = f"delete:{uuid.uuid4().hex}"
    ids = await inngest_client.send(
        inngest.Event(name=DELETE_EVENT, data={"name": name, "channel": channel})
    )
    return {"event_ids": list(ids), "channel": channel, "name": name}


@router.get("/store")
async def list_store() -> Any:
    """The local GGUF model store (queryable Postgres catalog the Studio reads).

    Each entry includes its family and the backends that can serve it faithfully.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        return ModelCatalog().list()
    except CatalogUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.get("/families")
async def list_families() -> Any:
    """Known model families and how each must be served."""
    from docie_bench.serving.model_store import FAMILIES

    return [
        {
            "name": fam.name,
            "vision": fam.vision,
            "needs_mmproj": fam.needs_mmproj,
            "ollama_faithful": fam.ollama_faithful,
            "template_delivery": str(fam.template_delivery),
        }
        for fam in FAMILIES.values()
    ]


@router.get("/benchmarks")
async def list_benchmarks() -> list[dict[str, Any]]:
    """List completed benchmark runs (no ControlPlane method — read runs_dir)."""
    runs_dir = get_settings().runs_dir
    results: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return results
    for entry in sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not entry.is_dir():
            continue
        record: dict[str, Any] = {"run": entry.name, "path": str(entry)}
        metrics_path = entry / "metrics.json"
        if metrics_path.exists():
            try:
                record["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record["metrics"] = None
        results.append(record)
    return results


__all__ = ["router"]
