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

import datetime as dt
import json
import os
import uuid
from collections.abc import Mapping
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
LOAD_EVENT = "serving/load.requested"
UNLOAD_EVENT = "serving/unload.requested"
PIN_EVENT = "serving/pin.requested"

# Snapshot-staleness gate: a published node snapshot is only trusted while it
# is at most this many reconcile intervals old. If the serving reconciler dies,
# its last snapshot must NOT keep backing "observed_available: true" sizing
# forever — /resources promises "never a stale number", and that promise has to
# cover the reconciler-died case, not just the never-published one.
SNAPSHOT_STALE_INTERVALS = 3.0
# Floor so a very short dev interval (e.g. 1s) does not flap the gate on one
# slow DB round-trip.
SNAPSHOT_STALE_FLOOR_S = 30.0


def _reconcile_interval_s() -> float:
    """The reconciler's cycle interval — same env knob worker.py reads."""
    try:
        interval = float(os.getenv("DOCIE_SERVING_RECONCILE_INTERVAL", "10"))
    except ValueError:
        return 10.0
    return interval if interval > 0 else 10.0


def snapshot_stale_after_s() -> float:
    return max(SNAPSHOT_STALE_INTERVALS * _reconcile_interval_s(), SNAPSHOT_STALE_FLOOR_S)


def _snapshot_age_s(snapshot: Mapping[str, Any], *, now: dt.datetime | None = None) -> float | None:
    """Seconds since the snapshot's ``updated_at`` (None when unparseable)."""
    raw = snapshot.get("updated_at")
    if not raw:
        return None
    try:
        stamp = dt.datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        # sqlite round-trips the reconciler's UTC stamp as a naive datetime.
        stamp = stamp.replace(tzinfo=dt.UTC)
    current = now if now is not None else dt.datetime.now(dt.UTC)
    return (current - stamp).total_seconds()


def _gate_snapshot_staleness(
    snapshot: dict[str, Any] | None, *, now: dt.datetime | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """(snapshot or None, staleness detail): drop a snapshot too old to trust.

    A snapshot older than ``snapshot_stale_after_s()`` (default 3x the
    reconcile interval, floored at 30s) means the serving reconciler stopped
    publishing — the number describes a dead past, so it degrades to the SAME
    honest "observed unavailable" state as never-published, with a detail
    saying how old the last measurement is. An unparseable/missing
    ``updated_at`` fails open (treated as fresh): the stamp is always written
    by ``publish_node_snapshot``, and refusing to serve over a formatting
    quirk would be a false outage.
    """
    if snapshot is None:
        return None, None
    age = _snapshot_age_s(snapshot, now=now)
    threshold = snapshot_stale_after_s()
    if age is not None and age > threshold:
        return None, (
            f"node snapshot is stale: last published {age:.0f}s ago "
            f"(> {threshold:.0f}s = {SNAPSHOT_STALE_INTERVALS:g}x the reconcile "
            f"interval) — is the serving service's reconciler still running?"
        )
    return snapshot, None


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
    when the database is unreachable, the reconciler has never published a
    snapshot, OR the last snapshot is older than the staleness gate allows
    (the reconciler died and its final number describes a dead past) — never
    a stale or locally-measured number. Auth matches the sibling serving
    reads (unauthenticated ops view; mutations stay evented).
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
        else:
            node, detail = _gate_snapshot_staleness(node)
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
    database degrades to empty inputs + a reason; a missing OR stale snapshot
    (staleness gate above) keeps the store list (footprints still price) and
    lets the engine mark fits unknown — never a fit computed against a dead
    reconciler's last number.
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
    if snapshot is None:
        return (
            models,
            None,
            placements,
            "no node snapshot published yet — is the serving service's reconciler running?",
        )
    snapshot, detail = _gate_snapshot_staleness(snapshot)
    return models, snapshot, placements, detail


@router.get("/sizing")
async def serving_sizing() -> dict[str, Any]:
    """Per-model fit table: how many MORE instances fit right now (PR-3, §3).

    Pure read over the observed surface: footprint per candidate instance is
    the PR-2 tracker's ``max(calibrated steady RSS, predicted)`` (mmproj-aware,
    calibration sidecars on the shared serving volume, KV priced at the deploy
    default context), free RAM is the reconciler-published node snapshot
    (hot deployments' RSS is already inside "used" — the engine never
    subtracts them again; loading deployments reserve only their not-yet-
    resident remainder; see the double-count guard in ``serving.sizing``),
    and the safety margin is the explicit, configurable
    ``serving_sizing_margin_fraction`` slice of total — the same margin the
    deploy path's restart fit gate holds back (the gate re-measures free RAM
    live at decision time; this table prices against the snapshot that same
    reader published last cycle).

    Honest degradation mirrors ``/resources``: ``observed_available: false`` +
    a ``detail`` reason when the database is down, the snapshot was never
    published, or the snapshot is stale (reconciler died) — footprints still
    price, ``fits_now`` stays null. Auth parity with the sibling serving
    reads (unauthenticated ops view).
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

    Same engine, same footprint math, same margin and same loading-placement
    reservation as ``/sizing`` — the two surfaces can never disagree (and the
    policy is the deploy path's fit gate's, priced against the last published
    snapshot; see ``serving.sizing``). A pure computation (nothing deploys,
    nothing mutates), so auth parity is with the sibling reads, not the
    evented mutations. 422 for a model not in the store or a staged model
    that cannot be priced — NEVER 404: the frontend's global convention maps
    404 to "endpoint doesn't exist yet" (``api.ts isUnavailableStatus``), and
    a store-removal racing the UI poll must surface the server's detail, not
    a bogus "endpoint unavailable". With no node snapshot the plan still
    prices (``total_predicted_bytes``) but ``ok`` / ``remaining_bytes`` stay
    null — never a verdict against a made-up number.
    """
    from docie_bench.serving.resources import FootprintStore
    from docie_bench.serving.sizing import (
        UnknownModelError,
        UnpriceableModelError,
        compute_whatif,
    )

    models, snapshot, placements, detail = _sizing_inputs()
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
            placements,
            footprints=FootprintStore(),
            margin_fraction=get_settings().serving_sizing_margin_fraction,
        )
    except (UnknownModelError, UnpriceableModelError) as exc:
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


async def _fire_lifecycle_event(
    name: str,
    *,
    event: str,
    prefix: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """404-gate ``name`` then fire one ``serving/*`` lifecycle event.

    Shared shape for delete/load/unload/pin: the api can neither spawn nor
    kill a runtime (different PID namespace, and only the serving service may
    write ``deployments.json``), so every mutation is an event handled on the
    single-replica ``serving`` service. Returns the event id(s) + channel to
    poll; 404 for an unknown deployment so a typo never queues a no-op job.
    """
    try:
        await _control_plane().deployment_status(name)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    channel = f"{prefix}:{uuid.uuid4().hex}"
    data: dict[str, Any] = {"name": name, "channel": channel, **(extra or {})}
    ids = await inngest_client.send(inngest.Event(name=event, data=data))
    return {"event_ids": list(ids), "channel": channel, "name": name}


@router.delete("/deployments/{name}")
async def delete_deployment(name: str, tenant: TenantDependency) -> dict[str, Any]:
    """Fire the real-teardown event (PR-1): a delete that actually deletes.

    Handled on ``serving`` (the only process holding the Popen handle), which
    kills the process, drops the record (freeing its port), and DELETEs the
    placement row — the ONLY path that deletes a row.
    """
    del tenant  # authenticated principal required; no per-tenant scoping (ops surface)
    return await _fire_lifecycle_event(name, event=DELETE_EVENT, prefix="delete")


@router.post("/deployments/{name}/load")
async def load_deployment(name: str, tenant: TenantDependency) -> dict[str, Any]:
    """Cold-start a deployment (PR-4): fire ``serving/load.requested``.

    The serving-side handler is idempotent (per-deployment load lock — an
    already-hot deployment is a no-op) and may evict LRU unpinned victims
    when that makes the load fit. Works on manual-cold deployments too: the
    Load button IS the explicit Start.
    """
    del tenant
    return await _fire_lifecycle_event(name, event=LOAD_EVENT, prefix="load")


@router.post("/deployments/{name}/unload")
async def unload_deployment(name: str, tenant: TenantDependency) -> dict[str, Any]:
    """Evict a deployment (PR-4): fire ``serving/unload.requested``.

    DISTINCT from stop/delete (design fix #3): the record, its port
    reservation and its placement row all SURVIVE — the row is UPDATEd to
    ``phase=evicted`` / ``activation=managed``, so the next request to it
    auto-reloads instead of failing.
    """
    del tenant
    return await _fire_lifecycle_event(name, event=UNLOAD_EVENT, prefix="unload")


class PinRequest(BaseModel):
    """Body of POST /deployments/{name}/pin."""

    pinned: bool = True


@router.post("/deployments/{name}/pin")
async def pin_deployment(
    name: str, request: PinRequest, tenant: TenantDependency
) -> dict[str, Any]:
    """Set/clear a deployment's eviction shield (PR-4): never evicted while
    pinned. An event (not an in-place write) because ``pinned`` lives in
    ``deployments.json`` and only the serving service writes that file."""
    del tenant
    return await _fire_lifecycle_event(
        name, event=PIN_EVENT, prefix="pin", extra={"pinned": request.pinned}
    )


@router.get("/mesh")
async def mesh_status() -> dict[str, Any]:
    """Live mesh-llm status: configured endpoint, reachability, served models.

    Backs the Studio's mesh surfacing (Agents backing-model datalist and any
    future Deploy card). Never 500s on an unreachable mesh — that is a normal
    ``healthy: false`` answer with a ``detail``.
    """
    from docie_bench.serving.mesh import mesh_view

    return await mesh_view()


@router.get("/store")
async def list_store() -> Any:
    """The local GGUF model store (queryable Postgres catalog the Studio reads).

    Each entry includes its family and the backends that can serve it faithfully.
    ``model_path``/``mmproj_path`` (container filesystem paths) are STRIPPED
    here: they are server-side sizing inputs (calibration key + projector
    pricing), and this is an unauthenticated surface — the browser never needs
    them, so it never sees them.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        entries = ModelCatalog().list()
    except CatalogUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    for entry in entries:
        entry.pop("model_path", None)
        entry.pop("mmproj_path", None)
    return entries


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
