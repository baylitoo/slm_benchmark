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
import logging
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import inngest
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docie_bench.inngest.client import inngest_client, send_or_503
from docie_bench.security import TenantDependency
from docie_bench.serving.control_plane import ControlPlane
from docie_bench.serving.failure import classify_failure
from docie_bench.settings import get_settings

logger = logging.getLogger("docie_bench.inngest.serving_api")

router = APIRouter(prefix="/v1/serving", tags=["serving"])

# Domain "not found" marker. The Studio treats a BARE 404/501 as "endpoint not
# built on this backend" and swallows the detail; this header tells it the
# route exists and the detail is a real, user-relevant answer ("deployment 'x'
# not found"), so typos and races stop rendering as "endpoint unavailable".
_DOMAIN_404 = {"X-Docie-Error": "not_found"}

DELETE_EVENT = "serving/delete.requested"
LOAD_EVENT = "serving/load.requested"
UNLOAD_EVENT = "serving/unload.requested"
REPAIR_EVENT = "serving/repair.requested"
RECONFIGURE_EVENT = "serving/reconfigure.requested"
RESIZE_EVENT = "serving/resize.requested"
PIN_EVENT = "serving/pin.requested"
# Scaling reuses the ordinary single-deploy job (deploy_model_job) — the scale
# endpoint just fans out one deploy event per new replica name.
DEPLOY_EVENT = "serving/deploy.requested"

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
            f"capacity measurement is stale: last published {age:.0f}s ago "
            f"(threshold {threshold:.0f}s) — is the serving service still running?"
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
    """Deployment records with their live observed state.

    Each record carries an ``observed`` object (phase / pid / rss_bytes /
    health_ok / last_probe_at / last_error / endpoint), refreshed every
    cycle — ``None`` per record when no observation has been published yet,
    and ``observed_available: false`` on all records when the observed state
    is unreachable (desired state only).
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
        obs = observed.get(name) if observed and name else None
        record["observed"] = obs
        # Derived, not stored (no migration): the published last_error already
        # carries the killing signal (OOM) and the fit-check reason. The
        # observed row's last_error is the superset (the reconciler appends the
        # withheld-restart reason to it), so prefer it over the raw record's.
        last_error = (obs or {}).get("last_error") or record.get("last_error")
        record["failure_kind"] = classify_failure(record.get("state"), last_error)
    return records


# The api NEVER measures RAM itself: a psutil call in this process would
# describe the api container's cgroup, not the serving node's. Numbers come
# from the ``serving_node`` row the in-``serving`` reconciler publishes every
# cycle. Auth: the whole serving router is mounted behind tenant_guard
# (api.py); reads and mutations alike require a key.
@router.get("/resources")
async def serving_resources() -> dict[str, Any]:
    """Node RAM snapshot + per-deployment memory usage (read-only).

    Serves the last published node measurement, taken inside the serving
    container (cgroup-v2 first; ``source: "cgroup" | "vm"`` flags a soft VM
    fallback). Degrades honestly: ``observed_available: false`` plus a
    ``detail`` reason when the measurement is missing, stale, or the
    database is unreachable — never a stale or locally-measured number.
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
                "no capacity measurement published yet — is the serving "
                "service running?"
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
            "no capacity measurement published yet — is the serving service running?",
        )
    snapshot, detail = _gate_snapshot_staleness(snapshot)
    return models, snapshot, placements, detail


@router.get("/sizing")
async def serving_sizing() -> dict[str, Any]:
    """Per-model fit table: how many MORE instances fit right now.

    Footprint per candidate instance is ``max(measured steady-state RSS,
    predicted)`` (mmproj-aware, KV priced at the deploy default context);
    free RAM is the last published node snapshot (running deployments'
    memory is already inside "used"; loading deployments reserve only the
    part not yet in memory); the safety margin is the configurable
    ``serving_sizing_margin_fraction`` slice of total — the same margin
    enforced at deploy time.

    Degrades honestly, mirroring ``/resources``: ``observed_available:
    false`` plus a ``detail`` reason when the database is down or the
    snapshot is missing or stale — footprints still price, ``fits_now``
    stays null.
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


@router.get("/activity")
async def serving_activity() -> dict[str, Any]:
    """Per-store-model request activity: a crude "how hot is this model
    right now" signal (``model_activity`` — see ``catalog.ModelActivity``).

    Purely observational today — nothing scales on this yet. It exists so
    an operator can see load next to the Sizing tab's fit numbers before
    anyone builds a decision on top of it. ``window_count`` resets whenever
    something reads-then-zeros the window (nothing does yet — a fresh
    process never sees fewer requests than actually happened, it just
    hasn't been zeroed), so treat it as "requests since window_started_at",
    not a live rate.

    Degrades honestly like ``/sizing``: an empty list plus a ``detail``
    reason when the database is down, never a 500.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        entries = ModelCatalog().list_activity()
    except CatalogUnavailableError:
        return {"entries": [], "detail": "activity unavailable: DATABASE_URL is not configured"}
    except Exception:  # noqa: BLE001 - a DB hiccup must not 500 this tile
        return {"entries": [], "detail": "activity unavailable: database error"}
    return {"entries": entries}


# Errors here are 422, NEVER 404: the Studio treats 404/501 as "endpoint not
# available" (api.ts isUnavailableStatus), so a store-removal racing the UI
# poll must surface the server's detail, not a bogus "endpoint unavailable".
@router.post("/sizing/whatif")
async def serving_sizing_whatif(request: WhatIfRequest) -> dict[str, Any]:
    """Price a hypothetical deployment mix → fits or an explicit deficit.

    Same engine, footprint math, margin and loading-placement reservation as
    ``/sizing``, so the two can never disagree. A pure computation — nothing
    deploys, nothing mutates. 422 for a model not in the store or a staged
    model that cannot be priced. With no node snapshot the plan still prices
    (``total_predicted_bytes``) but ``ok`` / ``remaining_bytes`` stay null.
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


@router.get("/ocr-cache")
async def ocr_cache_stats() -> dict[str, Any]:
    """OCR cache utilization: entry count, size on disk, oldest/newest entry age.

    OCRCache.get()/put() record a per-call cache_hit flag, but it was never
    aggregated or exposed anywhere -- an operator has no way to tell if the
    cache is helping, or how close it is to its ocr_cache_max_mb budget.
    Scans the shared cache directory directly (the same technique
    OCRCache.evict() already uses internally: root.glob("*.json") + stat), so
    this is accurate regardless of which api/worker replica served a given
    request -- unlike a hit/miss RATE counter, size-on-disk has no
    process-locality problem (every replica reads the same
    DOCIE_OCR_CACHE_DIR volume). A true hit-rate metric needs aggregation
    across replicas -- the same class of gap as an autoscale load signal --
    and is deliberately left for later rather than reported dishonestly here.
    """
    settings = get_settings()
    if not settings.ocr_cache_enabled:
        return {"enabled": False}

    root = Path(settings.ocr_cache_dir)
    entry_count = 0
    total_bytes = 0
    oldest_mtime: float | None = None
    newest_mtime: float | None = None
    if root.exists():
        for path in root.glob("*.json"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue  # evicted between glob() and stat() -- benign race
            entry_count += 1
            total_bytes += stat.st_size
            if oldest_mtime is None or stat.st_mtime < oldest_mtime:
                oldest_mtime = stat.st_mtime
            if newest_mtime is None or stat.st_mtime > newest_mtime:
                newest_mtime = stat.st_mtime

    max_bytes = settings.ocr_cache_max_mb * 1024 * 1024
    now = time.time()
    return {
        "enabled": True,
        "entry_count": entry_count,
        "total_bytes": total_bytes,
        "max_bytes": max_bytes,
        "utilization_pct": round(100 * total_bytes / max_bytes, 1) if max_bytes else None,
        "oldest_entry_age_seconds": round(now - oldest_mtime, 1) if oldest_mtime else None,
        "newest_entry_age_seconds": round(now - newest_mtime, 1) if newest_mtime else None,
    }


@router.get("/deployments/{name}")
async def deployment_status(name: str) -> Any:
    try:
        return await _control_plane().deployment_status(name)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc), headers=_DOMAIN_404) from exc


def _serving_home() -> Path:
    """The shared serving home (``DOCIE_SERVING_HOME``) on the serving-state
    volume that api, serving and worker all mount."""
    return Path(
        os.environ.get(
            "DOCIE_SERVING_HOME",
            Path.home() / ".local" / "share" / "docie-bench" / "serving",
        )
    )


def _serving_logs_dir() -> Path:
    """``<serving_home>/logs`` — the runtime stdout files the supervisor writes,
    readable by the api (it never spawned the process)."""
    return _serving_home() / "logs"


@router.get("/deployments/{name}/logs")
async def deployment_logs(name: str, lines: int = 200) -> dict[str, Any]:
    """Tail a deployment's runtime log (its stdout/stderr) for the Studio.

    Reads ``<serving_home>/logs/<name>.log`` on the shared volume — the same
    file the supervisor captures ``last_error`` from — so the operator can see
    WHY a deployment failed (bad endpoint, missing binary, OOM) without shell
    access. ``last_error`` is the one-line failure summary; ``lines`` is
    the raw tail (capped). Never 500s on a missing/rotated log — an absent file
    is an empty tail, not an error.
    """
    max_lines = max(1, min(int(lines), 1000))
    # Path containment: the name is joined into a filesystem path.
    logs_dir = _serving_logs_dir().resolve()
    log_path = (logs_dir / f"{name}.log").resolve()
    if logs_dir not in log_path.parents:
        raise HTTPException(status_code=400, detail="invalid deployment name")

    last_error: str | None = None
    try:
        record = await _control_plane().deployment_status(name)
        if isinstance(record, dict):
            last_error = record.get("last_error") or (
                (record.get("observed") or {}).get("last_error")
                if isinstance(record.get("observed"), dict)
                else None
            )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc), headers=_DOMAIN_404) from exc

    tail: list[str] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            tail = handle.read().splitlines()[-max_lines:]
    except FileNotFoundError:
        tail = []
    except OSError as exc:  # pragma: no cover - unreadable log must not 500
        tail = [f"(log unreadable: {exc})"]
    return {"name": name, "last_error": last_error, "lines": tail}


@router.get("/seed-progress")
async def seed_progress(channel: str) -> dict[str, Any]:
    """The latest download progress for a seed run's ``channel`` (realtime-free).

    The seed job persists its percentage to a sidecar on the shared volume; the
    Studio's polling fallback reads it here to render the same bar realtime would.
    ``progress`` is null when nothing has been written yet (or the download is
    done — the sidecar is cleared on settle)."""
    from docie_bench.serving.seed_progress import read_progress

    return {"channel": channel, "progress": read_progress(channel)}


async def trigger_deployment_load(name: str) -> tuple[str, float] | None:
    """Fire whichever event turns store model ``name`` into a live deployment,
    for a caller that hit "not live yet" and would rather wait than fail.

    Returns ``(name, eta_seconds)`` when a load/deploy was actually fired, or
    ``None`` when ``name`` is genuinely not a catalog entry at all (nothing to
    trigger — the caller's original error stands). Two cases, both real:

    * never deployed (no placement row at all) -- fires the same
      ``serving/deploy.requested`` a first :func:`scale_store_model` replica
      would, using the base name as the deployment name (the established
      convention: the first replica of ``name`` IS the bare store name).
    * deployed then evicted / not yet ready (a placement row exists but none
      is live) -- fires :data:`LOAD_EVENT`, same as ``POST
      /deployments/{name}/load``.

    Best-effort on the ETA: a size-aware budget when the catalog knows the
    weights' byte size, else the same conservative floor
    :func:`_autoload_target` (the async Studio worker's own auto-reload) uses
    when it cannot stat the file directly.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog
    from docie_bench.serving.lifecycle import load_timeout_s
    from docie_bench.serving.resources import DEFAULT_DEPLOY_CONTEXT_LENGTH

    try:
        catalog = ModelCatalog()
        entry = catalog.get(name)
        if entry is None:
            return None
        placements = catalog.list_placements_for_model(name)
    except CatalogUnavailableError:
        return None

    size_bytes = entry.get("size_bytes") if isinstance(entry, dict) else None
    eta = load_timeout_s(None, size_bytes=size_bytes if isinstance(size_bytes, int) else None)

    if not placements:
        channel = f"deploy:{uuid.uuid4().hex}"
        data = {
            "model": name,
            "deployment_name": name,
            "context_length": DEFAULT_DEPLOY_CONTEXT_LENGTH,
            "channel": channel,
        }
        await send_or_503(inngest_client, inngest.Event(name=DEPLOY_EVENT, data=data))
    else:
        try:
            await _fire_lifecycle_event(name, event=LOAD_EVENT, prefix="load")
        except HTTPException:
            # The deployment record vanished between the catalog read above and
            # this call (rare race) -- nothing sane left to trigger.
            return None
    return name, eta


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
        raise HTTPException(status_code=404, detail=str(exc), headers=_DOMAIN_404) from exc
    channel = f"{prefix}:{uuid.uuid4().hex}"
    data: dict[str, Any] = {"name": name, "channel": channel, **(extra or {})}
    ids = await send_or_503(inngest_client, inngest.Event(name=event, data=data))
    return {"event_ids": list(ids), "channel": channel, "name": name}


@router.delete("/deployments/{name}")
async def delete_deployment(name: str, tenant: TenantDependency) -> dict[str, Any]:
    """Delete a deployment: stop its process, free its port, remove the record.

    Handled on the serving service (the only process that can stop the
    runtime); also removes the deployment's observed placement row.
    """
    del tenant  # authenticated principal required; no per-tenant scoping (ops surface)
    return await _fire_lifecycle_event(name, event=DELETE_EVENT, prefix="delete")


@router.post("/deployments/{name}/load")
async def load_deployment(name: str, tenant: TenantDependency) -> dict[str, Any]:
    """Load (start) a deployment: fire ``serving/load.requested``.

    Idempotent — loading an already-running deployment is a no-op. May
    unload least-recently-used unpinned deployments when that makes the
    load fit. Also starts manually stopped deployments.
    """
    del tenant
    return await _fire_lifecycle_event(name, event=LOAD_EVENT, prefix="load")


@router.post("/deployments/{name}/unload")
async def unload_deployment(name: str, tenant: TenantDependency) -> dict[str, Any]:
    """Unload a deployment, keeping its configuration.

    Unlike delete, the record, its port reservation and its placement row
    all survive — the deployment is marked ``phase=evicted`` /
    ``activation=managed``, and the next request to it reloads it
    automatically instead of failing.
    """
    del tenant
    return await _fire_lifecycle_event(name, event=UNLOAD_EVENT, prefix="unload")


class ScaleRequest(BaseModel):
    """Body of POST /store/{name}/scale — the TARGET total replica count."""

    replicas: int = Field(ge=1, le=16)
    context_length: int | None = None


@router.post("/store/{name}/scale")
async def scale_store_model(
    name: str, request: ScaleRequest, tenant: TenantDependency
) -> dict[str, Any]:
    """Scale a store model to ``replicas`` total addressable deployments.

    Deploying the SAME store model several times means several records — the
    first is the bare store name, the rest are ``<name>-2``/``-3``/… on their
    own auto-allocated ports (control_plane.replica_names_to_add). Idempotent:
    if already at/above the target, nothing is spawned. This fans out one
    ordinary ``serving/deploy.requested`` per new replica (each carrying
    ``deployment_name``), so every deploy reuses the proven single-deploy job —
    timeouts, port reallocation, placement recording — with no long-running
    scale job. The fit check lives in the Sizing surface the UI drives; the
    reconciler is the runtime backstop.
    """
    del tenant  # authenticated principal required; ops surface, no per-tenant scoping
    from docie_bench.serving.control_plane import (
        count_replica_deployments,
        replica_names_to_add,
    )
    from docie_bench.serving.resources import DEFAULT_DEPLOY_CONTEXT_LENGTH

    records = await _control_plane().list_deployments()
    existing = [
        str(spec_name)
        for record in (records if isinstance(records, list) else [])
        if isinstance(record, dict)
        and (spec_name := (record.get("spec") or {}).get("name"))
    ]
    to_add = replica_names_to_add(name, existing, request.replicas)
    current = count_replica_deployments(name, existing)
    if not to_add:
        return {
            "model": name,
            "target": request.replicas,
            "current": current,
            "adding": [],
            "event_ids": [],
            "channel": None,
        }
    ctx_len = request.context_length or DEFAULT_DEPLOY_CONTEXT_LENGTH
    channel = f"scale:{uuid.uuid4().hex}"
    event_ids: list[str] = []
    for deployment_name in to_add:
        data = {
            "model": name,
            "deployment_name": deployment_name,
            "context_length": ctx_len,
            "channel": channel,
        }
        ids = await send_or_503(inngest_client, inngest.Event(name=DEPLOY_EVENT, data=data))
        event_ids.extend(ids)
    return {
        "model": name,
        "target": request.replicas,
        "current": current,
        "adding": to_add,
        "event_ids": event_ids,
        "channel": channel,
    }


class ResizeRequest(BaseModel):
    """Body of POST /store/{name}/resize — the TARGET context window."""

    context_length: int = Field(ge=128, le=1_048_576)


@router.post("/store/{name}/resize")
async def resize_store_model(
    name: str, request: ResizeRequest, tenant: TenantDependency
) -> dict[str, Any]:
    """Change a live store deployment's context window with zero downtime.

    Two-tier RAM honesty, same split as ``/sizing`` vs the worker's live fit
    gate: this synchronous pre-check prices ONE instance at the new context
    length against the last published node snapshot — the SAME engine
    ``/sizing/whatif`` uses — and returns 422 with the deficit up front when
    it would not fit, WITHOUT touching the running deployment. A missing or
    stale snapshot fails open here (nothing to judge against yet) rather than
    blocking the request; the actual resize runs on the ``serving`` service,
    which re-checks against LIVE measured memory (the authoritative gate,
    correct cgroup, holds the Popen handles) before spawning anything — this
    pre-check is a fast, honest convenience, never a substitute for it.

    422 also covers a deployment whose runtime does not honor a context
    override (only llama.cpp does) — checked before the RAM pre-check so an
    unsupported deployment never even prices a plan.
    """
    del tenant
    try:
        status = await _control_plane().deployment_status(name)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc), headers=_DOMAIN_404) from exc
    launch = (status.get("spec") or {}).get("launch") or {} if isinstance(status, dict) else {}
    if str(launch.get("runtime") or "") != "llamacpp":
        raise HTTPException(
            status_code=422,
            detail=(
                f"deployment {name!r} runs the {launch.get('runtime')!r} runtime, "
                f"which does not accept a context-length override (only llama.cpp "
                f"honors --ctx-size)"
            ),
        )
    model_name = str(launch.get("alias") or name)

    from docie_bench.serving.resources import FootprintStore
    from docie_bench.serving.sizing import UnknownModelError, UnpriceableModelError, compute_whatif

    models, snapshot, placements, _detail = _sizing_inputs()
    if models:
        try:
            report = compute_whatif(
                models,
                snapshot,
                [
                    {
                        "model": model_name,
                        "instances": 1,
                        "context_length": request.context_length,
                    }
                ],
                placements,
                footprints=FootprintStore(),
                margin_fraction=get_settings().serving_sizing_margin_fraction,
            )
        except (UnknownModelError, UnpriceableModelError):
            report = None
        if report is not None and report.ok is False:
            raise HTTPException(status_code=422, detail=report.as_dict())

    return await _fire_lifecycle_event(
        name,
        event=RESIZE_EVENT,
        prefix="resize",
        extra={"context_length": request.context_length},
    )


class RepairRequest(BaseModel):
    """Body of POST /deployments/{name}/repair.

    ``port=None`` (default) auto-reallocates a free port; an explicit port is
    honored verbatim. Recovery without delete+recreate.
    """

    port: int | None = None


@router.post("/deployments/{name}/repair")
async def repair_deployment(
    name: str, request: RepairRequest, tenant: TenantDependency
) -> dict[str, Any]:
    """Recover a stuck or failed deployment in place.

    Redeploys the same launch configuration on a (re)allocated port and
    resets the restart counter — recovery for a deployment stuck failing to
    bind its port (for example when another process still holds it), without
    losing the deployment's config. ``port=None`` picks a free port
    automatically; an explicit port is honored.
    """
    del tenant
    extra = {"port": request.port} if request.port is not None else None
    return await _fire_lifecycle_event(name, event=REPAIR_EVENT, prefix="repair", extra=extra)


class ReconfigureRequest(BaseModel):
    """Editable defaults for an existing deployment.

    The context window is a runtime allocation and therefore requires a
    process restart when the deployment is hot. ``max_tokens=None`` clears a
    deployment-specific output cap and restores the model-family default.
    """

    context_length: int = Field(ge=128, le=1_048_576)
    max_tokens: int | None = Field(default=None, ge=1, le=131_072)


@router.patch("/deployments/{name}")
async def reconfigure_deployment(
    name: str, request: ReconfigureRequest, tenant: TenantDependency
) -> dict[str, Any]:
    """Edit a deployment in place and restart it when currently running."""
    del tenant
    return await _fire_lifecycle_event(
        name,
        event=RECONFIGURE_EVENT,
        prefix="reconfigure",
        extra={
            "context_length": request.context_length,
            "max_tokens": request.max_tokens,
        },
    )


class PinRequest(BaseModel):
    """Body of POST /deployments/{name}/pin."""

    pinned: bool = True


@router.post("/deployments/{name}/pin")
async def pin_deployment(
    name: str, request: PinRequest, tenant: TenantDependency
) -> dict[str, Any]:
    # An event (not an in-place write) because ``pinned`` lives in
    # deployments.json and only the serving service writes that file.
    """Pin or unpin a deployment — pinned deployments are never auto-unloaded."""
    del tenant
    return await _fire_lifecycle_event(
        name, event=PIN_EVENT, prefix="pin", extra={"pinned": request.pinned}
    )


def _ondisk_store_view() -> list[dict[str, Any]]:
    """The seeded store read from the ON-DISK index — the authoritative list of
    what is actually deployable (``serve_store_model`` reads the same index).

    Independent of the Postgres catalog, so a model seeded without DATABASE_URL
    (or during a catalog hiccup) still shows in Models instead of silently
    vanishing — the store/catalog desync this closes. Size is measured from
    disk (a file's size, or a snapshot directory's tree sum).
    """
    from docie_bench.serving.catalog import available_backends
    from docie_bench.serving.model_store import FAMILIES, ModelStore

    store = ModelStore(_serving_home() / "models")
    view: list[dict[str, Any]] = []
    for entry in store.list():
        contract = FAMILIES.get(entry.family)
        path = entry.model_path
        try:
            if path.is_dir():
                size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            else:
                size = path.stat().st_size
        except OSError:
            size = None
        view.append(
            {
                "name": entry.name,
                "family": entry.family,
                "vision": bool(contract and contract.vision),
                "embedding": bool(contract and contract.embedding),
                "reranker": bool(contract and contract.reranker),
                "analyzer": bool(contract and contract.analyzer),
                "available_backends": available_backends(entry.family),
                "has_mmproj": entry.mmproj_path is not None,
                "source": entry.source,
                "size_bytes": size,
                "placement": None,
                "created_at": None,
                "updated_at": None,
            }
        )
    return view


@router.get("/store")
async def list_store() -> Any:
    """The local model store the Studio reads — GGUFs AND encoder snapshots.

    Sourced from the ON-DISK store index (authoritative: exactly what
    ``serve_store_model`` can deploy), then enriched with the Postgres catalog's
    placement/timestamps when a catalog is configured. Reading on-disk first
    means a model seeded without DATABASE_URL still appears — no store/catalog
    desync. ``model_path``/``mmproj_path`` (container filesystem paths) are
    internal sizing inputs and are never included in the response.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    entries = _ondisk_store_view()
    # Best-effort catalog enrichment (placement + timestamps); never fatal.
    try:
        catalog = {row["name"]: row for row in ModelCatalog().list()}
    except CatalogUnavailableError:
        catalog = {}
    except Exception:  # noqa: BLE001 - catalog hiccup must not blank the store list
        logger.warning("store catalog enrichment failed; serving on-disk view", exc_info=True)
        catalog = {}
    for entry in entries:
        extra = catalog.get(entry["name"])
        if extra:
            entry["placement"] = extra.get("placement")
            entry["created_at"] = extra.get("created_at")
            entry["updated_at"] = extra.get("updated_at")
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
            "tools": fam.tools,
            "embedding": fam.embedding,
            "reranker": fam.reranker,
            "multi_vector": fam.multi_vector,
            "analyzer": fam.analyzer,
            "encoder_backend": fam.encoder_backend,
            "transformers_runtime": fam.transformers_runtime,
            "trust_remote_code": fam.trust_remote_code,
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
