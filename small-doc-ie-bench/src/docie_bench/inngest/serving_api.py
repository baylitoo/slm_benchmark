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
from pathlib import Path
from typing import Any

import inngest
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docie_bench.inngest.client import inngest_client, send_or_503
from docie_bench.security import TenantDependency
from docie_bench.serving.control_plane import ControlPlane
from docie_bench.serving.failure import classify_failure
from docie_bench.serving.node_liveness import (
    SNAPSHOT_STALE_FLOOR_S as _SNAPSHOT_STALE_FLOOR_S,
)
from docie_bench.serving.node_liveness import (
    SNAPSHOT_STALE_INTERVALS as _SNAPSHOT_STALE_INTERVALS,
)
from docie_bench.serving.node_liveness import (
    reconcile_interval_s,
    snapshot_age_s,
    snapshot_stale_after_s,
)
from docie_bench.serving.paths import serving_home
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
PIN_EVENT = "serving/pin.requested"
# Scaling reuses the ordinary single-deploy job (deploy_model_job) — the scale
# endpoint just fans out one deploy event per new replica name.
DEPLOY_EVENT = "serving/deploy.requested"

# The snapshot-staleness gate now lives in serving/node_liveness.py, so the
# router can apply the SAME definition of "this node is still reporting" when
# deciding whether its placements should keep receiving traffic (#531). Two
# definitions of reporting would drift, and the capacity view and the router
# disagreeing about whether a node is alive is the worst version of that.
# Re-exported under the original names: these are this module's public surface.
SNAPSHOT_STALE_FLOOR_S = _SNAPSHOT_STALE_FLOOR_S
SNAPSHOT_STALE_INTERVALS = _SNAPSHOT_STALE_INTERVALS
_reconcile_interval_s = reconcile_interval_s
_snapshot_age_s = snapshot_age_s


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


def _nodes_that_stopped_reporting() -> frozenset[str]:
    """Nodes whose capacity snapshot has gone stale (best-effort)."""
    from docie_bench.serving.catalog import ModelCatalog
    from docie_bench.serving.node_liveness import stale_node_ids

    try:
        return stale_node_ids(ModelCatalog().list_node_snapshots())
    except Exception:  # noqa: BLE001 - attribution must not 500 the Board
        return frozenset()


def _records_for_other_nodes(
    local: list[Any], observed: dict[str, dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Catalog-derived records for deployments this process has no file for.

    The api reads one node's ``deployments.json`` -- the shared `serving-state`
    volume -- so every deployment on another machine is simply absent from its
    listing, presented as though the fleet were that one node (#562). These
    rows fill that gap from ``model_placement``, which carries the owner since
    #530.

    Deliberately partial and marked as such. A placement row has no desired
    state, restart policy or pin -- those live in the owning node's file, and
    publishing them is the phase-2 desired-state work (#534). What it does
    carry is enough to show the deployment exists, where it is, what state it
    is in and how it is behaving, which beats omitting it.
    """
    if not observed:
        return []
    known = {
        str((record.get("spec") or {}).get("name") or "")
        for record in local
        if isinstance(record, dict)
    }
    remote: list[dict[str, Any]] = []
    for name, row in sorted(observed.items()):
        if name in known:
            continue
        remote.append(
            {
                "spec": {
                    "name": name,
                    # model_name is the launch alias for a store deploy, which
                    # is what groups replicas (`_replica_base`).
                    "launch": {
                        "alias": row.get("model_name") or name,
                        "model": row.get("model_name"),
                    },
                },
                "state": row.get("state"),
                "endpoint": row.get("endpoint"),
                "last_error": row.get("last_error"),
                "node_id": row.get("node_id"),
                # The UI needs to know this record is a catalog projection and
                # not the owning node's own record, so it does not offer
                # actions that need the file it does not have.
                "record_source": "catalog",
            }
        )
    return remote


def _replica_base(record: dict[str, Any]) -> str | None:
    """The routable base a record belongs to: the launch ``--alias`` (shared by
    every replica of a scaled store model), falling back to the record name."""
    spec = record.get("spec") or {}
    launch = spec.get("launch") or {}
    base = launch.get("alias") or spec.get("name")
    return str(base) if base else None


@router.get("/deployments")
async def list_deployments() -> Any:
    """Deployment records with their live observed state.

    Each record carries an ``observed`` object (phase / pid / rss_bytes /
    health_ok / last_probe_at / last_error / endpoint), refreshed every
    cycle — ``None`` per record when no observation has been published yet,
    and ``observed_available: false`` on all records when the observed state
    is unreachable (desired state only).

    Replicas: each record also carries ``replica_group`` (the shared launch
    alias its requests are load-balanced behind) and ``replicas`` (how many
    records share that group right now). Derived from the records, never
    stored — no schema migration.
    """
    records = await _control_plane().list_deployments()
    observed = _observed_placements()
    if not isinstance(records, list):
        return records
    # Deployments owned by OTHER nodes, which this process has no file for
    # (#562). Without this the listing silently presents one node's disk as the
    # whole fleet; a missing deployment is worse than a partial one.
    records.extend(_records_for_other_nodes(records, observed))
    stale_nodes = _nodes_that_stopped_reporting()
    group_sizes: dict[str, int] = {}
    for record in records:
        if isinstance(record, dict) and (base := _replica_base(record)):
            group_sizes[base] = group_sizes.get(base, 0) + 1
    for record in records:
        if not isinstance(record, dict):
            continue
        spec = record.get("spec") or {}
        name = spec.get("name")
        record["observed_available"] = observed is not None
        obs = observed.get(name) if observed and name else None
        record["observed"] = obs
        # Which node runs this, and whether that node is still reporting. Both
        # come from the catalog, so a locally-read record gets attributed too
        # rather than being implicitly "here".
        record.setdefault("node_id", (obs or {}).get("node_id"))
        record.setdefault("record_source", "node")
        node_id = record.get("node_id")
        record["node_reporting"] = not (node_id and node_id in stale_nodes)
        base = _replica_base(record)
        record["replica_group"] = base
        record["replicas"] = group_sizes.get(base, 1) if base else 1
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
    node_count = 0
    try:
        catalog = ModelCatalog()
        # Every node that has published, newest measurement first. Reading the
        # list rather than "the" snapshot is what keeps this view honest once a
        # second node reports: it describes ONE machine, and says which.
        snapshots = catalog.list_node_snapshots()
        node_count = len(snapshots)
        if not snapshots:
            detail = (
                "no capacity measurement published yet — is the serving "
                "service running?"
            )
        else:
            node, detail = _gate_snapshot_staleness(snapshots[0])
            if node is not None and node_count > 1:
                detail = (
                    f"showing node {node['node_id']}; {node_count} nodes are "
                    f"publishing capacity — these numbers describe that one "
                    f"node, not the fleet total"
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
        # How many nodes published a snapshot, so the UI can show that `node`
        # is one of several instead of implying it is the whole capacity.
        "node_count": node_count,
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
    # Call-level (not per-item): mirrors compute_whatif's own n_parallel
    # parameter, which prices every row in the plan under the same llama-server
    # slot count (#248/#321) rather than per-row.
    n_parallel: int = Field(default=1, ge=1, le=32)


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

    With several nodes publishing, the snapshot is likewise withheld: a fit
    priced against one arbitrary node's free RAM is the guess #529 removes, and
    it would refuse a plan that fits elsewhere in the fleet. Pricing resumes
    once a deploy names its target node (placement ownership, #527 §5.C).
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        catalog = ModelCatalog()
        models = catalog.list()
        snapshots = catalog.list_node_snapshots()
        placements = list(catalog.list_placements())
    except CatalogUnavailableError:
        return [], None, [], "observed state unavailable: DATABASE_URL is not configured"
    except Exception:  # noqa: BLE001 - a DB hiccup must not 500 the Sizing tab
        return [], None, [], "observed state unavailable: database error"
    if len(snapshots) > 1:
        return (
            models,
            None,
            placements,
            f"{len(snapshots)} serving nodes are publishing capacity; a fit "
            f"cannot be priced against one of them without knowing which node "
            f"a deploy would land on",
        )
    snapshot = snapshots[0] if snapshots else None
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
            n_parallel=request.n_parallel,
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


@router.get("/deployments/{name}/slots")
async def deployment_slots(name: str) -> dict[str, Any]:
    """llama-server's own ``GET /slots`` introspection (#315), queried live
    and on demand for one deployment -- the same shape as
    ``code_interpreter_workers``'s live call to Judge0's own ``/workers``: a
    status card, not something baked into the reconciler's observed-state
    publish loop. 404s (the domain marker) for a deployment that is not
    llama.cpp, or one with no live endpoint yet, so the UI can render "not
    applicable" instead of a broken card.
    """
    try:
        record = await _control_plane().deployment_status(name)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc), headers=_DOMAIN_404) from exc
    launch = (record.get("spec") or {}).get("launch") or {} if isinstance(record, dict) else {}
    if launch.get("runtime") != "llamacpp":
        raise HTTPException(
            status_code=404,
            detail=f"deployment {name!r} is not a llama.cpp runtime",
            headers=_DOMAIN_404,
        )
    endpoint = record.get("endpoint") if isinstance(record, dict) else None
    if not endpoint:
        raise HTTPException(
            status_code=404,
            detail=f"deployment {name!r} has no live endpoint yet",
            headers=_DOMAIN_404,
        )
    from docie_bench.serving.runtime import fetch_llamacpp_slots

    request_headers: dict[str, str] = {}
    api_key_env = launch.get("api_key_env")
    if api_key_env and (api_key := os.environ.get(api_key_env)):
        request_headers["Authorization"] = f"Bearer {api_key}"
    slots = fetch_llamacpp_slots(endpoint, headers=request_headers)
    return {"name": name, "slots": list(slots)}


def _serving_logs_dir() -> Path:
    """``<serving_home>/logs`` — the runtime stdout files the supervisor writes,
    readable by the api (it never spawned the process)."""
    return serving_home() / "logs"


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


def _owning_node(name: str) -> str | None:
    """Which node runs ``name``, from the placement table (#566).

    Best-effort and advisory: an unreadable catalog means "unknown", which
    leaves the caller's own 404 in place rather than inventing an owner.
    """
    from docie_bench.serving.catalog import ModelCatalog

    try:
        for row in ModelCatalog().list_placements():
            if str(row.get("name") or "") == name:
                owner = str(row.get("node_id") or "")
                return owner or None
    except Exception:  # noqa: BLE001 - attribution must not change the status code
        return None
    return None


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
        # Not on THIS node's disk. It may still exist, on another node (#562):
        # these handlers read the local deployments.json, so a perfectly live
        # remote deployment reaches here. A 404 would tell the operator it does
        # not exist, which is both wrong and the kind of wrong that sends them
        # looking in the right place last.
        owner = _owning_node(name)
        if owner is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"deployment {name!r} runs on serving node {owner!r}, and "
                    f"lifecycle events are delivered to whichever node Inngest "
                    f"picks -- so this cannot act on it. Run it on that node "
                    f"(`docie-serving stop {name}`), or deploy with node_id so "
                    f"the deployment is managed as desired state (#535)"
                ),
                headers=_DOMAIN_404,
            ) from exc
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


async def existing_deployment_records() -> list[dict[str, Any]]:
    """Every deployment record in the shared on-disk state (replica math input
    for the scale endpoint and the replicated-deploy trigger).

    The records, not just their names: replica membership is decided by the
    launch alias (``control_plane.replica_group_of``), which a name alone does
    not carry.
    """
    records = await _control_plane().list_deployments()
    return [
        record
        for record in (records if isinstance(records, list) else [])
        if isinstance(record, dict) and (record.get("spec") or {}).get("name")
    ]


def _gib(value: int) -> str:
    return f"{value / (1024 ** 3):.1f} GiB"


def admit_replica_ram(
    model: str,
    added_instances: int,
    context_length: int | None,
    n_parallel: int = 1,
) -> None:
    """RAM admission gate for launching ``added_instances`` MORE instances of
    ``model``: N x the per-instance footprint must fit the sizing budget.

    Prices with the SAME engine as ``/sizing/whatif`` (``compute_whatif`` —
    footprint formula, safety margin, loading-placement reservation, and now
    also ``n_parallel``), so this gate and the Sizing tab can never disagree.
    Raises a 422 with the explicit deficit when the plan PROVABLY does not fit.

    Fail-open by design when nothing can be proven: no catalog / no (or stale)
    node snapshot / an unpriceable model all admit the deploy — the serving
    container's live per-instance fit gate (``lifecycle.assess_fit``) remains
    the runtime backstop, and refusing on ignorance would turn every degraded
    observability state into a deploy outage.
    """
    if added_instances < 1:
        return
    from docie_bench.serving.resources import FootprintStore
    from docie_bench.serving.sizing import (
        UnknownModelError,
        UnpriceableModelError,
        compute_whatif,
    )

    models, snapshot, placements, _detail = _sizing_inputs()
    if not models or snapshot is None:
        return
    try:
        report = compute_whatif(
            models,
            snapshot,
            [
                {
                    "model": model,
                    "instances": added_instances,
                    "context_length": context_length,
                }
            ],
            placements,
            footprints=FootprintStore(),
            margin_fraction=get_settings().serving_sizing_margin_fraction,
            n_parallel=n_parallel,
        )
    except (UnknownModelError, UnpriceableModelError):
        return
    if report.ok is False:
        item = report.per_item[0]
        free = report.free_effective_bytes or 0
        raise HTTPException(
            status_code=422,
            detail=(
                f"not enough RAM to launch {added_instances} more instance(s) of "
                f"{model!r}: {added_instances} x {_gib(item.footprint_bytes)} per "
                f"instance = {_gib(report.total_predicted_bytes)} needed, but only "
                f"{_gib(max(free, 0))} is available after the safety margin and "
                f"loading reservations (short by {_gib(report.deficit_bytes or 0)}). "
                f"Lower the replica count, or free RAM by unloading/deleting a "
                f"deployment (see the Sizing tab)."
            ),
        )


class ScaleRequest(BaseModel):
    """Body of POST /store/{name}/scale — the TARGET total replica count."""

    replicas: int = Field(ge=1, le=16)
    context_length: int | None = None
    # llama-server request slots (#248/#321): 32 is an operator sanity ceiling,
    # not a llama.cpp hard limit.
    n_parallel: int = Field(default=1, ge=1, le=32)
    cache_reuse: int | None = Field(default=None, ge=1)
    chat_template_file: str | None = None
    batch_size: int | None = Field(default=None, ge=1)
    ubatch_size: int | None = Field(default=None, ge=1)
    numa: str | None = None
    cpu_threads: int | None = Field(default=None, ge=1)
    cpu_threads_batch: int | None = Field(default=None, ge=1)
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    reasoning_budget: int | None = Field(default=None, ge=-1)
    reasoning_budget_message: str | None = None
    flash_attn: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)


@router.post("/store/{name}/scale")
async def scale_store_model(
    name: str, request: ScaleRequest, tenant: TenantDependency
) -> dict[str, Any]:
    """Scale a store model to ``replicas`` total addressable deployments.

    Deploying the SAME store model several times means several records — the
    first is the bare store name, the rest are ``<name>-2``/``-3``/… on their
    own auto-allocated ports (control_plane.replica_names_to_add). Idempotent:
    already at the target, nothing happens.

    Scale UP first passes the RAM admission gate (``admit_replica_ram`` — N x
    per-instance footprint against the live sizing budget; a provable deficit
    is a 422, an unknowable budget admits and leaves the serving container's
    fit gate as the backstop), then fans out one ordinary
    ``serving/deploy.requested`` per new replica (each carrying
    ``deployment_name``), so every deploy reuses the proven single-deploy job —
    timeouts, port reallocation, placement recording — with no long-running
    scale job.

    Scale DOWN fans out one ``serving/delete.requested`` per surplus replica —
    highest suffix first, the bare base record always survives — reusing the
    real teardown job (kill process, free port, DELETE placement row), so a
    drained replica leaves the load-balancing rotation exactly like a deleted
    deployment.
    """
    del tenant  # authenticated principal required; ops surface, no per-tenant scoping
    from docie_bench.serving.control_plane import (
        count_replica_deployments,
        replica_names_to_add,
        replica_names_to_remove,
    )
    from docie_bench.serving.resources import DEFAULT_DEPLOY_CONTEXT_LENGTH

    existing = await existing_deployment_records()
    to_add = replica_names_to_add(name, existing, request.replicas)
    to_remove = replica_names_to_remove(name, existing, request.replicas)
    current = count_replica_deployments(name, existing)
    if to_add:
        admit_replica_ram(
            name, len(to_add), request.context_length, n_parallel=request.n_parallel
        )
        ctx_len = request.context_length or DEFAULT_DEPLOY_CONTEXT_LENGTH
        channel = f"scale:{uuid.uuid4().hex}"
        event_ids: list[str] = []
        for deployment_name in to_add:
            data: dict[str, Any] = {
                "model": name,
                "deployment_name": deployment_name,
                "context_length": ctx_len,
                "channel": channel,
            }
            if request.n_parallel > 1:
                data["n_parallel"] = request.n_parallel
            if request.cache_reuse is not None:
                data["cache_reuse"] = request.cache_reuse
            if request.chat_template_file is not None:
                data["chat_template_file"] = request.chat_template_file
            if request.batch_size is not None:
                data["batch_size"] = request.batch_size
            if request.ubatch_size is not None:
                data["ubatch_size"] = request.ubatch_size
            if request.numa is not None:
                data["numa"] = request.numa
            if request.cpu_threads is not None:
                data["cpu_threads"] = request.cpu_threads
            if request.cpu_threads_batch is not None:
                data["cpu_threads_batch"] = request.cpu_threads_batch
            if request.cache_type_k is not None:
                data["cache_type_k"] = request.cache_type_k
            if request.cache_type_v is not None:
                data["cache_type_v"] = request.cache_type_v
            if request.reasoning_budget is not None:
                data["reasoning_budget"] = request.reasoning_budget
            if request.reasoning_budget_message is not None:
                data["reasoning_budget_message"] = request.reasoning_budget_message
            if request.flash_attn is not None:
                data["flash_attn"] = request.flash_attn
            if request.timeout_seconds is not None:
                data["timeout_seconds"] = request.timeout_seconds
            ids = await send_or_503(inngest_client, inngest.Event(name=DEPLOY_EVENT, data=data))
            event_ids.extend(ids)
        return {
            "model": name,
            "target": request.replicas,
            "current": current,
            "adding": to_add,
            "removing": [],
            "event_ids": event_ids,
            "channel": channel,
        }
    if to_remove:
        channel = f"scale:{uuid.uuid4().hex}"
        event_ids = []
        for deployment_name in to_remove:
            ids = await send_or_503(
                inngest_client,
                inngest.Event(
                    name=DELETE_EVENT,
                    data={"name": deployment_name, "channel": channel},
                ),
            )
            event_ids.extend(ids)
        return {
            "model": name,
            "target": request.replicas,
            "current": current,
            "adding": [],
            "removing": to_remove,
            "event_ids": event_ids,
            "channel": channel,
        }
    return {
        "model": name,
        "target": request.replicas,
        "current": current,
        "adding": [],
        "removing": [],
        "event_ids": [],
        "channel": None,
    }


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

    store = ModelStore(serving_home() / "models")
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
                "structured_extraction": bool(contract and contract.structured_extraction),
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
    # Models the catalog knows but THIS node has no blob for -- i.e. seeded on
    # another node (#562). Without them the Studio shows one node's disk as the
    # fleet's store, and a deploy of a model it cannot see simply fails.
    known = {entry["name"] for entry in entries}
    for name, row in sorted(catalog.items()):
        if name in known:
            continue
        entries.append(_store_entry_from_catalog(name, row))
    node_stores = _store_models_by_node()
    for entry in entries:
        extra = catalog.get(entry["name"])
        if extra:
            entry["placement"] = extra.get("placement")
            entry["created_at"] = extra.get("created_at")
            entry["updated_at"] = extra.get("updated_at")
        entry["nodes"] = _presence_across_nodes(str(entry["name"]), node_stores)
        entry.pop("model_path", None)
        entry.pop("mmproj_path", None)
    return entries


def _store_models_by_node() -> dict[str, list[str] | None]:
    """What each node reports holding, ``None`` for a node that has not said.

    Best-effort: an unreadable node table yields nothing, so the store listing
    degrades to un-annotated rather than failing.
    """
    from docie_bench.serving.catalog import ModelCatalog

    try:
        snapshots = ModelCatalog().list_node_snapshots()
    except Exception:  # noqa: BLE001 - annotation must not blank the store list
        return {}
    return {
        str(row.get("node_id") or ""): row.get("store_models")
        for row in snapshots
        if row.get("node_id")
    }


def _presence_across_nodes(
    name: str, node_stores: dict[str, list[str] | None]
) -> dict[str, list[str]]:
    """Which nodes hold ``name``, which do not, and which have not reported.

    The distinction matters for what an operator does next: `missing` means
    re-seed that node, `unreported` means wait for its next publish (or check
    whether it is reporting at all). Collapsing them into one list would turn
    "we do not know" into "it is not there".
    """
    present: list[str] = []
    missing: list[str] = []
    unreported: list[str] = []
    for node_id, models in sorted(node_stores.items()):
        if models is None:
            unreported.append(node_id)
        elif name in models:
            present.append(node_id)
        else:
            missing.append(node_id)
    return {"present": present, "missing": missing, "unreported": unreported}


def _store_entry_from_catalog(name: str, row: dict[str, Any]) -> dict[str, Any]:
    """A store row for a model seeded on another node (#562).

    Deliberately thin: size and mmproj presence are properties of a blob this
    process cannot see. What it can say is that the model exists, what family
    it is, and -- via the `nodes` annotation added by the caller -- where its
    blob actually is.
    """
    from docie_bench.serving.catalog import available_backends
    from docie_bench.serving.model_store import FAMILIES

    family = str(row.get("family") or "")
    contract = FAMILIES.get(family)
    return {
        "name": name,
        "family": family,
        "vision": bool(contract and contract.vision),
        "embedding": bool(contract and contract.embedding),
        "reranker": bool(contract and contract.reranker),
        "analyzer": bool(contract and contract.analyzer),
        "structured_extraction": bool(contract and contract.structured_extraction),
        "available_backends": available_backends(family),
        "has_mmproj": None,
        "source": row.get("source"),
        "size_bytes": row.get("size_bytes"),
        "placement": None,
        "created_at": None,
        "updated_at": None,
        # This process has no blob for it, so nothing here was read from disk.
        "store_view": "catalog",
    }


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
            "dense_embedding": fam.dense_embedding,
            "analyzer": fam.analyzer,
            "structured_extraction": fam.structured_extraction,
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
