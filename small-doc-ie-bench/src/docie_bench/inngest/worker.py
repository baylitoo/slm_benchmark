"""DocIE Studio Inngest worker (Connect mode).

Dials OUT to the Inngest Connect gateway over a WebSocket and serves the
functions; never needs to be publicly reachable. Run with the ``docie-worker``
console script or ``python -m docie_bench.inngest.worker``.

Local dev:   INNGEST_DEV=1 docie-worker   (with `npx inngest-cli@latest dev`)
Docker/prod: env from docker-compose (INNGEST_DEV=0 + signing/event keys).

Roles (PR-1, design doc §0.1/P1): ``DOCIE_WORKER_ROLE`` selects which Inngest
APP this process registers — one app id per role, because a Connect handshake
syncs its function list as the app's authoritative set and two fleets syncing
disjoint sets under one app id would overwrite each other's registration on
every (re)connect (see ``client.py``):

  * ``serving`` — the dedicated SINGLE-REPLICA lifecycle service: registers
    the ``docie-serving`` app (deploy / seed / delete) plus the background
    reconciler. The only process that spawns or kills runtime processes and
    the sole writer of ``deployments.json``. Never scale this above one
    replica — the reconciler additionally refuses to start when a live lease
    from another instance is found on the shared serving-state volume.
  * ``worker``  — registers the ``docie-studio`` app (extraction / benchmark /
    GC). Replica-safe; scale freely. Never spawns a runtime and never runs
    the reconciler.
  * ``all`` (default) — registers BOTH apps over the one connection, for
    local dev without the compose split (``connect()`` is multi-app by
    design). The reconciler is OFF by default in this role (opt in with
    ``DOCIE_SERVING_RECONCILE=1``) so an accidentally scaled legacy `worker`
    can never run N reconcilers.

Gateway endpoint: the Inngest server advertises its own gateway URL during the
handshake. In Docker that advertised host (e.g. 127.0.0.1) isn't reachable from
the worker container, so we rewrite it to ``INNGEST_CONNECT_GATEWAY_URL`` when
set (e.g. ws://inngest:8289). Locally, leave it unset and the SDK uses the
advertised URL as-is.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable
from typing import Any

from inngest.connect import connect

from docie_bench.inngest.functions import apps_for_role
from docie_bench.logging_config import configure_logging
from docie_bench.settings import get_settings

logger = logging.getLogger("docie_bench.inngest.worker")

# Compose services known to scale beyond one replica. A serving deploy that
# ADVERTISES one of these names records a round-robin endpoint (each replica
# gets its own A-record in compose DNS), which is exactly the pre-P1 failure
# the dedicated single-replica `serving` service removed.
_SCALED_SERVICE_NAMES = frozenset({"worker"})


def _gateway_rewrite() -> Callable[[str], str] | None:
    override = os.getenv("INNGEST_CONNECT_GATEWAY_URL", "").strip()
    if not override:
        return None
    logger.info("rewriting connect gateway endpoint -> %s", override)
    return lambda _advertised: override


def _role() -> str:
    return os.getenv("DOCIE_WORKER_ROLE", "all").strip().lower() or "all"


def _reconciler_enabled(role: str) -> bool:
    """The reconciler runs on the ``serving`` role; opt-in for ``all`` (dev).

    Never for ``worker``: a scaled worker replica running a reconciler is the
    exact multi-writer clobber P1 exists to prevent.
    """
    if role == "serving":
        return os.getenv("DOCIE_SERVING_RECONCILE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
    if role == "all":
        return os.getenv("DOCIE_SERVING_RECONCILE", "").strip().lower() in {"1", "true", "yes"}
    return False


def _warn_legacy_advertise_host(role: str) -> None:
    """Upgrade-trap guard (PR-1 service split): advertise must not be a scaled name.

    A ``.env`` written before the split carries
    ``DOCIE_SERVING_ADVERTISE_HOST=worker``; a serving-role process that keeps
    it would record every deploy's endpoint against the freely-scaled
    ``worker`` service name, whose compose DNS round-robins across replicas.
    Compose already forces the correct default (the serving service reads its
    own name from ``DOCIE_SERVING_ADVERTISE_HOST_OVERRIDE``, ignoring the
    legacy variable), so hitting this warning means a hand-rolled env is
    overriding it. The deploy-time ``_guard_deterministic_advertise`` will
    still fail such deploys once the service actually scales >1; this warning
    surfaces the misconfiguration at startup instead of at first deploy.
    """
    if role not in {"serving", "all"}:
        return
    advertise = get_settings().serving_advertise_host
    if advertise in _SCALED_SERVICE_NAMES:
        logger.warning(
            "DOCIE_SERVING_ADVERTISE_HOST=%r names a SCALED compose service: deploys "
            "would advertise a round-robin endpoint that may resolve to a replica "
            "that never ran the deploy. Your .env likely predates the PR-1 service "
            "split — set DOCIE_SERVING_ADVERTISE_HOST=serving (or delete the line; "
            "docker-compose.yml pins the serving service's own default).",
            advertise,
        )


def _build_reconciler(instance_id: str) -> Any:
    """Reconciler over the SHARED control plane's supervisor.

    Must use ``_serving_control_plane()`` (the lru-cached instance the deploy
    handlers use), never a fresh ``PersistentSupervisor``: only the shared
    instance holds the runtime adapters' ``_processes`` Popen handles that
    make in-namespace liveness authoritative, and only one instance may write
    ``deployments.json``. Carries the singleton lease on the shared serving
    home so a second (misconfigured, ``--scale serving=2``) replica refuses
    to start its reconciler instead of silently double-writing.
    """
    from docie_bench.inngest.functions import _serving_control_plane, _serving_home
    from docie_bench.serving.reconciler import ReconcilerLease, ServingReconciler

    control_plane = _serving_control_plane()
    supervisor = control_plane.supervisor.backend  # the shared PersistentSupervisor
    interval = float(os.getenv("DOCIE_SERVING_RECONCILE_INTERVAL", "10"))
    lease = ReconcilerLease(
        path=_serving_home() / "reconciler-lease.json",
        instance_id=instance_id,
        # Generous multiple of the cycle so one slow cycle (long health
        # timeouts) never lets a second replica steal a live lease.
        stale_after_s=max(6 * interval, 60.0),
    )
    return ServingReconciler(supervisor, interval_s=interval, lease=lease)


def _start_reconciler(instance_id: str) -> asyncio.Task[None] | None:
    """Claim the singleton lease and start the loop; refuse (None) if held.

    Refusal is deliberately loud but non-fatal: the replica still serves its
    Inngest functions (so a mis-scaled fleet degrades instead of flapping),
    but it will not run a second reconciler against the shared volume.
    """
    from docie_bench.serving.reconciler import ReconcilerSingletonError

    reconciler = _build_reconciler(instance_id)
    try:
        reconciler.claim_singleton()
    except ReconcilerSingletonError as exc:
        logger.critical(
            "reconciler DISABLED on this replica: %s (`--scale serving=N>1` is "
            "forbidden — the serving service is single-replica by design)",
            exc,
        )
        return None
    return asyncio.create_task(reconciler.run_forever(), name="serving-reconciler")


async def _serve() -> None:
    role = _role()
    apps = apps_for_role(role)
    _warn_legacy_advertise_host(role)
    instance_id = (
        os.getenv("INNGEST_INSTANCE_ID") or os.getenv("HOSTNAME") or f"docie-{role}"
    )
    connection = connect(
        apps,
        instance_id=instance_id,
        rewrite_gateway_endpoint=_gateway_rewrite(),
    )
    logger.info(
        "docie worker connecting (instance=%s, role=%s, apps=%s)",
        instance_id,
        role,
        ", ".join(
            f"{client.app_id}[{len(functions)} fns]" for client, functions in apps
        ),
    )
    reconciler_task: asyncio.Task[None] | None = None
    if _reconciler_enabled(role):
        reconciler_task = _start_reconciler(instance_id)
    try:
        # Blocks until the connection is closed (SIGTERM/SIGINT drain in-flight
        # steps). The reconciler runs beside it on this same loop, its blocking
        # cycles pushed to a worker thread so heartbeats keep flowing.
        await connection.start()
    finally:
        if reconciler_task is not None:
            reconciler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reconciler_task


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    # Enable the model-store catalog (and any DB-backed work) inside the worker.
    # No-op when DATABASE_URL is unset. For the serving role this also runs the
    # model_placement observed-columns migration before the first publish.
    from docie_bench.storage.db import init_engine

    init_engine()
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:  # pragma: no cover - graceful Ctrl-C
        logger.info("worker interrupted; shutting down")


if __name__ == "__main__":
    main()
