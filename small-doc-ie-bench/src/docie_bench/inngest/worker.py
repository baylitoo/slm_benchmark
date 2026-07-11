"""DocIE Studio Inngest worker (Connect mode).

Dials OUT to the Inngest Connect gateway over a WebSocket and serves the
functions; never needs to be publicly reachable. Run with the ``docie-worker``
console script or ``python -m docie_bench.inngest.worker``.

Local dev:   INNGEST_DEV=1 docie-worker   (with `npx inngest-cli@latest dev`)
Docker/prod: env from docker-compose (INNGEST_DEV=0 + signing/event keys).

Roles (PR-1, design doc §0.1/P1): ``DOCIE_WORKER_ROLE`` selects which function
set this process registers —

  * ``serving`` — the dedicated SINGLE-REPLICA lifecycle service: deploy /
    seed / delete functions plus the background reconciler. The only process
    that spawns or kills runtime processes and the sole writer of
    ``deployments.json``. Never scale this above one replica.
  * ``worker``  — extraction / benchmark / GC. Replica-safe; scale freely.
    Never spawns a runtime and never runs the reconciler.
  * ``all`` (default) — everything in one process, for local dev without the
    compose split. The reconciler is OFF by default in this role (opt in with
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

from docie_bench.inngest.client import APP_ID, inngest_client
from docie_bench.inngest.functions import functions_for_role
from docie_bench.logging_config import configure_logging
from docie_bench.settings import get_settings

logger = logging.getLogger("docie_bench.inngest.worker")


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


def _build_reconciler() -> Any:
    """Reconciler over the SHARED control plane's supervisor.

    Must use ``_serving_control_plane()`` (the lru-cached instance the deploy
    handlers use), never a fresh ``PersistentSupervisor``: only the shared
    instance holds the runtime adapters' ``_processes`` Popen handles that
    make in-namespace liveness authoritative, and only one instance may write
    ``deployments.json``.
    """
    from docie_bench.inngest.functions import _serving_control_plane
    from docie_bench.serving.reconciler import ServingReconciler

    control_plane = _serving_control_plane()
    supervisor = control_plane.supervisor.backend  # the shared PersistentSupervisor
    interval = float(os.getenv("DOCIE_SERVING_RECONCILE_INTERVAL", "10"))
    return ServingReconciler(supervisor, interval_s=interval)


async def _serve() -> None:
    role = _role()
    functions = functions_for_role(role)
    instance_id = (
        os.getenv("INNGEST_INSTANCE_ID") or os.getenv("HOSTNAME") or f"docie-{role}"
    )
    connection = connect(
        [(inngest_client, functions)],
        instance_id=instance_id,
        rewrite_gateway_endpoint=_gateway_rewrite(),
    )
    logger.info(
        "docie worker connecting (app_id=%s, instance=%s, role=%s, functions=%d)",
        APP_ID,
        instance_id,
        role,
        len(functions),
    )
    reconciler_task: asyncio.Task[None] | None = None
    if _reconciler_enabled(role):
        reconciler = _build_reconciler()
        reconciler_task = asyncio.create_task(
            reconciler.run_forever(), name="serving-reconciler"
        )
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
