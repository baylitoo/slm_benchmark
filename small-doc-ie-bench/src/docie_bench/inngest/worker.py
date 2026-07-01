"""DocIE Studio Inngest worker (Connect mode).

Dials OUT to the Inngest Connect gateway over a WebSocket and serves the
functions; never needs to be publicly reachable. Run with the ``docie-worker``
console script or ``python -m docie_bench.inngest.worker``.

Local dev:   INNGEST_DEV=1 docie-worker   (with `npx inngest-cli@latest dev`)
Docker/prod: env from docker-compose (INNGEST_DEV=0 + signing/event keys).

Gateway endpoint: the Inngest server advertises its own gateway URL during the
handshake. In Docker that advertised host (e.g. 127.0.0.1) isn't reachable from
the worker container, so we rewrite it to ``INNGEST_CONNECT_GATEWAY_URL`` when
set (e.g. ws://inngest:8289). Locally, leave it unset and the SDK uses the
advertised URL as-is.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable

from inngest.connect import connect

from docie_bench.inngest.client import APP_ID, inngest_client
from docie_bench.inngest.functions import functions
from docie_bench.logging_config import configure_logging
from docie_bench.settings import get_settings

logger = logging.getLogger("docie_bench.inngest.worker")


def _gateway_rewrite() -> Callable[[str], str] | None:
    override = os.getenv("INNGEST_CONNECT_GATEWAY_URL", "").strip()
    if not override:
        return None
    logger.info("rewriting connect gateway endpoint -> %s", override)
    return lambda _advertised: override


async def _serve() -> None:
    instance_id = (
        os.getenv("INNGEST_INSTANCE_ID") or os.getenv("HOSTNAME") or "docie-worker"
    )
    connection = connect(
        [(inngest_client, functions)],
        instance_id=instance_id,
        rewrite_gateway_endpoint=_gateway_rewrite(),
    )
    logger.info(
        "docie worker connecting (app_id=%s, instance=%s, functions=%d)",
        APP_ID,
        instance_id,
        len(functions),
    )
    # Blocks until the connection is closed (SIGTERM/SIGINT drain in-flight steps).
    await connection.start()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    # Enable the model-store catalog (and any DB-backed work) inside the worker.
    # No-op when DATABASE_URL is unset.
    from docie_bench.storage.db import init_engine

    init_engine()
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:  # pragma: no cover - graceful Ctrl-C
        logger.info("worker interrupted; shutting down")


if __name__ == "__main__":
    main()
