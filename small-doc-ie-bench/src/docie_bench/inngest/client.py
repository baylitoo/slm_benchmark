"""The shared Inngest client for DocIE Studio.

Dev vs production is driven by ``INNGEST_DEV`` (the Inngest convention):
unset / ``0`` / ``false`` => production (signature verification on, needs
``INNGEST_SIGNING_KEY`` + ``INNGEST_EVENT_KEY``); anything else => dev mode
(points at the local dev server, no keys). Connection / gateway discovery is
driven by env (``INNGEST_BASE_URL``, ``INNGEST_CONNECT_GATEWAY_URL``) so the
same code runs locally and in Docker without edits.
"""

from __future__ import annotations

import logging
import os

import inngest

logger = logging.getLogger("docie_bench.inngest")

APP_ID = os.getenv("INNGEST_APP_ID", "docie-studio")


def _is_dev() -> bool:
    val = os.getenv("INNGEST_DEV", "").strip().lower()
    return val not in ("", "0", "false", "no")


# Realtime is an experimental part of the SDK and moves fast; importing it is
# best-effort so the worker still serves functions if it is unavailable.
try:  # pragma: no cover - import guard
    from inngest.experimental import realtime as _realtime
except Exception:  # noqa: BLE001
    _realtime = None
    logger.info("inngest.experimental.realtime unavailable; realtime publishing disabled")

_middleware: list = []
if _realtime is not None:
    _mw = getattr(_realtime, "RealtimeMiddleware", None)
    if _mw is not None:
        _middleware.append(_mw())


# Point the client at the self-hosted server. In production mode the SDK
# otherwise defaults to Inngest Cloud (api.inngest.com) and the connect/start +
# event POSTs 401. INNGEST_BASE_URL handles both the API (connect/sync) and the
# event-ingest host for our single-binary self-hosted server. In dev mode
# (INNGEST_DEV=1) leave it unset so the SDK auto-discovers the local dev server.
_base_url = os.getenv("INNGEST_BASE_URL", "").strip() or None

inngest_client = inngest.Inngest(
    app_id=APP_ID,
    is_production=not _is_dev(),
    logger=logger,
    middleware=_middleware or None,
    api_base_url=_base_url,
    event_api_base_url=_base_url,
)

__all__ = ["inngest_client", "APP_ID"]
