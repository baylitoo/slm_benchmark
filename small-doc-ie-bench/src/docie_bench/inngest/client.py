"""The shared Inngest clients for DocIE Studio.

Two apps, one environment (PR-1, design doc §0.1/P1):

* ``inngest_client``  — app id ``docie-studio`` (``INNGEST_APP_ID``): the
  scaled ``worker`` fleet's app (extraction / benchmark / GC).
* ``serving_client``  — app id ``docie-serving`` (``INNGEST_SERVING_APP_ID``):
  the single-replica ``serving`` service's app (deploy / seed / delete).

Why two app ids instead of one app with per-role function subsets: an Inngest
app's function set is (re)registered on every worker sync — a Connect
handshake sends ``AppConfiguration{app_name, functions}`` as the app's
authoritative function list, exactly like an HTTP sync. Two fleets syncing
DISJOINT function sets under one app id therefore overwrite each other's
registration on every (re)connect: functions flap between registered and
archived, and an event may be routed while its function is deregistered. The
SDK's ``connect()`` API is explicitly multi-app (``apps: list[(client,
functions)]``) so split-role fleets register one app per role; event delivery
is unaffected because events route by NAME within the environment (the
event/signing keys), not by app.

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
from types import ModuleType
from typing import Any

import inngest

logger = logging.getLogger("docie_bench.inngest")

APP_ID = os.getenv("INNGEST_APP_ID", "docie-studio")
SERVING_APP_ID = os.getenv("INNGEST_SERVING_APP_ID", "docie-serving")


def _is_dev() -> bool:
    val = os.getenv("INNGEST_DEV", "").strip().lower()
    return val not in ("", "0", "false", "no")


# Realtime is an experimental part of the SDK and moves fast; importing it is
# best-effort so the worker still serves functions if it is unavailable.
try:  # pragma: no cover - import guard
    from inngest.experimental import realtime

    _realtime: ModuleType | None = realtime
except Exception:  # noqa: BLE001
    _realtime = None
    logger.info("inngest.experimental.realtime unavailable; realtime publishing disabled")


# Point the client at the self-hosted server. In production mode the SDK
# otherwise defaults to Inngest Cloud (api.inngest.com) and the connect/start +
# event POSTs 401. INNGEST_BASE_URL handles both the API (connect/sync) and the
# event-ingest host for our single-binary self-hosted server. In dev mode
# (INNGEST_DEV=1) leave it unset so the SDK auto-discovers the local dev server.
_base_url = os.getenv("INNGEST_BASE_URL", "").strip() or None


def _build_client(app_id: str) -> inngest.Inngest:
    """One configured client per app id (fresh middleware per client)."""
    middleware: list[Any] = []
    if _realtime is not None:
        realtime_middleware = getattr(_realtime, "RealtimeMiddleware", None)
        if realtime_middleware is not None:
            middleware.append(realtime_middleware())
    return inngest.Inngest(
        app_id=app_id,
        is_production=not _is_dev(),
        logger=logger,
        middleware=middleware or None,
        api_base_url=_base_url,
        event_api_base_url=_base_url,
    )


inngest_client = _build_client(APP_ID)
serving_client = _build_client(SERVING_APP_ID)

__all__ = ["inngest_client", "serving_client", "APP_ID", "SERVING_APP_ID"]
