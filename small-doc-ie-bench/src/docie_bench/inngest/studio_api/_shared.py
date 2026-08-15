"""Constants, response models, and helpers shared across the Studio API's
route submodules.

``default_run_store`` and ``MODELS_CONFIG_PATH`` are deliberately kept as
module-level names ONLY here, never re-imported by name (``from .foo import
X``) into another submodule -- every other submodule accesses them as
``_shared.default_run_store()`` / ``_shared.MODELS_CONFIG_PATH`` (attribute
access on this module object) instead. ``from X import Y`` binds a SEPARATE
local name at import time that a later ``monkeypatch.setattr(_shared, "Y",
...)`` can't reach; attribute access re-reads this module's own ``__dict__``
on every call, so one patch site here covers every route that touches the
run store or the models-config path, no matter which submodule it lives in
after the split (see tests/test_studio_artifacts.py's
``monkeypatch.setattr(studio_api._shared, ...)`` calls).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from docie_bench.inngest.realtime import (
    TOPIC_ERROR,
    TOPIC_PROGRESS,
    TOPIC_RESULT,
    TOPIC_STATUS,
)
from docie_bench.studio.store import RunStoreUnavailableError, default_run_store

__all__ = [
    "_DOMAIN_404",
    "DEFAULT_TOPICS",
    "EXTRACT_EVENT",
    "BENCHMARK_EVENT",
    "DEPLOY_EVENT",
    "SEED_EVENT",
    "MODELS_CONFIG_PATH",
    "TriggerResponse",
    "RunStoreUnavailableError",
    "default_run_store",
    "_record_event_owners",
    "_annotate_fits",
]

# Domain "not found" marker (see serving_api._DOMAIN_404): without it the
# Studio renders these 404s as "endpoint unavailable" and drops the detail.
_DOMAIN_404 = {"X-Docie-Error": "not_found"}

DEFAULT_TOPICS = [TOPIC_STATUS, TOPIC_PROGRESS, TOPIC_RESULT, TOPIC_ERROR]
EXTRACT_EVENT = "doc/extract.requested"
BENCHMARK_EVENT = "benchmark/run.requested"
DEPLOY_EVENT = "serving/deploy.requested"
SEED_EVENT = "serving/seed.requested"

# Same file the gateway (serving.gateway.DEFAULT_MODELS_CONFIG) and the benchmark
# worker (inngest.functions.MODELS_CONFIG_PATH) read -- this module's own constant
# rather than importing theirs, matching the existing convention (each of those two
# already duplicates the literal independently; see also serving.profile_resolver).
MODELS_CONFIG_PATH = Path("configs/models.yaml")


class TriggerResponse(BaseModel):
    event_ids: list[str]
    channel: str
    topics: list[str]


def _record_event_owners(event_ids: list[str], tenant_id: str) -> None:
    """Bind each triggered event id to its principal so ``/runs`` can be scoped.

    Best-effort: if the run index is unconfigured the run-status proxy stays
    unscoped in that degraded mode (see ``runs.event_runs``).
    """
    store = default_run_store()
    if not store.enabled:
        return
    try:
        for event_id in event_ids:
            store.record_event_owner(event_id=event_id, tenant_id=tenant_id)
    except RunStoreUnavailableError:
        pass


def _annotate_fits(cards: list[dict[str, Any]], budget: int | None) -> list[dict[str, Any]]:
    """Stamp ``fits_node`` (bool | None) + ``node_available_bytes`` on each card.

    ``None`` when the estimate or the budget is unknown -- the badge reads
    "unknown", never a false "fits". Pure so the fit logic is tested without a
    node snapshot / database."""
    for card in cards:
        est = card.get("size_est_bytes")
        card["node_available_bytes"] = budget
        card["fits_node"] = (
            est <= budget if isinstance(est, int) and isinstance(budget, int) else None
        )
    return cards
