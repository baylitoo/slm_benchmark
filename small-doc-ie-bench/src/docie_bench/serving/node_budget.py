"""Where this node's soft memory budget comes from (#539).

The budget is DocIE's declared share of a machine it does not own. These are
shared hosts: free host RAM is not DocIE's to take, because memory that is free
now belongs to whichever co-tenant asks for it next, and the loser of that race
is decided by the OOM killer.

#621 made the budget a setting, which meant changing it restarted the
container and unloaded every resident model -- the opposite of what tuning a
budget is for. This makes it a catalog value an operator edits live.

PRECEDENCE, highest first:

1. an explicit override in the node's catalog row;
2. the node's bootstrap setting (``DOCIE_SERVING_MEM_BUDGET``);
3. nothing -- the hard cgroup ceiling alone, which is already a real limit
   where one exists.

READ COST. ``read_node_memory`` is called fresh at every sizing decision, by
design, so a database round trip per call is not affordable. A short in-process
TTL sits in front of the catalog: a live edit reaches the next decision after
at most ``_TTL_S``, which is well inside the reconcile interval that publishes
the consequences anyway.

OUTAGE. A catalog that cannot be read must never WIDEN the budget -- that is
the one direction where being wrong means admitting a model into memory a
co-tenant is about to need. So the last answer the catalog gave is persisted
beside the node's other durable state and used while it is unreachable, marked
stale. With no cached answer at all the bootstrap setting applies, which is a
configured number rather than host RAM; a node in that position has never
successfully read its own row.

A validated "no override" is itself an answer and is cached as one. Falling
back to the bootstrap setting on a NULL the catalog actually returned is
correct; doing it because the catalog could not be reached is a guess.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from docie_bench.serving.paths import serving_home

logger = logging.getLogger("docie_bench.serving.node_budget")

BUDGET_CACHE_FILENAME = "node_budget.json"

#: How long a catalog answer is reused before re-reading. Short enough that a
#: live edit lands on the next decision rather than the next deploy, long
#: enough that the fit check does not query Postgres per candidate.
_TTL_S = 5.0


@dataclass(frozen=True)
class BudgetResolution:
    """The budget in force, and where it came from."""

    budget_bytes: int | None
    source: str
    """``catalog`` | ``catalog_cache`` | ``setting`` | ``none``."""

    stale: bool = False
    """True when the catalog could not be read and a cached answer is in use."""

    @property
    def detail(self) -> str:
        if self.source == "catalog":
            return "operator override, read from the catalog"
        if self.source == "catalog_cache":
            return (
                "operator override, from the last answer the catalog gave -- "
                "the catalog is currently unreachable, and a budget is never "
                "widened on an outage"
            )
        if self.source == "setting":
            return "this node's DOCIE_SERVING_MEM_BUDGET bootstrap setting"
        return "no budget: the hard cgroup ceiling alone"


def budget_cache_path() -> Path:
    return serving_home() / BUDGET_CACHE_FILENAME


def read_budget_cache() -> int | None | str:
    """The last answer the catalog gave, or ``"unknown"`` when there is none.

    Three outcomes, and they are three different facts: an int is a validated
    override, ``None`` is a validated "no override", and ``"unknown"`` means
    this node has never read its own row. Collapsing the last two would make a
    node that cannot reach the catalog behave as though it had been told it has
    no budget.
    """
    try:
        raw = budget_cache_path().read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    try:
        decoded = json.loads(raw)
    except ValueError:
        logger.warning("node budget cache is not valid JSON; treating as unknown")
        return "unknown"
    if not isinstance(decoded, dict) or "budget_bytes" not in decoded:
        return "unknown"
    value = decoded["budget_bytes"]
    if value is None:
        return None
    return int(value) if isinstance(value, int) and value > 0 else "unknown"


def write_budget_cache(budget_bytes: int | None) -> None:
    """Persist the catalog's answer, atomically, so an outage is defined."""
    path = budget_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps({"budget_bytes": budget_bytes}, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def setting_budget_bytes() -> int | None:
    """The node's bootstrap budget setting, or None."""
    from docie_bench.settings import get_settings

    try:
        value = int(get_settings().serving_mem_budget_bytes)
    except Exception:  # noqa: BLE001 - an unreadable knob must not break sizing
        return None
    return value if value > 0 else None


# (key, resolution, monotonic deadline). Process-local by design: every
# process that sizes re-reads for itself, and a stale entry costs at most
# _TTL_S.
#
# KEYED ON THE SETTING, not just the node. The setting is one of the inputs,
# so an entry computed under a different one is not a stale answer to this
# question -- it is an answer to another one. In production the setting changes
# only across a restart, so this costs an lru-cached read; in tests it is the
# difference between a suite that passes in any order and one that depends on
# which test ran first.
_cached: tuple[tuple[str | None, int | None], BudgetResolution, float] | None = None


def reset_budget_cache() -> None:
    """Drop the in-process TTL entry. For tests, and after a known write."""
    global _cached
    _cached = None


def resolve_budget(
    *,
    node_id: str | None = None,
    clock: float | None = None,
) -> BudgetResolution:
    """The budget in force for this node, with its provenance.

    Cached for ``_TTL_S``. ``clock`` is injectable so a test can drive the TTL
    without sleeping.
    """
    global _cached
    now = clock if clock is not None else time.monotonic()
    key = (node_id, setting_budget_bytes())
    if _cached is not None and _cached[0] == key and now < _cached[2]:
        return _cached[1]
    resolution = _resolve_uncached(node_id)
    _cached = (key, resolution, now + _TTL_S)
    return resolution


def _resolve_uncached(node_id: str | None) -> BudgetResolution:
    from docie_bench.serving.catalog import ModelCatalog
    from docie_bench.serving.node_identity import resolve_node_id

    setting = setting_budget_bytes()
    try:
        name = node_id if node_id is not None else resolve_node_id()
        row = ModelCatalog().read_node_budget(name)
    except Exception:  # noqa: BLE001 - an unreachable catalog must not break sizing
        # ONE branch for every failure, and the CACHE decides -- not the
        # exception type. "No DATABASE_URL" and "Postgres is unreachable"
        # arrive as the same `CatalogUnavailableError` whenever something has
        # disposed the engine, so reading the error as "single-node stack,
        # the setting is the whole answer" would quietly widen the budget of a
        # fleet node during an outage.
        #
        # A cached answer exists only if this node has successfully read its
        # own row, which proves a catalog exists. That is the evidence worth
        # branching on.
        logger.debug("node budget read failed; using the last known answer", exc_info=True)
        cached = read_budget_cache()
        if cached == "unknown":
            # Never read its row: a single-node stack, or a node that has not
            # reached the catalog once. The setting is all there is.
            return _from_setting(setting)
        if cached is None:
            # Validated "no override", so the setting applies -- but flagged
            # stale, because the override may have been set since.
            return _from_setting(setting, stale=True)
        return BudgetResolution(
            budget_bytes=int(cached), source="catalog_cache", stale=True
        )

    # A node with no row has no override, and that is a validated answer: the
    # catalog answered, it simply has nothing for this machine yet.
    override = row.get("mem_budget_bytes") if row else None
    value = int(override) if isinstance(override, int) and override > 0 else None
    write_budget_cache(value)
    if value is not None:
        return BudgetResolution(budget_bytes=value, source="catalog")
    return _from_setting(setting)


def _from_setting(setting: int | None, *, stale: bool = False) -> BudgetResolution:
    if setting is None:
        return BudgetResolution(budget_bytes=None, source="none", stale=stale)
    return BudgetResolution(budget_bytes=setting, source="setting", stale=stale)


__all__ = [
    "BUDGET_CACHE_FILENAME",
    "BudgetResolution",
    "budget_cache_path",
    "read_budget_cache",
    "reset_budget_cache",
    "resolve_budget",
    "setting_budget_bytes",
    "write_budget_cache",
]
