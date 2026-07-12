"""Sizing engine for the serving control plane (PR-3, design doc §3).

Answers one question per store model: **"how many MORE instances fit right
now?"** — and prices hypothetical deployment mixes (what-if) with exactly the
same math, so the number the UI shows is the number the deploy path's fit gate
would enforce.

Inputs are the observed surfaces the reconciler publishes (design doc §3): the
store models (``ModelCatalog.list``), the live observed placements, and the
single ``serving_node`` snapshot. The engine itself is pure — it measures
nothing and mutates nothing — so the api can serve it and tests can drive it
deterministically.

Footprint per candidate instance = the PR-2 tracker's calibrated working
footprint::

    footprint(X) = max(observed_steady_rss(X), predicted(X))
    predicted(X) = weights + kv_cache(ctx, n_parallel) + overhead (+ mmproj)

with weights from ``ModelStoreEntry.size_bytes`` (or an on-disk stat) and the
calibration read from the ``FootprintStore`` sidecars on the shared serving
volume — never the registry (design doc fix #6).

**The double-count trap, and the guard (design doc §3).** The snapshot's
``free_bytes`` is a *measured* number (cgroup working-set or psutil
``available``): the RSS of every RUNNING llama-server is **already inside
"used"**. Subtracting predicted footprints of running deployments from that
free number would price them twice, halving the apparent capacity. So this
engine prices **prospective instances only** against the measured free —
observed placements are consumed for display (running-instance counts) and are
deliberately NEVER re-subtracted. That choice is stated here once and asserted
by tests; the UI states it too.

Fit for a candidate model X (design doc §3)::

    fits_now(X) = floor( (free - safety_margin) / footprint(X) )

``safety_margin`` is explicit and configurable (default ~10% of total),
surfaced in every payload — an honest buffer, never a hidden fudge factor.

Honesty rules:
* no node snapshot (reconciler never ran / DB empty) => footprints are still
  priced, but ``fits_now`` is ``None`` and ``observed_available`` is False —
  never a locally-measured or fabricated free number.
* unpriceable model (no ``size_bytes``, unreadable GGUF, never calibrated) =>
  ``footprint_bytes=None`` + a reason, never a pretend number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docie_bench.serving.resources import (
    DEFAULT_CONTEXT_LENGTH,
    FootprintStore,
    footprint_bytes,
    predicted_footprint_for_model,
)

# Default safety margin as a fraction of node total RAM (design doc §3 brackets
# 10-15%; take the low end — the margin is visible, not padding on padding).
DEFAULT_MARGIN_FRACTION = 0.10

# Phases whose process is resident right now (RSS already inside "used").
LIVE_PHASES = frozenset({"hot", "loading"})


class UnknownModelError(ValueError):
    """A what-if plan names a model that is not in the store."""


class UnpriceableModelError(ValueError):
    """A what-if plan names a model whose footprint cannot be priced."""


@dataclass(frozen=True)
class ModelFit:
    """One fit-table row: how a store model prices and how many more fit."""

    name: str
    family: str | None
    predicted_bytes: int | None
    calibrated_bytes: int | None  # steady-state RSS sidecar, None = never run
    footprint_bytes: int | None  # max(calibrated, predicted); None = unpriceable
    running_instances: int  # display only — NEVER re-subtracted (double-count guard)
    fits_now: int | None  # None when unpriceable or no snapshot
    detail: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "predicted_bytes": self.predicted_bytes,
            "calibrated_bytes": self.calibrated_bytes,
            "calibrated": self.calibrated_bytes is not None,
            "footprint_bytes": self.footprint_bytes,
            "running_instances": self.running_instances,
            "fits_now": self.fits_now,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SizingReport:
    """The GET /v1/serving/sizing payload body (engine half)."""

    observed_available: bool
    total_bytes: int | None
    free_bytes: int | None
    source: str | None  # "cgroup" | "vm" — the soft-number badge input
    safety_margin_bytes: int | None
    free_effective_bytes: int | None  # free - margin (may be negative: honest)
    margin_fraction: float
    context_length: int
    n_parallel: int
    per_model: tuple[ModelFit, ...]
    detail: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_available": self.observed_available,
            "detail": self.detail,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "source": self.source,
            "safety_margin_bytes": self.safety_margin_bytes,
            "free_effective_bytes": self.free_effective_bytes,
            "assumptions": {
                "context_length": self.context_length,
                "n_parallel": self.n_parallel,
                "margin_fraction": self.margin_fraction,
            },
            "per_model": [fit.as_dict() for fit in self.per_model],
        }


@dataclass(frozen=True)
class WhatIfItem:
    """One priced line of a what-if plan."""

    model: str
    instances: int
    context_length: int
    footprint_bytes: int  # per instance
    subtotal_bytes: int  # footprint * instances
    calibrated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "instances": self.instances,
            "context_length": self.context_length,
            "footprint_bytes": self.footprint_bytes,
            "subtotal_bytes": self.subtotal_bytes,
            "calibrated": self.calibrated,
        }


@dataclass(frozen=True)
class WhatIfReport:
    """The POST /v1/serving/sizing/whatif payload body (engine half)."""

    observed_available: bool
    total_predicted_bytes: int
    free_effective_bytes: int | None
    safety_margin_bytes: int | None
    remaining_bytes: int | None  # free_effective - total_predicted
    ok: bool | None  # None = no snapshot to judge against (honest, not False)
    deficit_bytes: int | None  # >0 iff ok is False; how much RAM is missing
    margin_fraction: float
    per_item: tuple[WhatIfItem, ...]
    detail: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_available": self.observed_available,
            "detail": self.detail,
            "total_predicted_bytes": self.total_predicted_bytes,
            "free_effective_bytes": self.free_effective_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "remaining_bytes": self.remaining_bytes,
            "ok": self.ok,
            "deficit_bytes": self.deficit_bytes,
            "margin_fraction": self.margin_fraction,
            "per_item": [item.as_dict() for item in self.per_item],
        }


# ------------------------------------------------------------------- pricing


def safety_margin_bytes(total_bytes: int, margin_fraction: float) -> int:
    """The explicit headroom slice: a fraction of node TOTAL (not of free)."""
    return int(total_bytes * margin_fraction)


def _mmproj_bytes(row: Mapping[str, Any]) -> int:
    """Size of the store entry's vision projector (0 when none/unreadable).

    llama-server loads the projector fully resident for vision families, so a
    candidate instance must be priced with it (mmproj-aware footprint — same
    rule as the reconciler's restart fit gate). Unreadable degrades to 0: the
    fit table under-counting a projector beats refusing to price the model.
    """
    mmproj_path = row.get("mmproj_path")
    if not mmproj_path:
        return 0
    try:
        return Path(str(mmproj_path)).stat().st_size
    except OSError:
        return 0


def price_model(
    row: Mapping[str, Any],
    *,
    footprints: FootprintStore,
    context_length: int | None,
    n_parallel: int = 1,
) -> tuple[int | None, int | None, int | None]:
    """(predicted, calibrated, working footprint) for one store row.

    ``predicted`` from the PR-2 formula (store ``size_bytes`` or on-disk stat,
    mmproj-aware); ``calibrated`` from the steady-state RSS sidecar keyed by
    the store entry's launch model path; working footprint =
    ``max(calibrated, predicted)`` — the tracker's calibration rule. All three
    are ``None``-honest: a model with no known weights AND no calibration is
    unpriceable, never zero.
    """
    model_path = row.get("model_path")
    predicted = predicted_footprint_for_model(
        size_bytes=row.get("size_bytes"),
        model_path=str(model_path) if model_path else None,
        context_length=context_length,
        n_parallel=n_parallel,
        mmproj_bytes=_mmproj_bytes(row),
    )
    calibrated = footprints.get(str(model_path)) if model_path else None
    if predicted is None and calibrated is None:
        return None, None, None
    working = footprint_bytes(predicted if predicted is not None else 0, calibrated)
    return predicted, calibrated, working


def _running_instances(
    placements: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Live (hot/loading) instance count per store model — DISPLAY only.

    The double-count guard in one function: these placements' RSS is already
    inside the snapshot's "used", so the counts inform the operator but are
    never subtracted from free again.
    """
    counts: dict[str, int] = {}
    for placement in placements:
        model_name = placement.get("model_name")
        if model_name and placement.get("phase") in LIVE_PHASES:
            counts[model_name] = counts.get(model_name, 0) + 1
    return counts


def _check_margin(margin_fraction: float) -> None:
    if not 0.0 <= margin_fraction < 1.0:
        raise ValueError("margin_fraction must be in [0, 1)")


def _free_budget(
    snapshot: Mapping[str, Any] | None, margin_fraction: float
) -> tuple[int | None, int | None, int | None, int | None, str | None]:
    """(total, free, margin, free_effective, source) from the node snapshot.

    ``free_bytes`` is taken VERBATIM from the snapshot (cgroup working-set
    adjusted, or psutil ``available``) — the measurement already nets out
    running processes, which is exactly why nothing else may be subtracted for
    them (module docstring). ``free_effective`` may go negative when the node
    is over the margin: an honest red number beats a clamped zero.
    """
    if snapshot is None:
        return None, None, None, None, None
    total = int(snapshot["total_bytes"])
    free = int(snapshot["free_bytes"])
    margin = safety_margin_bytes(total, margin_fraction)
    return total, free, margin, free - margin, str(snapshot["source"])


# -------------------------------------------------------------------- engine


def compute_sizing(
    models: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any] | None,
    placements: Sequence[Mapping[str, Any]] = (),
    *,
    footprints: FootprintStore | None = None,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
    context_length: int | None = None,
    n_parallel: int = 1,
) -> SizingReport:
    """The per-model fit table: ``fits_now = floor(free_effective / footprint)``.

    ``models``/``placements``/``snapshot`` are the catalog's view dicts
    (``ModelCatalog.list`` / ``list_placements`` / ``get_node_snapshot``) so
    sizing reads the same observed surface the Board does. ``snapshot=None``
    degrades honestly: footprints still price, ``fits_now`` stays ``None``.
    """
    _check_margin(margin_fraction)
    store = footprints if footprints is not None else FootprintStore()
    context = context_length if context_length is not None else DEFAULT_CONTEXT_LENGTH
    total, free, margin, free_effective, source = _free_budget(snapshot, margin_fraction)
    running = _running_instances(placements)

    fits_rows: list[ModelFit] = []
    for row in models:
        name = str(row["name"])
        predicted, calibrated, working = price_model(
            row, footprints=store, context_length=context, n_parallel=n_parallel
        )
        detail: str | None = None
        fits_now: int | None = None
        if working is None:
            detail = (
                "unpriceable: no size_bytes in the store, the GGUF is not "
                "readable here, and no calibration has been observed"
            )
        elif free_effective is None:
            detail = "no node snapshot: footprint priced, fit unknown"
        else:
            # floor of an honest (possibly negative) budget, never below 0.
            fits_now = max(free_effective // working, 0)
        fits_rows.append(
            ModelFit(
                name=name,
                family=(str(row["family"]) if row.get("family") else None),
                predicted_bytes=predicted,
                calibrated_bytes=calibrated,
                footprint_bytes=working,
                running_instances=running.get(name, 0),
                fits_now=fits_now,
                detail=detail,
            )
        )

    return SizingReport(
        observed_available=snapshot is not None,
        total_bytes=total,
        free_bytes=free,
        source=source,
        safety_margin_bytes=margin,
        free_effective_bytes=free_effective,
        margin_fraction=margin_fraction,
        context_length=context,
        n_parallel=n_parallel,
        per_model=tuple(fits_rows),
        detail=None if snapshot is not None else "no node snapshot published",
    )


def compute_whatif(
    models: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any] | None,
    plan: Sequence[Mapping[str, Any]],
    *,
    footprints: FootprintStore | None = None,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
    n_parallel: int = 1,
) -> WhatIfReport:
    """Price a hypothetical deployment mix against the live free budget.

    ``plan`` items: ``{"model": <store name>, "instances": N,
    "context_length": ctx | None}``. Same footprint math as the fit table
    (max(calibrated, predicted), mmproj-aware) so the what-if answer matches
    what the deploy path's fit gate would decide. Raises
    :class:`UnknownModelError` for a model not in the store and
    :class:`UnpriceableModelError` when an item cannot be priced — a plan sum
    with silent zero-priced lines would be a lie, not an estimate.
    """
    _check_margin(margin_fraction)
    store = footprints if footprints is not None else FootprintStore()
    by_name = {str(row["name"]): row for row in models}
    _, _, margin, free_effective, _ = _free_budget(snapshot, margin_fraction)

    items: list[WhatIfItem] = []
    for entry in plan:
        model = str(entry["model"])
        row = by_name.get(model)
        if row is None:
            raise UnknownModelError(f"unknown store model {model!r}")
        instances = int(entry.get("instances", 1))
        if instances < 1:
            raise ValueError(f"instances must be >= 1 (got {instances} for {model!r})")
        raw_context = entry.get("context_length")
        context = int(raw_context) if raw_context else DEFAULT_CONTEXT_LENGTH
        _, calibrated, working = price_model(
            row, footprints=store, context_length=context, n_parallel=n_parallel
        )
        if working is None:
            raise UnpriceableModelError(
                f"cannot price {model!r}: no size_bytes in the store, no readable "
                f"GGUF, and no observed calibration"
            )
        items.append(
            WhatIfItem(
                model=model,
                instances=instances,
                context_length=context,
                footprint_bytes=working,
                subtotal_bytes=working * instances,
                calibrated=calibrated is not None,
            )
        )

    total_predicted = sum(item.subtotal_bytes for item in items)
    remaining: int | None = None
    ok: bool | None = None
    deficit: int | None = None
    if free_effective is not None:
        remaining = free_effective - total_predicted
        ok = remaining >= 0
        deficit = max(-remaining, 0)
    return WhatIfReport(
        observed_available=snapshot is not None,
        total_predicted_bytes=total_predicted,
        free_effective_bytes=free_effective,
        safety_margin_bytes=margin,
        remaining_bytes=remaining,
        ok=ok,
        deficit_bytes=deficit,
        margin_fraction=margin_fraction,
        per_item=tuple(items),
        detail=None if snapshot is not None else "no node snapshot published",
    )


__all__ = [
    "DEFAULT_MARGIN_FRACTION",
    "LIVE_PHASES",
    "ModelFit",
    "SizingReport",
    "UnknownModelError",
    "UnpriceableModelError",
    "WhatIfItem",
    "WhatIfReport",
    "compute_sizing",
    "compute_whatif",
    "price_model",
    "safety_margin_bytes",
]
