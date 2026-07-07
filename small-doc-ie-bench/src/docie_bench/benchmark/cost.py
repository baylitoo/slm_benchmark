"""Honest three-unit cost accounting for the benchmark.

Three DISTINCT units are tracked and NEVER averaged together:

1. abstract routing ``cost_units`` — a routing-budget unit (not money), owned by
   the router and surfaced only in the routing sub-table. Untouched here.
2. local ``tokens`` — real token usage from ``response.usage`` for LLM profiles.
   Populated for any profile whose upstream reports usage; N/A for DL adapters
   (Donut/docTR emit no tokens).
3. paid ``usd_per_doc`` — real money, computed ONLY for profiles that carry a
   ``pricing`` block. Local/DL profiles show N/A (never ``$0``) in the $ column.

The cost guard protects against runaway paid spend: a paid profile refuses to
run without a ``--cost-ceiling`` or a ``--dry-run-cost`` upper-bound estimate.
The estimate is a conservative UPPER bound (worst-case ``max_tokens`` output),
intended to gate spend, not to forecast it. Keys are never referenced here — only
profile names and dollar amounts appear in the estimate and its log lines.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from docie_bench.llm.model_profiles import ModelProfile

logger = logging.getLogger(__name__)

# Conservative per-document input-token estimate used for the pre-flight upper
# bound and as a fallback when a hosted response omits usage. Intentionally
# generous — the guard is a ceiling, not a forecast.
DEFAULT_EST_INPUT_TOKENS = 2000


class CostGuardError(RuntimeError):
    """A paid profile was selected without a cost ceiling or dry-run estimate."""


class CostCeilingExceeded(RuntimeError):  # noqa: N818 - reads as an event, mirrors design name
    """The pre-flight upper-bound estimate exceeds the configured cost ceiling."""

    def __init__(self, estimate: CostEstimate, ceiling: float) -> None:
        self.estimate = estimate
        self.ceiling = ceiling
        super().__init__(
            f"Estimated upper-bound cost ${estimate.total_usd:.4f} exceeds "
            f"cost ceiling ${ceiling:.4f}; raise --cost-ceiling or reduce the run"
        )


@dataclass(frozen=True)
class ProfileCostEstimate:
    profile_name: str
    docs: int
    est_input_tokens: int
    est_output_tokens: int
    usd: float


@dataclass(frozen=True)
class CostEstimate:
    total_usd: float
    per_profile: tuple[ProfileCostEstimate, ...]


def estimate_run_cost(
    profiles: Iterable[ModelProfile],
    doc_count: int,
    *,
    est_input_tokens: int = DEFAULT_EST_INPUT_TOKENS,
) -> CostEstimate:
    """Conservative UPPER-bound cost of running ``doc_count`` docs on paid profiles.

    Pure: performs no I/O and makes no upstream call. Only profiles with a
    ``pricing`` block contribute; unpriced local/DL/hosted profiles cost $0 here.
    Output tokens are charged at the profile's ``max_tokens`` cap (worst case) and
    input at ``est_input_tokens`` (deliberately generous) so the estimate is a
    ceiling to gate spend, never an under-count.
    """
    per: list[ProfileCostEstimate] = []
    for profile in profiles:
        pricing = getattr(profile, "pricing", None)
        if pricing is None:
            continue
        in_tokens = est_input_tokens * doc_count
        out_tokens = profile.max_tokens * doc_count
        usd = pricing.usd_for(in_tokens, out_tokens)
        # Log profile NAME and $ only — never the api_key.
        logger.info("cost_estimate profile=%s docs=%d usd=%.6f", profile.name, doc_count, usd)
        per.append(
            ProfileCostEstimate(
                profile_name=profile.name,
                docs=doc_count,
                est_input_tokens=in_tokens,
                est_output_tokens=out_tokens,
                usd=round(usd, 6),
            )
        )
    total = round(sum(item.usd for item in per), 6)
    return CostEstimate(total_usd=total, per_profile=tuple(per))


def enforce_cost_guard(
    profiles: Sequence[ModelProfile],
    doc_count: int,
    cost_ceiling: float | None,
) -> CostEstimate | None:
    """Refuse a paid run without a ceiling; abort if the estimate exceeds it.

    Returns the pre-flight estimate when a paid profile is present and cleared,
    or ``None`` when no selected profile is paid (nothing to guard). Raises
    ``CostGuardError`` (no ceiling) or ``CostCeilingExceeded`` (estimate over
    ceiling) BEFORE any task runs, so a real run never bills unexpectedly.
    """
    paid = [profile for profile in profiles if getattr(profile, "is_paid", False)]
    if not paid:
        return None
    if cost_ceiling is None:
        names = ", ".join(sorted(profile.name for profile in paid))
        raise CostGuardError(
            f"Paid profile(s) selected ({names}) but no spend guard set. Re-run with "
            "--cost-ceiling USD (aborts if the pre-flight estimate exceeds it) or "
            "--dry-run-cost (prints an upper-bound estimate and makes no API call)."
        )
    estimate = estimate_run_cost(paid, doc_count)
    if estimate.total_usd > cost_ceiling:
        raise CostCeilingExceeded(estimate, cost_ceiling)
    return estimate


def row_total_tokens(row: Mapping[str, Any]) -> int | None:
    """Total tokens for one prediction row, or ``None`` when usage is absent.

    Reads the ``usage`` captured from ``response.usage``. Falls back to
    prompt+completion when ``total_tokens`` is missing; ``None`` (not 0) when no
    usable usage exists so DL adapters leave the tokens average rather than
    dragging it to zero.
    """
    usage = row.get("usage")
    if not isinstance(usage, Mapping):
        return None
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)):
        return int(total)
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None and completion is None:
        return None
    return int(prompt or 0) + int(completion or 0)


@dataclass(frozen=True)
class ProfileCostSummary:
    avg_tokens: float | None
    avg_cost_usd_per_doc: float | None
    cost_estimated: bool


def summarize_profile_cost(
    rows: Sequence[Mapping[str, Any]],
    pricing: Any | None,
) -> ProfileCostSummary:
    """Fold per-row token usage into ``avg_tokens`` and (paid only) ``$``/doc.

    ``avg_tokens`` is computed for any profile that reported usage (local tokens
    unit). ``avg_cost_usd_per_doc`` is populated ONLY when ``pricing`` is set —
    otherwise it stays ``None`` (N/A, never $0). If a paid row lacks a usable
    token split, the per-doc cost falls back to a counted/estimated split and the
    whole summary is flagged ``cost_estimated`` so callers can mark it approximate
    rather than silently reporting $0.
    """
    token_values = [tokens for row in rows if (tokens := row_total_tokens(row)) is not None]
    avg_tokens = round(sum(token_values) / len(token_values), 1) if token_values else None
    if pricing is None:
        return ProfileCostSummary(
            avg_tokens=avg_tokens, avg_cost_usd_per_doc=None, cost_estimated=False
        )

    costs: list[float] = []
    estimated = False
    for row in rows:
        usage = row.get("usage")
        prompt: float | None = None
        completion: float | None = None
        if isinstance(usage, Mapping):
            raw_prompt = usage.get("prompt_tokens")
            raw_completion = usage.get("completion_tokens")
            prompt = float(raw_prompt) if isinstance(raw_prompt, (int, float)) else None
            completion = float(raw_completion) if isinstance(raw_completion, (int, float)) else None
        if prompt is None or completion is None:
            estimated = True
            total = row_total_tokens(row)
            if prompt is None:
                prompt = float(DEFAULT_EST_INPUT_TOKENS)
            if completion is None:
                completion = max(0.0, float(total) - prompt) if total is not None else 0.0
        costs.append(pricing.usd_for(prompt, completion))
    avg_cost = round(sum(costs) / len(costs), 6) if costs else None
    return ProfileCostSummary(
        avg_tokens=avg_tokens, avg_cost_usd_per_doc=avg_cost, cost_estimated=estimated
    )
