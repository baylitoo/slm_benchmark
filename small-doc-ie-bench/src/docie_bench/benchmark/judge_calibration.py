"""Judge <-> human calibration.

The LLM judge (``judge.py``) emits faithfulness/completeness scores that are
promoted to first-class regression metrics. Those scores are only trustworthy as
a *blocking* gate once we have measured how well they agree with human labels.

This module keeps that measurement honest and deterministic: it consumes a small
human-labeled calibration set (human scores paired with the judge's recorded
scores on the same documents), reports agreement (mean absolute error plus an
informational correlation), and decides whether the judge is calibrated enough to
block a regression. Computing the agreement needs no live model — the judge
scores are recorded ahead of time — so the gate is fully unit-testable.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import StatisticsError, correlation, mean
from typing import Any

# Reuse the same small-sample threshold the comparison gate warns at, so a
# calibration set below it can never certify the judge as a blocking gate. This
# counts REAL paired labels per dimension, not padded rows: a file can carry 30
# rows yet only a handful of dimensions actually labelled.
MIN_CALIBRATION_SAMPLES = 30
# The judge is trusted to BLOCK a regression only when its mean absolute error
# against human labels is at or below this on every scored dimension.
DEFAULT_MAX_JUDGE_MAE = 0.15
# Low MAE alone is satisfied by a near-constant judge (always ~0.8), which never
# actually tracks the human labels. Require a minimum judge<->human correlation
# per dimension so a constant/zero-variance judge (correlation None) cannot
# certify as a blocking gate.
MIN_CALIBRATION_CORRELATION = 0.3

JUDGE_DIMENSIONS = ("faithfulness", "completeness")

# Per-dimension failure reasons, in the order they gate (most fundamental first).
_DIMENSION_GATE_REASON = {
    "insufficient_pairs": "insufficient_calibration_samples",
    "agreement_below_threshold": "agreement_below_threshold",
    "correlation_below_threshold": "correlation_below_threshold",
}


def load_calibration(path: Path) -> list[dict[str, Any]]:
    """Load calibration records from ``{"records": [...]}`` or a bare list file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Calibration file must contain a records list: {path}")
    return [record for record in records if isinstance(record, dict)]


def _dimension_agreement(
    records: list[dict[str, Any]], dimension: str
) -> dict[str, float | int | None]:
    pairs = [
        (float(record[f"judge_{dimension}"]), float(record[f"human_{dimension}"]))
        for record in records
        if record.get(f"judge_{dimension}") is not None
        and record.get(f"human_{dimension}") is not None
    ]
    if not pairs:
        return {"n": 0, "mae": None, "correlation": None}
    judge_scores = [judge for judge, _human in pairs]
    human_scores = [human for _judge, human in pairs]
    mae = mean(abs(judge - human) for judge, human in pairs)
    try:
        # Undefined for < 2 points or zero variance (e.g. all-1.0 human labels).
        corr: float | None = correlation(judge_scores, human_scores)
    except (StatisticsError, ValueError):
        corr = None
    return {"n": len(pairs), "mae": mae, "correlation": corr}


def compute_judge_agreement(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report per-dimension judge<->human agreement over the calibration set."""
    dimensions = {
        dimension: _dimension_agreement(records, dimension) for dimension in JUDGE_DIMENSIONS
    }
    maes = [
        dimension["mae"]
        for dimension in dimensions.values()
        if dimension["mae"] is not None
    ]
    sample_count = len(records)
    return {
        "sample_count": sample_count,
        "dimensions": dimensions,
        # Gate on the worst-agreeing dimension so a good average can't hide a bad one.
        "worst_mae": max(maes) if maes else None,
        "small_sample": sample_count < MIN_CALIBRATION_SAMPLES,
    }


def _evaluate_dimension(
    dimension: dict[str, Any], *, max_mae: float, min_correlation: float
) -> dict[str, Any]:
    """Decide whether a single judged dimension is calibrated enough to block.

    Gates on the REAL paired-label count first, so a dimension with too few (or
    zero) pairs is never calibrated regardless of how the file is padded, then on
    agreement (MAE) and discriminative tracking (correlation).
    """
    n = int(dimension.get("n") or 0)
    mae = dimension.get("mae")
    corr = dimension.get("correlation")
    if n < MIN_CALIBRATION_SAMPLES:
        reason = "insufficient_pairs"
    elif mae is None or mae > max_mae:
        reason = "agreement_below_threshold"
    elif corr is None or corr < min_correlation:
        # None correlation == constant/zero-variance judge: no discriminative
        # agreement, so it may not block despite a small MAE.
        reason = "correlation_below_threshold"
    else:
        reason = "within_agreement"
    return {
        "n": n,
        "mae": mae,
        "correlation": corr,
        "calibrated": reason == "within_agreement",
        "reason": reason,
    }


def evaluate_calibration(
    report: dict[str, Any],
    *,
    max_mae: float = DEFAULT_MAX_JUDGE_MAE,
    min_correlation: float = MIN_CALIBRATION_CORRELATION,
) -> dict[str, Any]:
    """Decide whether the judge may BLOCK a regression.

    A dimension may block only when EVERY gate holds for it: at least
    ``MIN_CALIBRATION_SAMPLES`` real paired labels, worst-case MAE within
    ``max_mae``, and judge<->human correlation at least ``min_correlation``. The
    judge certifies as a blocking gate only when every scored dimension passes, so
    padding rows or a near-constant judge can never fail open into a block.
    """
    dimensions = report.get("dimensions") or {}
    per_dimension = {
        name: _evaluate_dimension(
            dimensions.get(name) or {"n": 0, "mae": None, "correlation": None},
            max_mae=max_mae,
            min_correlation=min_correlation,
        )
        for name in JUDGE_DIMENSIONS
    }
    base = {
        "sample_count": report.get("sample_count", 0),
        "worst_mae": report.get("worst_mae"),
        "max_mae": max_mae,
        "min_samples": MIN_CALIBRATION_SAMPLES,
        "min_correlation": min_correlation,
        "dimensions": per_dimension,
    }
    if all(dimension["calibrated"] for dimension in per_dimension.values()):
        return {**base, "calibrated": True, "reason": "within_agreement"}
    failing = {
        dimension["reason"]
        for dimension in per_dimension.values()
        if not dimension["calibrated"]
    }
    # Surface the most fundamental failure across dimensions.
    reason = next(
        gate_reason
        for dimension_reason, gate_reason in _DIMENSION_GATE_REASON.items()
        if dimension_reason in failing
    )
    return {**base, "calibrated": False, "reason": reason}


def calibration_gate(
    path: Path | None, *, max_mae: float = DEFAULT_MAX_JUDGE_MAE
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Resolve a calibration file into ``(report, gate)``.

    ``path is None`` (the default for a comparison) yields an uncalibrated gate so
    the judge cannot block a regression until agreement has been measured.
    """
    if path is None:
        gate = {
            "calibrated": False,
            "reason": "no_calibration_provided",
            "max_mae": max_mae,
            "min_samples": MIN_CALIBRATION_SAMPLES,
        }
        return None, gate
    report = compute_judge_agreement(load_calibration(path))
    return report, evaluate_calibration(report, max_mae=max_mae)
