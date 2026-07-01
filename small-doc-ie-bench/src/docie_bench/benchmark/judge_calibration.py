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
# calibration set below it can never certify the judge as a blocking gate.
MIN_CALIBRATION_SAMPLES = 30
# The judge is trusted to BLOCK a regression only when its mean absolute error
# against human labels is at or below this on every scored dimension.
DEFAULT_MAX_JUDGE_MAE = 0.15

JUDGE_DIMENSIONS = ("faithfulness", "completeness")


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


def evaluate_calibration(
    report: dict[str, Any], *, max_mae: float = DEFAULT_MAX_JUDGE_MAE
) -> dict[str, Any]:
    """Decide whether the judge may BLOCK a regression.

    Requires BOTH a large-enough set (``sample_count >= MIN_CALIBRATION_SAMPLES``)
    AND worst-dimension MAE within ``max_mae``. A tiny set can show spuriously high
    agreement, so it is never allowed to certify the judge.
    """
    worst_mae = report.get("worst_mae")
    base = {
        "sample_count": report.get("sample_count", 0),
        "worst_mae": worst_mae,
        "max_mae": max_mae,
        "min_samples": MIN_CALIBRATION_SAMPLES,
    }
    if report.get("small_sample", True):
        return {**base, "calibrated": False, "reason": "insufficient_calibration_samples"}
    if worst_mae is None:
        return {**base, "calibrated": False, "reason": "no_paired_labels"}
    calibrated = worst_mae <= max_mae
    reason = "within_agreement" if calibrated else "agreement_below_threshold"
    return {**base, "calibrated": calibrated, "reason": reason}


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
