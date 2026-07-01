import json
from pathlib import Path

import pytest

from docie_bench.benchmark.judge_calibration import (
    MIN_CALIBRATION_SAMPLES,
    calibration_gate,
    compute_judge_agreement,
    evaluate_calibration,
    load_calibration,
)


def _records(n: int, *, error: float) -> list[dict[str, float]]:
    # Human labels spread monotonically across the 0-1 range so correlation is
    # well-defined and positive; the judge tracks them at a fixed offset to
    # control MAE independently.
    records = []
    for i in range(n):
        span = i / max(n - 1, 1)
        human_faithfulness = round(0.5 + 0.4 * span, 4)
        human_completeness = round(0.4 + 0.4 * span, 4)
        records.append(
            {
                "doc_id": f"doc-{i}",
                "human_faithfulness": human_faithfulness,
                "judge_faithfulness": round(min(1.0, human_faithfulness + error), 4),
                "human_completeness": human_completeness,
                "judge_completeness": round(min(1.0, human_completeness + error), 4),
            }
        )
    return records


def test_agreement_reports_per_dimension_mae_and_correlation():
    records = [
        {"judge_faithfulness": 0.9, "human_faithfulness": 1.0,
         "judge_completeness": 0.5, "human_completeness": 0.7},
        {"judge_faithfulness": 0.6, "human_faithfulness": 0.5,
         "judge_completeness": 0.8, "human_completeness": 0.9},
    ]

    report = compute_judge_agreement(records)

    assert report["sample_count"] == 2
    assert report["dimensions"]["faithfulness"]["mae"] == pytest.approx(0.1)
    assert report["dimensions"]["completeness"]["mae"] == pytest.approx(0.15)
    assert report["worst_mae"] == pytest.approx(0.15)


def test_correlation_undefined_on_zero_variance_does_not_crash():
    # A curated set with constant human labels has zero variance; correlation is
    # undefined and must degrade to None rather than raising.
    records = [
        {"judge_faithfulness": 0.9, "human_faithfulness": 1.0},
        {"judge_faithfulness": 0.7, "human_faithfulness": 1.0},
    ]

    report = compute_judge_agreement(records)

    assert report["dimensions"]["faithfulness"]["correlation"] is None
    assert report["dimensions"]["faithfulness"]["mae"] == pytest.approx(0.2)


def test_small_calibration_set_is_never_blocking():
    report = compute_judge_agreement(_records(5, error=0.0))
    gate = evaluate_calibration(report)

    assert report["small_sample"] is True
    assert gate["calibrated"] is False
    assert gate["reason"] == "insufficient_calibration_samples"


def test_large_low_error_set_certifies_the_judge():
    report = compute_judge_agreement(_records(MIN_CALIBRATION_SAMPLES, error=0.05))
    gate = evaluate_calibration(report, max_mae=0.15)

    assert report["small_sample"] is False
    assert gate["calibrated"] is True
    assert gate["reason"] == "within_agreement"


def test_large_high_error_set_is_not_calibrated():
    report = compute_judge_agreement(_records(MIN_CALIBRATION_SAMPLES, error=0.4))
    gate = evaluate_calibration(report, max_mae=0.15)

    assert gate["calibrated"] is False
    assert gate["reason"] == "agreement_below_threshold"


def test_padded_file_without_per_dimension_pairs_does_not_certify():
    # 30 rows, but only 3 real faithfulness pairs and ZERO completeness pairs. A
    # gate keyed off len(records)/worst_mae would fail open and certify the judge;
    # the per-dimension pair count must catch it.
    records = [
        {"doc_id": f"faith-{i}", "human_faithfulness": 0.8, "judge_faithfulness": 0.85}
        for i in range(3)
    ]
    records += [{"doc_id": f"pad-{i}"} for i in range(27)]

    report = compute_judge_agreement(records)
    gate = evaluate_calibration(report)

    assert report["dimensions"]["faithfulness"]["n"] == 3
    assert report["dimensions"]["completeness"]["n"] == 0
    assert gate["calibrated"] is False
    assert gate["reason"] == "insufficient_calibration_samples"
    assert gate["dimensions"]["faithfulness"]["reason"] == "insufficient_pairs"
    assert gate["dimensions"]["completeness"]["reason"] == "insufficient_pairs"


def test_near_constant_judge_is_not_calibrated_despite_low_mae():
    # Enough pairs and a tiny MAE, but the judge sits on a constant so its
    # correlation with the human labels is undefined — no discriminative
    # agreement, so it may not block.
    records = [
        {
            "doc_id": f"doc-{i}",
            "human_faithfulness": 0.8,
            "judge_faithfulness": 0.82,
            "human_completeness": 0.6,
            "judge_completeness": 0.62,
        }
        for i in range(MIN_CALIBRATION_SAMPLES)
    ]

    report = compute_judge_agreement(records)
    gate = evaluate_calibration(report)

    assert report["dimensions"]["faithfulness"]["n"] == MIN_CALIBRATION_SAMPLES
    assert report["dimensions"]["faithfulness"]["mae"] == pytest.approx(0.02)
    assert report["dimensions"]["faithfulness"]["correlation"] is None
    assert gate["calibrated"] is False
    assert gate["reason"] == "correlation_below_threshold"


def test_missing_calibration_path_yields_non_blocking_gate():
    report, gate = calibration_gate(None)

    assert report is None
    assert gate["calibrated"] is False
    assert gate["reason"] == "no_calibration_provided"


def test_load_calibration_accepts_records_wrapper(tmp_path: Path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"records": [{"judge_faithfulness": 0.9, "human_faithfulness": 0.9}]}),
        encoding="utf-8",
    )

    assert load_calibration(path) == [{"judge_faithfulness": 0.9, "human_faithfulness": 0.9}]


def test_load_calibration_rejects_non_list(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"records": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="records list"):
        load_calibration(path)


def test_cli_judge_calibration_reports_gate(tmp_path: Path):
    from typer.testing import CliRunner

    from docie_bench.cli import app

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"records": _records(4, error=0.0)}), encoding="utf-8")

    result = CliRunner().invoke(app, ["benchmark", "judge-calibration", str(path)])

    assert result.exit_code == 0, result.output
    assert "insufficient_calibration_samples" in result.output
    assert "non-blocking" in result.output
