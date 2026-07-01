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
    # Judge scores offset from human labels by a fixed error to control MAE.
    return [
        {
            "doc_id": f"doc-{i}",
            "human_faithfulness": 0.8,
            "judge_faithfulness": min(1.0, 0.8 + error),
            "human_completeness": 0.6,
            "judge_completeness": min(1.0, 0.6 + error),
        }
        for i in range(n)
    ]


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
