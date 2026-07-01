import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from docie_bench.benchmark.comparison import (
    compare_runs,
    list_baselines,
    promote_baseline,
    resolve_run,
)
from docie_bench.cli import app


def _write_metrics(path: Path, rows: list[dict]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    metrics = path / "metrics.json"
    metrics.write_text(json.dumps({"rows": rows, "summary": []}), encoding="utf-8")
    return metrics


def _row(
    doc_id: str,
    *,
    correct: bool,
    latency: float = 10,
    schema: str = "invoice",
    language: str = "en",
) -> dict:
    return {
        "doc_id": doc_id,
        "model_profile": "model",
        "schema_name": schema,
        "language": language,
        "ok": True,
        "latency_ms": latency,
        "validation": {"valid": True},
        "score": {
            "field_accuracy": float(correct),
            "avg_similarity": float(correct),
            "fields": [
                {
                    "field": "invoice_number",
                    "correct": correct,
                    "similarity": float(correct),
                }
            ],
        },
    }


def _write_budgets(path: Path, budgets: list[dict]) -> Path:
    path.write_text(yaml.safe_dump({"regression_budgets": budgets}), encoding="utf-8")
    return path


def _judge_row(doc_id: str, *, faithfulness: float) -> dict:
    return {
        "doc_id": doc_id,
        "model_profile": "model",
        "schema_name": "invoice",
        "language": "en",
        "ok": True,
        "latency_ms": 10,
        "validation": {"valid": True},
        "score": {},
        "judge": {
            "overall_faithfulness": faithfulness,
            "overall_completeness": faithfulness,
        },
    }


def _write_calibration(path: Path, *, count: int, error: float) -> Path:
    records = [
        {
            "doc_id": f"cal-{i}",
            "human_faithfulness": 0.8,
            "judge_faithfulness": min(1.0, 0.8 + error),
            "human_completeness": 0.6,
            "judge_completeness": min(1.0, 0.6 + error),
        }
        for i in range(count)
    ]
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    return path


def test_compare_reports_improvements_across_dimensions(tmp_path):
    baseline = _write_metrics(tmp_path / "baseline", [_row("one", correct=False)])
    candidate = _write_metrics(tmp_path / "candidate", [_row("one", correct=True)])

    result = compare_runs(baseline, candidate, output_dir=tmp_path / "out")
    comparison = json.loads(result.comparison_path.read_text(encoding="utf-8"))

    assert result.verdict == "pass"
    aggregate = next(
        item
        for item in comparison["comparisons"]
        if item["dimension"] == "aggregate" and item["metric"] == "field_accuracy"
    )
    assert aggregate["delta"] == 1.0
    assert aggregate["signed_improvement"] == 1.0
    assert "small_sample" in aggregate["warnings"]
    assert any(
        item["dimension"] == "schema_name" and item["group"] == {"schema_name": "invoice"}
        for item in comparison["comparisons"]
    )
    assert any(item["dimension"] == "field" for item in comparison["comparisons"])


def test_regression_budget_fails_and_identifies_root_causes(tmp_path):
    baseline = _write_metrics(tmp_path / "baseline", [_row("one", correct=True)])
    candidate = _write_metrics(tmp_path / "candidate", [_row("one", correct=False)])
    budgets = _write_budgets(
        tmp_path / "budgets.yaml",
        [{"name": "accuracy", "metric": "field_accuracy", "max_regression": 0.1}],
    )

    result = compare_runs(
        baseline, candidate, output_dir=tmp_path / "out", budgets_path=budgets
    )
    comparison = json.loads(result.comparison_path.read_text(encoding="utf-8"))

    assert result.verdict == "fail"
    assert result.exit_code == 1
    assert comparison["budget_checks"][0]["reason"] == "budget_exceeded"
    assert comparison["root_causes"]["documents"]
    assert comparison["root_causes"]["fields"]
    assert "Regressing Fields" in result.report_path.read_text(encoding="utf-8")


def test_lower_is_better_budget_handles_latency_increase(tmp_path):
    baseline = _write_metrics(tmp_path / "baseline", [_row("one", correct=True, latency=10)])
    candidate = _write_metrics(tmp_path / "candidate", [_row("one", correct=True, latency=25)])
    budgets = _write_budgets(
        tmp_path / "budgets.yaml",
        [{"name": "latency", "metric": "avg_latency_ms", "max_regression": 5}],
    )

    result = compare_runs(
        baseline, candidate, output_dir=tmp_path / "out", budgets_path=budgets
    )
    check = json.loads(result.verdict_path.read_text(encoding="utf-8"))["checks"][0]

    assert result.verdict == "fail"
    assert check["signed_improvement"] == -15


def test_partial_overlap_is_paired_and_warned(tmp_path):
    baseline = _write_metrics(
        tmp_path / "baseline", [_row("shared", correct=True), _row("old", correct=True)]
    )
    candidate = _write_metrics(
        tmp_path / "candidate", [_row("shared", correct=True), _row("new", correct=False)]
    )

    result = compare_runs(baseline, candidate, output_dir=tmp_path / "out")
    comparison = json.loads(result.comparison_path.read_text(encoding="utf-8"))
    aggregate = next(
        item
        for item in comparison["comparisons"]
        if item["dimension"] == "aggregate" and item["metric"] == "field_accuracy"
    )

    assert aggregate["paired_samples"] == 1
    assert aggregate["baseline_only"] == 1
    assert aggregate["candidate_only"] == 1
    assert "partial_overlap" in aggregate["warnings"]


def test_incompatible_runs_produce_error_verdict(tmp_path):
    baseline = _write_metrics(tmp_path / "baseline", [_row("old", correct=True)])
    candidate = _write_metrics(
        tmp_path / "candidate",
        [{**_row("new", correct=True), "model_profile": "different-model"}],
    )

    result = compare_runs(baseline, candidate, output_dir=tmp_path / "out")
    comparison = json.loads(result.comparison_path.read_text(encoding="utf-8"))

    assert result.verdict == "error"
    assert comparison["compatibility_errors"]


def test_missing_budget_metric_is_error_unless_allowed(tmp_path):
    baseline = _write_metrics(tmp_path / "baseline", [_row("one", correct=True)])
    candidate = _write_metrics(tmp_path / "candidate", [_row("one", correct=True)])
    budgets = _write_budgets(
        tmp_path / "budgets.yaml",
        [{"name": "missing", "metric": "judge_faithfulness", "max_regression": 0}],
    )

    result = compare_runs(
        baseline, candidate, output_dir=tmp_path / "out", budgets_path=budgets
    )

    assert result.verdict == "error"


def test_uncalibrated_judge_regression_does_not_block(tmp_path):
    # Same judge regression beyond budget; no calibration set -> non-blocking warn.
    baseline = _write_metrics(tmp_path / "baseline", [_judge_row("one", faithfulness=0.9)])
    candidate = _write_metrics(tmp_path / "candidate", [_judge_row("one", faithfulness=0.5)])
    budgets = _write_budgets(
        tmp_path / "budgets.yaml",
        [{"metric": "judge_faithfulness", "max_regression": 0, "min_paired_samples": 1}],
    )

    result = compare_runs(baseline, candidate, output_dir=tmp_path / "out", budgets_path=budgets)
    verdict = json.loads(result.verdict_path.read_text(encoding="utf-8"))

    assert result.verdict == "pass"
    assert verdict["judge_calibration"]["reason"] == "no_calibration_provided"
    assert verdict["checks"][0]["status"] == "warn"
    assert verdict["checks"][0]["reason"] == "judge_uncalibrated_non_blocking"


def test_calibrated_judge_regression_blocks(tmp_path):
    # Identical regression, but a large low-error calibration set lets the judge block.
    baseline = _write_metrics(tmp_path / "baseline", [_judge_row("one", faithfulness=0.9)])
    candidate = _write_metrics(tmp_path / "candidate", [_judge_row("one", faithfulness=0.5)])
    budgets = _write_budgets(
        tmp_path / "budgets.yaml",
        [{"metric": "judge_faithfulness", "max_regression": 0, "min_paired_samples": 1}],
    )
    calibration = _write_calibration(tmp_path / "calibration.json", count=30, error=0.05)

    result = compare_runs(
        baseline,
        candidate,
        output_dir=tmp_path / "out",
        budgets_path=budgets,
        calibration_path=calibration,
    )
    verdict = json.loads(result.verdict_path.read_text(encoding="utf-8"))

    assert result.verdict == "fail"
    assert verdict["judge_calibration"]["calibrated"] is True
    assert verdict["checks"][0]["status"] == "fail"
    assert verdict["checks"][0]["reason"] == "budget_exceeded"


def test_baseline_promotion_is_versioned_and_resolvable(tmp_path):
    metrics = _write_metrics(tmp_path / "run", [_row("one", correct=True)])
    registry = tmp_path / "registry"

    first = promote_baseline(metrics, "main", registry_dir=registry)
    second = promote_baseline(metrics, "main", registry_dir=registry)
    entries = list_baselines(registry)

    assert first["version"] != second["version"]
    assert len(entries) == 2
    assert entries[0]["current"] is True
    assert resolve_run("main", registry_dir=registry).is_file()
    assert resolve_run(f"main@{first['version']}", registry_dir=registry).is_file()


def test_compare_cli_returns_nonzero_and_writes_machine_verdict(tmp_path):
    baseline = _write_metrics(tmp_path / "baseline", [_row("one", correct=True)])
    candidate = _write_metrics(tmp_path / "candidate", [_row("one", correct=False)])
    budgets = _write_budgets(
        tmp_path / "budgets.yaml",
        [{"metric": "field_accuracy", "max_regression": 0}],
    )
    output = tmp_path / "comparison"

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "compare",
            str(baseline),
            str(candidate),
            "--budgets",
            str(budgets),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert json.loads((output / "verdict.json").read_text(encoding="utf-8"))["verdict"] == "fail"


def test_invalid_metrics_contract_is_rejected(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    candidate = _write_metrics(tmp_path / "candidate", [_row("one", correct=True)])

    with pytest.raises(ValueError, match="missing rows"):
        compare_runs(invalid, candidate, output_dir=tmp_path / "out")
