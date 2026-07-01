from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from docie_bench.benchmark.judge_calibration import DEFAULT_MAX_JUDGE_MAE, calibration_gate
from docie_bench.benchmark.reproducibility import atomic_write_json, atomic_write_text

# Judge scores are only trusted to BLOCK a regression once calibrated against
# human labels (see judge_calibration). Uncalibrated, they may only warn.
JUDGE_METRICS = {"judge_faithfulness", "judge_completeness"}
HIGHER_IS_BETTER = {
    "ok_rate",
    "valid_rate",
    "field_accuracy",
    "avg_similarity",
    "evidence_coverage",
    "judge_faithfulness",
    "judge_completeness",
    "throughput_docs_per_min",
}
LOWER_IS_BETTER = {
    "hallucination_rate",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
}
DIMENSIONS = ("aggregate", "model_profile", "schema_name", "language", "document", "field")


@dataclass(frozen=True)
class Observation:
    key: str
    dimensions: dict[str, str]
    metrics: dict[str, float]


@dataclass(frozen=True)
class ComparisonResult:
    verdict: str
    exit_code: int
    comparison_path: Path
    verdict_path: Path
    report_path: Path


def compare_runs(
    baseline: Path,
    candidate: Path,
    *,
    output_dir: Path,
    budgets_path: Path | None = None,
    calibration_path: Path | None = None,
    max_judge_mae: float = DEFAULT_MAX_JUDGE_MAE,
) -> ComparisonResult:
    baseline_path = _metrics_path(baseline)
    candidate_path = _metrics_path(candidate)
    baseline_metrics = _load_json(baseline_path)
    candidate_metrics = _load_json(candidate_path)
    baseline_observations = _observations(baseline_metrics)
    candidate_observations = _observations(candidate_metrics)
    comparisons = _compare_observations(baseline_observations, candidate_observations)
    budgets = _load_budgets(budgets_path)
    # Gap (a): an uncalibrated judge must not block a release. Measure judge<->human
    # agreement (default: no calibration -> judge budgets only warn).
    calibration_report, judge_calibration = calibration_gate(
        calibration_path, max_mae=max_judge_mae
    )
    checks = _evaluate_budgets(
        comparisons, budgets, judge_calibrated=bool(judge_calibration["calibrated"])
    )
    failures = [check for check in checks if check["status"] == "fail"]
    errors = [check for check in checks if check["status"] == "error"]
    incompatible = not comparisons
    verdict = "error" if errors or incompatible else ("fail" if failures else "pass")
    exit_code = 0 if verdict == "pass" else 1

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": _source_metadata(baseline_path),
        "candidate": _source_metadata(candidate_path),
        "verdict": verdict,
        "comparisons": comparisons,
        "budget_checks": checks,
        "judge_calibration": {"gate": judge_calibration, "report": calibration_report},
        "compatibility_errors": (
            ["No comparable matched observations were found"] if incompatible else []
        ),
        "root_causes": _root_causes(comparisons),
    }
    verdict_payload = {
        "contract_version": 1,
        "verdict": verdict,
        "exit_code": exit_code,
        "checks": checks,
        "judge_calibration": judge_calibration,
        "failed_checks": failures,
        "error_checks": errors,
        "compatibility_errors": (
            ["No comparable matched observations were found"] if incompatible else []
        ),
    }
    comparison_path = output_dir / "comparison.json"
    verdict_path = output_dir / "verdict.json"
    report_path = output_dir / "comparison.md"
    atomic_write_json(comparison_path, payload, indent=2)
    atomic_write_json(verdict_path, verdict_payload, indent=2)
    atomic_write_text(report_path, _markdown_report(payload))
    return ComparisonResult(
        verdict,
        exit_code,
        comparison_path,
        verdict_path,
        report_path,
    )


def promote_baseline(run: Path, name: str, *, registry_dir: Path) -> dict[str, Any]:
    if not name or any(part in name for part in ("/", "\\", "..")):
        raise ValueError("Baseline name must be a non-empty path-safe name")
    metrics_path = _metrics_path(run)
    content = metrics_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    promoted_at = datetime.now(UTC)
    version = promoted_at.strftime("%Y%m%dT%H%M%S%fZ") + "-" + digest[:8]
    target_dir = registry_dir / name / version
    target_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(metrics_path, target_dir / "metrics.json")
    entry = {
        "name": name,
        "version": version,
        "promoted_at": promoted_at.isoformat(),
        "source": str(metrics_path.resolve()),
        "sha256": digest,
        "metrics_path": str(Path(name) / version / "metrics.json"),
    }
    registry = _load_registry(registry_dir)
    registry.setdefault("baselines", {}).setdefault(name, []).append(entry)
    registry["baselines"][name][-1]["current"] = True
    for old in registry["baselines"][name][:-1]:
        old["current"] = False
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return entry


def list_baselines(registry_dir: Path) -> list[dict[str, Any]]:
    registry = _load_registry(registry_dir)
    return [
        entry
        for versions in registry.get("baselines", {}).values()
        for entry in reversed(versions)
    ]


def resolve_run(value: str, *, registry_dir: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    name, _, requested_version = value.partition("@")
    versions = _load_registry(registry_dir).get("baselines", {}).get(name, [])
    matches = [
        entry
        for entry in versions
        if not requested_version or entry["version"] == requested_version
    ]
    if not matches:
        raise ValueError(f"Run or named baseline not found: {value}")
    metrics_path = Path(matches[-1]["metrics_path"])
    return metrics_path if metrics_path.is_absolute() else registry_dir / metrics_path


def _metrics_path(run: Path) -> Path:
    path = run / "metrics.json" if run.is_dir() else run
    if not path.is_file():
        raise ValueError(f"Metrics file not found: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise ValueError(f"Incompatible metrics file (missing rows): {path}")
    return value


def _source_metadata(path: Path) -> dict[str, str]:
    content = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()}


def _observations(metrics: dict[str, Any]) -> list[Observation]:
    observations: list[Observation] = []
    for row_index, row in enumerate(metrics["rows"]):
        profile = str(row.get("model_profile", "unknown"))
        doc_id = str(row.get("doc_id", f"row-{row_index}"))
        schema = str(row.get("schema_name", "unknown"))
        language = str(row.get("language") or "unknown")
        base_dimensions = {
            "model_profile": profile,
            "schema_name": schema,
            "language": language,
            "document": doc_id,
        }
        doc_metrics = {
            "ok_rate": float(bool(row.get("ok"))),
            "valid_rate": float(bool(row.get("validation", {}).get("valid"))),
            "avg_latency_ms": float(row.get("latency_ms", 0)),
        }
        score = row.get("score") or {}
        for metric in ("evidence_coverage", "hallucination_rate"):
            if score.get(metric) is not None:
                doc_metrics[metric] = float(score[metric])
        judge = row.get("judge") or {}
        for metric in ("judge_faithfulness", "judge_completeness"):
            source_key = metric.removeprefix("judge_")
            if judge.get(f"overall_{source_key}") is not None:
                doc_metrics[metric] = float(judge[f"overall_{source_key}"])
        observations.append(
            Observation(f"document:{profile}:{doc_id}", base_dimensions, doc_metrics)
        )
        for field in score.get("fields", []):
            field_name = str(field.get("field", "unknown"))
            field_metrics = {
                "field_accuracy": float(bool(field.get("correct"))),
                "avg_similarity": float(field.get("similarity", 0)),
            }
            observations.append(
                Observation(
                    f"field:{profile}:{doc_id}:{field_name}",
                    {**base_dimensions, "field": field_name},
                    field_metrics,
                )
            )
    return observations


def _compare_observations(
    baseline: list[Observation], candidate: list[Observation]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        baseline_groups = _group_observations(baseline, dimension)
        candidate_groups = _group_observations(candidate, dimension)
        for group_key in sorted(set(baseline_groups) | set(candidate_groups)):
            base_by_key = {obs.key: obs for obs in baseline_groups.get(group_key, [])}
            cand_by_key = {obs.key: obs for obs in candidate_groups.get(group_key, [])}
            all_metrics = {
                metric
                for obs in (*base_by_key.values(), *cand_by_key.values())
                for metric in obs.metrics
            }
            for metric in sorted(all_metrics):
                paired_keys = sorted(
                    key
                    for key in set(base_by_key) & set(cand_by_key)
                    if metric in base_by_key[key].metrics and metric in cand_by_key[key].metrics
                )
                if not paired_keys:
                    continue
                baseline_values = [base_by_key[key].metrics[metric] for key in paired_keys]
                candidate_values = [cand_by_key[key].metrics[metric] for key in paired_keys]
                deltas = [
                    candidate - base
                    for base, candidate in zip(baseline_values, candidate_values, strict=True)
                ]
                delta = mean(deltas)
                direction = "higher" if metric in HIGHER_IS_BETTER else "lower"
                warnings = []
                if len(paired_keys) < 30:
                    warnings.append("small_sample")
                baseline_metric_keys = {
                    key for key, observation in base_by_key.items() if metric in observation.metrics
                }
                candidate_metric_keys = {
                    key for key, observation in cand_by_key.items() if metric in observation.metrics
                }
                unmatched_baseline = len(baseline_metric_keys - candidate_metric_keys)
                unmatched_candidate = len(candidate_metric_keys - baseline_metric_keys)
                if unmatched_baseline or unmatched_candidate:
                    warnings.append("partial_overlap")
                results.append(
                    {
                        "dimension": dimension,
                        "group": _group_payload(dimension, group_key),
                        "metric": metric,
                        "direction": direction,
                        "baseline": mean(baseline_values),
                        "candidate": mean(candidate_values),
                        "delta": delta,
                        "signed_improvement": delta if direction == "higher" else -delta,
                        "paired_samples": len(paired_keys),
                        "baseline_only": unmatched_baseline,
                        "candidate_only": unmatched_candidate,
                        "confidence_interval_95": _confidence_interval(deltas),
                        "sign_test_p_value": _sign_test(deltas),
                        "warnings": warnings,
                    }
                )
    return results


def _group_observations(
    observations: list[Observation], dimension: str
) -> dict[str, list[Observation]]:
    groups: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        if dimension == "aggregate":
            key = "all"
        elif dimension == "field":
            if "field" not in observation.dimensions:
                continue
            key = observation.dimensions["field"]
        else:
            key = observation.dimensions[dimension]
        groups[key].append(observation)
    return groups


def _group_payload(dimension: str, key: str) -> dict[str, str]:
    return {} if dimension == "aggregate" else {dimension: key}


def _confidence_interval(deltas: list[float]) -> list[float]:
    center = mean(deltas)
    if len(deltas) < 2:
        return [center, center]
    margin = 1.96 * stdev(deltas) / math.sqrt(len(deltas))
    return [center - margin, center + margin]


def _sign_test(deltas: list[float]) -> float | None:
    positive = sum(delta > 0 for delta in deltas)
    negative = sum(delta < 0 for delta in deltas)
    n = positive + negative
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(positive, negative) + 1)) / (2**n)
    return float(min(1.0, 2 * tail))


def _load_budgets(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    budgets = payload.get("regression_budgets", payload.get("budgets", []))
    if not isinstance(budgets, list):
        raise ValueError("Budget config must contain a regression_budgets list")
    return budgets


def _evaluate_budgets(
    comparisons: list[dict[str, Any]],
    budgets: list[dict[str, Any]],
    *,
    judge_calibrated: bool = True,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for index, budget in enumerate(budgets):
        name = str(budget.get("name", f"budget-{index + 1}"))
        metric = budget.get("metric")
        dimension = budget.get("dimension", "aggregate")
        selector = budget.get("selector", {})
        max_regression = float(budget.get("max_regression", budget.get("max_drop", 0)))
        min_paired = int(budget.get("min_paired_samples", 1))
        matches = [
            item
            for item in comparisons
            if item["metric"] == metric
            and item["dimension"] == dimension
            and all(item["group"].get(key) == str(value) for key, value in selector.items())
        ]
        if not matches:
            checks.append(
                {
                    "name": name,
                    "status": "pass" if budget.get("allow_missing", False) else "error",
                    "reason": "no_matching_comparison",
                }
            )
            continue
        for item in matches:
            enough_samples = item["paired_samples"] >= min_paired
            within_budget = item["signed_improvement"] >= -max_regression
            passed = enough_samples and within_budget
            status = "pass" if passed else "fail"
            reason = (
                "within_budget"
                if passed
                else ("insufficient_paired_samples" if not enough_samples else "budget_exceeded")
            )
            # Gap (a): an uncalibrated judge may flag but not block. Downgrade a
            # judge-metric failure to a non-blocking warning; leave passes alone.
            if not passed and metric in JUDGE_METRICS and not judge_calibrated:
                status = "warn"
                reason = "judge_uncalibrated_non_blocking"
            checks.append(
                {
                    "name": name,
                    "status": status,
                    "metric": metric,
                    "dimension": dimension,
                    "group": item["group"],
                    "signed_improvement": item["signed_improvement"],
                    "max_regression": max_regression,
                    "paired_samples": item["paired_samples"],
                    "min_paired_samples": min_paired,
                    "reason": reason,
                }
            )
    return checks


def _root_causes(comparisons: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    regressions = [item for item in comparisons if item["signed_improvement"] < 0]
    return {
        "documents": sorted(
            (item for item in regressions if item["dimension"] == "document"),
            key=lambda item: item["signed_improvement"],
        )[:20],
        "fields": sorted(
            (item for item in regressions if item["dimension"] == "field"),
            key=lambda item: item["signed_improvement"],
        )[:20],
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Comparison",
        "",
        f"**Verdict:** `{payload['verdict'].upper()}`",
        "",
        f"- Baseline: `{payload['baseline']['path']}`",
        f"- Candidate: `{payload['candidate']['path']}`",
        "",
    ]
    if payload["compatibility_errors"]:
        lines.extend(
            [
                "## Compatibility Errors",
                "",
                *[f"- {error}" for error in payload["compatibility_errors"]],
                "",
            ]
        )
    lines.extend(
        [
        "## Budget Checks",
        "",
        "| Status | Name | Metric | Group | Reason |",
        "|---|---|---|---|---|",
        ]
    )
    for check in payload["budget_checks"]:
        lines.append(
            f"| {check['status'].upper()} | {check['name']} | {check.get('metric', '-')} | "
            f"`{json.dumps(check.get('group', {}), sort_keys=True)}` | {check['reason']} |"
        )
    if not payload["budget_checks"]:
        lines.append("| PASS | No budgets configured | - | `{}` | comparison only |")
    lines.extend(
        [
            "",
            "## Aggregate Deltas",
            "",
            "| Metric | Baseline | Candidate | Delta | N | Warnings |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in payload["comparisons"]:
        if item["dimension"] == "aggregate":
            lines.append(
                f"| {item['metric']} | {item['baseline']:.4f} | {item['candidate']:.4f} | "
                f"{item['delta']:+.4f} | {item['paired_samples']} | "
                f"{', '.join(item['warnings']) or '-'} |"
            )
    for title, key in (("Regressing Documents", "documents"), ("Regressing Fields", "fields")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Group | Metric | Signed improvement | N |",
                "|---|---|---:|---:|",
            ]
        )
        for item in payload["root_causes"][key]:
            lines.append(
                f"| `{json.dumps(item['group'], sort_keys=True)}` | {item['metric']} | "
                f"{item['signed_improvement']:+.4f} | {item['paired_samples']} |"
            )
        if not payload["root_causes"][key]:
            lines.append("| - | - | 0 | 0 |")
    return "\n".join(lines) + "\n"


def _load_registry(registry_dir: Path) -> dict[str, Any]:
    path = registry_dir / "registry.json"
    if not path.exists():
        return {"contract_version": 1, "baselines": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid baseline registry: {path}")
    return value
