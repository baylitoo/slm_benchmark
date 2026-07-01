from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import cache
from typing import Any

from rapidfuzz import fuzz


@dataclass(frozen=True)
class MetricConfig:
    string_similarity_threshold: float = 0.7  # soft: accept near-matches
    numeric_relative_tolerance: float = 0.001
    numeric_absolute_tolerance: float = 0.01


class ValidityGateError(RuntimeError):
    """Raised when a profile's valid_rate falls below the configured threshold.

    A run that silently scores zeros because every extraction was invalid (e.g.
    the empty-content defect) is a false negative; the gate turns it into a loud,
    actionable failure instead.
    """

    def __init__(self, threshold: float, failures: list[dict[str, Any]]) -> None:
        self.threshold = threshold
        self.failures = failures
        detail = ", ".join(
            f"{item['model_profile']} valid_rate={item['valid_rate']:.3f}" for item in failures
        )
        super().__init__(
            f"Validity gate failed: {len(failures)} profile(s) below valid_rate "
            f"threshold {threshold:.3f} ({detail})"
        )


def evaluate_validity_gate(
    summary_rows: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Return the summary rows whose ``valid_rate`` is below ``threshold``.

    A non-positive threshold disables the gate (returns ``[]``). Rows without a
    numeric ``valid_rate`` are skipped rather than treated as failures.
    """
    if threshold <= 0:
        return []
    failures: list[dict[str, Any]] = []
    for row in summary_rows:
        valid_rate = row.get("valid_rate")
        if not isinstance(valid_rate, (int, float)):
            continue
        if valid_rate < threshold:
            failures.append(row)
    return failures


def evaluate_constrained_gate(
    summary_rows: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Return summary rows whose ``constrained_rate`` fell below ``threshold``.

    Surfaces the constrained->unconstrained downgrade that post-repair
    ``valid_rate`` is blind to: a profile can score ``valid_rate=1.0`` while
    every strong (schema-constrained) decode was silently downgraded to a weaker
    rung or none+repair. A non-positive ``threshold`` disables the check
    (returns ``[]``); rows without a numeric ``constrained_rate`` (no comparable
    LLM-decoded row — e.g. OCR/pipeline or routed profiles) are skipped rather
    than treated as failures. Report-only: unlike ``valid_rate`` this never
    fails a run by default, so existing runs keep passing.
    """
    if threshold <= 0:
        return []
    failures: list[dict[str, Any]] = []
    for row in summary_rows:
        constrained_rate = row.get("constrained_rate")
        if not isinstance(constrained_rate, (int, float)):
            continue
        if constrained_rate < threshold:
            failures.append(row)
    return failures


def get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    if isinstance(current, dict) and "value" in current:
        return current.get("value")
    return current


def compare_values(expected: Any, actual: Any, cfg: MetricConfig) -> tuple[bool, str, float]:
    """Return (correct, reason, similarity).

    similarity is a continuous 0–1 score used for avg_similarity in aggregates.
    correct is the binary decision at cfg.string_similarity_threshold.
    """
    if expected is None:
        s = 1.0 if actual is None else 0.0
        return actual is None, "null", s
    if actual is None:
        return False, "missing", 0.0
    exp_num = _decimal_or_none(expected)
    act_num = _decimal_or_none(actual)
    if exp_num is not None and act_num is not None:
        diff = abs(exp_num - act_num)
        if diff <= Decimal(str(cfg.numeric_absolute_tolerance)):
            return True, "numeric_abs", 1.0
        denom = max(abs(exp_num), Decimal("1"))
        rel = float(diff / denom)
        if rel <= cfg.numeric_relative_tolerance:
            return True, "numeric_rel", 1.0
        # Partial credit for amounts: closer = higher score
        sim = max(0.0, 1.0 - min(rel, 1.0))
        return False, "numeric_mismatch", sim
    exp = str(expected).strip().casefold()
    act = str(actual).strip().casefold()
    if exp == act:
        return True, "exact", 1.0
    # Containment: verbatim extraction includes surrounding context.
    # "LYON" ∈ "LYON (69)"  or  "05HK12345" ∈ "N° 05HK12345"
    if exp in act or act in exp:
        return True, "contained", 1.0
    # Token-set ratio: ignores extra tokens and word order.
    token_score = fuzz.token_set_ratio(exp, act) / 100
    char_score = fuzz.ratio(exp, act) / 100
    best = max(token_score, char_score)
    if token_score >= cfg.string_similarity_threshold:
        return True, f"token_set:{token_score:.3f}", token_score
    if char_score >= cfg.string_similarity_threshold:
        return True, f"fuzzy:{char_score:.3f}", char_score
    return False, f"string_mismatch:{char_score:.3f}", best


def _decimal_or_none(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", "."))
    except (InvalidOperation, ValueError):
        return None


def score_prediction(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    cfg: MetricConfig | None = None,
    *,
    evidence_applicable: bool = True,
) -> dict[str, Any]:
    cfg = cfg or MetricConfig()
    rows: list[dict[str, Any]] = []
    table_scores: list[dict[str, Any]] = []
    correct = 0
    similarity_sum = 0.0
    for path, expected in ground_truth.items():
        actual = get_path(prediction, path)
        if isinstance(expected, list):
            table = score_table(path, expected, actual if isinstance(actual, list) else [], cfg)
            table_scores.append(table)
            rows.extend(table["cells"])
            correct += table["cell_correct"]
            similarity_sum += table["cell_similarity_sum"]
            continue
        ok, reason, sim = compare_values(expected, actual, cfg)
        correct += int(ok)
        similarity_sum += sim
        rows.append(
            {
                "field": path,
                "expected": expected,
                "actual": actual,
                "correct": ok,
                "similarity": round(sim, 4),
                "reason": reason,
            }
        )
    total = len(rows)
    row_expected = sum(table["row_expected"] for table in table_scores)
    row_predicted = sum(table["row_predicted"] for table in table_scores)
    row_correct = sum(table["row_correct"] for table in table_scores)
    row_precision = (
        row_correct / row_predicted if row_predicted else (1.0 if not row_expected else 0.0)
    )
    row_recall = row_correct / row_expected if row_expected else (1.0 if not row_predicted else 0.0)
    return {
        "field_total": total,
        "field_correct": correct,
        "field_accuracy": correct / total if total else None,
        "avg_similarity": round(similarity_sum / total, 4) if total else None,
        "fields": rows,
        "table_total": len(table_scores),
        "row_expected": row_expected,
        "row_predicted": row_predicted,
        "row_correct": row_correct,
        "row_precision": row_precision,
        "row_recall": row_recall,
        "row_f1": (
            2 * row_precision * row_recall / (row_precision + row_recall)
            if row_precision + row_recall
            else 0.0
        ),
        "tables": table_scores,
        **score_evidence(prediction, evidence_applicable=evidence_applicable),
    }


def score_table(
    path: str,
    expected_rows: list[Any],
    actual_rows: list[Any],
    cfg: MetricConfig,
) -> dict[str, Any]:
    """Align repeated rows by maximum cell similarity, then score aligned cells."""
    expected = [row for row in expected_rows if isinstance(row, dict)]
    actual = [row for row in actual_rows if isinstance(row, dict)]
    similarities = [
        [_row_similarity(expected_row, actual_row, cfg) for actual_row in actual]
        for expected_row in expected
    ]
    alignment = _align_rows(similarities)
    cells: list[dict[str, Any]] = []
    row_correct = 0
    similarity_sum = 0.0
    for expected_index, expected_row in enumerate(expected):
        actual_index = alignment.get(expected_index)
        actual_row = actual[actual_index] if actual_index is not None else {}
        aligned_correct = True
        for field, expected_value in _flatten_row(expected_row).items():
            actual_value = get_path(actual_row, field)
            ok, reason, similarity = compare_values(expected_value, actual_value, cfg)
            aligned_correct = aligned_correct and ok
            similarity_sum += similarity
            cells.append(
                {
                    "field": f"{path}.{expected_index}.{field}",
                    "expected": expected_value,
                    "actual": actual_value,
                    "correct": ok,
                    "similarity": round(similarity, 4),
                    "reason": reason,
                    "actual_row": actual_index,
                }
            )
        row_correct += int(aligned_correct and actual_index is not None)
    return {
        "field": path,
        "row_expected": len(expected),
        "row_predicted": len(actual),
        "row_correct": row_correct,
        "alignment": [
            {"expected_row": expected_index, "actual_row": actual_index}
            for expected_index, actual_index in sorted(alignment.items())
        ],
        "cell_total": len(cells),
        "cell_correct": sum(int(cell["correct"]) for cell in cells),
        "cell_similarity_sum": similarity_sum,
        "cells": cells,
    }


def _flatten_row(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_row(value, path))
        else:
            flattened[path] = value
    return flattened


def _row_similarity(expected: dict[str, Any], actual: dict[str, Any], cfg: MetricConfig) -> float:
    fields = _flatten_row(expected)
    if not fields:
        return 0.0
    similarity_sum = sum(
        compare_values(value, get_path(actual, path), cfg)[2] for path, value in fields.items()
    )
    return similarity_sum / len(fields)


def _align_rows(similarities: list[list[float]]) -> dict[int, int]:
    """Return one-to-one expected-to-actual alignment maximizing total similarity."""
    if not similarities or not similarities[0]:
        return {}
    expected_count = len(similarities)
    actual_count = len(similarities[0])
    if actual_count > 16:
        return _greedy_alignment(similarities)

    @cache
    def best(expected_index: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if expected_index == expected_count:
            return 0.0, ()
        best_score, best_pairs = best(expected_index + 1, used_mask)
        for actual_index in range(actual_count):
            bit = 1 << actual_index
            if used_mask & bit:
                continue
            tail_score, tail_pairs = best(expected_index + 1, used_mask | bit)
            candidate = similarities[expected_index][actual_index] + tail_score
            if candidate > best_score:
                best_score = candidate
                best_pairs = ((expected_index, actual_index), *tail_pairs)
        return best_score, best_pairs

    return dict(best(0, 0)[1])


def _greedy_alignment(similarities: list[list[float]]) -> dict[int, int]:
    candidates = sorted(
        (
            (score, expected_index, actual_index)
            for expected_index, row in enumerate(similarities)
            for actual_index, score in enumerate(row)
        ),
        reverse=True,
    )
    alignment: dict[int, int] = {}
    used_actual: set[int] = set()
    for _score, expected_index, actual_index in candidates:
        if expected_index not in alignment and actual_index not in used_actual:
            alignment[expected_index] = actual_index
            used_actual.add(actual_index)
    return alignment


def score_evidence(
    prediction: dict[str, Any], *, evidence_applicable: bool = True
) -> dict[str, Any]:
    fields = _evidence_fields(prediction)
    total = len(fields)
    evidence_rows = _evidence_rows(prediction)
    if not evidence_applicable:
        # Evidence grounding cites OCR blocks; the vision path has none, so grounding
        # is structurally impossible — report N/A rather than 100% "hallucinated".
        return {
            "evidence_applicable": False,
            "evidence_field_total": total,
            "evidence_grounded": 0,
            "evidence_coverage": None,
            "hallucination_rate": None,
            "ungrounded_fields": [],
            "evidence_row_total": len(evidence_rows),
            "evidence_rows_grounded": 0,
            "evidence_row_coverage": None,
            "ungrounded_rows": [],
        }
    grounded = [path for path, field in fields if field.get("evidence_ids")]
    ungrounded = [path for path, field in fields if not field.get("evidence_ids")]
    grounded_total = len(grounded)
    grounded_rows = [
        path
        for path, row_fields in evidence_rows
        if all(field.get("evidence_ids") for field in row_fields)
    ]
    return {
        "evidence_applicable": True,
        "evidence_field_total": total,
        "evidence_grounded": grounded_total,
        "evidence_coverage": grounded_total / total if total else None,
        "hallucination_rate": len(ungrounded) / total if total else None,
        "ungrounded_fields": ungrounded,
        "evidence_row_total": len(evidence_rows),
        "evidence_rows_grounded": len(grounded_rows),
        "evidence_row_coverage": len(grounded_rows) / len(evidence_rows) if evidence_rows else None,
        "ungrounded_rows": [path for path, _fields in evidence_rows if path not in grounded_rows],
    }


def _evidence_fields(obj: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    if isinstance(obj, list):
        fields: list[tuple[str, dict[str, Any]]] = []
        for index, item in enumerate(obj):
            fields.extend(_evidence_fields(item, f"{path}.{index}" if path else str(index)))
        return fields
    if not isinstance(obj, dict):
        return []
    if obj.get("value") is not None or obj.get("amount") is not None:
        return [(path, obj)]
    fields = []
    for key, value in obj.items():
        child_path = f"{path}.{key}" if path else key
        fields.extend(_evidence_fields(value, child_path))
    return fields


def _evidence_rows(obj: Any, path: str = "") -> list[tuple[str, list[dict[str, Any]]]]:
    rows: list[tuple[str, list[dict[str, Any]]]] = []
    if isinstance(obj, list):
        for index, item in enumerate(obj):
            row_path = f"{path}.{index}" if path else str(index)
            fields = [field for _field_path, field in _evidence_fields(item, row_path)]
            if fields:
                rows.append((row_path, fields))
            rows.extend(_evidence_rows(item, row_path))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            rows.extend(_evidence_rows(value, child_path))
    return rows
