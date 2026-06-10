from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from rapidfuzz import fuzz


@dataclass(frozen=True)
class MetricConfig:
    string_similarity_threshold: float = 0.7  # soft: accept near-matches
    numeric_relative_tolerance: float = 0.001
    numeric_absolute_tolerance: float = 0.01


def get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
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
) -> dict[str, Any]:
    cfg = cfg or MetricConfig()
    rows: list[dict[str, Any]] = []
    correct = 0
    similarity_sum = 0.0
    for path, expected in ground_truth.items():
        actual = get_path(prediction, path)
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
    return {
        "field_total": total,
        "field_correct": correct,
        "field_accuracy": correct / total if total else None,
        "avg_similarity": round(similarity_sum / total, 4) if total else None,
        "fields": rows,
        **score_evidence(prediction),
    }


def score_evidence(prediction: dict[str, Any]) -> dict[str, Any]:
    fields = _evidence_fields(prediction)
    grounded = [path for path, field in fields if field.get("evidence_ids")]
    ungrounded = [path for path, field in fields if not field.get("evidence_ids")]
    total = len(fields)
    grounded_total = len(grounded)
    return {
        "evidence_field_total": total,
        "evidence_grounded": grounded_total,
        "evidence_coverage": grounded_total / total if total else None,
        "hallucination_rate": len(ungrounded) / total if total else None,
        "ungrounded_fields": ungrounded,
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
