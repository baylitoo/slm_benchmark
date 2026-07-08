"""Provenance segmentation reducer.

Splits per-field scoring into *asserted* (printed on the document) vs *derived*
(computed from other labels, e.g. ``total_ttc.amount`` = subtotal + vat) buckets.

This is a pure downstream reduction over ``score["fields"]`` plus a parallel
``label_provenance`` sidecar. It deliberately lives OUTSIDE ``metrics.py`` so the
shared scoring module needs zero edits: ``summarize`` calls the aggregator and
merges the new (additive) keys into each summary row.

Invariant held *by construction* — every scored field is assigned to exactly one
bucket, so::

    asserted_total   + derived_total   == field_total
    asserted_correct + derived_correct == field_correct
"""

from __future__ import annotations

from typing import Any

ASSERTED = "asserted"
DERIVED = "derived"


def resolve_provenance(field: str, label_provenance: dict[str, str]) -> str:
    """Return the provenance bucket for a scored ``field`` dotted path.

    Matching precedence:

    1. Exact match on the full dotted path (handles scalars such as
       ``subtotal.amount``; also lets ``subtotal.amount`` win over a bare
       ``subtotal`` prefix).
    2. Longest dotted-prefix match where the prefix is followed by ``.`` in the
       field (table cells: ``line_items.0.line_total.amount`` -> ``line_items``).
       The dotted boundary prevents ``subtotal`` from swallowing a sibling like
       ``subtotal_note``.
    3. Default ``"asserted"`` — missing keys are backward-compatible.
    """
    exact = label_provenance.get(field)
    if exact is not None:
        return exact
    best_prefix = ""
    best_value = ASSERTED
    for key, value in label_provenance.items():
        if field.startswith(f"{key}.") and len(key) > len(best_prefix):
            best_prefix = key
            best_value = value
    return best_value


def segment_score_fields(
    fields: list[dict[str, Any]],
    label_provenance: dict[str, str],
) -> dict[str, int]:
    """Bucket one row's scored ``fields`` into asserted/derived correct+total.

    Single pass, one bucket per field, so the totals reconcile exactly with the
    row's ``field_total``/``field_correct``.
    """
    counts = {
        "asserted_total": 0,
        "asserted_correct": 0,
        "derived_total": 0,
        "derived_correct": 0,
    }
    for row in fields:
        resolved = resolve_provenance(row["field"], label_provenance)
        bucket = DERIVED if resolved == DERIVED else ASSERTED
        counts[f"{bucket}_total"] += 1
        counts[f"{bucket}_correct"] += int(bool(row.get("correct")))
    return counts


def aggregate_provenance_segments(ok_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate provenance segmentation across a profile's ``ok`` rows.

    Restricted to rows that carry a ``score`` (mirrors the existing
    ``field_total`` aggregation, which sums over ok rows only) so failed rows —
    which have no ``score["fields"]`` — never distort the buckets.

    Returns additive summary keys; ``field_accuracy_{asserted,derived}`` are
    ``None`` when the corresponding bucket is empty (guards 0/0 like the existing
    ``field_accuracy``).
    """
    asserted_total = asserted_correct = derived_total = derived_correct = 0
    for row in ok_rows:
        score = row.get("score") or {}
        fields = score.get("fields")
        if not fields:
            continue
        counts = segment_score_fields(fields, row.get("label_provenance") or {})
        asserted_total += counts["asserted_total"]
        asserted_correct += counts["asserted_correct"]
        derived_total += counts["derived_total"]
        derived_correct += counts["derived_correct"]
    return {
        "field_asserted_total": asserted_total,
        "field_asserted_correct": asserted_correct,
        "field_accuracy_asserted": (asserted_correct / asserted_total if asserted_total else None),
        "field_derived_total": derived_total,
        "field_derived_correct": derived_correct,
        "field_accuracy_derived": (derived_correct / derived_total if derived_total else None),
    }
