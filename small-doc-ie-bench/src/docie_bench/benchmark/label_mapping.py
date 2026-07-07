"""Declarative annotation -> ground-truth + provenance mapping.

A dataset onboards by writing a YAML spec (see
``scripts/label_mapping/*.yaml``) instead of a bespoke build script. The spec
declares, for each label, its ``source`` path in the raw annotation, the dotted
``target`` path in ``ground_truth``, a ``transform``, and its ``provenance``
(``asserted`` = printed on the document, ``derived`` = computed from other
labels). ``apply_mapping`` is the generic engine reused by every dataset; it is
the sole *producer* of the ``label_provenance`` sidecar that
``benchmark/provenance.py`` later segments on.

Precedence rule: when both a printed total and a derivable subtotal+tax exist,
the asserted (printed) mapping wins and the ``derived`` rule is a fallback — it
only fires for targets not already asserted. Reconciliation between the two is an
audit concern (``label_audit.py``), never a label overwrite here.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

TRANSFORMS = ("identity", "normalize_amount", "to_iso_date", "normalize_quantity")
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d.%m.%Y",
    "%Y/%m/%d",
)


class MappingError(ValueError):
    """Raised when a mapping spec is structurally invalid."""


def _get_source(annotation: Any, path: str) -> Any:
    """Read a dotted path from a raw annotation (dict/list), None if absent."""
    current: Any = annotation
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
        if current is None:
            return None
    return current


def _normalize_amount(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(Decimal(str(value).replace(",", ".").strip())), 2)
    except (InvalidOperation, ValueError):
        return None


def _normalize_quantity(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(Decimal(str(value).replace(",", ".").strip()))
    except (InvalidOperation, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _to_iso_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()  # noqa: DTZ007
        except ValueError:
            continue
    return None


def _apply_transform(name: str, value: Any) -> Any:
    if name == "identity":
        return value
    if name == "normalize_amount":
        return _normalize_amount(value)
    if name == "normalize_quantity":
        return _normalize_quantity(value)
    if name == "to_iso_date":
        return _to_iso_date(value)
    raise MappingError(f"Unknown transform {name!r}; expected one of {TRANSFORMS}")


def _set_dotted(target_map: dict[str, Any], dotted: str, value: Any) -> None:
    target_map[dotted] = value


def _apply_derived(
    rule: dict[str, Any],
    ground_truth: dict[str, Any],
) -> tuple[str, Any] | None:
    """Evaluate one derived rule over already-computed asserted targets.

    Skips (returns None) rather than crashing when inputs are missing, mirroring
    the historical subtotal+vat -> subtotal-only -> nothing fallback. The first
    input is the anchor: if it is absent the whole target is skipped; otherwise
    the formula runs over whichever inputs are present.
    """
    target = rule["target"]
    formula = rule.get("formula", "sum")
    inputs = rule.get("inputs", [])
    if not inputs:
        raise MappingError(f"Derived rule for {target!r} needs a non-empty 'inputs' list")
    values = [ground_truth.get(key) for key in inputs]
    if values[0] is None:
        return None  # anchor missing -> no derivation (skip, do not crash)
    present: list[float] = [
        amount for v in values if v is not None and (amount := _normalize_amount(v)) is not None
    ]
    if not present:
        return None
    if formula == "sum":
        return target, round(sum(present), 2)
    raise MappingError(f"Unknown derived formula {formula!r} for target {target!r}")


def apply_mapping(
    annotation: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Map a raw ``annotation`` to ``(ground_truth, label_provenance)`` per ``spec``.

    ``ground_truth`` is a flat dotted-path -> value map (line-item tables become a
    ``list[dict]`` under a single target key, matching ``score_prediction``).
    ``label_provenance`` is keyed by the SAME dotted paths (table targets use the
    list's key; ``benchmark/provenance.py`` prefix-buckets the per-cell paths).
    """
    ground_truth: dict[str, Any] = {}
    label_provenance: dict[str, str] = {}

    for mapping in spec.get("mappings", []):
        source = mapping["source"]
        target = mapping["target"]
        provenance = mapping.get("provenance", "asserted")
        raw = _get_source(annotation, source)
        value = _apply_transform(mapping.get("transform", "identity"), raw)
        if value is None:
            continue
        _set_dotted(ground_truth, target, value)
        label_provenance[target] = provenance

    for table in spec.get("tables", []):
        source_rows = _get_source(annotation, table["source"])
        if not isinstance(source_rows, list):
            continue
        target = table["target"]
        provenance = table.get("provenance", "asserted")
        columns = table.get("columns", [])
        built_rows: list[dict[str, Any]] = []
        for source_row in source_rows:
            if not isinstance(source_row, dict):
                continue
            built: dict[str, Any] = {}
            for column in columns:
                cell = _apply_transform(
                    column.get("transform", "identity"),
                    _get_source(source_row, column["source"]),
                )
                if cell is not None:
                    _assign_nested(built, column["target"], cell)
            if built:
                built_rows.append(built)
        if built_rows:
            ground_truth[target] = built_rows
            label_provenance[target] = provenance

    for rule in spec.get("derived", []):
        # Precedence: asserted labels win — a derived rule only fills a target
        # that was not already asserted (Voxel51 has no printed total, so this
        # fires and total_ttc.amount is recorded as derived).
        if rule["target"] in ground_truth:
            continue
        result = _apply_derived(rule, ground_truth)
        if result is None:
            continue
        target, value = result
        ground_truth[target] = value
        label_provenance[target] = rule.get("provenance", "derived")

    return ground_truth, label_provenance


def _assign_nested(row: dict[str, Any], dotted: str, value: Any) -> None:
    """Assign ``value`` at a dotted path inside a table-row dict.

    ``line_total.amount`` -> ``{"line_total": {"amount": value}}`` so the row
    matches the nested prediction shape ``score_table`` flattens.
    """
    parts = dotted.split(".")
    current = row
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value
