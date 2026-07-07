"""Offline label audit — runs at manifest-build time, off the benchmark hot path.

Emits a ``label_audit.json`` sidecar of findings *about* the labels without ever
overwriting them. Three families of checks per document:

* range      — amounts are non-negative; subtotal does not exceed the total.
* consistency — sum of line-item totals reconciles with the subtotal, within the
  same numeric tolerance ``metrics.py`` uses for scoring (single source of truth).
* reconciliation — ``total_ttc.reconciled``: does subtotal + tax match a PRINTED
  (asserted) total? ``None`` when no printed total exists (Voxel51: the total is
  a derived hypothesis, so this is ``null`` by design, documenting that nothing
  verifies it — it is NOT a label bug to be "fixed").

Also cross-checks that every ``label_provenance`` key maps to a real
``ground_truth`` key, so a renamed/typo'd provenance path is caught here rather
than silently defaulting to ``asserted`` at scoring time.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docie_bench.benchmark.metrics import MetricConfig

if TYPE_CHECKING:
    from docie_bench.benchmark.dataset import DatasetItem

AUDIT_FORMAT_VERSION = 1


def _amount(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("amount", value.get("value"))
    try:
        return Decimal(str(value).replace(",", ".").strip())
    except (InvalidOperation, ValueError):
        return None


def _within_tolerance(left: Decimal, right: Decimal, cfg: MetricConfig) -> bool:
    diff = abs(left - right)
    if diff <= Decimal(str(cfg.numeric_absolute_tolerance)):
        return True
    denom = max(abs(left), Decimal("1"))
    return float(diff / denom) <= cfg.numeric_relative_tolerance


def _line_total_sum(ground_truth: dict[str, Any]) -> Decimal | None:
    rows = ground_truth.get("line_items")
    if not isinstance(rows, list):
        return None
    total = Decimal("0")
    seen = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        line_total = row.get("line_total")
        amount = _amount(line_total)
        if amount is None:
            continue
        total += amount
        seen = True
    return total if seen else None


def flattened_keys(ground_truth: dict[str, Any]) -> set[str]:
    """Keys a provenance path may legitimately reference.

    Scalar dotted keys, plus each table key and its per-cell paths
    (``line_items``, ``line_items.0.line_total.amount``, ...), so the subset
    check accepts both an exact table key and a specific cell path.
    """
    keys: set[str] = set()
    for key, value in ground_truth.items():
        keys.add(key)
        if isinstance(value, list):
            for index, row in enumerate(value):
                if isinstance(row, dict):
                    for cell in _flatten(row):
                        keys.add(f"{key}.{index}.{cell}")
    return keys


def _flatten(row: dict[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in row.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths.extend(_flatten(value, path))
        else:
            paths.append(path)
    return paths


def audit_item(item: DatasetItem, cfg: MetricConfig | None = None) -> dict[str, Any]:
    """Return an audit record for one dataset item. Never mutates the item."""
    cfg = cfg or MetricConfig()
    ground_truth = item.ground_truth
    provenance = item.label_provenance
    findings: list[str] = []

    subtotal = _amount(ground_truth.get("subtotal.amount"))
    vat = _amount(ground_truth.get("vat.amount") or ground_truth.get("tax.amount"))
    total = _amount(ground_truth.get("total_ttc.amount"))

    for name, amount in (("subtotal", subtotal), ("vat", vat), ("total_ttc", total)):
        if amount is not None and amount < 0:
            findings.append(f"negative_amount:{name}")
    if (
        subtotal is not None
        and total is not None
        and subtotal > total
        and not _within_tolerance(subtotal, total, cfg)
    ):
        findings.append("subtotal_exceeds_total")

    line_sum = _line_total_sum(ground_truth)
    line_items_consistent: bool | None = None
    if line_sum is not None and subtotal is not None:
        line_items_consistent = _within_tolerance(line_sum, subtotal, cfg)
        if not line_items_consistent:
            findings.append("line_items_sum_mismatch")

    # Reconciliation only means something against a PRINTED (asserted) total.
    total_provenance = provenance.get("total_ttc.amount", "asserted")
    printed_total_present = total is not None and total_provenance == "asserted"
    reconciled: bool | None
    if not printed_total_present or subtotal is None:
        reconciled = None
    else:
        derived = subtotal + (vat if vat is not None else Decimal("0"))
        assert total is not None  # narrowed by printed_total_present
        reconciled = _within_tolerance(derived, total, cfg)
        if not reconciled:
            findings.append("total_not_reconciled")

    gt_keys = flattened_keys(ground_truth)
    unknown_provenance_keys = sorted(k for k in provenance if k not in gt_keys)
    if unknown_provenance_keys:
        findings.append("provenance_keys_not_in_ground_truth")

    return {
        "doc_id": item.doc_id,
        "checks": {
            "subtotal": _as_float(subtotal),
            "vat": _as_float(vat),
            "total_ttc": _as_float(total),
            "line_items_sum": _as_float(line_sum),
            "line_items_consistent": line_items_consistent,
            "total_ttc_reconciled": reconciled,
            "total_provenance": total_provenance,
        },
        "unknown_provenance_keys": unknown_provenance_keys,
        "findings": findings,
        "ok": not findings,
    }


def _as_float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def audit_dataset(
    items: list[DatasetItem], cfg: MetricConfig | None = None
) -> dict[str, Any]:
    cfg = cfg or MetricConfig()
    records = [audit_item(item, cfg) for item in items]
    return {
        "audit_format_version": AUDIT_FORMAT_VERSION,
        "tolerance": {
            "numeric_relative_tolerance": cfg.numeric_relative_tolerance,
            "numeric_absolute_tolerance": cfg.numeric_absolute_tolerance,
        },
        "documents": len(records),
        "documents_with_findings": sum(not r["ok"] for r in records),
        "items": records,
    }


def write_label_audit(
    path: Path, items: list[DatasetItem], cfg: MetricConfig | None = None
) -> dict[str, Any]:
    report = audit_dataset(items, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
