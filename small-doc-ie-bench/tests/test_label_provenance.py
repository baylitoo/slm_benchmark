"""PR-5: dataset label provenance — segmentation, audit, declarative mapping."""

from __future__ import annotations

from pathlib import Path

import yaml

from docie_bench.benchmark.dataset import DatasetItem, load_dataset
from docie_bench.benchmark.label_audit import audit_item
from docie_bench.benchmark.label_mapping import apply_mapping
from docie_bench.benchmark.metrics import score_prediction
from docie_bench.benchmark.provenance import (
    aggregate_provenance_segments,
    resolve_provenance,
    segment_score_fields,
)
from docie_bench.benchmark.runner import summarize

SPEC_PATH = (
    Path(__file__).parent.parent / "scripts" / "label_mapping" / "voxel51_invoice.yaml"
)


def _spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


# --- (1) reducer invariants -------------------------------------------------


def test_reducer_holds_both_invariants_on_synthetic_score():
    fields = [
        {"field": "invoice_number", "correct": True},
        {"field": "subtotal.amount", "correct": True},
        {"field": "vat.amount", "correct": False},
        {"field": "total_ttc.amount", "correct": False},  # derived
        {"field": "line_items.0.line_total.amount", "correct": True},  # derived table
        {"field": "line_items.0.description", "correct": False},  # derived table
    ]
    provenance = {
        "total_ttc.amount": "derived",
        "line_items": "derived",
    }
    score = {
        "field_total": len(fields),
        "field_correct": sum(f["correct"] for f in fields),
        "fields": fields,
    }
    counts = segment_score_fields(fields, provenance)

    assert counts["asserted_total"] + counts["derived_total"] == score["field_total"]
    assert counts["asserted_correct"] + counts["derived_correct"] == score["field_correct"]
    # asserted: invoice_number(1/1), subtotal(1), vat(0) -> 2 correct / 3
    assert (counts["asserted_correct"], counts["asserted_total"]) == (2, 3)
    # derived: total_ttc(0), line_total(1), line_items.0.description(0) -> 1 / 3
    assert (counts["derived_correct"], counts["derived_total"]) == (1, 3)


def test_aggregate_field_accuracy_derived_none_when_no_derived():
    ok_rows = [
        {
            "score": {"fields": [{"field": "invoice_number", "correct": True}]},
            "label_provenance": {},
        }
    ]
    seg = aggregate_provenance_segments(ok_rows)
    assert seg["field_accuracy_asserted"] == 1.0
    assert seg["field_derived_total"] == 0
    assert seg["field_accuracy_derived"] is None


# --- (2) prefix bucketing ---------------------------------------------------


def test_prefix_bucketing_and_exact_match_precedence():
    provenance = {
        "line_items": "derived",
        "subtotal.amount": "asserted",
        "subtotal": "derived",  # must NOT swallow subtotal.amount (exact wins)
    }
    assert resolve_provenance("line_items.0.line_total.amount", provenance) == "derived"
    assert resolve_provenance("subtotal.amount", provenance) == "asserted"
    # A sibling that merely shares a string prefix is not a dotted-boundary match.
    assert resolve_provenance("subtotal_note", provenance) == "asserted"
    # Unknown field defaults to asserted.
    assert resolve_provenance("vendor_name", provenance) == "asserted"


# --- (3) declarative mapping (round trip -> hand-specified target) -----------


def test_apply_mapping_produces_expected_ground_truth_and_provenance():
    annotation = {
        "doc_id": "inv-1",
        "file_path": "inv-1.pdf",
        "invoice_number": "INV-1",
        "vendor": {"name": "ACME SARL"},
        "invoice_date": "05/02/2026",
        "totals": {"subtotal": "1000,00", "vat": "200"},
        "line_items": [
            {"description": "Widget", "quantity": "2", "line_total": "600.00"},
            {"description": "Gadget", "quantity": "1", "line_total": "400"},
        ],
    }
    ground_truth, provenance = apply_mapping(annotation, _spec())

    assert ground_truth == {
        "invoice_number": "INV-1",
        "vendor_name": "ACME SARL",
        "invoice_date": "2026-02-05",
        "subtotal.amount": 1000.0,
        "vat.amount": 200.0,
        "line_items": [
            {"description": "Widget", "quantity": 2, "line_total": {"amount": 600.0}},
            {"description": "Gadget", "quantity": 1, "line_total": {"amount": 400.0}},
        ],
        "total_ttc.amount": 1200.0,  # DERIVED subtotal + vat, no printed total
    }
    assert provenance["total_ttc.amount"] == "derived"
    assert provenance["subtotal.amount"] == "asserted"
    assert provenance["line_items"] == "asserted"


def test_apply_mapping_derived_skips_missing_anchor_without_crashing():
    # No subtotal -> derived total is skipped entirely (mirrors historical elif).
    annotation = {"invoice_number": "INV-2", "totals": {"vat": "50"}}
    ground_truth, provenance = apply_mapping(annotation, _spec())
    assert "total_ttc.amount" not in ground_truth
    assert "total_ttc.amount" not in provenance


def test_apply_mapping_derived_falls_back_to_subtotal_only():
    annotation = {"invoice_number": "INV-3", "totals": {"subtotal": "900"}}
    ground_truth, _ = apply_mapping(annotation, _spec())
    assert ground_truth["total_ttc.amount"] == 900.0


# --- (4) label audit --------------------------------------------------------


def test_audit_reconciled_null_when_total_derived():
    item = DatasetItem(
        doc_id="inv-1",
        file_path="inv-1.pdf",
        ground_truth={
            "subtotal.amount": 1000.0,
            "vat.amount": 200.0,
            "total_ttc.amount": 1200.0,
        },
        label_provenance={"total_ttc.amount": "derived"},
    )
    record = audit_item(item)
    assert record["checks"]["total_ttc_reconciled"] is None
    assert "total_not_reconciled" not in record["findings"]


def test_audit_consistency_flag_fires_on_line_item_mismatch():
    item = DatasetItem(
        doc_id="inv-2",
        file_path="inv-2.pdf",
        ground_truth={
            "subtotal.amount": 1000.0,
            "line_items": [
                {"line_total": {"amount": 300.0}},
                {"line_total": {"amount": 300.0}},  # sums to 600, not 1000
            ],
        },
    )
    record = audit_item(item)
    assert record["checks"]["line_items_consistent"] is False
    assert "line_items_sum_mismatch" in record["findings"]


def test_audit_reconciles_printed_total_and_flags_mismatch():
    # A printed (asserted) total that does NOT equal subtotal + vat is flagged.
    item = DatasetItem(
        doc_id="inv-3",
        file_path="inv-3.pdf",
        ground_truth={
            "subtotal.amount": 1000.0,
            "vat.amount": 200.0,
            "total_ttc.amount": 1500.0,
        },
        label_provenance={"total_ttc.amount": "asserted"},
    )
    record = audit_item(item)
    assert record["checks"]["total_ttc_reconciled"] is False
    assert "total_not_reconciled" in record["findings"]


def test_audit_flags_provenance_key_not_in_ground_truth():
    item = DatasetItem(
        doc_id="inv-4",
        file_path="inv-4.pdf",
        ground_truth={"subtotal.amount": 100.0},
        label_provenance={"subtotal.amont": "asserted"},  # typo
    )
    record = audit_item(item)
    assert record["unknown_provenance_keys"] == ["subtotal.amont"]
    assert "provenance_keys_not_in_ground_truth" in record["findings"]


# --- (5) backward compatibility + end-to-end through summarize --------------


def test_manifest_row_without_provenance_scores_and_buckets_as_asserted(tmp_path: Path):
    doc = tmp_path / "inv.txt"
    doc.write_text("invoice", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"doc_id":"a","file_path":"inv.txt","ground_truth":{"invoice_number":"INV-1"}}\n',
        encoding="utf-8",
    )
    items = load_dataset(manifest)
    assert items[0].label_provenance == {}  # default, backward compatible

    score = score_prediction(items[0].ground_truth, {"invoice_number": {"value": "INV-1"}})
    row = {"score": score, "label_provenance": items[0].label_provenance}
    seg = aggregate_provenance_segments([row])
    assert seg["field_asserted_total"] == score["field_total"]
    assert seg["field_derived_total"] == 0
    assert seg["field_accuracy_derived"] is None


def test_summarize_segments_field_accuracy_by_provenance():
    gt = {"invoice_number": "INV-1", "subtotal.amount": "1000", "total_ttc.amount": "1200"}
    pred = {
        "invoice_number": {"value": "INV-1"},
        "subtotal": {"amount": "1000"},
        "total_ttc": {"amount": "999"},  # wrong -> derived miss
    }
    score = score_prediction(gt, pred)
    rows = [
        {
            "model_profile": "p",
            "ok": True,
            "latency_ms": 1,
            "validation": {"valid": True},
            "score": score,
            "label_provenance": {"total_ttc.amount": "derived"},
        }
    ]
    summary = summarize(rows)["summary"][0]

    # Invariant carried through summarize.
    assert (
        summary["field_asserted_total"] + summary["field_derived_total"]
        == score["field_total"]
    )
    assert (
        summary["field_asserted_correct"] + summary["field_derived_correct"]
        == score["field_correct"]
    )
    # invoice_number + subtotal.amount asserted & correct.
    assert summary["field_accuracy_asserted"] == 1.0
    # total_ttc.amount derived & wrong -> the NuExtract-style derived-total gap,
    # now a visible segment rather than a hidden fraction of field_accuracy.
    assert summary["field_derived_total"] == 1
    assert summary["field_accuracy_derived"] == 0.0
