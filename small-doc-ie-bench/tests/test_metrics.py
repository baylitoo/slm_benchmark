import pytest

from docie_bench.benchmark.metrics import score_evidence, score_prediction
from docie_bench.benchmark.runner import summarize


def test_score_prediction_nested_value():
    pred = {
        "invoice_number": {"value": "INV-1"},
        "total_ttc": {"amount": "1200.00", "currency": "EUR"},
    }
    gt = {"invoice_number": "INV-1", "total_ttc.amount": "1200"}
    score = score_prediction(gt, pred)
    assert score["field_correct"] == 2


def test_score_evidence_reports_coverage_and_ungrounded_fields():
    pred = {
        "invoice_number": {"value": "INV-1", "evidence_ids": ["b1"]},
        "vendor_name": {"value": "Invented Corp", "evidence_ids": []},
        "due_date": None,
    }

    score = score_evidence(pred)

    assert score["evidence_field_total"] == 2
    assert score["evidence_grounded"] == 1
    assert score["evidence_coverage"] == 0.5
    assert score["hallucination_rate"] == 0.5
    assert score["ungrounded_fields"] == ["vendor_name"]


def test_summarize_aggregates_evidence_metrics():
    rows = [
        {
            "model_profile": "test",
            "ok": True,
            "latency_ms": 10,
            "validation": {"valid": True},
            "score": {
                "field_total": 1,
                "field_correct": 1,
                "avg_similarity": 1.0,
                "evidence_field_total": 4,
                "evidence_grounded": 3,
            },
        }
    ]

    summary = summarize(rows)["summary"][0]

    assert summary["evidence_coverage"] == 0.75
    assert summary["hallucination_rate"] == 0.25


def test_summarize_aggregates_routing_metrics():
    rows = [
        {
            "model_profile": "routed",
            "ok": True,
            "latency_ms": 10,
            "validation": {"valid": True},
            "score": {},
            "routing": {
                "terminal_decision": "accept",
                "attempts": 2,
                "fallback_count": 1,
                "budget_exhausted": False,
                "latency_ms": 30,
                "total_tokens": 100,
                "cost_units": 0.2,
                "stages": [{"status": "success"}, {"status": "success"}],
            },
        },
        {
            "model_profile": "routed",
            "ok": False,
            "latency_ms": 20,
            "routing": {
                "terminal_decision": "escalate",
                "attempts": 1,
                "fallback_count": 0,
                "budget_exhausted": True,
                "latency_ms": 10,
                "total_tokens": 20,
                "cost_units": 0.1,
                "stages": [{"status": "error"}],
            },
        },
    ]

    summary = summarize(rows)["summary"][0]

    assert summary["routing_accept_rate"] == 0.5
    assert summary["routing_escalation_rate"] == 0.5
    assert summary["routing_fallback_rate"] == 0.5
    assert summary["routing_budget_exhaustion_rate"] == 0.5
    assert summary["avg_routing_attempts"] == 1.5
    assert summary["avg_routing_latency_ms"] == 20
    assert summary["avg_routing_tokens"] == 60
    assert summary["avg_routing_cost_units"] == pytest.approx(0.15)
    assert summary["routing_stage_failure_rate"] == pytest.approx(1 / 3)


def test_score_evidence_vision_path_is_not_applicable():
    # Vision extractions have no OCR blocks to cite, so grounding is N/A — not 100%
    # "hallucinated". Reporting None lets the report distinguish "ungrounded" from "absent".
    pred = {"invoice_number": {"value": "INV-1", "evidence_ids": []}}
    ev = score_evidence(pred, evidence_applicable=False)
    assert ev["evidence_applicable"] is False
    assert ev["hallucination_rate"] is None
    assert ev["evidence_coverage"] is None
    assert ev["ungrounded_fields"] == []
    # The OCR path still computes grounding as before.
    assert score_evidence(pred)["hallucination_rate"] == 1.0


def test_score_prediction_threads_evidence_applicable_without_touching_field_score():
    gt = {"invoice_number": "INV-1"}
    pred = {"invoice_number": {"value": "INV-1", "evidence_ids": []}}
    score = score_prediction(gt, pred, evidence_applicable=False)
    assert score["evidence_applicable"] is False
    assert score["hallucination_rate"] is None
    assert score["field_correct"] == 1  # field scoring is unaffected by the evidence flag
