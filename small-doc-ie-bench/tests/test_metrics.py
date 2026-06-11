from docie_bench.benchmark.metrics import score_evidence, score_prediction
from docie_bench.benchmark.runner import summarize


def test_score_prediction_nested_value():
    pred = {"invoice_number": {"value": "INV-1"}, "total_ttc": {"amount": "1200.00", "currency": "EUR"}}
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
