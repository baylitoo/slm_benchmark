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


def test_score_prediction_aligns_reordered_line_items_and_penalizes_extra_rows():
    gt = {
        "line_items": [
            {"description": "Consulting", "quantity": "2", "line_total.amount": "200.00"},
            {"description": "Hosting", "quantity": "1", "line_total.amount": "50.00"},
        ]
    }
    pred = {
        "line_items": [
            {
                "description": {"value": "Hosting", "evidence_ids": ["row-2"]},
                "quantity": {"value": "1", "evidence_ids": ["row-2"]},
                "line_total": {"amount": "50", "currency": "EUR", "evidence_ids": ["row-2"]},
            },
            {
                "description": {"value": "Consulting", "evidence_ids": ["row-1"]},
                "quantity": {"value": "2", "evidence_ids": ["row-1"]},
                "line_total": {"amount": "200", "currency": "EUR", "evidence_ids": ["row-1"]},
            },
            {"description": {"value": "Invented", "evidence_ids": []}},
        ]
    }

    score = score_prediction(gt, pred)

    assert score["field_correct"] == 6
    assert score["row_correct"] == 2
    assert score["row_recall"] == 1.0
    assert score["row_precision"] == 2 / 3
    assert score["tables"][0]["alignment"] == [
        {"expected_row": 0, "actual_row": 1},
        {"expected_row": 1, "actual_row": 0},
    ]
    assert score["evidence_row_coverage"] == 2 / 3
    assert score["ungrounded_rows"] == ["line_items.2"]


def test_score_prediction_marks_missing_table_rows_and_nested_ground_truth():
    score = score_prediction(
        {
            "line_items": [
                {"description": "Consulting", "line_total": {"amount": "200.00"}},
                {"description": "Hosting", "line_total": {"amount": "50.00"}},
            ]
        },
        {
            "line_items": [
                {
                    "description": {"value": "Consulting"},
                    "line_total": {"amount": "200.00", "currency": "EUR"},
                }
            ]
        },
    )

    assert score["row_expected"] == 2
    assert score["row_predicted"] == 1
    assert score["row_correct"] == 1
    assert score["row_recall"] == 0.5
    assert score["field_correct"] == 2
