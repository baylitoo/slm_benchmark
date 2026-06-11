from docie_bench.benchmark.report import write_report


def test_report_surfaces_evidence_metrics_and_ungrounded_fields(tmp_path):
    metrics = {
        "summary": [
            {
                "model_profile": "test",
                "docs": 1,
                "concurrency": 1,
                "wall_seconds": 1.0,
                "throughput_docs_per_min": 60.0,
                "valid_rate": 1.0,
                "field_accuracy": 1.0,
                "row_f1": 0.5,
                "evidence_coverage": 0.5,
                "evidence_row_coverage": 0.5,
                "hallucination_rate": 0.5,
                "avg_latency_ms": 10,
                "p50_latency_ms": 10,
                "p95_latency_ms": 10,
                "routing_accept_rate": 0.5,
                "routing_fallback_rate": 0.5,
                "routing_escalation_rate": 0.0,
                "routing_budget_exhaustion_rate": 0.0,
                "routing_stage_failure_rate": 0.0,
                "avg_routing_attempts": 1.5,
                "avg_routing_latency_ms": 12,
                "avg_routing_tokens": 100,
                "avg_routing_cost_units": 0.25,
            }
        ],
        "rows": [
            {
                "doc_id": "doc-1",
                "model_profile": "test",
                "score": {
                    "ungrounded_fields": ["vendor_name"],
                    "tables": [
                        {
                            "field": "line_items",
                            "row_correct": 1,
                            "row_expected": 2,
                            "row_predicted": 1,
                        }
                    ],
                },
                "validation": {"warnings": ["sum(line_items.line_total) does not match subtotal"]},
            }
        ],
    }

    report = write_report(tmp_path, metrics).read_text(encoding="utf-8")

    assert "Evidence coverage" in report
    assert "Hallucination rate" in report
    assert "Routing Summary" in report
    assert "Cost units" in report
    assert "vendor_name" in report
    assert "Row F1" in report
    assert "Row evidence coverage" in report
    assert "Arithmetic Validation Warnings" in report
    assert "sum(line_items.line_total)" in report
    assert "line_items" in report
