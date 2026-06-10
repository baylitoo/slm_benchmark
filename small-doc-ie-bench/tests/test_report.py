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
                "evidence_coverage": 0.5,
                "hallucination_rate": 0.5,
                "avg_latency_ms": 10,
                "p50_latency_ms": 10,
                "p95_latency_ms": 10,
            }
        ],
        "rows": [
            {
                "doc_id": "doc-1",
                "model_profile": "test",
                "score": {"ungrounded_fields": ["vendor_name"]},
            }
        ],
    }

    report = write_report(tmp_path, metrics).read_text(encoding="utf-8")

    assert "Evidence coverage" in report
    assert "Hallucination rate" in report
    assert "vendor_name" in report
