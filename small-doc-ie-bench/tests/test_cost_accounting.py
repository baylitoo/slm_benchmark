"""Three-unit cost accounting + honest schema coverage (no live stack).

Fabricated prediction rows exercise the fold that was previously dropped at row
build: local `tokens` come from row usage; paid `$/doc` only when a profile has
pricing (local shows N/A, never $0); the abstract routing `cost_units` path is
unchanged. Also: score_prediction excludes unsupported ground-truth fields from
the denominator (schema_coverage), the report renders the new columns with N/A,
pricing parses/classifies a profile as paid, and docTR registers via the factory.
"""

from __future__ import annotations

from docie_bench.benchmark.metrics import score_prediction
from docie_bench.benchmark.report import write_report
from docie_bench.benchmark.runner import summarize
from docie_bench.llm.model_profiles import ModelProfile, Pricing, load_model_profiles
from docie_bench.ocr.doctr_backend import DocTRBackend
from docie_bench.ocr.factory import get_ocr_backend


def _row(
    profile: str,
    *,
    usage: dict | None = None,
    routing: dict | None = None,
    ingestion_path: str = "ocr:tesseract",
    score: dict | None = None,
) -> dict:
    row: dict = {
        "model_profile": profile,
        "ok": True,
        "ingestion_path": ingestion_path,
        "validation": {"valid": True},
        "latency_ms": 10,
        "score": score if score is not None else {"field_total": 2, "field_correct": 2},
    }
    if usage is not None:
        row["usage"] = usage
    if routing is not None:
        row["routing"] = routing
    return row


def _paid_profile() -> ModelProfile:
    return ModelProfile(
        name="hosted",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="secret",
        pricing=Pricing(input_usd_per_1k=0.00015, output_usd_per_1k=0.0006),
    )


def _local_profile() -> ModelProfile:
    return ModelProfile(name="local", model="m", base_url="http://x/v1", api_key="k")


# ── local tokens vs paid $ ───────────────────────────────────────────────────


def test_local_profile_reports_tokens_but_cost_is_na() -> None:
    rows = [
        _row("local", usage={"total_tokens": 100}),
        _row("local", usage={"total_tokens": 200}),
    ]
    metrics = summarize(rows, profiles={"local": _local_profile()})
    entry = metrics["summary"][0]
    assert entry["avg_tokens"] == 150.0
    # Local profile has no pricing => $ column is N/A (None), NEVER $0.
    assert entry["avg_cost_usd_per_doc"] is None


def test_paid_profile_carries_usage_and_computes_cost() -> None:
    rows = [_row("hosted", usage={"prompt_tokens": 1000, "completion_tokens": 500})]
    metrics = summarize(rows, profiles={"hosted": _paid_profile()})
    entry = metrics["summary"][0]
    # 1000/1k*0.00015 + 500/1k*0.0006 = 0.00015 + 0.0003 = 0.00045
    assert entry["avg_cost_usd_per_doc"] == 0.00045
    assert entry["avg_tokens"] == 1500.0
    assert entry["cost_estimated"] is False


def test_missing_profiles_map_leaves_cost_na() -> None:
    rows = [_row("hosted", usage={"prompt_tokens": 10, "completion_tokens": 5})]
    entry = summarize(rows)["summary"][0]
    assert entry["avg_cost_usd_per_doc"] is None
    assert entry["avg_tokens"] == 15.0


def test_paid_row_without_usage_split_flags_estimated() -> None:
    rows = [_row("hosted", usage=None)]
    entry = summarize(rows, profiles={"hosted": _paid_profile()})["summary"][0]
    assert entry["cost_estimated"] is True
    assert entry["avg_cost_usd_per_doc"] is not None  # falls back, not silently $0
    assert entry["avg_tokens"] is None  # no usage => tokens N/A


# ── routing cost_units unchanged ─────────────────────────────────────────────


def test_routing_cost_units_path_unchanged() -> None:
    routed = {
        "terminal_decision": "accept",
        "total_tokens": 100,
        "cost_units": 0.25,
        "attempts": 1,
        "latency_ms": 12,
        "stages": [],
    }
    rows = [_row("routed:v1", routing=routed, ingestion_path="routed")]
    entry = summarize(rows)["summary"][0]
    assert entry["avg_routing_cost_units"] == 0.25
    assert entry["avg_routing_tokens"] == 100
    # The paid-$ unit is distinct and does not borrow the routing cost_units value.
    assert entry["avg_cost_usd_per_doc"] is None


# ── schema coverage (anti silent-zero) ───────────────────────────────────────


def test_score_prediction_supported_fields_excludes_unsupported() -> None:
    gt = {"vendor_name": "ACME", "total": "42.00", "invoice_number": "INV-1"}
    pred = {"vendor_name": {"value": "ACME"}, "total": {"value": "42.00"}}
    score = score_prediction(gt, pred, supported_fields=["vendor_name", "total"])
    # invoice_number is unsupported -> excluded from the denominator, not scored 0.
    assert score["field_total"] == 2
    assert score["field_correct"] == 2
    assert score["field_accuracy"] == 1.0
    assert score["schema_coverage"] == 2 / 3
    assert score["unsupported_fields"] == ["invoice_number"]


def test_score_prediction_default_scores_all_fields_unchanged() -> None:
    gt = {"vendor_name": "ACME", "invoice_number": "INV-1"}
    pred = {"vendor_name": {"value": "ACME"}}
    score = score_prediction(gt, pred)
    assert score["field_total"] == 2  # unchanged: missing field scored, not dropped
    assert score["schema_coverage"] is None
    assert score["unsupported_fields"] == []


def test_summarize_aggregates_schema_coverage() -> None:
    rows = [
        _row(
            "donut_cord",
            score={"field_total": 2, "field_correct": 2, "schema_coverage": 2 / 3},
        )
    ]
    entry = summarize(rows)["summary"][0]
    assert entry["schema_coverage"] == round(2 / 3, 4)


# ── report columns ───────────────────────────────────────────────────────────


def test_report_renders_cost_and_coverage_columns(tmp_path) -> None:  # noqa: ANN001
    metrics = {
        "summary": [
            {
                "model_profile": "local",
                "ingestion_path": "ocr:tesseract",
                "docs": 1,
                "concurrency": 1,
                "wall_seconds": 1.0,
                "throughput_docs_per_min": 60.0,
                "valid_rate": 1.0,
                "field_accuracy": 1.0,
                "row_f1": None,
                "evidence_coverage": None,
                "evidence_row_coverage": None,
                "hallucination_rate": None,
                "avg_tokens": 150.0,
                "avg_cost_usd_per_doc": None,  # local => N/A
                "cost_estimated": False,
                "schema_coverage": None,
                "avg_latency_ms": 10,
                "p50_latency_ms": 10,
                "p95_latency_ms": 10,
            }
        ],
        "rows": [],
    }
    report = write_report(tmp_path, metrics).read_text(encoding="utf-8")
    assert "Cost $/doc" in report
    assert "Tokens/doc" in report
    assert "Schema coverage" in report
    # Routing cost_units keeps its own sub-table column (distinct unit).
    assert "Cost units" in report


# ── pricing load + classification, docTR factory ─────────────────────────────


def test_load_pricing_classifies_paid(tmp_path) -> None:  # noqa: ANN001
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "profiles:\n"
        "  paid_api:\n"
        "    model: gpt-4o-mini\n"
        "    base_url: https://api.openai.com/v1\n"
        "    api_key_env: OPENAI_API_KEY\n"
        "    pricing:\n"
        "      input_usd_per_1k: 0.00015\n"
        "      output_usd_per_1k: 0.0006\n"
        "  free_local:\n"
        "    model: qwen\n"
        "    base_url: http://localhost:11434/v1\n",
        encoding="utf-8",
    )
    profiles = load_model_profiles(cfg)
    assert profiles["paid_api"].is_paid is True
    assert profiles["paid_api"].pricing is not None
    assert profiles["free_local"].is_paid is False


def test_doctr_registers_as_ocr_backend() -> None:
    backend = get_ocr_backend("doctr")
    assert isinstance(backend, DocTRBackend)
    assert backend.name == "doctr"
