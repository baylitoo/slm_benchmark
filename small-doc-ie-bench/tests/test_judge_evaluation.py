import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from docie_bench.benchmark.judge import EvaluationMode, build_judge_prompt, judge_extraction
from docie_bench.benchmark.runner import run_benchmark, summarize
from docie_bench.cli import app
from docie_bench.llm.model_profiles import ModelProfile, load_judge_profile


def test_summarize_includes_judge_metrics_without_ground_truth():
    rows = [
        {
            "model_profile": "extractor",
            "ok": True,
            "latency_ms": 10,
            "validation": {"valid": True},
            "judge": {
                "overall_faithfulness": 0.9,
                "overall_completeness": 0.7,
                "judge_model": "judge-model",
            },
        }
    ]

    metrics = summarize(rows, eval_mode=EvaluationMode.LLM_JUDGE)
    summary = metrics["summary"][0]

    assert metrics["eval_mode"] == "llm_judge"
    assert summary["field_accuracy"] is None
    assert summary["judge_faithfulness"] == 0.9
    assert summary["judge_completeness"] == 0.7
    assert summary["judge_model"] == "judge-model"


def test_both_mode_reports_judge_ground_truth_calibration_delta():
    rows = [
        {
            "model_profile": "extractor",
            "ok": True,
            "latency_ms": 10,
            "validation": {"valid": True},
            "score": {"field_total": 4, "field_correct": 3, "avg_similarity": 0.8},
            "judge": {
                "overall_faithfulness": 0.9,
                "overall_completeness": 0.7,
                "judge_model": "judge-model",
            },
        }
    ]

    summary = summarize(rows, eval_mode=EvaluationMode.BOTH)["summary"][0]

    assert summary["field_accuracy"] == 0.75
    assert summary["judge_field_accuracy_delta"] == 0.15


def test_load_judge_profile_uses_separate_config_selector(tmp_path: Path):
    config = tmp_path / "models.yaml"
    config.write_text(
        """
judge:
  profile: auditor
profiles:
  extractor:
    model: extraction-model
    base_url: http://extractor/v1
  auditor:
    model: judge-model
    base_url: http://judge/v1
""",
        encoding="utf-8",
    )

    judge = load_judge_profile(config)

    assert judge.name == "auditor"
    assert judge.model == "judge-model"


def test_judge_extraction_sends_source_and_adds_model_metadata(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, profile):
            captured["profile"] = profile

        async def chat_json(self, **kwargs):
            captured.update(kwargs)
            return (
                {
                    "field_scores": {},
                    "overall_faithfulness": 0.8,
                    "overall_completeness": 0.6,
                    "issues": [],
                },
                None,
                {},
            )

        async def aclose(self):
            return None

    monkeypatch.setattr("docie_bench.benchmark.judge.OpenAICompatibleClient", FakeClient)
    profile = ModelProfile(
        name="auditor", model="judge-model", base_url="http://judge/v1", api_key="x"
    )

    result = asyncio.run(
        judge_extraction(
            profile=profile,
            document_text="Invoice INV-1",
            extraction={"id": "INV-1"},
        )
    )

    assert "Invoice INV-1" in captured["user_prompt"]
    assert result["overall_faithfulness"] == 0.8
    assert result["judge_model"] == "judge-model"
    assert "SOURCE DOCUMENT" in build_judge_prompt("text", {"field": "value"})


def test_cli_accepts_unlabeled_document_in_judge_mode(monkeypatch, tmp_path: Path):
    document = tmp_path / "invoice.txt"
    document.write_text("Invoice INV-1", encoding="utf-8")
    models = tmp_path / "models.yaml"
    models.write_text("profiles: {}", encoding="utf-8")

    async def fake_run_benchmark(**kwargs):
        assert kwargs["dataset_path"] is None
        assert kwargs["document_path"] == document
        assert kwargs["eval_mode"] is EvaluationMode.LLM_JUDGE
        return SimpleNamespace(
            run_dir=tmp_path,
            predictions_path=tmp_path / "predictions.jsonl",
            metrics_path=tmp_path / "metrics.json",
            report_path=tmp_path / "report.html",
        )

    monkeypatch.setattr("docie_bench.cli.run_benchmark", fake_run_benchmark)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "run",
            "--document",
            str(document),
            "--models-config",
            str(models),
            "--eval-mode",
            "llm_judge",
            "--judge-profile",
            "auditor",
        ],
    )

    assert result.exit_code == 0, result.output


def test_runner_evaluates_unlabeled_document_and_excludes_judge_profile(
    monkeypatch,
    tmp_path: Path,
):
    document = tmp_path / "invoice.txt"
    document.write_text("Invoice INV-1", encoding="utf-8")
    config = tmp_path / "models.yaml"
    config.write_text("profiles: {}", encoding="utf-8")
    extractor = ModelProfile(
        name="extractor", model="extractor-model", base_url="http://extractor/v1", api_key="x"
    )
    auditor = ModelProfile(
        name="auditor", model="judge-model", base_url="http://judge/v1", api_key="x"
    )
    called_profiles = []

    class FakeExtractionService:
        def __init__(self, profile):
            called_profiles.append(profile.name)

        async def extract_from_file(self, **kwargs):
            return SimpleNamespace(
                schema_name="invoice",
                dynamic_schema=None,
                latency_ms=5,
                validation=SimpleNamespace(model_dump=lambda: {"valid": True}),
                result={"invoice_number": {"value": "INV-1", "evidence_ids": ["b1"]}},
            )

    class FakeBackend:
        def extract(self, path):
            return [SimpleNamespace(text=path.read_text(encoding="utf-8"))]

    async def fake_judge_extraction(**kwargs):
        assert kwargs["profile"] is auditor
        return {
            "overall_faithfulness": 1.0,
            "overall_completeness": 0.8,
            "judge_model": "judge-model",
        }

    monkeypatch.setattr(
        "docie_bench.benchmark.runner.get_settings",
        lambda: SimpleNamespace(default_ocr_backend="pdf_text", runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(
        "docie_bench.benchmark.runner.load_model_profiles",
        lambda path: {"extractor": extractor, "auditor": auditor},
    )
    monkeypatch.setattr(
        "docie_bench.benchmark.runner.load_judge_profile",
        lambda path, profile_name: auditor,
    )
    monkeypatch.setattr("docie_bench.benchmark.runner.ExtractionService", FakeExtractionService)
    monkeypatch.setattr(
        "docie_bench.benchmark.runner.get_ocr_backend",
        lambda *args, **kwargs: FakeBackend(),
    )
    monkeypatch.setattr("docie_bench.benchmark.runner.judge_extraction", fake_judge_extraction)
    monkeypatch.setattr(
        "docie_bench.benchmark.runner.write_report",
        lambda run_dir, metrics: run_dir / "report.html",
    )

    result = asyncio.run(
        run_benchmark(
            dataset_path=None,
            document_path=document,
            models_config_path=config,
            eval_mode=EvaluationMode.LLM_JUDGE,
            output_dir=tmp_path / "output",
        )
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert called_profiles == ["extractor"]
    assert metrics["summary"][0]["field_accuracy"] is None
    assert metrics["summary"][0]["judge_faithfulness"] == 1.0
