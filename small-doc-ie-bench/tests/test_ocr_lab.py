from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from docie_bench.cli import app
from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.ocr.artifact import OCRPageImage, quality_signals
from docie_bench.ocr.metrics import layout_preservation, score_ocr
from docie_bench.ocr.runner import run_ocr_benchmark, summarize_ocr
from docie_bench.schemas.common import OCRBlock


def test_common_artifact_supports_page_images_and_quality_signals() -> None:
    image = OCRPageImage.from_bytes(page=1, media_type="image/png", data=b"png")
    quality = quality_signals(
        [OCRBlock(id="b1", text="uncertain", confidence=0.2, source="tesseract")]
    )

    assert image.sha256
    assert image.data_base64 == "cG5n"
    assert quality.low_quality is True
    assert "low_mean_confidence" in quality.reasons


def test_ocr_metrics_cover_text_and_layout() -> None:
    exact = [
        OCRBlock(id="b1", text="hello world", page=1, source="manual"),
        OCRBlock(id="b2", text="second line", page=1, source="manual"),
    ]
    score = score_ocr("hello world\nsecond line", exact)

    assert score["character_error_rate"] == 0
    assert score["word_error_rate"] == 0
    assert score["layout_preservation"] == 1
    assert layout_preservation("hello world\nsecond line", "hello\nworld second line") == 0


def test_summarize_correlates_ocr_with_extraction_accuracy(tmp_path: Path) -> None:
    extraction = tmp_path / "metrics.json"
    extraction.write_text(
        json.dumps(
            {
                "rows": [
                    {"doc_id": "a", "score": {"field_accuracy": 0.2}},
                    {"doc_id": "b", "score": {"field_accuracy": 0.9}},
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "doc_id": "a",
            "backend": "fake",
            "ok": True,
            "cache_hit": False,
            "latency_ms": 10,
            "quality": {"low_quality": True},
            "score": {
                "character_error_rate": 0.8,
                "word_error_rate": 0.8,
                "character_accuracy": 0.2,
                "layout_preservation": 0.5,
            },
        },
        {
            "doc_id": "b",
            "backend": "fake",
            "ok": True,
            "cache_hit": False,
            "latency_ms": 20,
            "quality": {"low_quality": False},
            "score": {
                "character_error_rate": 0.1,
                "word_error_rate": 0.1,
                "character_accuracy": 0.9,
                "layout_preservation": 1.0,
            },
        },
    ]

    metrics = summarize_ocr(rows, extraction)

    assert metrics["summary"][0]["low_quality_rate"] == 0.5
    assert metrics["correlations"][0]["ocr_character_accuracy_vs_field_accuracy"] == 1.0


def test_ocr_runner_reuses_cache_and_cli_does_not_need_models(tmp_path: Path) -> None:
    document = tmp_path / "document.txt"
    document.write_text("hello\nworld", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "doc_id": "doc",
                "file_path": "document.txt",
                "ocr_reference_text": "hello\nworld",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"

    first = run_ocr_benchmark(
        dataset_path=manifest,
        backends=["pdf_text"],
        output_dir=tmp_path / "first",
        cache_dir=cache_dir,
    )
    second = run_ocr_benchmark(
        dataset_path=manifest,
        backends=["pdf_text"],
        output_dir=tmp_path / "second",
        cache_dir=cache_dir,
    )
    second_metrics = json.loads(second.metrics_path.read_text(encoding="utf-8"))

    assert first.report_path.exists()
    assert second_metrics["summary"][0]["cache_hit_rate"] == 1.0
    assert second_metrics["summary"][0]["character_error_rate"] == 0

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "ocr",
            "run",
            "--dataset",
            str(manifest),
            "--output-dir",
            str(tmp_path / "cli"),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cli" / "ocr-metrics.json").exists()


@pytest.mark.asyncio
async def test_extraction_path_reuses_cached_ocr(monkeypatch, tmp_path: Path) -> None:
    document = tmp_path / "document.txt"
    document.write_text("hello", encoding="utf-8")
    calls = 0
    captured = []

    class FakeBackend:
        name = "fake"

        def version(self):
            return "1"

        def configuration(self):
            return {}

        def extract(self, path):
            nonlocal calls
            calls += 1
            return [OCRBlock(id="b1", text="hello", source="manual")]

    async def fake_extract_blocks(self, **kwargs):
        captured.append(kwargs)
        return "response"

    settings = SimpleNamespace(
        ocr_cache_enabled=True,
        ocr_cache_dir=tmp_path / "cache",
        ocr_cache_max_mb=1,
    )
    monkeypatch.setattr(
        "docie_bench.ocr.service.get_ocr_backend", lambda *args, **kwargs: FakeBackend()
    )
    monkeypatch.setattr("docie_bench.extract.service.get_settings", lambda: settings)
    monkeypatch.setattr(ExtractionService, "_extract_blocks", fake_extract_blocks)
    profile = ModelProfile(
        name="test", model="test", base_url="http://example.test/v1", api_key="test"
    )
    service = ExtractionService(profile)

    for _ in range(2):
        assert (
            await service.extract_from_file(
                path=document, ocr_backend_name="fake", schema_name="invoice"
            )
            == "response"
        )

    assert calls == 1
    assert captured[0]["blocks"][0].text == "hello"
