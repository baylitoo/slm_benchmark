from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from docie_bench.benchmark.reproducibility import (
    ResumeDriftError,
    append_jsonl,
    load_jsonl_recover,
    write_manifest,
)
from docie_bench.benchmark.runner import run_benchmark


def _benchmark_inputs(tmp_path: Path, count: int = 2) -> tuple[Path, Path]:
    rows = []
    for index in range(count):
        document = tmp_path / f"doc-{index}.txt"
        document.write_text(f"Invoice INV-{index}", encoding="utf-8")
        rows.append(
            {
                "doc_id": f"doc-{index}",
                "file_path": document.name,
                "schema_name": "invoice",
                "ground_truth": {"invoice_number": f"INV-{index}"},
            }
        )
    dataset = tmp_path / "manifest.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    models = tmp_path / "models.yaml"
    models.write_text(
        """
profiles:
  extractor:
    model: deterministic-model
    base_url: http://user:password@extractor/v1
    temperature: 0
""",
        encoding="utf-8",
    )
    return dataset, models


def _install_runner_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    calls: list[str] = []

    class FakeExtractionService:
        def __init__(self, profile: Any) -> None:
            self.profile = profile

        async def extract_from_file(self, **kwargs: Any) -> Any:
            path = kwargs["path"]
            calls.append(path.name)
            await asyncio.sleep(0.005 if path.stem.endswith("0") else 0)
            value = path.read_text(encoding="utf-8").split()[-1]
            return SimpleNamespace(
                schema_name="invoice",
                dynamic_schema=None,
                latency_ms=5,
                validation=SimpleNamespace(model_dump=lambda: {"valid": True}),
                result={"invoice_number": {"value": value, "evidence_ids": ["b1"]}},
            )

    monkeypatch.setattr(
        "docie_bench.benchmark.runner.get_settings",
        lambda: SimpleNamespace(default_ocr_backend="pdf_text", runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr("docie_bench.benchmark.runner.ExtractionService", FakeExtractionService)
    return calls


def test_task_ids_are_stable_across_concurrency_and_manifests_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset, models = _benchmark_inputs(tmp_path)
    _install_runner_fakes(monkeypatch, tmp_path)

    first = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            model_profile="extractor",
            output_dir=tmp_path / "run-1",
            concurrency=1,
        )
    )
    second = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            model_profile="extractor",
            output_dir=tmp_path / "run-2",
            concurrency=4,
        )
    )

    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    assert first_manifest["task_ids"] == second_manifest["task_ids"]
    assert first_manifest["input_fingerprint"] == second_manifest["input_fingerprint"]
    assert "api_key" not in json.dumps(first_manifest["inputs"]["selected_profiles"])
    assert "password" not in json.dumps(first_manifest)
    assert first_manifest["invocation"]["concurrency"] == 1
    assert second_manifest["invocation"]["concurrency"] == 4
    assert "Immutable run manifest" in first.report_path.read_text(encoding="utf-8")


def test_str_dataset_path_from_cli_does_not_assume_path_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The Typer CLI passes --dataset through as a str (manifest path or registry
    # reference), so run_benchmark must not call Path-only methods on it. Every
    # other test passes a Path, which masked this. See issue #41.
    dataset, models = _benchmark_inputs(tmp_path)
    _install_runner_fakes(monkeypatch, tmp_path)

    result = asyncio.run(
        run_benchmark(
            dataset_path=str(dataset),
            models_config_path=models,
            model_profile="extractor",
            output_dir=tmp_path / "run",
        )
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    # source_hash is the resolved manifest's content hash, not a hash of the string.
    assert manifest["inputs"]["dataset"]["source_hash"].startswith("sha256:")
    assert manifest["invocation"]["dataset_path"] == str(dataset)


def test_resume_repairs_partial_tail_and_executes_only_missing_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset, models = _benchmark_inputs(tmp_path)
    calls = _install_runner_fakes(monkeypatch, tmp_path)
    output = tmp_path / "run"
    result = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            model_profile="extractor",
            output_dir=output,
            concurrency=2,
        )
    )
    original_rows = load_jsonl_recover(result.predictions_path)
    result.predictions_path.write_text(
        json.dumps(original_rows[0], sort_keys=True) + "\n" + '{"task_id":',
        encoding="utf-8",
    )
    calls.clear()

    resumed = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            model_profile="extractor",
            output_dir=output,
            concurrency=1,
            resume=True,
        )
    )

    rows = load_jsonl_recover(resumed.predictions_path)
    metrics = json.loads(resumed.metrics_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert len({row["task_id"] for row in rows}) == 2
    assert calls == ["doc-1.txt"]
    assert metrics["reproducibility"]["tasks_skipped"] == 1
    assert metrics["reproducibility"]["tasks_executed"] == 1
    assert metrics["reproducibility"]["warnings"] == ["Concurrency changed from 2 to 1"]


def test_resume_refuses_document_drift_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset, models = _benchmark_inputs(tmp_path, count=1)
    calls = _install_runner_fakes(monkeypatch, tmp_path)
    output = tmp_path / "run"
    asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            model_profile="extractor",
            output_dir=output,
        )
    )
    calls.clear()
    (tmp_path / "doc-0.txt").write_text("Changed invoice", encoding="utf-8")

    with pytest.raises(ResumeDriftError, match="dataset"):
        asyncio.run(
            run_benchmark(
                dataset_path=dataset,
                models_config_path=models,
                model_profile="extractor",
                output_dir=output,
                resume=True,
            )
        )
    assert calls == []


def test_resume_refuses_model_config_drift_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset, models = _benchmark_inputs(tmp_path, count=1)
    calls = _install_runner_fakes(monkeypatch, tmp_path)
    output = tmp_path / "run"
    asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            model_profile="extractor",
            output_dir=output,
        )
    )
    calls.clear()
    models.write_text(models.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

    with pytest.raises(ResumeDriftError, match="models_config_hash"):
        asyncio.run(
            run_benchmark(
                dataset_path=dataset,
                models_config_path=models,
                model_profile="extractor",
                output_dir=output,
                resume=True,
            )
        )
    assert calls == []


def test_concurrent_run_records_terminal_lifecycle_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset, models = _benchmark_inputs(tmp_path, count=3)
    _install_runner_fakes(monkeypatch, tmp_path)
    result = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            model_profile="extractor",
            output_dir=tmp_path / "run",
            concurrency=3,
        )
    )

    events = load_jsonl_recover(result.run_dir / "task-events.jsonl")
    completed = [event for event in events if event["state"] == "completed"]
    assert len(completed) == 3
    assert len({event["task_id"] for event in completed}) == 3


def test_failed_task_records_reason_and_is_not_retried_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset, models = _benchmark_inputs(tmp_path, count=1)
    calls: list[str] = []

    class FailingExtractionService:
        def __init__(self, profile: Any) -> None:
            pass

        async def extract_from_file(self, **kwargs: Any) -> Any:
            calls.append(kwargs["path"].name)
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        "docie_bench.benchmark.runner.get_settings",
        lambda: SimpleNamespace(default_ocr_backend="pdf_text", runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr("docie_bench.benchmark.runner.ExtractionService", FailingExtractionService)
    output = tmp_path / "run"
    result = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            model_profile="extractor",
            output_dir=output,
        )
    )
    asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            model_profile="extractor",
            output_dir=output,
            resume=True,
        )
    )

    events = load_jsonl_recover(result.run_dir / "task-events.jsonl")
    failed = [event for event in events if event["state"] == "failed"]
    assert calls == ["doc-0.txt"]
    assert len(failed) == 1
    assert "model unavailable" in failed[0]["reason"]


def test_jsonl_recovery_rejects_corruption_before_final_record(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    append_jsonl(path, {"task_id": "one"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")
        handle.write(json.dumps({"task_id": "two"}) + "\n")

    with pytest.raises(ValueError, match="Corrupt JSONL record 2"):
        load_jsonl_recover(path)


def test_manifest_creation_is_exclusive_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(path, {"version": 1})

    with pytest.raises(FileExistsError, match="immutable"):
        write_manifest(path, {"version": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
