from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from docie_bench.benchmark.registry import (
    dataset_hash,
    dataset_statistics,
    detect_leakage,
    load_registry,
    migrate_manifest,
    register_dataset_version,
    resolve_dataset,
    validate_dataset,
)
from docie_bench.benchmark.runner import run_benchmark
from docie_bench.cli import app


def _write_manifest(
    root: Path,
    rows: list[tuple[str, str, str]],
    *,
    reverse: bool = False,
) -> Path:
    files = root / "files"
    files.mkdir(parents=True)
    manifest_rows = []
    for doc_id, split, text in rows:
        path = files / f"{doc_id}.txt"
        path.write_text(text, encoding="utf-8")
        manifest_rows.append(
            {
                "doc_id": doc_id,
                "file_path": f"files/{doc_id}.txt",
                "schema_name": "invoice",
                "language": "en",
                "split": split,
                "ground_truth": {"invoice_number": doc_id},
            }
        )
    if reverse:
        manifest_rows.reverse()
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in manifest_rows) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_hash_is_stable_across_manifest_location_paths_and_row_order(tmp_path: Path):
    rows = [("a", "train", "Invoice A"), ("b", "test", "Invoice B")]
    first = _write_manifest(tmp_path / "first", rows)
    second = _write_manifest(tmp_path / "second", rows, reverse=True)

    assert dataset_hash(resolve_dataset(first).items) == dataset_hash(resolve_dataset(second).items)


def test_registry_versions_resolve_latest_and_detect_tampering(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "dataset", [("a", "test", "Invoice A")])
    registry_path = tmp_path / "datasets.yaml"

    entry = register_dataset_version(
        registry_path=registry_path,
        name="invoices",
        version="1.0.0",
        manifest_path=manifest,
        description="Test invoices",
    )
    resolved = resolve_dataset("invoices", registry_path=registry_path)

    assert resolved.reference == "invoices@1.0.0"
    assert resolved.dataset_hash == entry.dataset_hash
    assert load_registry(registry_path).datasets["invoices"].latest == "1.0.0"

    (manifest.parent / "files" / "a.txt").write_text("Changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_dataset("invoices@1.0.0", registry_path=registry_path)


def test_registry_rejects_invalid_and_duplicate_versions(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "dataset", [("a", "test", "Invoice A")])
    registry_path = tmp_path / "datasets.yaml"

    with pytest.raises(ValueError, match="Dataset name"):
        register_dataset_version(
            registry_path=registry_path,
            name="invalid@name",
            version="1.0.0",
            manifest_path=manifest,
        )
    register_dataset_version(
        registry_path=registry_path,
        name="invoices",
        version="1.0.0",
        manifest_path=manifest,
    )
    with pytest.raises(ValueError, match="already exists"):
        register_dataset_version(
            registry_path=registry_path,
            name="invoices",
            version="1.0.0",
            manifest_path=manifest,
        )


def test_validation_reports_duplicates_missing_files_and_statistics(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "dataset", [("a", "test", "Invoice A")])
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    row["file_path"] = "files/missing.txt"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(manifest)

    assert report["valid"] is False
    assert any("Duplicate doc_id" in error for error in report["errors"])
    assert any("file does not exist" in error for error in report["errors"])


def test_validation_rejects_empty_dataset(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")

    report = validate_dataset(manifest)

    assert report["valid"] is False
    assert report["errors"] == ["Dataset must contain at least one document"]


def test_exact_and_near_duplicate_leakage_are_cross_split_only(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "dataset",
        [
            ("train-exact", "train", "Invoice number 123 total 99 EUR"),
            ("test-exact", "test", "Invoice number 123 total 99 EUR"),
            ("train-near", "train", "Customer Acme invoice 456 total 1000 euros due tomorrow"),
            ("test-near", "test", "Customer Acme invoice 456 total 1001 euros due tomorrow"),
            ("train-same-split", "train", "Invoice number 123 total 99 EUR"),
        ],
    )
    items = resolve_dataset(manifest).items

    leakage = detect_leakage(items, near_duplicate_threshold=0.9)

    assert len(leakage["exact_duplicates"]) == 2
    assert len(leakage["near_duplicates"]) == 1
    assert validate_dataset(manifest, near_duplicate_threshold=0.9)["valid"] is False


def test_migration_adds_splits_and_preserves_resolvable_paths(tmp_path: Path):
    source = _write_manifest(tmp_path / "legacy", [("a", "unspecified", "Invoice A")])
    destination = tmp_path / "versions" / "1.0.0" / "manifest.jsonl"

    migrate_manifest(source, destination, split_map={"a": "validation"})
    resolved = resolve_dataset(destination)

    assert resolved.items[0].split == "validation"
    assert Path(resolved.items[0].file_path).read_text(encoding="utf-8") == "Invoice A"
    assert validate_dataset(destination)["valid"] is True


def test_statistics_cover_splits_schemas_languages_and_labels(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "dataset",
        [("a", "train", "Invoice A"), ("b", "test", "Invoice B")],
    )

    stats = dataset_statistics(resolve_dataset(manifest).items)

    assert stats["documents"] == 2
    assert stats["splits"] == {"test": 1, "train": 1}
    assert stats["schemas"] == {"invoice": 2}
    assert stats["languages"] == {"en": 2}
    assert stats["labeled_documents"] == 2


def test_dataset_cli_versions_inspects_and_validates(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "dataset", [("a", "test", "Invoice A")])
    registry = tmp_path / "datasets.yaml"
    runner = CliRunner()

    versioned = runner.invoke(
        app,
        [
            "dataset",
            "version",
            "invoices",
            "1.0.0",
            "--manifest",
            str(manifest),
            "--registry",
            str(registry),
        ],
    )
    inspected = runner.invoke(
        app, ["dataset", "inspect", "invoices@1.0.0", "--registry", str(registry)]
    )
    validated = runner.invoke(
        app, ["dataset", "validate", "invoices@1.0.0", "--registry", str(registry)]
    )

    assert versioned.exit_code == 0, versioned.output
    assert inspected.exit_code == 0, inspected.output
    assert validated.exit_code == 0, validated.output
    assert "dataset_hash" in inspected.output


def test_dataset_validate_cli_reports_invalid_direct_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")

    result = CliRunner().invoke(app, ["dataset", "validate", str(manifest)])

    assert result.exit_code == 1
    assert "Dataset must contain at least one document" in result.output


def test_benchmark_records_resolved_dataset_identity(monkeypatch, tmp_path: Path):
    manifest = _write_manifest(
        tmp_path / "dataset",
        [("a", "test", "Invoice A"), ("b", "train", "Invoice B")],
    )
    config = tmp_path / "models.yaml"
    config.write_text("profiles: {}", encoding="utf-8")
    profile = SimpleNamespace(name="extractor", vision=False)

    class FakeExtractionService:
        def __init__(self, selected_profile):
            assert selected_profile is profile

        async def extract_from_file(self, **kwargs):
            return SimpleNamespace(
                schema_name="invoice",
                dynamic_schema=None,
                latency_ms=1,
                validation=SimpleNamespace(model_dump=lambda: {"valid": True}),
                result={"invoice_number": {"value": "a", "evidence_ids": ["b1"]}},
            )

    monkeypatch.setattr(
        "docie_bench.benchmark.runner.get_settings",
        lambda: SimpleNamespace(default_ocr_backend="pdf_text", runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(
        "docie_bench.benchmark.runner.load_model_profiles", lambda path: {"extractor": profile}
    )
    monkeypatch.setattr("docie_bench.benchmark.runner.ExtractionService", FakeExtractionService)
    monkeypatch.setattr(
        "docie_bench.benchmark.runner.write_report",
        lambda run_dir, metrics: run_dir / "report.html",
    )

    result = asyncio.run(
        run_benchmark(
            dataset_path=manifest,
            models_config_path=config,
            output_dir=tmp_path / "output",
            split="test",
        )
    )
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert metrics["dataset"]["dataset_hash"].startswith("sha256:")
    assert metrics["dataset"]["manifest_path"] == str(manifest.resolve())
    assert metrics["dataset"]["selected_split"] == "test"
    assert [row["doc_id"] for row in metrics["rows"]] == ["a"]
    assert metrics["rows"][0]["split"] == "test"


def test_registry_yaml_is_human_readable(tmp_path: Path):
    manifest = _write_manifest(tmp_path / "dataset", [("a", "test", "Invoice A")])
    registry = tmp_path / "datasets.yaml"

    register_dataset_version(
        registry_path=registry,
        name="invoices",
        version="1.0.0",
        manifest_path=manifest,
    )
    raw = yaml.safe_load(registry.read_text(encoding="utf-8"))

    assert raw["registry_version"] == 1
    assert raw["datasets"]["invoices"]["versions"]["1.0.0"]["statistics"]["documents"] == 1
