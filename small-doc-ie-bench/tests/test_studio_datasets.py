"""GET /v1/studio/datasets — the registered-dataset listing the Benchmark form
picks from, instead of a free-text guess at a name that may not exist."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import docie_bench.api as api
import docie_bench.benchmark.registry as registry_module


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def test_lists_registered_datasets_with_latest_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        """
registry_version: 1
datasets:
  sample:
    description: Bundled multilingual invoice and identity document sample
    latest: "1.0.0"
    versions:
      "1.0.0":
        manifest: sample_dataset/manifest.jsonl
        dataset_hash: "sha256:abc"
        created_at: "2026-06-10T21:56:08+00:00"
        statistics:
          documents: 9
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_module, "DEFAULT_REGISTRY_PATH", registry_path)

    resp = client.get("/v1/studio/datasets")

    assert resp.status_code == 200
    (entry,) = resp.json()
    assert entry["name"] == "sample"
    assert entry["latest"] == "1.0.0"
    assert entry["versions"] == ["1.0.0"]
    assert entry["documents"] == 9
    assert "multilingual" in entry["description"]


def test_missing_registry_file_degrades_to_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    monkeypatch.setattr(registry_module, "DEFAULT_REGISTRY_PATH", tmp_path / "nope.yaml")

    resp = client.get("/v1/studio/datasets")

    assert resp.status_code == 200
    assert resp.json() == []


def test_dataset_with_no_latest_version_has_null_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    registry_path = tmp_path / "datasets.yaml"
    registry_path.write_text(
        "registry_version: 1\ndatasets:\n  empty_one:\n    versions: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_module, "DEFAULT_REGISTRY_PATH", registry_path)

    (entry,) = client.get("/v1/studio/datasets").json()

    assert entry["name"] == "empty_one"
    assert entry["latest"] is None
    assert entry["documents"] is None
