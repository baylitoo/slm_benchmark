"""GET/POST /v1/studio/model-profiles -- the models.yaml listing + pipeline-profile
authoring routes. Read-side is what the Benchmark tab's pickers can now reference
instead of a free-text guess; write-side is the missing counterpart to #180-183's
read-side wiring (benchmark/gateway could already RUN a kind="pipeline" profile,
nothing let an operator CREATE one without hand-editing the file on the server)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import docie_bench.api as api
import docie_bench.inngest.studio_api as studio_api

FIXTURE = """\
profiles:
  extractor_llm:
    model: qwen3:4b
    base_url: http://localhost:11434/v1
    api_key: local-not-used

  vision_llm:
    model: vl-model
    base_url: http://localhost:8088/v1
    api_key: local-not-used
    vision: true
"""


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


@pytest.fixture
def models_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(FIXTURE, encoding="utf-8")
    monkeypatch.setattr(studio_api, "MODELS_CONFIG_PATH", path)
    return path


def test_lists_profiles_from_models_yaml(models_yaml: Path, client: TestClient) -> None:
    resp = client.get("/v1/studio/model-profiles")

    assert resp.status_code == 200
    names = {p["name"] for p in resp.json()}
    assert names == {"extractor_llm", "vision_llm"}
    vision = next(p for p in resp.json() if p["name"] == "vision_llm")
    assert vision["vision"] is True
    assert vision["kind"] == "passthrough"


def test_missing_models_yaml_degrades_to_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    monkeypatch.setattr(studio_api, "MODELS_CONFIG_PATH", tmp_path / "nope.yaml")

    resp = client.get("/v1/studio/model-profiles")

    assert resp.status_code == 200
    assert resp.json() == []


def test_creates_pipeline_profile_with_ocr_backend(
    models_yaml: Path, client: TestClient
) -> None:
    resp = client.post(
        "/v1/studio/model-profiles/pipeline",
        json={"name": "invoice_pipeline", "extractor": "extractor_llm", "ocr_backend": "tesseract"},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "invoice_pipeline"
    assert body["kind"] == "pipeline"
    assert body["options"] == {"extractor": "extractor_llm", "ocr_backend": "tesseract"}

    # Actually landed on disk, reachable via the listing route too.
    names = {p["name"] for p in client.get("/v1/studio/model-profiles").json()}
    assert "invoice_pipeline" in names


def test_creates_pipeline_profile_with_ocr_model(
    models_yaml: Path, client: TestClient
) -> None:
    resp = client.post(
        "/v1/studio/model-profiles/pipeline",
        json={"name": "vlm_pipeline", "extractor": "extractor_llm", "ocr_model": "vision_llm"},
    )

    assert resp.status_code == 201
    assert resp.json()["options"] == {"extractor": "extractor_llm", "ocr_model": "vision_llm"}


def test_duplicate_name_is_409(models_yaml: Path, client: TestClient) -> None:
    resp = client.post(
        "/v1/studio/model-profiles/pipeline",
        json={"name": "extractor_llm", "extractor": "extractor_llm", "ocr_backend": "tesseract"},
    )

    assert resp.status_code == 409


def test_invalid_extractor_is_422(models_yaml: Path, client: TestClient) -> None:
    resp = client.post(
        "/v1/studio/model-profiles/pipeline",
        json={"name": "p", "extractor": "nope", "ocr_backend": "tesseract"},
    )

    assert resp.status_code == 422


def test_both_ocr_backend_and_ocr_model_is_422(models_yaml: Path, client: TestClient) -> None:
    resp = client.post(
        "/v1/studio/model-profiles/pipeline",
        json={
            "name": "p",
            "extractor": "extractor_llm",
            "ocr_backend": "tesseract",
            "ocr_model": "vision_llm",
        },
    )

    assert resp.status_code == 422
