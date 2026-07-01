"""End-to-end negotiation through ``run_benchmark`` against a stubbed runtime.

No live model: ``httpx.MockTransport`` stands in for the endpoint. The stub
returns empty content for ``json_schema`` (the defect) and valid JSON for
``json_object``, proving the full path (runner -> ExtractionService ->
OpenAICompatibleClient ladder) yields non-zero ``ok_rate``, records the
effective style, persists the probe into the manifest, and fails loudly when the
validity gate is tripped.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from docie_bench.benchmark.metrics import ValidityGateError
from docie_bench.benchmark.runner import run_benchmark
from docie_bench.llm.model_gateway import reset_gateway_state
from docie_bench.settings import get_settings


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    # Keep OCR off-disk-cache and out of the repo tree; the extraction still runs
    # real OCR on the .txt document, only the write-through cache is disabled.
    monkeypatch.setenv("OCR_CACHE_ENABLED", "false")
    monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path / "ocr-cache"))
    reset_gateway_state()
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    reset_gateway_state()


def _install_stub_transport(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    real_client = httpx.AsyncClient

    def factory(*_args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        kwargs.pop("timeout", None)
        kwargs.pop("headers", None)
        return real_client(
            base_url=kwargs.get("base_url", ""), transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr("docie_bench.llm.openai_client.httpx.AsyncClient", factory)


def _inputs(tmp_path: Path, *, style_probe: bool = True) -> tuple[Path, Path]:
    document = tmp_path / "doc-0.txt"
    document.write_text("Invoice INV-0\nTotal 1200.00 EUR\n", encoding="utf-8")
    dataset = tmp_path / "manifest.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "doc_id": "doc-0",
                "file_path": document.name,
                "schema_name": "invoice",
                "ground_truth": {"invoice_number": "INV-0"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    models = tmp_path / "models.yaml"
    models.write_text(
        """
profiles:
  stub:
    model: stub-model
    base_url: http://stub/v1
    api_key: local-not-used
    response_format_style: openai_json_schema
    capability_discovery: optional
    retry_max_attempts: 1
""",
        encoding="utf-8",
    )
    return dataset, models


_VALID_EXTRACTION = json.dumps(
    {
        "invoice_number": {"value": "INV-0", "evidence_ids": []},
        "total_ttc": {"amount": "1200.00", "currency": "EUR", "evidence_ids": []},
    }
)


def _completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def _downgrade_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/models"):
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "stub-model",
                        "capabilities": {
                            "response_format_styles": ["openai_json_schema", "json_object"]
                        },
                    }
                ]
            },
        )
    payload = json.loads(request.read().decode("utf-8"))
    rf = payload.get("response_format")
    style = rf.get("type") if rf else "none"
    # The empty-content defect: json_schema yields nothing; json_object works.
    if style == "json_schema":
        return _completion("")
    return _completion(_VALID_EXTRACTION)


def test_benchmark_nonzero_ok_rate_and_probe_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset, models = _inputs(tmp_path)
    _install_stub_transport(monkeypatch, _downgrade_handler)

    result = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            output_dir=tmp_path / "run",
            probe=True,
        )
    )

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    summary = metrics["summary"][0]
    # CI-style assertion: the ladder makes ok_rate non-zero despite json_schema
    # returning empty content.
    assert summary["ok_rate"] == 1.0

    rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # The EFFECTIVE (downgraded) style is recorded into predictions.
    assert rows[0]["response_format_style"] == "json_object"

    # The probe result is persisted into the manifest for reproducibility.
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    probes = manifest["capability_probes"]
    assert probes["enabled"] is True
    assert probes["results"]["stub"]["effective_style"] == "json_object"
    assert probes["results"]["stub"]["rejected_styles"] == ["openai_json_schema"]
    # Probe output is NOT folded into the input fingerprint.
    assert "capability_probes" not in manifest["inputs"]


def test_benchmark_validity_gate_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset, models = _inputs(tmp_path)

    # Every style returns empty content, so no extraction ever parses: valid_rate
    # collapses to 0. Without the gate this would silently score zeros.
    def empty_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "stub-model"}]})
        return _completion("")

    _install_stub_transport(monkeypatch, empty_handler)

    run_dir = tmp_path / "run"
    with pytest.raises(ValidityGateError, match="valid_rate"):
        asyncio.run(
            run_benchmark(
                dataset_path=dataset,
                models_config_path=models,
                output_dir=run_dir,
                probe=False,
                valid_rate_threshold=0.5,
            )
        )

    # Metrics were persisted BEFORE the raise, so the failure is debuggable.
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validity_gate"]["passed"] is False
    assert metrics["validity_gate"]["failing_profiles"][0]["model_profile"] == "stub"
    assert metrics["summary"][0]["valid_rate"] == 0.0
