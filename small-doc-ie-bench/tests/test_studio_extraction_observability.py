"""The Studio (Inngest worker) extraction path must emit the same observability
side effects as the sync API: a durable audit row plus the Prometheus request
counter. Otherwise every DocIE Studio extraction is invisible to the
Observability tab / Grafana.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from docie_bench.inngest import functions
from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation
from docie_bench.telemetry import EXTRACTION_REQUESTS

_TEST_PROFILE = "inngest-test-profile"


def _make_response() -> ExtractionResponse:
    return ExtractionResponse(
        request_id="req-test-1",
        schema_name="invoice",
        model_profile=_TEST_PROFILE,
        document_hash="deadbeef",
        result={"invoice_number": {"value": "INV-1"}},
        validation=ExtractionValidation(valid=True, errors=[], warnings=[]),
        latency_ms=42,
    )


class _FakeService:
    """Stand-in for ExtractionService whose extract_* calls skip the model."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def extract_from_text(self, **kwargs) -> ExtractionResponse:
        return _make_response()

    async def extract_from_file(self, **kwargs) -> ExtractionResponse:
        return _make_response()


def _counter_value() -> float:
    return (
        EXTRACTION_REQUESTS.labels("invoice", _TEST_PROFILE, "true")._value.get()
    )


@pytest.fixture
def audit_calls(monkeypatch):
    calls: list[dict] = []

    def _fake_save(response, *, tenant_id=None):
        calls.append({"response": response, "tenant_id": tenant_id})

    # record_extraction resolves save_extraction_audit as a module global in
    # docie_bench.storage.audit, so patch it there.
    monkeypatch.setattr("docie_bench.storage.audit.save_extraction_audit", _fake_save)
    monkeypatch.setattr(functions, "ExtractionService", _FakeService)
    return calls


def test_studio_text_extraction_records_audit_and_metric(audit_calls):
    before = _counter_value()

    result = asyncio.run(
        functions._run_extraction({"text": "Invoice total 10", "tenant_id": "acme"})
    )

    assert result["model_profile"] == _TEST_PROFILE
    # Audit row written exactly once, tenant-scoped from the event data.
    assert len(audit_calls) == 1
    assert audit_calls[0]["tenant_id"] == "acme"
    # Prometheus request counter advanced by exactly one.
    assert _counter_value() == pytest.approx(before + 1)


def test_studio_file_extraction_records_audit_and_metric(audit_calls):
    before = _counter_value()
    content_b64 = base64.b64encode(b"%PDF-1.4 fake").decode("ascii")

    result = asyncio.run(
        functions._run_extraction(
            {"content_b64": content_b64, "filename": "doc.pdf", "tenant_id": "acme"}
        )
    )

    assert result["model_profile"] == _TEST_PROFILE
    assert len(audit_calls) == 1
    assert audit_calls[0]["tenant_id"] == "acme"
    assert _counter_value() == pytest.approx(before + 1)
