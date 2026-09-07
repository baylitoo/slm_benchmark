"""Regression: extraction_complete's docie_latency_ms (and ExtractionResponse.
latency_ms) lumped queue-wait time (spent behind the ModelGateway's
per-(base_url, model) semaphore) together with actual generation time, with no
way to tell them apart from a single request's numbers -- see the parallelism
audit that also produced the blocking-OCR-call fix. ExtractionService now
reads OpenAICompatibleClient.last_queue_wait_ms and threads it through as
ExtractionResponse.queue_wait_ms (and a matching docie_queue_wait_ms /
docie_generation_ms log split), defaulting to None for any client stand-in
(a benchmark FakeClient, an adapter) that doesn't expose the attribute --
mirrors the existing last_response_format_style getattr(..., None) convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile


def _profile() -> ModelProfile:
    return ModelProfile(
        name="test", model="test-model", base_url="http://test", api_key="test"
    )


class _ClientWithWait:
    last_response_format_style = "json_schema"
    last_queue_wait_ms = 1234

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], None, dict[str, Any]]:
        return {"document_type": "invoice"}, None, {}

    async def aclose(self) -> None:
        return None


class _ClientWithoutWait:
    """Stands in for a benchmark FakeClient that never heard of
    last_queue_wait_ms -- must not raise, must surface as None."""

    last_response_format_style = "json_schema"

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], None, dict[str, Any]]:
        return {"document_type": "invoice"}, None, {}

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_queue_wait_ms_surfaces_on_the_response(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "docie_bench.extract.service.OpenAICompatibleClient", _ClientWithWait
    )
    service = ExtractionService(_profile())

    response = await service.extract_from_text(
        text="irrelevant", ocr_blocks=None, schema_name="invoice"
    )

    assert response.queue_wait_ms == 1234


@pytest.mark.asyncio
async def test_queue_wait_ms_is_none_for_a_client_without_the_attribute(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "docie_bench.extract.service.OpenAICompatibleClient", _ClientWithoutWait
    )
    service = ExtractionService(_profile())

    response = await service.extract_from_text(
        text="irrelevant", ocr_blocks=None, schema_name="invoice"
    )

    assert response.queue_wait_ms is None
