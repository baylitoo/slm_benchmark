"""Per-token logprob confidence (#335): pure aggregation logic, request-wiring
through `OpenAICompatibleClient.chat_json`, and end-to-end attachment through
`ExtractionService`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from docie_bench.extract.logprob_confidence import (
    attach_model_confidence,
    compute_field_confidences,
    find_value_span,
    min_logprob_in_span,
    reconstruct_generated_text,
)
from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_gateway import reset_gateway_state
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.openai_client import OpenAICompatibleClient
from docie_bench.serving.runtime import LlamaCppRuntime, OllamaRuntime, RuntimeFeature, VLLMRuntime


@pytest.fixture(autouse=True)
def _reset_gateway() -> None:
    reset_gateway_state()


def _profile(**overrides: Any) -> ModelProfile:
    values: dict[str, Any] = {
        "name": "test",
        "model": "test-model",
        "base_url": "http://model.test/v1",
        "api_key": "test",
        "response_format_style": "openai_json_schema",
        "retry_max_attempts": 1,
        "retry_backoff_base_seconds": 0,
        "retry_backoff_max_seconds": 0,
        "queue_timeout_seconds": 1,
    }
    values.update(overrides)
    return ModelProfile(**values)


# ---------------------------------------------------------------------------
# RuntimeFeature advertisement parity
# ---------------------------------------------------------------------------


def test_logprobs_feature_advertised_llamacpp_only() -> None:
    assert RuntimeFeature.LOGPROBS in LlamaCppRuntime.features
    assert RuntimeFeature.LOGPROBS not in VLLMRuntime.features
    assert RuntimeFeature.LOGPROBS not in OllamaRuntime.features


# ---------------------------------------------------------------------------
# Pure aggregation logic
# ---------------------------------------------------------------------------


def test_reconstruct_generated_text_concatenates_tokens_in_order() -> None:
    tokens = [{"token": "ab", "logprob": -0.1}, {"token": "cd", "logprob": -0.2}]
    result = reconstruct_generated_text(tokens)
    assert result is not None
    text, spans = result
    assert text == "abcd"
    assert spans == [(0, 2, -0.1), (2, 4, -0.2)]


def test_reconstruct_generated_text_none_on_empty_or_malformed() -> None:
    assert reconstruct_generated_text(None) is None
    assert reconstruct_generated_text([]) is None
    assert reconstruct_generated_text([{"token": "x"}]) is None  # missing logprob
    assert reconstruct_generated_text([{"logprob": -0.1}]) is None  # missing token


def test_find_value_span_prefers_key_correlated_match() -> None:
    text = '{"currency":"EUR","invoice_number":"INV-1"}'
    span = find_value_span(text, "invoice_number", "INV-1")
    assert span is not None
    start, end = span
    assert text[start:end] == '"INV-1"'


def test_find_value_span_numeric_and_missing() -> None:
    text = '{"vat_rate":20}'
    span = find_value_span(text, "vat_rate", 20)
    assert span is not None
    start, end = span
    assert text[start:end] == "20"
    assert find_value_span(text, "missing_field", "nope") is None


def test_min_logprob_in_span_picks_the_lowest_overlapping_token() -> None:
    spans = [(0, 5, -0.1), (5, 10, -9.0), (10, 15, -0.2)]
    assert min_logprob_in_span(spans, 3, 12) == -9.0
    assert min_logprob_in_span(spans, 20, 25) is None


def _hand_built_invoice_response() -> tuple[dict[str, Any], dict[str, Any]]:
    """A minimal flat invoice payload plus a matching llama-server-shaped
    response, split into three hand-picked token pieces so the MIN-aggregated
    result for `invoice_number` is verifiable by inspection: the middle token
    deliberately carries a very low logprob and its span overlaps the
    `"INV-42"` value, so it -- not the surrounding tokens -- must win the MIN.
    """
    content = '{"invoice_number":"INV-42","subtotal":{"amount":100,"currency":"EUR"}}'
    value_start = content.index('"INV-42"')
    split1 = value_start + 4  # inside the quoted value
    split2 = value_start + 8  # exactly at the end of the quoted value
    tokens = [
        {"token": content[:split1], "logprob": -0.1},
        {"token": content[split1:split2], "logprob": -7.5},
        {"token": content[split2:], "logprob": -0.2},
    ]
    flat_result = {
        "invoice_number": "INV-42",
        "subtotal": {"amount": 100, "currency": "EUR"},
    }
    raw_response = {
        "choices": [
            {
                "message": {"content": content},
                "logprobs": {"content": tokens},
            }
        ]
    }
    return flat_result, raw_response


def test_compute_field_confidences_min_aggregates_and_skips_nested() -> None:
    flat_result, raw_response = _hand_built_invoice_response()
    confidences = compute_field_confidences(flat_result, raw_response)
    assert confidences["invoice_number"] == -7.5
    # MoneyField's raw value is a nested object -- not resolved this round.
    assert confidences["subtotal"] is None


def test_compute_field_confidences_all_none_on_reconstruction_mismatch() -> None:
    flat_result, raw_response = _hand_built_invoice_response()
    # Corrupt one token so concatenation no longer reproduces `message.content`.
    raw_response["choices"][0]["logprobs"]["content"][0]["token"] = "WRONG"  # noqa: S105
    confidences = compute_field_confidences(flat_result, raw_response)
    assert confidences == {"invoice_number": None, "subtotal": None}


def test_compute_field_confidences_none_without_logprobs() -> None:
    flat_result, raw_response = _hand_built_invoice_response()
    del raw_response["choices"][0]["logprobs"]
    confidences = compute_field_confidences(flat_result, raw_response)
    assert confidences == {"invoice_number": None, "subtotal": None}


def test_attach_model_confidence_walks_wrappers_lists_and_nested() -> None:
    normalized = {
        "document_type": "invoice",
        "invoice_number": {"value": "INV-42", "evidence_ids": [], "confidence": 0.5},
        "subtotal": {"amount": 100, "currency": "EUR", "evidence_ids": [], "confidence": 0.5},
        "line_items": [
            {"description": {"value": "Widget", "evidence_ids": [], "confidence": 0.5}}
        ],
        "extraction_notes": ["a note"],
    }
    attach_model_confidence(normalized, {"invoice_number": -7.5})
    assert normalized["invoice_number"]["model_confidence"] == -7.5
    assert normalized["subtotal"]["model_confidence"] is None
    assert normalized["line_items"][0]["description"]["model_confidence"] is None
    # Non-wrapper leaves are left untouched, not crashed on.
    assert normalized["document_type"] == "invoice"
    assert normalized["extraction_notes"] == ["a note"]


# ---------------------------------------------------------------------------
# Request wiring: OpenAICompatibleClient.chat_json
# ---------------------------------------------------------------------------


def _completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


async def _mock_client(profile: ModelProfile, handler: Any) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(profile)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=profile.base_url, transport=httpx.MockTransport(handler)
    )
    return client


async def test_chat_json_sends_logprobs_when_requested() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read().decode("utf-8")))
        return _completion('{"ok": true}')

    client = await _mock_client(_profile(), handler)
    try:
        await client.chat_json(
            system_prompt="s",
            user_prompt="u",
            schema_name="test",
            schema={"type": "object"},
            request_logprobs=True,
        )
    finally:
        await client.aclose()

    assert sent[0]["logprobs"] is True
    assert sent[0]["top_logprobs"] == 1


async def test_chat_json_omits_logprobs_by_default() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read().decode("utf-8")))
        return _completion('{"ok": true}')

    client = await _mock_client(_profile(), handler)
    try:
        await client.chat_json(
            system_prompt="s",
            user_prompt="u",
            schema_name="test",
            schema={"type": "object"},
        )
    finally:
        await client.aclose()

    assert "logprobs" not in sent[0]
    assert "top_logprobs" not in sent[0]


# ---------------------------------------------------------------------------
# End-to-end: ExtractionService attaches `model_confidence`
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, profile: ModelProfile, responses: list[Any], calls: list[dict[str, Any]]):
        self.profile = profile
        self._responses = responses
        self._calls = calls

    async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        self._calls.append(kwargs)
        return self._responses.pop(0)

    async def aclose(self) -> None:
        return None


def _install_fake_client(monkeypatch: Any, responses: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def factory(profile: ModelProfile) -> _FakeClient:
        return _FakeClient(profile, responses, calls)

    monkeypatch.setattr("docie_bench.extract.service.OpenAICompatibleClient", factory)
    return calls


async def test_extraction_attaches_model_confidence_when_enabled_and_llamacpp(
    monkeypatch: Any,
) -> None:
    flat_result, raw_response = _hand_built_invoice_response()
    calls = _install_fake_client(monkeypatch, [(flat_result, None, raw_response)])

    service = ExtractionService(_profile(runtime="llamacpp", logprob_confidence=True))
    response = await service.extract_from_text(
        text="", ocr_blocks=[], schema_name="invoice", schema_mode="static"
    )

    assert calls[0]["request_logprobs"] is True
    assert response.result["invoice_number"]["model_confidence"] == -7.5
    # Nested MoneyField value -- unresolved this round, honest-null.
    assert response.result["subtotal"]["model_confidence"] is None
    # A field the model never emitted at all validates to a bare `None`
    # (Pydantic's `TextField | None = None` default, not a wrapper instance)
    # -- the walk must not crash on it.
    assert response.result["vendor_name"] is None


async def test_extraction_omits_model_confidence_when_flag_off(monkeypatch: Any) -> None:
    flat_result, raw_response = _hand_built_invoice_response()
    calls = _install_fake_client(monkeypatch, [(flat_result, None, raw_response)])

    service = ExtractionService(_profile(runtime="llamacpp", logprob_confidence=False))
    response = await service.extract_from_text(
        text="", ocr_blocks=[], schema_name="invoice", schema_mode="static"
    )

    assert calls[0]["request_logprobs"] is False
    assert "model_confidence" not in response.result["invoice_number"]
    assert "model_confidence" not in response.result["subtotal"]


async def test_extraction_omits_model_confidence_for_non_llamacpp_runtime(
    monkeypatch: Any,
) -> None:
    flat_result, raw_response = _hand_built_invoice_response()
    calls = _install_fake_client(monkeypatch, [(flat_result, None, raw_response)])

    service = ExtractionService(_profile(runtime="vllm", logprob_confidence=True))
    response = await service.extract_from_text(
        text="", ocr_blocks=[], schema_name="invoice", schema_mode="static"
    )

    assert calls[0]["request_logprobs"] is False
    assert "model_confidence" not in response.result["invoice_number"]


async def test_extraction_unresolvable_span_is_none_not_a_crash(monkeypatch: Any) -> None:
    flat_result, raw_response = _hand_built_invoice_response()
    # Break token reconstruction -- the caller must degrade to all-None, never raise.
    raw_response["choices"][0]["logprobs"]["content"][0]["token"] = "WRONG"  # noqa: S105
    calls = _install_fake_client(monkeypatch, [(flat_result, None, raw_response)])

    service = ExtractionService(_profile(runtime="llamacpp", logprob_confidence=True))
    response = await service.extract_from_text(
        text="", ocr_blocks=[], schema_name="invoice", schema_mode="static"
    )

    assert calls[0]["request_logprobs"] is True
    assert response.validation.valid
    assert response.result["invoice_number"]["model_confidence"] is None
    assert response.result["subtotal"]["model_confidence"] is None


async def test_extraction_missing_runtime_declaration_defaults_to_off(monkeypatch: Any) -> None:
    flat_result, raw_response = _hand_built_invoice_response()
    calls = _install_fake_client(monkeypatch, [(flat_result, None, raw_response)])

    # logprob_confidence=True but `runtime` was never declared -- fail closed.
    service = ExtractionService(_profile(logprob_confidence=True))
    response = await service.extract_from_text(
        text="", ocr_blocks=[], schema_name="invoice", schema_mode="static"
    )

    assert calls[0]["request_logprobs"] is False
    assert "model_confidence" not in response.result["invoice_number"]
