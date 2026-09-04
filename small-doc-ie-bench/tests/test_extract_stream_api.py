"""``POST /v1/extract/stream`` (#397): SSE delta/reset/result framing for the
Playground's live-preview extraction path.

Model calls are faked at the ExtractionService boundary -- these tests pin
the ENDPOINT's own event framing (delta events precede one final `result`
event carrying the post-processed response, `reset` clears the buffer,
`/v1/extract/text` stays untouched), not chat_json's negotiation ladder
(covered by tests/test_chat_json_streaming.py).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docie_bench import api
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation


def _response(**overrides: Any) -> ExtractionResponse:
    values: dict[str, Any] = {
        "request_id": "req-1",
        "schema_name": "invoice",
        "model_profile": "fake-model",
        "document_hash": None,
        # A NuExtract-normalized value the raw deltas below never contain --
        # proves the frontend must read THIS event, not its own accumulated
        # delta text, as the result.
        "result": {"invoice_number": {"value": "INV-1"}},
        "validation": ExtractionValidation(valid=True),
        "latency_ms": 5,
    }
    values.update(overrides)
    return ExtractionResponse(**values)


class _FakeStreamingService:
    """Stands in for ExtractionService: captures the on_delta/on_reset
    callbacks it was constructed with and fires them exactly like a real
    streaming chat_json call would, before returning the canned response."""

    deltas_to_emit: list[str] = []
    emit_reset = False

    def __init__(self, profile: ModelProfile, **kwargs: Any) -> None:
        self.profile = profile
        self.on_delta = kwargs.get("on_delta")
        self.on_reset = kwargs.get("on_reset")

    async def _run(self, **_: Any) -> ExtractionResponse:
        for piece in _FakeStreamingService.deltas_to_emit:
            if self.on_delta:
                self.on_delta(piece)
        if _FakeStreamingService.emit_reset and self.on_reset:
            self.on_reset()
        return _response(model_profile=self.profile.name)

    async def extract_from_text(self, **kwargs: Any) -> ExtractionResponse:
        return await self._run(**kwargs)

    async def extract_from_file(self, **kwargs: Any) -> ExtractionResponse:
        return await self._run(**kwargs)


def _profile(name: str = "fake-model") -> ModelProfile:
    return ModelProfile(name=name, base_url="http://fake", model=name, api_key="x")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _FakeStreamingService.deltas_to_emit = ['{"invoice', '_number": "INV-1"}']
    _FakeStreamingService.emit_reset = False

    async def fake_resolve_or_error(model: str, *, session_id: str | None = None):
        return _profile(model or "studio_default")

    monkeypatch.setattr(api, "_resolve_or_error", fake_resolve_or_error)
    monkeypatch.setattr(api, "ExtractionService", _FakeStreamingService)
    monkeypatch.setattr(api, "record_extraction", lambda *a, **k: None)
    monkeypatch.setattr(api.recency, "stamp_served_profile", lambda *a, **k: None)
    return TestClient(api.app)


def _events(response) -> list[dict[str, Any]]:
    out = []
    for line in response.iter_lines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        out.append(json.loads(line[len("data: ") :]))
    return out


def test_deltas_stream_before_the_authoritative_result_event(client: TestClient) -> None:
    response = client.post(
        "/v1/extract/stream",
        json={"text": "Invoice INV-1, total 42 EUR", "deployment": "fake-model"},
    )
    assert response.status_code == 200
    events = _events(response)
    assert [e["type"] for e in events] == ["phase", "delta", "delta", "result"]
    delta_text = "".join(e["text"] for e in events if e["type"] == "delta")
    assert delta_text == '{"invoice_number": "INV-1"}'
    result_event = events[-1]
    # The final event is the post-processed ExtractionResponse, not a replay
    # of the raw delta text.
    assert result_event["result"]["result"] == {"invoice_number": {"value": "INV-1"}}
    assert result_event["result"]["model_profile"] == "fake-model"


def test_reset_event_precedes_no_further_deltas_when_a_retry_clears_the_buffer(
    client: TestClient,
) -> None:
    _FakeStreamingService.emit_reset = True
    response = client.post(
        "/v1/extract/stream",
        json={"text": "Invoice INV-1", "deployment": "fake-model"},
    )
    events = _events(response)
    # deltas, THEN reset, THEN the result -- reset only ever clears what
    # already streamed, never blocks the eventual result.
    assert [e["type"] for e in events] == ["phase", "delta", "delta", "reset", "result"]


def test_missing_text_and_content_is_a_422_before_any_sse_framing(client: TestClient) -> None:
    response = client.post("/v1/extract/stream", json={"deployment": "fake-model"})
    assert response.status_code == 422


def test_extract_text_endpoint_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing blocking route must still take chat_json's on_delta=None
    path -- this endpoint's ExtractionService construction never passes
    on_delta/on_reset."""
    captured: dict[str, Any] = {}

    class _Fake:
        def __init__(self, profile: ModelProfile, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs

        async def extract_from_text(self, **_: Any) -> ExtractionResponse:
            return _response()

    async def fake_resolve_profile(name: str | None) -> ModelProfile:
        return _profile(name or "studio_default")

    monkeypatch.setattr(api, "resolve_profile", fake_resolve_profile)
    monkeypatch.setattr(api, "ExtractionService", _Fake)
    monkeypatch.setattr(api, "record_extraction", lambda *a, **k: None)
    monkeypatch.setattr(api.recency, "stamp_served_profile", lambda *a, **k: None)
    client = TestClient(api.app)

    response = client.post("/v1/extract/text", json={"text": "hi"})
    assert response.status_code == 200
    assert "on_delta" not in captured["kwargs"]
    assert "on_reset" not in captured["kwargs"]
