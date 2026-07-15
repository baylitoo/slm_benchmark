"""Encoder family: shim server contract, guard parsing, guarded security proxy."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docie_bench.agents.api import configure_http_transport
from docie_bench.agents.api import router as agents_router
from docie_bench.agents.guard import (
    GuardAnalysisError,
    _parse_entities,
    labels_from_entities,
)
from docie_bench.encoders.server import create_encoder_app
from docie_bench.llm.model_profiles import ModelProfile

# ── encoder shim server (fake backend injected) ─────────────────────────────


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], float]] = []

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        self.calls.append((text, labels, threshold))
        needle = "Jean Dupont"
        start = text.find(needle)
        if start < 0:
            return []
        return [
            {"type": "person", "value": needle, "start": start, "end": start + len(needle), "score": 0.99}
        ]


@pytest.fixture()
def encoder_client() -> tuple[TestClient, FakeBackend]:
    backend = FakeBackend()
    app = create_encoder_app(model_id="fake-encoder", backend=backend)
    return TestClient(app), backend


def test_encoder_healthz_and_models(encoder_client) -> None:
    client, _ = encoder_client
    assert client.get("/healthz").json()["kind"] == "encoder"
    models = client.get("/v1/models").json()
    assert [m["id"] for m in models["data"]] == ["fake-encoder"]


def test_encoder_chat_returns_entities_json(encoder_client) -> None:
    client, backend = encoder_client
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Report by Jean Dupont today"}]},
    )
    assert response.status_code == 200
    body = response.json()
    payload = json.loads(body["choices"][0]["message"]["content"])
    assert payload == body["docie_encoder"]
    assert payload["entities"][0]["type"] == "person"
    assert payload["entities"][0]["value"] == "Jean Dupont"
    # Defaults applied when the request carries no labels/threshold.
    _, labels, threshold = backend.calls[-1]
    assert "person" in labels
    assert threshold == 0.5


def test_encoder_chat_honours_labels_and_threshold(encoder_client) -> None:
    client, backend = encoder_client
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "labels": ["project name"],
            "threshold": 0.8,
        },
    )
    _, labels, threshold = backend.calls[-1]
    assert labels == ["project name"]
    assert threshold == 0.8


def test_encoder_chat_requires_user_text(encoder_client) -> None:
    client, _ = encoder_client
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "system", "content": "x"}]}
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_encoder_joins_multimodal_text_parts(encoder_client) -> None:
    client, backend = encoder_client
    client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Report by"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                        {"type": "text", "text": "Jean Dupont"},
                    ],
                }
            ]
        },
    )
    text, _, _ = backend.calls[-1]
    assert text == "Report by\nJean Dupont"


# ── guard parsing (pure) ─────────────────────────────────────────────────────


def test_labels_from_entities_lowercases_and_despaces() -> None:
    assert labels_from_entities(["CREDIT_CARD", "EMAIL"]) == ["credit card", "email"]
    assert labels_from_entities(None) is None


def test_parse_entities_relocates_bad_spans_and_drops_hallucinations() -> None:
    text = "mail of Jean Dupont"
    payload = {
        "entities": [
            {"type": "person", "value": "Jean Dupont", "start": 0, "end": 5, "score": 0.9},
            {"type": "person", "value": "Marie Curie", "start": 0, "end": 11, "score": 0.9},
        ]
    }
    parsed = _parse_entities(payload, text)
    assert len(parsed) == 1
    assert (parsed[0].start, parsed[0].end) == (8, 19)  # relocated via find


def test_parse_entities_overlap_keeps_higher_score() -> None:
    text = "Jean Dupont"
    payload = {
        "entities": [
            {"type": "person", "value": "Jean Dupont", "start": 0, "end": 11, "score": 0.9},
            {"type": "organization", "value": "Dupont", "start": 5, "end": 11, "score": 0.4},
        ]
    }
    parsed = _parse_entities(payload, text)
    assert [e.type for e in parsed] == ["PERSON"]


def test_parse_entities_normalizes_label_to_placeholder_type() -> None:
    text = "call 0612345678"
    payload = {
        "entities": [
            {"type": "phone number", "value": "0612345678", "start": 5, "end": 15, "score": 0.8}
        ]
    }
    assert _parse_entities(payload, text)[0].type == "PHONE_NUMBER"


def test_parse_entities_requires_entities_list() -> None:
    with pytest.raises(GuardAnalysisError):
        _parse_entities({"nope": []}, "text")


# ── security proxy with a guard encoder (end to end via MockTransport) ──────

UPSTREAM = ModelProfile(name="alpha", model="up-alpha", base_url="http://upstream/v1", api_key="k")
GUARD = ModelProfile(name="guard-encoder", model="fake-encoder", base_url="http://guard/v1", api_key="k")


@pytest.fixture()
def guarded_api(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "guard-encoder":
            return GUARD
        if model_profile in (None, "alpha"):
            return UPSTREAM
        return replace(UPSTREAM, name=str(model_profile), model=str(model_profile))

    monkeypatch.setattr(
        "docie_bench.agents.runtime.resolve_extraction_profile", fake_resolver
    )

    captured: list[httpx.Request] = []
    state = {"guard_status": 200}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        if request.url.host == "guard":
            if state["guard_status"] != 200:
                return httpx.Response(state["guard_status"], json={"error": "guard down"})
            text = body["messages"][-1]["content"]
            needle = "Jean Dupont"
            start = text.find(needle)
            entities = (
                []
                if start < 0
                else [
                    {
                        "type": "person",
                        "value": needle,
                        "start": start,
                        "end": start + len(needle),
                        "score": 0.99,
                    }
                ]
            )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-encoder",
                    "object": "chat.completion",
                    "model": "fake-encoder",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"entities": entities}),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "docie_encoder": {"entities": entities},
                },
            )
        last = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"echo: {last}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(agents_router)
    client = TestClient(app)
    yield client, captured, state
    configure_http_transport(None)


def _create_guarded(client: TestClient, **option_overrides: object) -> None:
    options: dict[str, object] = {
        "mode": "placeholder",
        "guard_model": "guard-encoder",
        "guard_labels": ["person"],
    }
    options.update(option_overrides)
    response = client.post(
        "/v1/agents",
        json={
            "name": "guarded",
            "template": "proxy-security",
            "model_profile": "alpha",
            "options": options,
        },
    )
    assert response.status_code == 201, response.text


def test_guarded_proxy_masks_encoder_entities(guarded_api) -> None:
    client, captured, _ = guarded_api
    _create_guarded(client)
    response = client.post(
        "/v1/agents/guarded/chat/completions",
        json={"messages": [{"role": "user", "content": "Report by Jean Dupont today"}]},
    )
    assert response.status_code == 200, response.text

    guard_calls = [r for r in captured if r.url.host == "guard"]
    upstream_calls = [r for r in captured if r.url.host == "upstream"]
    assert json.loads(guard_calls[0].content)["labels"] == ["person"]
    sent = json.loads(upstream_calls[0].content)
    assert sent["messages"][-1]["content"] == "Report by [PERSON_1] today"

    report = response.json()["docie_agent"]["pii"]
    assert report["analyzer"] == "guard:guard-encoder"
    assert report["entities"] == [{"type": "PERSON", "count": 1}]
    assert "Jean Dupont" not in json.dumps(response.json()["docie_agent"])


def test_guarded_proxy_fails_closed_when_guard_down(guarded_api) -> None:
    client, captured, state = guarded_api
    _create_guarded(client)
    state["guard_status"] = 500
    response = client.post(
        "/v1/agents/guarded/chat/completions",
        json={"messages": [{"role": "user", "content": "mail jean@acme.fr"}]},
    )
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "guard_unavailable"
    # Nothing may reach the backing model when the analyzer is dead.
    assert [r for r in captured if r.url.host == "upstream"] == []


def test_guarded_proxy_optional_regex_fallback(guarded_api) -> None:
    client, captured, state = guarded_api
    _create_guarded(client, guard_fallback="regex")
    state["guard_status"] = 500
    response = client.post(
        "/v1/agents/guarded/chat/completions",
        json={"messages": [{"role": "user", "content": "mail jean@acme.fr"}]},
    )
    assert response.status_code == 200, response.text
    sent = json.loads([r for r in captured if r.url.host == "upstream"][0].content)
    assert sent["messages"][-1]["content"] == "mail [EMAIL_1]"
    report = response.json()["docie_agent"]["pii"]
    assert report["degraded_to_regex"] is True


def test_guarded_proxy_block_mode_uses_encoder_findings(guarded_api) -> None:
    client, captured, _ = guarded_api
    _create_guarded(client, mode="block")
    response = client.post(
        "/v1/agents/guarded/chat/completions",
        json={"messages": [{"role": "user", "content": "Report by Jean Dupont"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "pii_blocked"
    assert [r for r in captured if r.url.host == "upstream"] == []
