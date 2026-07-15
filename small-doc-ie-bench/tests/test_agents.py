"""Agents: PII analyzer, registry persistence, and the OpenAI-compatible API."""

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docie_bench.agents import pii
from docie_bench.agents.api import configure_http_transport
from docie_bench.agents.api import router as agents_router
from docie_bench.agents.registry import (
    AgentConflictError,
    AgentNotFoundError,
    AgentRegistry,
)
from docie_bench.agents.spec import AgentSpec
from docie_bench.llm.model_profiles import ModelProfile

# ── PII analyzer ─────────────────────────────────────────────────────────────

SAMPLE = (
    "Contact jean.dupont@acme.fr or +33 6 12 34 56 78. "
    "IBAN DE89 3704 0044 0532 0130 00, card 4111 1111 1111 1111, ip 10.0.0.1."
)


def test_analyze_detects_expected_types() -> None:
    types = {entity.type for entity in pii.analyze(SAMPLE)}
    assert types == {"EMAIL", "PHONE", "IBAN", "CREDIT_CARD", "IP_ADDRESS"}


def test_analyze_respects_entity_filter() -> None:
    types = {entity.type for entity in pii.analyze(SAMPLE, ["EMAIL"])}
    assert types == {"EMAIL"}


def test_luhn_rejects_non_card_digit_runs() -> None:
    # 16 digits failing the Luhn checksum must not be flagged as a card.
    assert pii.analyze("order ref 1234 5678 9012 3456", ["CREDIT_CARD"]) == []


def test_invalid_iban_checksum_rejected() -> None:
    assert pii.analyze("IBAN DE00 3704 0044 0532 0130 00", ["IBAN"]) == []


def test_anonymize_stable_placeholders_and_restore() -> None:
    text = "mail jean@acme.fr, again jean@acme.fr, other marie@acme.fr"
    found = pii.analyze(text, ["EMAIL"])
    masked, mapping = pii.anonymize(text, found)
    assert masked == "mail [EMAIL_1], again [EMAIL_1], other [EMAIL_2]"
    assert mapping == {"[EMAIL_1]": "jean@acme.fr", "[EMAIL_2]": "marie@acme.fr"}
    assert pii.deanonymize(masked, mapping) == text


def test_anonymize_shares_placeholders_across_calls() -> None:
    mapping: dict[str, str] = {}
    first, _ = pii.anonymize("a jean@acme.fr", pii.analyze("a jean@acme.fr"), placeholders=mapping)
    second, _ = pii.anonymize("b jean@acme.fr", pii.analyze("b jean@acme.fr"), placeholders=mapping)
    assert "[EMAIL_1]" in first and "[EMAIL_1]" in second


# ── registry ─────────────────────────────────────────────────────────────────


def _spec(name: str = "pii-proxy", **overrides: object) -> AgentSpec:
    base: dict[str, object] = {
        "name": name,
        "kind": "proxy_security",
        "model_profile": "alpha",
        "options": {"mode": "placeholder"},
    }
    base.update(overrides)
    return AgentSpec.model_validate(base)


def test_registry_crud_roundtrip(tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "agents.json")
    assert registry.list() == []
    registry.create(_spec())
    assert [spec.name for spec in registry.list()] == ["pii-proxy"]

    with pytest.raises(AgentConflictError):
        registry.create(_spec())

    updated = registry.update("pii-proxy", {"enabled": False})
    assert updated.enabled is False
    assert registry.get("pii-proxy").enabled is False

    registry.delete("pii-proxy")
    with pytest.raises(AgentNotFoundError):
        registry.get("pii-proxy")


def test_registry_update_keeps_name_and_created_at(tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "agents.json")
    created = registry.create(_spec())
    updated = registry.update("pii-proxy", {"name": "sneaky", "description": "d"})
    assert updated.name == "pii-proxy"
    assert updated.created_at == created.created_at


# ── API (router mounted standalone; upstream via MockTransport) ─────────────

UPSTREAM = ModelProfile(
    name="alpha", model="up-alpha", base_url="http://upstream/v1", api_key="k"
)


@pytest.fixture()
def api(tmp_path, monkeypatch) -> tuple[TestClient, list[httpx.Request]]:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile in (None, "alpha"):
            return UPSTREAM
        return replace(UPSTREAM, name=str(model_profile), model=str(model_profile))

    monkeypatch.setattr(
        "docie_bench.agents.runtime.resolve_extraction_profile", fake_resolver
    )

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
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
    yield client, captured
    configure_http_transport(None)


def _create_proxy(client: TestClient, **overrides: object) -> dict:
    payload: dict[str, object] = {
        "name": "pii-proxy",
        "template": "proxy-security",
        "model_profile": "alpha",
    }
    payload.update(overrides)
    response = client.post("/v1/agents", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_templates_listed(api) -> None:
    client, _ = api
    ids = {t["id"] for t in client.get("/v1/agents/templates").json()}
    assert ids == {"proxy-security", "ocr-agent", "custom"}


def test_create_from_template_and_list(api) -> None:
    client, _ = api
    created = _create_proxy(client)
    assert created["kind"] == "proxy_security"
    assert created["endpoint"] == "/v1/agents/pii-proxy"
    assert created["options"]["mode"] == "placeholder"

    listed = client.get("/v1/agents").json()
    assert [a["name"] for a in listed] == ["pii-proxy"]

    duplicate = client.post(
        "/v1/agents", json={"name": "pii-proxy", "template": "proxy-security"}
    )
    assert duplicate.status_code == 409


def test_create_requires_kind_or_template(api) -> None:
    client, _ = api
    assert client.post("/v1/agents", json={"name": "x"}).status_code == 400
    assert (
        client.post("/v1/agents", json={"name": "x", "template": "nope"}).status_code == 400
    )


def test_openai_models_lists_enabled_agents_only(api) -> None:
    client, _ = api
    _create_proxy(client)
    client.post("/v1/agents", json={"name": "off", "template": "custom", "enabled": False})
    data = client.get("/v1/agents/models").json()
    assert data["object"] == "list"
    assert [m["id"] for m in data["data"]] == ["pii-proxy"]


def test_proxy_masks_pii_before_upstream(api) -> None:
    client, captured = api
    _create_proxy(client)
    response = client.post(
        "/v1/agents/chat/completions",
        json={
            "model": "pii-proxy",
            "messages": [{"role": "user", "content": "email jean@acme.fr please"}],
        },
    )
    assert response.status_code == 200, response.text
    sent = json.loads(captured[-1].content)
    assert sent["messages"][-1]["content"] == "email [EMAIL_1] please"
    body = response.json()
    assert body["docie_agent"]["pii"]["detected"] == 1
    assert body["docie_agent"]["pii"]["entities"] == [{"type": "EMAIL", "count": 1}]
    # The raw value never appears in the report.
    assert "jean@acme.fr" not in json.dumps(body["docie_agent"])


def test_proxy_block_mode_refuses(api) -> None:
    client, captured = api
    _create_proxy(client, name="blocker", options={"mode": "block"})
    response = client.post(
        "/v1/agents/blocker/chat/completions",
        json={"messages": [{"role": "user", "content": "card 4111 1111 1111 1111"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "pii_blocked"
    assert captured == []  # nothing reached the backing model


def test_proxy_restore_pii_round_trips(api) -> None:
    client, _ = api
    _create_proxy(client, name="restorer", options={"restore_pii": True})
    response = client.post(
        "/v1/agents/restorer/chat/completions",
        json={"messages": [{"role": "user", "content": "mail jean@acme.fr"}]},
    )
    content = response.json()["choices"][0]["message"]["content"]
    assert content == "echo: mail jean@acme.fr"


def test_custom_agent_injects_system_prompt(api) -> None:
    client, captured = api
    client.post(
        "/v1/agents",
        json={
            "name": "helper",
            "template": "custom",
            "model_profile": "alpha",
            "system_prompt": "You are terse.",
        },
    )
    response = client.post(
        "/v1/agents/helper/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    sent = json.loads(captured[-1].content)
    assert sent["messages"][0] == {"role": "system", "content": "You are terse."}
    assert sent["model"] == "up-alpha"


def test_platform_route_requires_known_agent(api) -> None:
    client, _ = api
    response = client.post(
        "/v1/agents/chat/completions",
        json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


def test_disabled_agent_refuses_completions(api) -> None:
    client, _ = api
    _create_proxy(client, name="off", enabled=False)
    response = client.post(
        "/v1/agents/off/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 403
    assert response.json()["error"]["type"] == "agent_disabled"


def test_stream_wraps_completion_as_sse(api) -> None:
    client, _ = api
    _create_proxy(client)
    response = client.post(
        "/v1/agents/pii-proxy/chat/completions",
        json={
            "model": "pii-proxy",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text


def test_update_and_delete_agent(api) -> None:
    client, _ = api
    _create_proxy(client)
    updated = client.put("/v1/agents/pii-proxy", json={"enabled": False}).json()
    assert updated["enabled"] is False
    assert client.delete("/v1/agents/pii-proxy").status_code == 200
    assert client.get("/v1/agents/pii-proxy").status_code == 404


def test_per_agent_models_advertises_single_id(api) -> None:
    client, _ = api
    _create_proxy(client)
    data = client.get("/v1/agents/pii-proxy/models").json()
    assert [m["id"] for m in data["data"]] == ["pii-proxy"]
