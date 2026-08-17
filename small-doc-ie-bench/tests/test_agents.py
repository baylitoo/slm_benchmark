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


def test_resolve_ocr_mode_derives_and_honors_explicit() -> None:
    from docie_bench.agents.runtime import _resolve_ocr_mode

    assert _resolve_ocr_mode({}) == "ocr"  # nothing → plain OCR
    assert _resolve_ocr_mode({"extractor": "slm"}) == "ocr_extract"  # legacy pipeline
    assert _resolve_ocr_mode({"mode": "vision"}) == "vision"  # explicit wins
    assert _resolve_ocr_mode({"mode": "bogus"}) == "ocr"  # unknown → derive


def test_inject_response_format_unknown_schema_raises() -> None:
    from docie_bench.agents.runtime import AgentError, _inject_response_format

    with pytest.raises(AgentError):
        _inject_response_format({}, "not-a-real-schema")


def test_inject_response_format_builds_json_schema() -> None:
    from docie_bench.agents.runtime import _inject_response_format

    body: dict = {}
    _inject_response_format(body, "invoice")
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "invoice"
    assert rf["json_schema"]["strict"] is True
    schema = rf["json_schema"]["schema"]
    assert isinstance(schema, dict)
    assert schema["required"] == list(schema["properties"])


def test_inject_response_format_builds_saved_dynamic_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docie_bench.agents.runtime import _inject_response_format
    from docie_bench.studio import dynamic_schemas

    monkeypatch.setattr(
        dynamic_schemas,
        "get_dynamic_schema",
        lambda name: {
            "name": name,
            "spec": {
                "document_type": name,
                "fields": [
                    {"name": "full_name", "type": "string"},
                    {
                        "name": "addresses",
                        "type": "list",
                        "fields": [{"name": "city", "type": "string"}],
                    },
                ],
            },
        },
    )

    body: dict = {}
    _inject_response_format(body, "contact_record")

    response_format = body["response_format"]
    schema = response_format["json_schema"]["schema"]
    assert response_format["json_schema"]["name"] == "contact_record"
    assert set(schema["properties"]) == {"full_name", "addresses"}
    assert "evidence_ids" not in json.dumps(schema)


def _completion(model: str, content: str = "{}") -> dict:
    return {
        "id": "c",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


async def test_schema_fallback_downgrades_on_grammar_400() -> None:
    # llama.cpp fails to compile a deep schema's grammar (400 "failed to parse
    # grammar"); the vision forward downgrades json_schema -> json_object.
    from docie_bench.agents.runtime import _post_chat_with_schema

    seen: list[dict | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body.get("response_format"))
        rf = body.get("response_format") or {}
        if rf.get("type") == "json_schema":
            return httpx.Response(
                400,
                json={"error": {"message": "failed to parse grammar"}},
            )
        return httpx.Response(200, json=_completion(body["model"]))

    msgs = {"messages": [{"role": "user", "content": "x"}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _post_chat_with_schema(
            UPSTREAM, msgs, "invoice", http_client=client
        )
    # Ladder: grammar+prefill, grammar (no prefill), then json_object.
    assert [s and s["type"] for s in seen] == ["json_schema", "json_schema", "json_object"]
    assert result["choices"][0]["message"]["content"] == "{}"


async def test_schema_fallback_grammar_without_prefill_on_sampler_400() -> None:
    # Some models fail grammar-SAMPLER init only when the assistant turn is
    # prefilled ("Failed to initialize samplers"). The ladder then retries the
    # SAME grammar without the prefill — keeping schema enforcement — before
    # dropping to json_object. Verified live on lfm2.5-vl-1.6b-extract.
    from docie_bench.agents.runtime import _post_chat_with_schema

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        prefilled = (body.get("messages") or [])[-1:] == [{"role": "assistant", "content": "{"}]
        if body.get("response_format", {}).get("type") == "json_schema" and prefilled:
            return httpx.Response(
                400, json={"error": {"message": "Failed to initialize samplers: std::exception"}}
            )
        return httpx.Response(200, json=_completion(body["model"]))

    msgs = {"messages": [{"role": "user", "content": "x"}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await _post_chat_with_schema(UPSTREAM, msgs, "invoice", http_client=client)
    # 1st = grammar + prefill (400 sampler), 2nd = grammar WITHOUT prefill (200).
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["messages"][-1] == {"role": "assistant", "content": "{"}
    assert seen[1]["response_format"]["type"] == "json_schema"  # grammar KEPT
    assert seen[1]["messages"][-1] != {"role": "assistant", "content": "{"}  # no prefill
    assert len(seen) == 2  # no need to reach json_object


async def test_schema_instruction_names_fields_in_prompt() -> None:
    # The schema's fields must ride in the PROMPT (a system message), so the model
    # knows what to extract even under json_object — not just echo "OCR" garbage.
    from docie_bench.agents.runtime import _post_chat_with_schema

    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(json.loads(request.content)["model"]))

    msgs = {"messages": [{"role": "user", "content": "x"}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await _post_chat_with_schema(UPSTREAM, msgs, "invoice", http_client=client)

    first = sent[0]["messages"][0]
    assert first["role"] == "system"
    assert "invoice_number" in first["content"]  # real invoice fields named
    assert "total_ttc" in first["content"]


async def test_schema_fallback_reraises_non_grammar_400() -> None:
    # A 400 that is NOT a grammar/response_format problem must NOT be swallowed.
    from docie_bench.agents.runtime import AgentError, _post_chat_with_schema

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "context length exceeded"}})

    msgs = {"messages": [{"role": "user", "content": "x"}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AgentError) as exc:
            await _post_chat_with_schema(UPSTREAM, msgs, "invoice", http_client=client)
    assert exc.value.status_code == 400
    assert "context length" in str(exc.value)


async def test_schema_fallback_downgrades_on_empty_200_content() -> None:
    # The small-Ollama-and-friends defect: a strong response_format style
    # compiles fine (200, not 400) but the model emits nothing. Before this
    # fix, the agent forward accepted that as a "successful" empty
    # completion on the very first attempt -- openai_client.chat_json's
    # ladder already treats this as downgradable; the agent path needs its
    # own check since it forwards through _post_chat directly.
    from docie_bench.agents.runtime import _post_chat_with_schema

    seen: list[dict | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        rf = body.get("response_format")
        seen.append(rf)
        if rf and rf.get("type") == "json_schema":
            return httpx.Response(200, json=_completion(body["model"], content=""))
        return httpx.Response(200, json=_completion(body["model"], content='{"ok":true}'))

    msgs = {"messages": [{"role": "user", "content": "x"}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _post_chat_with_schema(UPSTREAM, msgs, "invoice", http_client=client)
    # Both json_schema attempts (prefill, no-prefill) came back empty ->
    # downgraded to json_object, which answered.
    assert [s and s["type"] for s in seen] == ["json_schema", "json_schema", "json_object"]
    assert result["choices"][0]["message"]["content"] == '{"ok":true}'


async def test_schema_fallback_raises_when_every_rung_is_empty() -> None:
    # Even the unconstrained final rung came back empty: there is no weaker
    # style left, so this must raise rather than hand back an empty
    # "successful" completion.
    from docie_bench.agents.runtime import AgentError, _post_chat_with_schema

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=_completion(body["model"], content=""))

    msgs = {"messages": [{"role": "user", "content": "x"}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AgentError) as exc:
            await _post_chat_with_schema(UPSTREAM, msgs, "invoice", http_client=client)
    assert "empty content" in str(exc.value)


def test_ocr_agent_vision_mode_forwards_image_with_schema(api) -> None:
    client, captured = api
    created = client.post(
        "/v1/agents",
        json={
            "name": "doc-vision",
            "kind": "ocr",
            "options": {"mode": "vision", "vision_model": "gemma-vl", "schema": "invoice"},
        },
    )
    assert created.status_code == 201, created.text

    resp = client.post(
        "/v1/agents/chat/completions",
        json={
            "model": "doc-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "extract the invoice"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    sent = json.loads(captured[-1].content)
    # Forwarded to the VISION deployment, not the agent name.
    assert sent["model"] == "gemma-vl"
    # Schema injected as response_format (GBNF structuring).
    assert sent["response_format"]["json_schema"]["name"] == "invoice"
    # The image reaches the model untouched (no OCR step).
    # The schema-constrained call prefills an assistant "{" (suppresses a
    # reasoning ramble); the image rides the user turn just before it.
    assert sent["messages"][-1] == {"role": "assistant", "content": "{"}
    assert any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for message in sent["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )
    assert resp.json()["docie_agent"]["mode"] == "vision"


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


def test_agent_requests_and_pii_metrics_recorded(api) -> None:
    from prometheus_client import REGISTRY

    def sample(name: str, labels: dict[str, str]) -> float:
        return REGISTRY.get_sample_value(name, labels) or 0.0

    client, _ = api
    _create_proxy(client)
    ok_labels = {"agent": "pii-proxy", "kind": "proxy_security", "outcome": "ok"}
    email_labels = {"agent": "pii-proxy", "entity_type": "EMAIL"}
    ok_before = sample("docie_agent_requests_total", ok_labels)
    email_before = sample("docie_agent_pii_detected_total", email_labels)

    client.post(
        "/v1/agents/pii-proxy/chat/completions",
        json={"messages": [{"role": "user", "content": "mail jean@acme.fr"}]},
    )
    assert sample("docie_agent_requests_total", ok_labels) == ok_before + 1
    assert sample("docie_agent_pii_detected_total", email_labels) == email_before + 1

    # A gate refusal lands under its own outcome, never "ok".
    _create_proxy(client, name="gate", options={"mode": "block"})
    blocked_labels = {"agent": "gate", "kind": "proxy_security", "outcome": "pii_blocked"}
    blocked_before = sample("docie_agent_requests_total", blocked_labels)
    client.post(
        "/v1/agents/gate/chat/completions",
        json={"messages": [{"role": "user", "content": "card 4111 1111 1111 1111"}]},
    )
    assert sample("docie_agent_requests_total", blocked_labels) == blocked_before + 1
