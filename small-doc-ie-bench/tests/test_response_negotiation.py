"""Response-format negotiation: the downgrade ladder in ``chat_json``.

These tests stub the runtime with ``httpx.MockTransport`` — the fake endpoint
returns empty content for ``json_schema`` (exactly the small-Ollama defect) and
proves the ladder downgrades to ``json_object`` and yields a valid parse without
a live model.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from docie_bench.llm.model_gateway import (
    InvalidModelResponseError,
    ModelGatewayError,
    reset_gateway_state,
)
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.openai_client import OpenAICompatibleClient
from docie_bench.llm.response_format import style_ladder


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


def _completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


async def _client(
    profile: ModelProfile, handler: Any
) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(profile)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=profile.base_url, transport=httpx.MockTransport(handler)
    )
    return client


async def _chat(client: OpenAICompatibleClient) -> dict[str, Any]:
    result, _usage, _raw = await client.chat_json(
        system_prompt="system",
        user_prompt="user",
        schema_name="test",
        schema={"type": "object"},
    )
    return result


def test_style_ladder_downgrades_generic_but_not_purpose_built() -> None:
    assert style_ladder("openai_json_schema") == ("openai_json_schema", "json_object", "none")
    assert style_ladder("json_object") == ("json_object", "none")
    # Purpose-built styles must stay singletons — never auto-downgraded.
    assert style_ladder("nuextract3") == ("nuextract3",)
    assert style_ladder("nuextract3_think") == ("nuextract3_think",)
    assert style_ladder("vllm_guided_json") == ("vllm_guided_json",)


async def test_json_schema_empty_content_downgrades_to_json_object() -> None:
    styles_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("utf-8")
        import json

        payload = json.loads(body)
        rf = payload.get("response_format")
        style = rf.get("type") if rf else "none"
        styles_seen.append(style)
        # Small-Ollama defect: json_schema returns EMPTY content.
        if style == "json_schema":
            return _completion("")
        # json_object honoured -> valid JSON.
        return _completion('{"invoice_number": {"value": "INV-1"}}')

    client = await _client(_profile(capability_discovery="disabled"), handler)
    try:
        result = await _chat(client)
    finally:
        await client.aclose()

    assert result == {"invoice_number": {"value": "INV-1"}}
    # It tried json_schema first, saw empty, then downgraded to json_object.
    assert styles_seen == ["json_schema", "json_object"]
    # The EFFECTIVE style is recorded so unconstrained decoding is distinguishable.
    assert client.last_response_format_style == "json_object"


def _grammar_error() -> httpx.Response:
    # Mirrors this project's Ollama: a hard HTTP 400 (not empty 200) when a
    # strong response_format style cannot be compiled into a sampler grammar.
    return httpx.Response(
        400,
        json={
            "error": {
                "message": (
                    "Failed to initialize samplers: failed to parse grammar: "
                    "unexpected end of input"
                )
            }
        },
    )


async def test_json_schema_grammar_400_downgrades_to_json_object() -> None:
    styles_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.read().decode("utf-8"))
        rf = payload.get("response_format")
        style = rf.get("type") if rf else "none"
        styles_seen.append(style)
        # Ollama defect: json_schema hard-fails with a grammar-compilation 400.
        if style == "json_schema":
            return _grammar_error()
        # json_object honoured -> valid JSON.
        return _completion('{"invoice_number": {"value": "INV-1"}}')

    client = await _client(_profile(capability_discovery="disabled"), handler)
    try:
        result = await _chat(client)
    finally:
        await client.aclose()

    assert result == {"invoice_number": {"value": "INV-1"}}
    # It tried json_schema first, got a grammar 400, then downgraded to json_object.
    assert styles_seen == ["json_schema", "json_object"]
    assert client.last_response_format_style == "json_object"


async def test_generic_style_non_grammar_400_still_raises() -> None:
    # A 400 whose body is NOT a grammar/schema-compilation failure is a genuine
    # bad request; it must raise, not trigger a downgrade. This is what makes the
    # marker gate meaningful (vs. blindly downgrading every generic 400).
    styles_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.read().decode("utf-8"))
        rf = payload.get("response_format")
        styles_seen.append(rf.get("type") if rf else "none")
        return httpx.Response(
            400, json={"error": {"message": "invalid request: unknown field 'foo'"}}
        )

    client = await _client(_profile(capability_discovery="disabled"), handler)
    try:
        with pytest.raises(ModelGatewayError):
            await _chat(client)
    finally:
        await client.aclose()
    # No downgrade attempted: only the declared style was tried before raising.
    assert styles_seen == ["json_schema"]


async def test_generic_400_mentioning_json_schema_still_raises() -> None:
    # The markers are narrow (only true compile-failure phrases). A genuine
    # bad-request 400 that merely ECHOES the style name ("json_schema") — e.g. an
    # unsupported-style error — must still raise, not spuriously downgrade. This
    # guards the SHARED sync /v1/extract + CLI path from masking real 400s.
    styles_seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.read().decode("utf-8"))
        rf = payload.get("response_format")
        styles_seen.append(rf.get("type") if rf else "none")
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "response_format.type 'json_schema' is not supported by this model"
                }
            },
        )

    client = await _client(_profile(capability_discovery="disabled"), handler)
    try:
        with pytest.raises(ModelGatewayError):
            await _chat(client)
    finally:
        await client.aclose()
    # 'json_schema' appears in the body but is not a compile failure -> no downgrade.
    assert styles_seen == ["json_schema"]


async def test_singleton_style_grammar_400_raises_without_downgrade() -> None:
    # A purpose-built style has no weaker rung; a grammar 400 must surface as an
    # error (retryable by the gateway) rather than silently switching families.
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _grammar_error()

    client = await _client(
        _profile(response_format_style="vllm_guided_json", capability_discovery="disabled"),
        handler,
    )
    try:
        with pytest.raises(ModelGatewayError):
            await _chat(client)
    finally:
        await client.aclose()
    assert calls == 1
    assert client.last_response_format_style is None


async def test_effective_style_matches_declared_when_honored() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _completion('{"ok": true}')

    client = await _client(_profile(capability_discovery="disabled"), handler)
    try:
        assert await _chat(client) == {"ok": True}
    finally:
        await client.aclose()
    assert client.last_response_format_style == "openai_json_schema"


async def test_ladder_falls_through_to_none_repair_rung() -> None:
    # Every structured style returns empty; only the terminal 'none' rung, which
    # relies on the parse-and-repair path, gets usable content.
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.read().decode("utf-8"))
        if payload.get("response_format") is not None:
            return _completion("")
        return _completion('Here is the answer: {"total": 10} thanks')

    client = await _client(_profile(capability_discovery="disabled"), handler)
    try:
        assert await _chat(client) == {"total": 10}
    finally:
        await client.aclose()
    assert client.last_response_format_style == "none"


async def test_singleton_style_does_not_downgrade_and_raises() -> None:
    # A purpose-built style has no weaker rung; empty content must surface as an
    # error (which the gateway would retry) rather than silently corrupting the
    # request by switching to json_object.
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _completion("")

    client = await _client(
        _profile(response_format_style="vllm_guided_json", capability_discovery="disabled"),
        handler,
    )
    try:
        with pytest.raises(InvalidModelResponseError):
            await _chat(client)
    finally:
        await client.aclose()
    # Only the declared style was attempted, no downgrade to another family.
    assert calls == 1
    assert client.last_response_format_style is None
