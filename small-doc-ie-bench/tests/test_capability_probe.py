"""Capability probe: canary, module-level cache, TTL, and invalidation.

The cache is the load-bearing part of the "don't burn tokens on every call"
requirement AND the "stale cache re-introduces the empty-content bug" failure
mode. These tests pin its TTL / fingerprint / reset semantics against a stubbed
runtime.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from docie_bench.llm.capability_probe import (
    CapabilityProbe,
    get_cached_probe,
    probe_endpoint,
    profile_probe_fingerprint,
    reset_probe_cache,
    run_capability_probes,
    store_probe,
)
from docie_bench.llm.model_gateway import reset_gateway_state
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.openai_client import OpenAICompatibleClient


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_gateway_state()
    reset_probe_cache()


def _profile(**overrides: Any) -> ModelProfile:
    values: dict[str, Any] = {
        "name": "test",
        "model": "test-model",
        "base_url": "http://model.test/v1",
        "api_key": "test",
        "response_format_style": "openai_json_schema",
        "retry_max_attempts": 1,
        "queue_timeout_seconds": 1,
    }
    values.update(overrides)
    return ModelProfile(**values)


async def _client(profile: ModelProfile, handler: Any) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(profile)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=profile.base_url, transport=httpx.MockTransport(handler)
    )
    return client


def _make_probe(profile: ModelProfile, *, probed_at: float) -> CapabilityProbe:
    return CapabilityProbe(
        base_url=profile.base_url,
        model=profile.model,
        declared_style=profile.response_format_style,
        effective_style="json_object",
        confirmed_styles=("json_object",),
        rejected_styles=("openai_json_schema",),
        advertised_styles=None,
        vision=None,
        source="probe",
        fingerprint=profile_probe_fingerprint(profile),
        probed_at=probed_at,
    )


def test_fingerprint_changes_when_style_changes() -> None:
    base = _profile()
    changed = _profile(response_format_style="json_object")
    assert profile_probe_fingerprint(base) != profile_probe_fingerprint(changed)


def test_cache_hit_within_ttl_but_miss_after_expiry() -> None:
    profile = _profile()
    store_probe(_make_probe(profile, probed_at=100.0))

    assert get_cached_probe(profile, ttl_seconds=60, now=150.0) is not None
    # 100 -> 200 exceeds the 60s TTL: must re-probe.
    assert get_cached_probe(profile, ttl_seconds=60, now=200.0) is None


def test_cache_invalidated_by_profile_change() -> None:
    profile = _profile()
    store_probe(_make_probe(profile, probed_at=100.0))
    # Same endpoint (base_url+model) but the declared style changed -> the cached
    # fingerprint no longer matches, so a stale result cannot be reused.
    changed = _profile(response_format_style="json_object")
    assert get_cached_probe(changed, ttl_seconds=600, now=120.0) is None


def test_reset_gateway_state_clears_probe_cache() -> None:
    profile = _profile()
    store_probe(_make_probe(profile, probed_at=0.0))
    reset_gateway_state()
    assert get_cached_probe(profile, ttl_seconds=600, now=0.0) is None


def _handler_factory() -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "test-model",
                            "capabilities": {
                                "vision": False,
                                "response_format_styles": [
                                    "openai_json_schema",
                                    "json_object",
                                ],
                            },
                        }
                    ]
                },
            )
        import json

        payload = json.loads(request.read().decode("utf-8"))
        rf = payload.get("response_format")
        style = rf.get("type") if rf else "none"
        # json_schema returns empty; json_object is honoured.
        if style == "json_schema":
            return httpx.Response(
                200, json={"choices": [{"message": {"content": ""}}]}
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"ok": "yes"}'}}]}
        )

    return handler


async def test_probe_endpoint_confirms_json_object_after_json_schema_rejected() -> None:
    profile = _profile(capability_discovery="optional")
    client = await _client(profile, _handler_factory())
    try:
        probe = await probe_endpoint(client, now=500.0)
    finally:
        await client.aclose()

    assert probe.rejected_styles == ("openai_json_schema",)
    assert probe.confirmed_styles == ("json_object",)
    assert probe.effective_style == "json_object"
    assert probe.source == "probe"
    assert probe.advertised_styles == ("json_object", "openai_json_schema")
    assert probe.vision is False
    # Cached: a second call within TTL does not re-probe.
    assert get_cached_probe(profile, now=500.0) is probe


async def test_probe_skips_purpose_built_style() -> None:
    profile = _profile(response_format_style="vllm_guided_json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        raise AssertionError("purpose-built styles must not be canaried")

    client = await _client(profile, handler)
    try:
        probe = await probe_endpoint(client, now=0.0)
    finally:
        await client.aclose()

    assert probe.source == "skipped"
    assert probe.effective_style == "vllm_guided_json"
    assert probe.confirmed_styles == ()


async def test_probe_is_best_effort_on_transport_error() -> None:
    profile = _profile()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("endpoint down")

    client = await _client(profile, handler)
    try:
        probe = await probe_endpoint(client, now=0.0)
    finally:
        await client.aclose()

    # No crash; recorded as an error so the manifest documents the failure.
    assert probe.source == "error"
    assert probe.advertised_styles is None


@pytest.mark.parametrize("status", [503, 429])
async def test_probe_transient_status_does_not_reject_style(status: int) -> None:
    # A single transient blip (503/429) on the one canary must NOT be recorded as
    # a style rejection: doing so would prune the strongest constrained style for
    # EVERY document in the run and downgrade to weaker decoding. It is
    # inconclusive, exactly like the runtime path treats the same status.
    profile = _profile(capability_discovery="optional")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "test-model",
                            "capabilities": {
                                "response_format_styles": ["openai_json_schema", "json_object"]
                            },
                        }
                    ]
                },
            )
        return httpx.Response(status, json={"error": "temporarily unavailable"})

    client = await _client(profile, handler)
    try:
        probe = await probe_endpoint(client, now=0.0)
    finally:
        await client.aclose()

    # The transient status is inconclusive: nothing is rejected/pruned.
    assert probe.rejected_styles == ()
    assert probe.source == "error"
    # The strongest constrained style is still on the ladder for real documents.
    assert "openai_json_schema" in client._negotiated_ladder()


async def test_probe_permanent_status_rejects_style() -> None:
    # A permanent 4xx ("style unsupported") is a genuine rejection signal, unlike
    # a transient status — the ladder legitimately downgrades to json_object.
    profile = _profile(capability_discovery="optional")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "test-model",
                            "capabilities": {
                                "response_format_styles": ["openai_json_schema", "json_object"]
                            },
                        }
                    ]
                },
            )
        import json as _json

        payload = _json.loads(request.read().decode("utf-8"))
        rf = payload.get("response_format")
        style = rf.get("type") if rf else "none"
        if style == "json_schema":
            return httpx.Response(400, json={"error": "response_format not supported"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": "yes"}'}}]})

    client = await _client(profile, handler)
    try:
        probe = await probe_endpoint(client, now=0.0)
    finally:
        await client.aclose()

    assert probe.rejected_styles == ("openai_json_schema",)
    assert probe.confirmed_styles == ("json_object",)
    assert probe.source == "probe"


async def test_run_capability_probes_skips_non_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passthrough = _profile(name="pt", capability_discovery="optional")
    solution = _profile(name="ocr_kind", kind="ocr", model="", base_url="")

    factories: dict[str, OpenAICompatibleClient] = {}

    def factory(profile: ModelProfile) -> OpenAICompatibleClient:
        client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
        client.profile = profile
        client._client = httpx.AsyncClient(
            base_url=profile.base_url or "http://x",
            transport=httpx.MockTransport(_handler_factory()),
        )
        from docie_bench.llm.model_gateway import ModelGateway

        client._gateway = ModelGateway(profile, client._client)
        client.last_response_format_style = None
        factories[profile.name] = client
        return client

    results = await run_capability_probes([passthrough, solution], client_factory=factory)

    assert set(results) == {"pt"}  # ocr kind is not probed
    assert results["pt"]["effective_style"] == "json_object"
