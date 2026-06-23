from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi.testclient import TestClient

from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.gateway import (
    GatewayRoutingError,
    create_gateway_app,
    resolve_profile,
)


def _profile(name: str, model: str, base_url: str) -> ModelProfile:
    return ModelProfile(name=name, model=model, base_url=base_url, api_key="k")


# Routing table: alpha/beta distinct; think shares boom-free; dup1/dup2 collide.
_PROFILES = {
    "alpha": _profile("alpha", "up-alpha", "http://up-a/v1"),
    "beta": _profile("beta", "up-beta", "http://up-b/v1"),
    "boom": _profile("boom", "up-boom", "http://up-b/v1"),
    # Two profiles, same upstream model, SAME base_url -> forward identically.
    "nux": _profile("nux", "nuextract3", "http://up-c/v1"),
    "nux_think": _profile("nux_think", "nuextract3", "http://up-c/v1"),
}

# Two profiles, same upstream model, DIFFERENT base_urls -> ambiguous.
_AMBIGUOUS = {
    "x_local": _profile("x_local", "shared", "http://up-a/v1"),
    "x_remote": _profile("x_remote", "shared", "http://up-b/v1"),
}


# ── resolver (pure) ──────────────────────────────────────────────────────────


def test_resolve_by_profile_name() -> None:
    assert resolve_profile("alpha", _PROFILES).model == "up-alpha"


def test_resolve_by_unique_upstream_id() -> None:
    # The benchmark sends the upstream id (e.g. qwen2.5:1.5b), not the profile name.
    assert resolve_profile("up-beta", _PROFILES).name == "beta"


def test_resolve_shared_upstream_same_base_is_not_ambiguous() -> None:
    # nux / nux_think share base_url -> forwarding is identical, pick one.
    assert resolve_profile("nuextract3", _PROFILES).base_url == "http://up-c/v1"


def test_resolve_unknown_model_raises_404() -> None:
    with pytest.raises(GatewayRoutingError) as exc:
        resolve_profile("does-not-exist", _PROFILES)
    assert exc.value.status_code == 404


def test_resolve_ambiguous_across_base_urls_raises_409() -> None:
    with pytest.raises(GatewayRoutingError) as exc:
        resolve_profile("shared", _AMBIGUOUS)
    assert exc.value.status_code == 409


def test_resolve_missing_model_raises_400() -> None:
    with pytest.raises(GatewayRoutingError) as exc:
        resolve_profile("", _PROFILES)
    assert exc.value.status_code == 400


# ── app (real forwarding via MockTransport) ──────────────────────────────────


async def _sse_chunks() -> AsyncIterator[bytes]:
    # A real async stream (not eager bytes) so the gateway's aiter_raw passthrough
    # is exercised the way a live llama-server / Ollama stream behaves.
    yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    yield b"data: [DONE]\n\n"


def _client(profiles: dict[str, ModelProfile], captured: list[httpx.Request]) -> TestClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        if body["model"] == "up-boom":
            return httpx.Response(400, json={"error": {"message": "upstream said no"}})
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse_chunks(),
            )
        return httpx.Response(200, json={"id": "c", "model": body["model"], "choices": []})

    app = create_gateway_app(profiles=profiles, transport=httpx.MockTransport(handler))
    return TestClient(app)


def test_healthz_and_models_list() -> None:
    with _client(_PROFILES, []) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    # /v1/models advertises profile names, sorted.
    assert ids == sorted(_PROFILES)


def test_chat_routes_by_name_and_rewrites_upstream_model() -> None:
    captured: list[httpx.Request] = []
    with _client(_PROFILES, captured) as client:
        resp = client.post("/v1/chat/completions", json={"model": "alpha", "messages": []})
    assert resp.status_code == 200
    assert str(captured[0].url) == "http://up-a/v1/chat/completions"
    # The upstream receives the upstream id, not the profile name.
    assert json.loads(captured[0].content)["model"] == "up-alpha"
    assert captured[0].headers["authorization"] == "Bearer k"


def test_chat_routes_by_upstream_id() -> None:
    captured: list[httpx.Request] = []
    with _client(_PROFILES, captured) as client:
        resp = client.post("/v1/chat/completions", json={"model": "up-beta", "messages": []})
    assert resp.status_code == 200
    assert str(captured[0].url) == "http://up-b/v1/chat/completions"


def test_chat_stream_passthrough_preserves_sse() -> None:
    with _client(_PROFILES, []) as client:
        resp = client.post(
            "/v1/chat/completions", json={"model": "alpha", "stream": True, "messages": []}
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert b"[DONE]" in resp.content


def test_chat_unknown_model_returns_openai_404() -> None:
    with _client(_PROFILES, []) as client:
        resp = client.post("/v1/chat/completions", json={"model": "ghost", "messages": []})
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "model_not_found"


def test_chat_passes_through_upstream_error_status() -> None:
    with _client(_PROFILES, []) as client:
        resp = client.post("/v1/chat/completions", json={"model": "boom", "messages": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "upstream said no"
