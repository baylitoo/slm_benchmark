"""Generic OpenAI chat surface (POST /v1/chat/completions on the main API)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docie_bench.agents.api import configure_http_transport
from docie_bench.chat_api import router as chat_router
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.profile_resolver import ProfileResolutionError

UPSTREAM = ModelProfile(
    name="lfm2.5-350m", model="lfm2.5-350m-served", base_url="http://upstream/v1", api_key="k"
)


@pytest.fixture()
def api(monkeypatch) -> tuple[TestClient, list[httpx.Request]]:
    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "lfm2.5-350m":
            return UPSTREAM
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)

    captured: list[httpx.Request] = []

    async def _sse_chunks() -> AsyncIterator[bytes]:
        for piece in ("Hel", "lo", "!"):
            yield f'data: {{"choices":[{{"delta":{{"content":"{piece}"}}}}]}}\n\n'.encode()
        yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse_chunks()
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
    app.include_router(chat_router)
    client = TestClient(app)
    yield client, captured
    configure_http_transport(None)


def test_chat_forwards_with_served_model_id(api) -> None:
    client, captured = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hello"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "echo: hello"
    sent = json.loads(captured[-1].content)
    # The upstream sees ITS served id, never the deployment name.
    assert sent["model"] == "lfm2.5-350m-served"
    assert sent["messages"][0]["role"] == "system"
    assert captured[-1].url.host == "upstream"


def test_chat_unknown_model_is_openai_404(api) -> None:
    client, _ = api
    response = client.post(
        "/v1/chat/completions",
        json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


def test_chat_requires_model(api) -> None:
    client, _ = api
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 400


def test_chat_stream_proxies_real_upstream_chunks(api) -> None:
    """A stream: true request gets the upstream's actual token-by-token SSE
    frames relayed as they arrive — not one buffered chunk built after the
    full completion finishes. The mock upstream emits 4 separate SSE events;
    all 4 must show up verbatim, in order, and the forwarded request must
    still carry stream: true (previously stripped before reaching upstream)."""
    client, captured = api
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "lfm2.5-350m",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert events == [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"delta":{"content":"!"}}]}',
        "data: [DONE]",
    ]
    sent = json.loads(captured[-1].content)
    assert sent["stream"] is True


# ── store: model that isn't live yet: load-on-demand, not a bare error ──────


@pytest.fixture
def api_store(monkeypatch) -> TestClient:
    from docie_bench.serving.placement_resolver import (
        PlacementNotFoundError,
        PlacementNotReadyError,
    )

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "store:never-deployed":
            raise PlacementNotFoundError("store model 'never-deployed' is not in the catalog")
        if model_profile == "store:evicted":
            raise PlacementNotReadyError("store model 'evicted' placement is 'stopped'")
        if model_profile == "store:truly-unseeded":
            raise PlacementNotFoundError("store model 'truly-unseeded' is not in the catalog")
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)

    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def test_chat_store_model_not_live_triggers_load_and_returns_202(
    api_store, monkeypatch
) -> None:
    async def fake_trigger(name: str) -> tuple[str, float] | None:
        assert name in ("never-deployed", "evicted")
        return name, 30.0

    monkeypatch.setattr("docie_bench.chat_api.trigger_deployment_load", fake_trigger)

    for model in ("store:never-deployed", "store:evicted"):
        response = api_store.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["status"] == "loading"
        assert body["eta_seconds"] == 30.0
        assert "30s" in body["message"]


def test_chat_bare_evicted_deployment_name_also_triggers_load(
    api_store, monkeypatch
) -> None:
    """The Playground's Chat/Vision pickers submit a raw deployment name,
    never store:-prefixed (unlike an explicit store:<name> API call) — an
    evicted deployment picked there must still cold-start, not 404, or the
    picker offering evicted deployments at all would be dead UI."""

    async def fake_trigger(name: str) -> tuple[str, float] | None:
        assert name == "gemma-2-2b-it"
        return name, 45.0

    monkeypatch.setattr("docie_bench.chat_api.trigger_deployment_load", fake_trigger)

    response = api_store.post(
        "/v1/chat/completions",
        json={"model": "gemma-2-2b-it", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "loading"
    assert body["deployment"] == "gemma-2-2b-it"


def test_chat_store_model_genuinely_unseeded_still_404s(api_store, monkeypatch) -> None:
    async def fake_trigger(name: str) -> tuple[str, float] | None:
        return None  # not a catalog entry at all -- nothing to trigger

    monkeypatch.setattr("docie_bench.chat_api.trigger_deployment_load", fake_trigger)

    response = api_store.post(
        "/v1/chat/completions",
        json={"model": "store:truly-unseeded", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"
