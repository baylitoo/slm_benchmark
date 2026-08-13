"""Embedding family launch flags + the /v1/embeddings surface."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docie_bench.agents.api import configure_http_transport
from docie_bench.chat_api import router as chat_router
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.model_store import FAMILIES, get_family
from docie_bench.serving.profile_resolver import ProfileResolutionError


def test_embedding_family_contract() -> None:
    fam = get_family("embedding")
    assert fam.embedding is True
    assert "--embedding" in fam.llama_server_args
    assert "--pooling" in fam.llama_server_args


def test_only_embedding_family_is_flagged() -> None:
    flagged = [name for name, f in FAMILIES.items() if f.embedding]
    assert flagged == ["embedding"]


EMB = ModelProfile(
    name="lfm-embed", model="lfm-embed-served", base_url="http://upstream/v1", api_key="k"
)


@pytest.fixture()
def api(monkeypatch) -> tuple[TestClient, list[httpx.Request]]:
    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "lfm-embed":
            return EMB
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        assert request.url.path.endswith("/embeddings")
        body = json.loads(request.content)
        n = 1 if isinstance(body["input"], str) else len(body["input"])
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]}
                    for i in range(n)
                ],
            },
        )

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    yield client, captured
    configure_http_transport(None)


def test_embeddings_forwards_with_served_id(api) -> None:
    client, captured = api
    response = client.post(
        "/v1/embeddings", json={"model": "lfm-embed", "input": "invoice total 5400 EUR"}
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 1
    sent = json.loads(captured[-1].content)
    assert sent["model"] == "lfm-embed-served"
    assert captured[-1].url.path == "/v1/embeddings"


def test_embeddings_batch_input(api) -> None:
    client, _ = api
    response = client.post(
        "/v1/embeddings", json={"model": "lfm-embed", "input": ["a", "b", "c"]}
    )
    assert len(response.json()["data"]) == 3


def test_embeddings_requires_model_and_input(api) -> None:
    client, _ = api
    assert client.post("/v1/embeddings", json={"input": "x"}).status_code == 400
    assert client.post("/v1/embeddings", json={"model": "lfm-embed"}).status_code == 400
    assert client.post("/v1/embeddings", json={"model": "lfm-embed", "input": []}).status_code == 400


def test_embeddings_unknown_model_is_404(api) -> None:
    client, _ = api
    response = client.post("/v1/embeddings", json={"model": "ghost", "input": "x"})
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


def test_embeddings_stamps_recency_for_the_served_deployment(api, monkeypatch) -> None:
    # PR-4 recency must be stamped by EVERY surface that serves traffic (see
    # recency.py's docstring) -- api.py's extract path already does this;
    # chat_api's /v1/embeddings didn't, so a deployment driven only through
    # this surface read as idle forever and became the first idle-TTL victim.
    client, _ = api
    calls: list[str | None] = []
    monkeypatch.setattr(
        "docie_bench.chat_api.recency.stamp_served_profile", lambda name: calls.append(name)
    )
    response = client.post(
        "/v1/embeddings", json={"model": "lfm-embed", "input": "invoice total 5400 EUR"}
    )
    assert response.status_code == 200, response.text
    assert calls == ["lfm-embed"]


def test_embeddings_store_model_not_live_triggers_load_and_returns_202(monkeypatch) -> None:
    # /v1/embeddings shares chat_api._resolve_or_error with /v1/chat/completions
    # and /v1/rerank -- this proves the load-on-demand wiring actually reaches
    # this endpoint too, not just that the shared helper works in isolation.
    from docie_bench.serving.placement_resolver import PlacementNotFoundError

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        raise PlacementNotFoundError("store model 'cold-embedder' is not in the catalog")

    async def fake_trigger(name: str) -> tuple[str, float] | None:
        assert name == "cold-embedder"
        return name, 15.0

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)
    monkeypatch.setattr("docie_bench.chat_api.trigger_deployment_load", fake_trigger)

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    response = client.post(
        "/v1/embeddings", json={"model": "store:cold-embedder", "input": "x"}
    )
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "loading"
