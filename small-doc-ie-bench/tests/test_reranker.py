"""Reranker family launch flags + the /v1/rerank surface."""

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


def test_reranker_family_contract() -> None:
    fam = get_family("reranker")
    assert fam.reranker is True
    assert fam.embedding is False  # a distinct type, not embedding=True
    # llama.cpp requires --reranking AND --embedding --pooling rank together;
    # --reranking alone leaves the endpoint unable to score (verified against
    # the server's own --help and README).
    assert "--reranking" in fam.llama_server_args
    assert "--embedding" in fam.llama_server_args
    assert "--pooling" in fam.llama_server_args
    assert "rank" in fam.llama_server_args
    # Ollama has no equivalent rerank pooling mode.
    assert fam.ollama_faithful is False


def test_only_reranker_family_is_flagged() -> None:
    flagged = [name for name, f in FAMILIES.items() if f.reranker]
    assert flagged == ["reranker"]


RERANK = ModelProfile(
    name="lfm-rerank", model="lfm-rerank-served", base_url="http://upstream/v1", api_key="k"
)


@pytest.fixture
def api(monkeypatch) -> tuple[TestClient, list[httpx.Request]]:
    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "lfm-rerank":
            return RERANK
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        assert request.url.path.endswith("/rerank")
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "results": [
                    {"index": i, "relevance_score": 1.0 - i * 0.1}
                    for i in range(len(body["documents"]))
                ],
            },
        )

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    yield client, captured
    configure_http_transport(None)


def test_rerank_forwards_with_served_id(api) -> None:
    client, captured = api
    response = client.post(
        "/v1/rerank",
        json={
            "model": "lfm-rerank",
            "query": "invoice total",
            "documents": ["FACTURE total 5400 EUR", "unrelated weather report"],
        },
    )
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["relevance_score"] >= results[1]["relevance_score"]
    sent = json.loads(captured[-1].content)
    assert sent["model"] == "lfm-rerank-served"
    assert captured[-1].url.path == "/v1/rerank"


def test_rerank_forwards_top_n(api) -> None:
    client, captured = api
    client.post(
        "/v1/rerank",
        json={"model": "lfm-rerank", "query": "x", "documents": ["a", "b"], "top_n": 1},
    )
    sent = json.loads(captured[-1].content)
    assert sent["top_n"] == 1


def test_rerank_requires_model_query_and_documents(api) -> None:
    client, _ = api
    assert client.post(
        "/v1/rerank", json={"query": "x", "documents": ["a"]}
    ).status_code == 400
    assert client.post(
        "/v1/rerank", json={"model": "lfm-rerank", "documents": ["a"]}
    ).status_code == 400
    assert client.post(
        "/v1/rerank", json={"model": "lfm-rerank", "query": "x"}
    ).status_code == 400
    assert client.post(
        "/v1/rerank", json={"model": "lfm-rerank", "query": "x", "documents": []}
    ).status_code == 400


def test_rerank_unknown_model_is_404(api) -> None:
    client, _ = api
    response = client.post(
        "/v1/rerank", json={"model": "ghost", "query": "x", "documents": ["a"]}
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


def test_rerank_upstream_error_names_the_flags_needed(api, monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "not a reranker"})

    configure_http_transport(httpx.MockTransport(handler))
    client, _ = api
    response = client.post(
        "/v1/rerank", json={"model": "lfm-rerank", "query": "x", "documents": ["a"]}
    )
    assert response.status_code == 400
    assert "--reranking" in response.json()["error"]["message"]
