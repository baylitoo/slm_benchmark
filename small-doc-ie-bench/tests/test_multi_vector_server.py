"""The multi-vector (ColBERT / PyLate) /v1/rerank shim -- request/response
shaping over an injected fake backend. Never imports sentence-transformers or
torch: the real backend is exercised only by the version-guard test below,
via a stubbed module."""

from __future__ import annotations

import sys
import types

import pytest
from fastapi.testclient import TestClient

from docie_bench.multi_vector_server.server import (
    MULTI_VECTOR_MIN_VERSION,
    SentenceTransformersMultiVectorBackend,
    create_multi_vector_app,
    parse_rerank_request,
    rank_results,
)


class _FakeBackend:
    """Scores each document by how many query words it contains -- a real,
    order-sensitive signal so ranking is meaningful, and deterministic."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        words = set(query.lower().split())
        return [
            sum(1.0 for w in words if w in doc.lower()) / max(len(words), 1) for doc in documents
        ]


@pytest.fixture
def api() -> tuple[TestClient, _FakeBackend]:
    backend = _FakeBackend()
    app = create_multi_vector_app(model_id="mxbai-edge-colbert-v0-32m", backend=backend)
    return TestClient(app), backend


# -- pure shaping ------------------------------------------------------------


def test_rank_results_sorts_descending_and_keeps_document_index() -> None:
    out = rank_results([0.2, 0.9, 0.5], top_n=None)
    assert out == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.5},
        {"index": 0, "relevance_score": 0.2},
    ]


def test_rank_results_ties_keep_document_order_as_the_deterministic_tiebreak() -> None:
    # A stable sort: equal scores keep the caller's ordering, never a hash
    # order that could flip between runs.
    out = rank_results([0.5, 0.5, 0.5], top_n=None)
    assert [r["index"] for r in out] == [0, 1, 2]


def test_rank_results_top_n_truncates_after_sorting_not_before() -> None:
    # top_n=1 must be the BEST document, not the first document.
    out = rank_results([0.1, 0.9, 0.5], top_n=1)
    assert out == [{"index": 1, "relevance_score": 0.9}]


def test_rank_results_top_n_none_or_nonpositive_means_all() -> None:
    assert len(rank_results([0.1, 0.2], top_n=None)) == 2
    assert len(rank_results([0.1, 0.2], top_n=0)) == 2


def test_parse_rerank_request_happy_path() -> None:
    query, docs, top_n = parse_rerank_request(
        {"model": "ignored", "query": "q", "documents": ["a", "b"], "top_n": 1}
    )
    assert (query, docs, top_n) == ("q", ["a", "b"], 1)


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("not a dict", "JSON object"),
        ({"documents": ["a"]}, "'query'"),
        ({"query": "  ", "documents": ["a"]}, "'query'"),
        ({"query": "q"}, "'documents'"),
        ({"query": "q", "documents": []}, "'documents'"),
        ({"query": "q", "documents": ["a", 42]}, r"documents\[1\]"),
        ({"query": "q", "documents": ["a"], "top_n": "x"}, "'top_n'"),
        ({"query": "q", "documents": ["a"], "top_n": 0}, "'top_n'"),
    ],
)
def test_parse_rerank_request_rejects_malformed_bodies(body: object, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        parse_rerank_request(body)


def test_parse_rerank_request_rejects_non_string_documents_as_text_only() -> None:
    # The deliberate scope decision: page-image documents (ColPali) are NOT
    # supported by this runtime, and the error says so.
    with pytest.raises(ValueError, match="TEXT documents only"):
        parse_rerank_request({"query": "q", "documents": [{"image": "..."}]})


# -- HTTP surface ------------------------------------------------------------


def test_healthz_and_models(api: tuple[TestClient, _FakeBackend]) -> None:
    client, _ = api
    health = client.get("/healthz").json()
    assert health == {"status": "ok", "model": "mxbai-edge-colbert-v0-32m", "kind": "multi_vector"}
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == "mxbai-edge-colbert-v0-32m"


def test_rerank_returns_llama_server_shaped_results(
    api: tuple[TestClient, _FakeBackend],
) -> None:
    # The wire contract chat_api.py's proxy forwards to and every consumer
    # reads: results sorted by relevance_score desc, index into the request.
    client, backend = api
    response = client.post(
        "/v1/rerank",
        json={
            "model": "mxbai-edge-colbert-v0-32m",
            "query": "invoice total",
            "documents": ["unrelated weather report", "FACTURE invoice total 5400 EUR"],
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["index"] == 1
    assert results[0]["relevance_score"] > results[1]["relevance_score"]
    assert backend.calls == [
        ("invoice total", ["unrelated weather report", "FACTURE invoice total 5400 EUR"])
    ]


def test_rerank_honours_top_n(api: tuple[TestClient, _FakeBackend]) -> None:
    client, _ = api
    response = client.post(
        "/v1/rerank",
        json={"query": "invoice", "documents": ["invoice", "weather", "invoice again"], "top_n": 1},
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_rerank_is_served_on_every_llama_server_alias(
    api: tuple[TestClient, _FakeBackend],
) -> None:
    # chat_api.py posts to f"{base_url}/rerank" where base_url ends in /v1 --
    # so /v1/rerank is the one that matters -- but llama-server also answers
    # /rerank and /v1/reranking; a caller written against it must just work.
    client, _ = api
    body = {"query": "q", "documents": ["a"]}
    for path in ("/v1/rerank", "/rerank", "/v1/reranking"):
        assert client.post(path, json=body).status_code == 200, path


def test_rerank_400s_on_malformed_body_with_openai_error_shape(
    api: tuple[TestClient, _FakeBackend],
) -> None:
    client, backend = api
    response = client.post("/v1/rerank", json={"query": "q", "documents": []})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "'documents'" in error["message"]
    assert backend.calls == []  # rejected before touching the backend

    response = client.post(
        "/v1/rerank", content=b"not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert "valid JSON" in response.json()["error"]["message"]


# -- the real backend's guards (no model load; the class is stubbed) --------


def _install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    fake = types.ModuleType("sentence_transformers")
    fake.__version__ = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)


def test_backend_refuses_a_too_old_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    # MultiVectorEncoder landed in 6.0: a 5.x install has the package but not
    # the class. The version is checked up front so the deploy fails at
    # startup with the actionable reason, not with an AttributeError.
    _install_fake_sentence_transformers(monkeypatch, "5.4.0")
    with pytest.raises(RuntimeError, match=f">= {MULTI_VECTOR_MIN_VERSION}|too old"):
        SentenceTransformersMultiVectorBackend("some/model")


def test_backend_pins_meanmaxsim_and_scores_via_the_encoder_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Drives the real backend against a stubbed MultiVectorEncoder to pin the
    # exact upstream API used (encode_query / encode_document / similarity)
    # and that similarity_fn_name is set to the BOUNDED variant.
    calls: dict[str, object] = {}

    class _Scores:
        def __init__(self, rows: list[list[float]]) -> None:
            self._rows = rows

        def __getitem__(self, i: int) -> _Row:
            return _Row(self._rows[i])

    class _Row:
        def __init__(self, row: list[float]) -> None:
            self._row = row

        def tolist(self) -> list[float]:
            return self._row

    class _FakeEncoder:
        def __init__(self, model_id: str) -> None:
            calls["model_id"] = model_id
            self.similarity_fn_name = "maxsim"

        def encode_query(self, queries: list[str]) -> object:
            calls["queries"] = queries
            return "Q"

        def encode_document(self, documents: list[str]) -> object:
            calls["documents"] = documents
            return "D"

        def similarity(self, q: object, d: object) -> _Scores:
            calls["similarity_args"] = (q, d)
            return _Scores([[0.25, 0.75]])

    fake = types.ModuleType("sentence_transformers")
    fake.__version__ = "6.0.0"  # type: ignore[attr-defined]
    fake.MultiVectorEncoder = _FakeEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    backend = SentenceTransformersMultiVectorBackend("mixedbread-ai/mxbai-edge-colbert-v0-32m")
    scores = backend.rerank("q", ["d1", "d2"])

    assert calls["model_id"] == "mixedbread-ai/mxbai-edge-colbert-v0-32m"
    assert backend._model.similarity_fn_name == "meanmaxsim"  # noqa: SLF001
    assert calls["queries"] == ["q"]
    assert calls["documents"] == ["d1", "d2"]
    assert calls["similarity_args"] == ("Q", "D")
    assert scores == [0.25, 0.75]
