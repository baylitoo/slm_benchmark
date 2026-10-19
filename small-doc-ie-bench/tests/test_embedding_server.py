"""Embeddings protocol behavior, using deterministic text-derived vectors."""

from __future__ import annotations

import base64
import math
import struct
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from docie_bench.embedding_server.server import (
    PROMPTS,
    EmbeddingBatch,
    SentenceTransformersEmbeddingBackend,
    create_embedding_app,
)


class _TextBackend:
    """A deterministic encoder, not a mock of the request/response wiring."""

    def embed(self, texts: list[str], *, prompt_name: str | None) -> EmbeddingBatch:
        formatted = [PROMPTS.get(prompt_name, "") + text for text in texts]
        vectors = []
        for text in formatted:
            vector = [0.0] * 768
            for position, character in enumerate(text):
                vector[position % 768] += ord(character)
            # Nonzero tail distinguishes normalize-then-truncate from the
            # required truncate-then-normalize, even for short test strings.
            vector[-1] += 1000.0
            vectors.append(vector)
        return EmbeddingBatch(vectors, sum(len(text.split()) + 2 for text in formatted))


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_embedding_app(model_id="google/embeddinggemma-2", backend=_TextBackend())
    with TestClient(app) as api:
        yield api


def test_health_and_discovery(client: TestClient) -> None:
    assert client.get("/healthz").json() == {
        "status": "ok", "model": "google/embeddinggemma-2", "kind": "dense_embedding"
    }
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "google/embeddinggemma-2"


def test_float_response_shape_order_and_actual_usage(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"input": ["apple pear", "banana"]})
    assert response.status_code == 200
    result = response.json()
    assert result["object"] == "list"
    assert result["model"] == "google/embeddinggemma-2"
    assert result["usage"] == {"prompt_tokens": 7, "total_tokens": 7}
    assert [row["index"] for row in result["data"]] == [0, 1]
    for row, text in zip(result["data"], ("apple pear", "banana"), strict=True):
        assert row["object"] == "embedding"
        assert len(row["embedding"]) == 768
        assert math.hypot(*row["embedding"]) == pytest.approx(1.0)
        single = client.post("/v1/embeddings", json={"input": text}).json()
        assert row["embedding"] == single["data"][0]["embedding"]
    assert result["data"][0]["embedding"] != result["data"][1]["embedding"]


@pytest.mark.parametrize("dimensions", [128, 256, 512, 768])
def test_truncation_renormalizes_and_base64_is_little_endian_float32(
    client: TestClient, dimensions: int
) -> None:
    body = {"input": "semantic search", "dimensions": dimensions}
    float_response = client.post("/v1/embeddings", json=body)
    assert float_response.status_code == 200
    vector = float_response.json()["data"][0]["embedding"]
    assert len(vector) == dimensions
    assert math.hypot(*vector) == pytest.approx(1.0)
    response = client.post("/v1/embeddings", json={**body, "encoding_format": "base64"})
    assert response.status_code == 200
    raw = base64.b64decode(response.json()["data"][0]["embedding"], validate=True)
    assert len(raw) == 4 * dimensions
    assert struct.unpack(f"<{dimensions}f", raw) == pytest.approx(vector, rel=1e-6)


@pytest.mark.parametrize(
    "body",
    [
        None, [], "text", {}, {"input": []}, {"input": ""}, {"input": "  "},
        {"input": ["ok", ""]}, {"input": [1, 2]}, {"input": [[1, 2]]},
        {"input": {"text": "x", "image": "file.png"}},
        {"input": [{"image_url": "https://example.org/image.png"}]},
        {"input": ["text", False]}, {"input": "text", "dimensions": True},
        {"input": "text", "dimensions": 128.0}, {"input": "text", "dimensions": "128"},
        {"input": "text", "dimensions": 0}, {"input": "text", "dimensions": 384},
        {"input": "text", "dimensions": 769}, {"input": "text", "dimensions": None},
        {"input": "text", "encoding_format": "int8"},
        {"input": "text", "encoding_format": None},
        {"input": "text", "prompt_name": "query"},
        {"input": "text", "prompt_name": []},
    ],
)
def test_invalid_inputs_are_openai_400s(client: TestClient, body: object) -> None:
    response = client.post("/v1/embeddings", json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["message"]


def test_invalid_json_is_openai_400(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings", content="{broken", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert "valid JSON" in response.json()["error"]["message"]


@pytest.mark.parametrize("vector", [[0.0] * 768, [float("nan")] * 768, [1.0] * 767])
def test_invalid_backend_embeddings_never_escape_as_success(vector: list[float]) -> None:
    class InvalidBackend:
        def embed(self, texts: list[str], *, prompt_name: str | None) -> EmbeddingBatch:
            return EmbeddingBatch([vector], 1)

    with TestClient(create_embedding_app(model_id="broken", backend=InvalidBackend())) as api:
        response = api.post("/v1/embeddings", json={"input": "text"})
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "server_error"


class _Counts:
    def __init__(self, counts: list[int]) -> None:
        self.counts = counts

    def sum(self, *, dim: int) -> _Counts:
        return self

    def tolist(self) -> list[int]:
        return self.counts


class _Vectors:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    def tolist(self) -> list[list[float]]:
        return self.vectors


class _TextModel:
    """Executable tokenizer/encoder stand-in for testing context boundaries."""

    def preprocess(
        self, texts: list[str], *, prompt: str, processing_kwargs: object
    ) -> dict[str, _Counts]:
        return {"attention_mask": _Counts([len((prompt + text).split()) + 2 for text in texts])}

    def encode(self, texts: list[str], **kwargs: object) -> _Vectors:
        prompt_name = kwargs.get("prompt_name")
        assert prompt_name is None or isinstance(prompt_name, str)
        batch = _TextBackend().embed(texts, prompt_name=prompt_name)
        return _Vectors(batch.vectors)


@pytest.mark.parametrize("prompt_name", [None, "SearchQuery", "Document"])
def test_backend_context_limit_counts_prompt_and_special_tokens(prompt_name: str | None) -> None:
    backend = object.__new__(SentenceTransformersEmbeddingBackend)
    backend._model = _TextModel()
    backend._max_tokens = 12
    prefix_tokens = len(PROMPTS.get(prompt_name, "").split())
    text = " ".join(["word"] * (10 - prefix_tokens))
    with TestClient(create_embedding_app(model_id="bounded", backend=backend)) as api:
        accepted = api.post("/v1/embeddings", json={"input": text, "prompt_name": prompt_name})
        assert accepted.status_code == 200
        assert accepted.json()["usage"] == {"prompt_tokens": 12, "total_tokens": 12}
        rejected = api.post(
            "/v1/embeddings", json={"input": text + " extra", "prompt_name": prompt_name}
        )
        assert rejected.status_code == 400
        assert "12-token context limit" in rejected.json()["error"]["message"]
