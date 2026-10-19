"""Text-only EmbeddingGemma 2 runtime with an OpenAI embeddings HTTP surface.

Prompt selection is explicit: omit ``prompt_name`` for already formatted text,
otherwise use ``SearchQuery`` or ``Document``. Media and token-ID inputs are not
accepted. Output is normalized after optional Matryoshka truncation.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib.metadata
import importlib.util
import math
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from docie_bench.openai_protocol import openai_error

SUPPORTED_DIMENSIONS = (128, 256, 512, 768)
PROMPTS = {
    "SearchQuery": "task: search result | query: ",
    "Document": "title: none | text: ",
}
_INSTALL_HINT = "pip install 'small-doc-ie-bench[embeddings]' (separate from the encoders extra)"


def dependency_error() -> str | None:
    """Probe without importing torch or downloading a model."""
    for module, distribution, minimum in (
        ("torch", "torch", None),
        ("torchvision", "torchvision", None),
        ("sentence_transformers", "sentence-transformers", (6, 1)),
        ("transformers", "transformers", (5, 19)),
    ):
        if importlib.util.find_spec(module) is None:
            return f"{distribution} is not installed; {_INSTALL_HINT}"
        if minimum is not None:
            try:
                installed = importlib.metadata.version(distribution)
                release = tuple(int(part) for part in installed.split(".")[:2])
            except (importlib.metadata.PackageNotFoundError, ValueError):
                return f"cannot determine {distribution} version; {_INSTALL_HINT}"
            if release < minimum:
                required = ".".join(str(part) for part in minimum)
                return (
                    f"{distribution} {installed} is too old for EmbeddingGemma 2; "
                    f"requires >= {required}; {_INSTALL_HINT}"
                )
    return None


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    prompt_tokens: int


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str], *, prompt_name: str | None) -> EmbeddingBatch:
        """Return 768-dimensional vectors in input order and actual token usage."""
        ...


def parse_embedding_request(body: Any) -> tuple[list[str], str | None, int, str]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    inputs = body.get("input")
    texts = [inputs] if isinstance(inputs, str) else inputs
    if not isinstance(texts, list) or not texts:
        raise ValueError("'input' must be a non-empty string or a non-empty list of strings")
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(
                f"input[{index}] must be text; token IDs, images, audio and video are not supported"
            )
        if not text.strip():
            raise ValueError(f"input[{index}] must be a non-empty string")
    prompt_name = body.get("prompt_name")
    if prompt_name is not None and (
        not isinstance(prompt_name, str) or prompt_name not in PROMPTS
    ):
        raise ValueError("'prompt_name' must be 'SearchQuery' or 'Document'")
    dimensions = body.get("dimensions", 768)
    if type(dimensions) is not int or dimensions not in SUPPORTED_DIMENSIONS:
        raise ValueError("'dimensions' must be one of 768, 512, 256, 128")
    encoding = body.get("encoding_format", "float")
    if encoding not in ("float", "base64"):
        raise ValueError("'encoding_format' must be 'float' or 'base64'")
    return texts, prompt_name, dimensions, encoding


def normalize_vector(vector: list[float], dimensions: int) -> list[float]:
    if len(vector) != 768 or not all(math.isfinite(value) for value in vector):
        raise RuntimeError(
            "embedding backend returned invalid or non-finite 768-dimensional output"
        )
    truncated = vector[:dimensions]
    norm = math.hypot(*truncated)
    if norm == 0:
        raise RuntimeError("embedding backend returned a zero-length vector")
    return [float(value / norm) for value in truncated]


class SentenceTransformersEmbeddingBackend:
    """Load only the text backbone, never in float16."""

    def __init__(self, model_id: str) -> None:
        reason = dependency_error()
        if reason is not None:
            raise RuntimeError(reason)
        import torch
        from sentence_transformers import SentenceTransformer

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self._model = SentenceTransformer(
            model_id,
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"torch_dtype": dtype},
        )
        self._max_tokens = min(8192, self._model.max_seq_length)

    def embed(self, texts: list[str], *, prompt_name: str | None) -> EmbeddingBatch:
        prompt = PROMPTS[prompt_name] if prompt_name is not None else ""
        tokens = 0
        # Count exactly the processor's non-padding tokens, including the prompt
        # and special tokens. Disable truncation here so oversized text is rejected
        # rather than silently embedded only in part.
        for start in range(0, len(texts), 32):
            features = self._model.preprocess(
                texts[start : start + 32],
                prompt=prompt,
                processing_kwargs={"text": {"truncation": False}},
            )
            counts = features["attention_mask"].sum(dim=1).tolist()
            for offset, count in enumerate(counts):
                if count > self._max_tokens:
                    raise ValueError(
                        f"input[{start + offset}] exceeds the {self._max_tokens}-token "
                        "context limit (including prompt and special tokens)"
                    )
                tokens += int(count)
        # An empty explicit prompt suppresses checkpoint defaults; no task is
        # silently selected for caller-formatted input.
        prompt_kwargs = {"prompt_name": prompt_name} if prompt_name else {"prompt": ""}
        vectors = self._model.encode(
            texts,
            **prompt_kwargs,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return EmbeddingBatch(vectors=vectors.tolist(), prompt_tokens=tokens)


def create_embedding_app(
    *, model_id: str, backend: EmbeddingBackend | None = None
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.backend is None:
            app.state.backend = SentenceTransformersEmbeddingBackend(model_id)
        yield

    app = FastAPI(title="docie dense embeddings", lifespan=lifespan)
    app.state.backend = backend
    app.state.model_id = model_id

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"status": "ok", "model": model_id, "kind": "dense_embedding"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": 0, "owned_by": "docie-embeddings"}
            ],
        }

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Any:
        try:
            body = await request.json()
        except ValueError:
            return openai_error(
                "request body must be valid JSON", status_code=400,
                error_type="invalid_request_error",
            )
        try:
            texts, prompt_name, dimensions, encoding = parse_embedding_request(body)
            backend_impl: EmbeddingBackend = app.state.backend
            batch = await asyncio.to_thread(backend_impl.embed, texts, prompt_name=prompt_name)
        except ValueError as exc:
            return openai_error(str(exc), status_code=400, error_type="invalid_request_error")
        try:
            if len(batch.vectors) != len(texts):
                raise RuntimeError("embedding backend returned the wrong number of vectors")
            data: list[dict[str, Any]] = []
            for index, vector in enumerate(batch.vectors):
                normalized = normalize_vector(vector, dimensions)
                embedding: list[float] | str = normalized
                if encoding == "base64":
                    embedding = base64.b64encode(
                        struct.pack(f"<{dimensions}f", *normalized)
                    ).decode("ascii")
                data.append({"object": "embedding", "index": index, "embedding": embedding})
        except RuntimeError as exc:
            return openai_error(str(exc), status_code=500, error_type="server_error")
        return JSONResponse({
            "object": "list", "model": model_id, "data": data,
            "usage": {"prompt_tokens": batch.prompt_tokens, "total_tokens": batch.prompt_tokens},
        })

    return app
