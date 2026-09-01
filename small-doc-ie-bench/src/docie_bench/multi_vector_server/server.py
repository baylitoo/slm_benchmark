"""``/v1/rerank`` shim server over a sentence-transformers ``MultiVectorEncoder``.

The multi-vector (ColBERT / PyLate) serving path (see the package docstring).
``MultiVectorEncoder`` loads a late-interaction checkpoint and scores query
vs. documents with MaxSim over per-token embeddings -- so a safetensors-only
retriever that llama.cpp cannot serve is still servable behind the SAME
rerank wire contract as ``llama-server --reranking``.

Contract (mirrors ``docie_bench.transformers_server`` so every deploy
surface, health probe and reconciler overlay is inherited unchanged; and
mirrors llama-server's rerank endpoint so ``chat_api.py``'s proxy forwards to
either reranker family unchanged):

* ``POST /v1/rerank`` -- body ``{"query": str, "documents": [str, ...],
  "top_n": int?}`` (``model`` is accepted and ignored: this process serves
  exactly one). Response ``{"results": [{"index": int, "relevance_score":
  float}, ...]}`` sorted by score descending, ``index`` into the request's
  ``documents``. Also served at ``/rerank`` and ``/v1/reranking`` --
  llama-server's own aliases -- so any caller written against it works.
* ``GET /v1/models`` / ``GET /healthz`` -- the usual discovery/liveness pair.

``relevance_score`` is the MEAN-MaxSim (``similarity_fn_name="meanmaxsim"``:
MaxSim divided by the query token count), i.e. an average per-query-token
cosine similarity -- BOUNDED, unlike raw MaxSim which is an unbounded sum
that grows with query length. Bounded scores are what every existing
consumer of ``relevance_score`` (the Playground Rerank tab, a cross-encoder
reranker's sigmoid) already assumes; a raw ``47.3`` next to a ``0.92`` would
read as broken. Ranking is identical either way (a positive constant divisor
never reorders).

``backend`` is an injection seam: tests pass any object with the same
``rerank`` signature, so the request/response shaping is covered without
importing torch. The real backend (:class:`SentenceTransformersMultiVector
Backend`) is lazy -- a missing / too-old ``sentence-transformers`` install
fails the deploy at startup with an actionable message, never 500s the
first caller.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

MULTI_VECTOR_MIN_VERSION = 6


class MultiVectorBackend(Protocol):
    """One synchronous rerank: ``documents`` scored against ``query``.

    Returns one float per document, in DOCUMENT ORDER (the server sorts and
    attaches indices). Runs in a worker thread so the event loop is never
    blocked by CPU encoding.
    """

    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


# ---------------------------------------------------------------------------
# Pure request/response shaping (unit-tested without any model).
# ---------------------------------------------------------------------------


def rank_results(scores: list[float], *, top_n: int | None) -> list[dict[str, Any]]:
    """Turn per-document scores into the rerank ``results`` list.

    Sorted by score descending; ties keep document order (stable sort), so a
    caller's original ordering is the deterministic tiebreak, never a hash
    order. ``top_n`` truncates AFTER sorting -- the top-N of the ranking, not
    the first N documents. ``None``/non-positive means "all".
    """
    ranked = sorted(enumerate(scores), key=lambda pair: -pair[1])
    if top_n is not None and top_n > 0:
        ranked = ranked[:top_n]
    return [{"index": index, "relevance_score": float(score)} for index, score in ranked]


def parse_rerank_request(body: Any) -> tuple[str, list[str], int | None]:
    """Validate a rerank body -> ``(query, documents, top_n)``.

    Raises ValueError with a caller-facing message on any malformed field.
    Mirrors ``chat_api.py``'s own pre-forward validation so a request that
    passes the proxy passes here too, and one that reaches this process
    directly gets the same errors.
    """
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("missing required 'query' field (a non-empty string)")
    documents = body.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("missing required 'documents' field (a non-empty list of strings)")
    for position, document in enumerate(documents):
        if not isinstance(document, str):
            raise ValueError(
                f"documents[{position}] is not a string -- this runtime reranks TEXT "
                "documents only (page-image documents are not supported)"
            )
    top_n_raw = body.get("top_n")
    top_n: int | None
    if top_n_raw is None:
        top_n = None
    else:
        try:
            top_n = int(top_n_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("'top_n' must be an integer") from exc
        if top_n < 1:
            raise ValueError("'top_n' must be positive")
    return query, documents, top_n


# ---------------------------------------------------------------------------
# The real backend (lazy import -- the heavyweight path).
# ---------------------------------------------------------------------------


class SentenceTransformersMultiVectorBackend:
    """``MultiVectorEncoder`` backend, loaded once at construction.

    ``similarity_fn_name`` is pinned to ``"meanmaxsim"`` (see the module
    docstring for why bounded scores matter on the shared rerank surface).
    """

    def __init__(self, model_id: str) -> None:
        try:
            import sentence_transformers
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "the multi-vector serving backend requires sentence-transformers >= "
                f"{MULTI_VECTOR_MIN_VERSION} (shipped in the 'encoders' extra): "
                "pip install 'small-doc-ie-bench[encoders]'"
            ) from exc
        installed = getattr(sentence_transformers, "__version__", "0")
        try:
            major = int(str(installed).split(".", 1)[0])
        except ValueError:  # pragma: no cover - malformed version string
            major = 0
        if major < MULTI_VECTOR_MIN_VERSION:
            raise RuntimeError(
                f"sentence-transformers {installed} is too old: MultiVectorEncoder "
                f"landed in {MULTI_VECTOR_MIN_VERSION}.0 -- upgrade with "
                "pip install 'small-doc-ie-bench[encoders]'"
            )
        try:
            from sentence_transformers import MultiVectorEncoder
        except ImportError as exc:  # pragma: no cover - guarded by the version check
            raise RuntimeError(
                f"sentence-transformers {installed} has no MultiVectorEncoder"
            ) from exc

        self.model_id = model_id
        self._model = MultiVectorEncoder(model_id)
        # Bounded per-query-token average, not the unbounded raw MaxSim sum.
        self._model.similarity_fn_name = "meanmaxsim"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        query_embeddings = self._model.encode_query([query])
        document_embeddings = self._model.encode_document(documents)
        # similarity(queries, documents) -> [n_queries, n_documents]; one query.
        scores = self._model.similarity(query_embeddings, document_embeddings)[0]
        return [float(score) for score in scores.tolist()]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def _openai_error(message: str, *, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": error_type}},
    )


def create_multi_vector_app(
    *,
    model_id: str,
    backend: MultiVectorBackend | None = None,
) -> FastAPI:
    """Build the multi-vector rerank shim app. ``backend=None`` loads one at startup."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load at startup (not first request) so a missing/too-old extra or
        # missing weights fail the deploy immediately instead of 500ing the
        # first caller.
        if app.state.backend is None:
            app.state.backend = SentenceTransformersMultiVectorBackend(model_id)
        yield

    app = FastAPI(
        title="docie multi-vector",
        summary="ColBERT / PyLate late-interaction retriever behind the rerank surface "
        "(sentence-transformers MultiVectorEncoder, MaxSim scoring).",
        lifespan=lifespan,
    )
    app.state.backend = backend
    app.state.model_id = model_id

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"status": "ok", "model": app.state.model_id, "kind": "multi_vector"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "docie-multi-vector",
                }
            ],
        }

    async def _rerank(request: Request) -> Any:
        try:
            body = await request.json()
        except ValueError:
            return _openai_error(
                "request body must be valid JSON",
                status_code=400,
                error_type="invalid_request_error",
            )
        try:
            query, documents, top_n = parse_rerank_request(body)
        except ValueError as exc:
            return _openai_error(
                str(exc), status_code=400, error_type="invalid_request_error"
            )
        backend_impl: MultiVectorBackend = app.state.backend
        try:
            scores = await asyncio.to_thread(backend_impl.rerank, query, documents)
        except ValueError as exc:
            return _openai_error(
                str(exc), status_code=400, error_type="invalid_request_error"
            )
        return JSONResponse(
            {
                "model": app.state.model_id,
                "object": "list",
                "results": rank_results(scores, top_n=top_n),
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
        )

    # llama-server serves /rerank with /v1/rerank and /v1/reranking as aliases;
    # register all three so any caller written against it just works.
    app.post("/v1/rerank")(_rerank)
    app.post("/rerank")(_rerank)
    app.post("/v1/reranking")(_rerank)

    return app
