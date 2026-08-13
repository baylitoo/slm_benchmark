"""Retrieval demo: embed a corpus, cosine-rank it, then rerank the shortlist.

Shows why the two-stage pattern earns its keep. A fast, cheap first pass over
every document (embeddings + cosine similarity) narrows a large corpus down to
a shortlist; a slower, more accurate cross-encoder reranker (LFM2.5-ColBERT-
350M, or any deployment registered with family "reranker") then reorders that
shortlist by the query's actual relevance, not just semantic distance. The two
scores routinely disagree at the margin -- that gap is the reranker's value.

Requires an embedding deployment and a reranker deployment already live (see
DocIE Studio's Deploy tab, or POST /v1/serving/store/<name>/scale).

Example:
  python scripts/rag_demo.py --embed-model lfm25-embedding-350m \
      --rerank-model lfm25-colbert-350m --query "What is the invoice total?"
"""

from __future__ import annotations

import argparse
import math

import httpx

# A deliberately mixed corpus: some docs are lexically close to the query
# (share words like "total"/"invoice") but semantically off-topic, others are
# on-topic without sharing vocabulary -- the kind of set where cosine-only
# search and a cross-encoder rerank can disagree.
DEFAULT_CORPUS = [
    "FACTURE total 5400 EUR TTC, echeance 30 jours.",
    "Weather forecast for tomorrow: light rain, high of 18C.",
    "Invoice total 5400 EUR, net 30 payment terms.",
    "Our return policy allows exchanges within 30 days of purchase.",
    "Quarterly board meeting minutes: revenue up 12% year over year.",
    "The customer's shipping address was updated on file.",
    "Grand total due: five thousand four hundred euros, VAT included.",
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def embed(base_url: str, model: str, texts: list[str]) -> list[list[float]]:
    resp = httpx.post(
        f"{base_url}/v1/embeddings", json={"model": model, "input": texts}, timeout=120
    )
    resp.raise_for_status()
    return [row["embedding"] for row in resp.json()["data"]]


def rerank(base_url: str, model: str, query: str, documents: list[str]) -> list[dict]:
    resp = httpx.post(
        f"{base_url}/v1/rerank",
        json={"model": model, "query": query, "documents": documents},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["results"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--embed-model", required=True, help="An embedding deployment name.")
    parser.add_argument("--rerank-model", required=True, help="A reranker deployment name.")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--top-k", type=int, default=4, help="Shortlist size handed to the reranker."
    )
    args = parser.parse_args()

    corpus = DEFAULT_CORPUS
    vectors = embed(args.base_url, args.embed_model, [args.query, *corpus])
    query_vec, doc_vecs = vectors[0], vectors[1:]

    cosine_ranked = sorted(
        zip(corpus, doc_vecs, strict=True),
        key=lambda pair: cosine(query_vec, pair[1]),
        reverse=True,
    )
    shortlist = [doc for doc, _ in cosine_ranked[: args.top_k]]

    print(f"Query: {args.query!r}\n")
    print(f"Stage 1 -- cosine similarity over {len(corpus)} docs ({args.embed_model}):")
    for doc, vec in cosine_ranked:
        marker = ">" if doc in shortlist else " "
        print(f"  {marker} {cosine(query_vec, vec):.4f}  {doc}")

    reranked = rerank(args.base_url, args.rerank_model, args.query, shortlist)
    reranked.sort(key=lambda r: r["relevance_score"], reverse=True)

    print(f"\nStage 2 -- reranked shortlist ({args.rerank_model}):")
    for r in reranked:
        print(f"    {r['relevance_score']:.4f}  {shortlist[r['index']]}")

    cosine_top = shortlist[0]
    rerank_top = shortlist[reranked[0]["index"]]
    if cosine_top != rerank_top:
        print("\nReranker disagreed with cosine on the top result:")
        print(f"  cosine top:   {cosine_top!r}")
        print(f"  reranked top: {rerank_top!r}")


if __name__ == "__main__":
    main()
