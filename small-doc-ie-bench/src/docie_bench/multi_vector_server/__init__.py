"""Multi-vector (late-interaction) reranking — the ColBERT / PyLate runtime.

A ColBERT-style retriever keeps one embedding PER TOKEN (versus a dense
embedding model's single pooled vector) and scores a query against a document
with MaxSim: for each query token, the max similarity over the document's
tokens, summed. That preserves the token-level matching a single vector has
to average away. These checkpoints ship as safetensors (PyLate, Stanford
ColBERT), which llama.cpp cannot load -- so unlike a GGUF reranker (family
``reranker``, ``llama-server --reranking``) they need a Python runtime.

That runtime is sentence-transformers >= 6's ``MultiVectorEncoder``
(``encode_query`` / ``encode_document`` / ``similarity``), which loads every
PyLate, Stanford ColBERT, and native checkpoint format behind one API. This
package is a thin ``/v1/rerank`` shim over it -- the same wire contract as
llama-server's rerank endpoint, so ``chat_api.py``'s proxy forwards to either
family unchanged.

Text-only by design: ColPali-style page-image documents are deliberately not
supported here (see the project decision), even though the underlying encoder
API is modality-agnostic.
"""

from docie_bench.multi_vector_server.server import create_multi_vector_app

__all__ = ["create_multi_vector_app"]
