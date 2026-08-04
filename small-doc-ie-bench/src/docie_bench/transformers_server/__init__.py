"""Generic transformers (AutoModel) serving — the last-resort runtime.

A model with no GGUF (or an architecture llama.cpp cannot serve) can still be
served here via Hugging Face ``transformers``: ``AutoProcessor`` +
``AutoModelForImageTextToText`` / ``AutoModelForCausalLM`` load the model, its
chat template and its (multimodal) processor straight from the repo, so
onboarding needs NO per-model family contract — the generic-add behavior the
GGUF/llama.cpp path can't offer.

Deliberately the LAST RESORT: unquantized weights use ~2-3x the RAM of a GGUF
Q4 and CPU inference is much slower. Prefer a GGUF repo whenever one exists.
Custom-code checkpoints (``config.json`` ``auto_map``) additionally require
``trust_remote_code`` — arbitrary code execution on the serving node, so it is
opt-in per deployment and off by default.
"""

from docie_bench.transformers_server.server import create_transformers_app

__all__ = ["create_transformers_app"]
