"""Canonical GGUF model store shared by llama.cpp (llama-server) and Ollama.

The goal is a single `models/` directory that *both* runtimes can serve from,
where each model **family** declares how its prompt template must be delivered so
the model is served correctly. The same GGUF is never downloaded twice: a store
entry can be *seeded* from a model already pulled by Ollama by hard-linking its
content-addressed blobs into the store under stable `*.gguf` filenames.

Layout::

    <root>/<name>/model.gguf      # weights
    <root>/<name>/mmproj.gguf     # optional vision projector
    <root>/index.json             # name -> entry metadata

Why a canonical store and not the raw Ollama blob? Two reasons surfaced while
integrating NuExtract3 (verified 2026-06-17):

* Ollama stores GGUFs as extension-less ``blobs/sha256-<hex>`` files; the
  llama.cpp runtime adapter requires a ``.gguf`` path, so it cannot point at a
  blob directly. Stable ``*.gguf`` names fix that.
* Ollama silently drops ``chat_template_kwargs`` over GGUF, so families that
  deliver their template that way (NuExtract3) cannot be served *faithfully* by
  Ollama at all — they must go through ``llama-server --jinja``. The family
  contract records this so the store refuses to emit a misleading Ollama
  Modelfile for such families.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from filelock import FileLock, Timeout

from docie_bench.serving.resources import DEFAULT_DEPLOY_CONTEXT_LENGTH
from docie_bench.serving.storage_backend import (
    LocalDiskBackend,
    StorageBackend,
    StorageBackendError,
)

INDEX_LOCK_NAME = ".index.lock"
"""Guards the index's read-modify-write; lives in the store root."""


class TemplateDelivery(StrEnum):
    """How a model family receives the extraction template/schema."""

    # Template passed via extra_body.chat_template_kwargs; requires the GGUF's
    # embedded jinja template, i.e. `llama-server --jinja`. (NuExtract3.)
    CHAT_TEMPLATE_KWARGS = "chat_template_kwargs"
    # Template baked into the prompt text, e.g. NuExtract v1's
    # `<|input|>### Template: ... ### Text: ...<|output|>` format.
    INPROMPT_NUEXTRACT_V1 = "inprompt_nuextract_v1"
    # Standard chat models: schema delivered via OpenAI `response_format`.
    OPENAI_JSON_SCHEMA = "openai_json_schema"


@dataclass(frozen=True)
class FamilyContract:
    """How one model *family* must be launched and prompted to respect its template.

    A contract is runtime-agnostic intent; the store/runtime translate it into
    concrete launch flags and extraction-client settings.
    """

    name: str
    template_delivery: TemplateDelivery
    # Extraction-client wiring (see docie_bench.llm).
    response_format_style: str
    prompt_profile: str
    # Extra llama-server flags required to honour the template (e.g. --jinja).
    llama_server_args: tuple[str, ...] = ()
    needs_mmproj: bool = False
    vision: bool = False
    # OpenAI tool calling (`tools`/`tool_choice` -> `tool_calls`). llama-server
    # only renders tools through its jinja engine, so this is True ONLY for
    # families whose launch args carry --jinja AND whose contract is
    # conversational (an extraction contract like nuextract3 has --jinja but
    # tool calling is meaningless there). The chat proxy forwards tools
    # verbatim either way -- this flag is the HONEST capability signal the
    # families API exposes, so a caller can tell "tools ignored by a
    # non-jinja launch" apart from "model chose not to call".
    tools: bool = False
    # An embedding model: GGUF pooling or native dense embeddings, answered on
    # /v1/embeddings — never chat/extraction. Deployments of an
    # embedding family are typed "embedding" and routed by the embeddings API.
    embedding: bool = False
    # Native dense embeddings served from a snapshot by SentenceTransformer.
    dense_embedding: bool = False
    # A reranker / late-interaction retriever (e.g. LFM2.5-ColBERT-350M): served
    # with llama-server's --reranking + --embedding --pooling rank (llama.cpp
    # requires both), and answered on /v1/rerank — never chat/extraction/plain
    # embeddings. Deployments of a reranker family are typed "reranker" and
    # routed by the rerank API, same shape as the embedding family above.
    reranker: bool = False
    # An analyzer (encoder) family: a transformers/safetensors checkpoint served
    # by the ENCODER runtime (docie encoder), NOT llama.cpp. Stored as a
    # directory snapshot rather than a single GGUF, and typed "encoder".
    # ``encoder_backend`` selects the analyzer library (gliner | gliner2 |
    # gliformer | gliclass); it is
    # the store-driven replacement for the server's name-based auto-detection.
    analyzer: bool = False
    encoder_backend: str | None = None
    # An analyzer that ALSO answers a schema: GLiFormer's structure() head reads
    # text and returns records, so the model belongs in the extraction picker
    # even though it is served by the encoder runtime. Being an analyzer says
    # how it is launched; this says what it can be asked for.
    structured_extraction: bool = False
    # A transformers/AutoModel family: a safetensors snapshot served by the
    # TRANSFORMERS runtime (docie transformers), NOT llama.cpp. The LAST-RESORT
    # path for a model with no GGUF / an arch llama.cpp cannot serve — stored as
    # a directory snapshot like an analyzer, and typed "transformers".
    # ``trust_remote_code`` opts a custom-code checkpoint into executing the
    # repo's Python on the serving node (SECURITY: off by default).
    transformers_runtime: bool = False
    trust_remote_code: bool = False
    # A multi-vector / late-interaction retriever (ColBERT, PyLate) served by
    # the MULTI-VECTOR runtime (sentence-transformers >= 6's
    # MultiVectorEncoder), NOT llama.cpp: a safetensors snapshot, answered on
    # /v1/rerank with MaxSim scoring over per-token embeddings. Distinct from
    # ``reranker`` (a GGUF served by llama-server --reranking): same API
    # surface, different runtime and artifact shape.
    multi_vector: bool = False
    stop_sequences: tuple[str, ...] = ()
    # Generation defaults inherited by a family-synthesized profile (a store deploy
    # whose served id matches no models.yaml profile). These are the single source
    # of truth for a family's tuning, so such a deployment runs with the family's
    # intended params — NOT the bare ModelProfile defaults (900 / 0.0 / 180), which
    # would silently degrade a model like NuExtract3 that needs long outputs.
    default_temperature: float = 0.0
    default_max_tokens: int = 900
    default_timeout_seconds: float = 180.0
    # Can Ollama serve this family *faithfully* (respecting its template)?
    # False for CHAT_TEMPLATE_KWARGS families: Ollama drops chat_template_kwargs.
    ollama_faithful: bool = True

    @property
    def snapshot(self) -> bool:
        """Is this a multi-file safetensors SNAPSHOT family (stored as a
        directory, served by a Python runtime via ``from_pretrained``) rather
        than a single-GGUF llama.cpp family?

        The one place that answers "snapshot or GGUF?" -- every seed/store/
        deploy branch that used to spell out ``analyzer or transformers_runtime``
        keys on this instead, so adding a new snapshot family (multi_vector was
        the third) means one flag here, not a widened disjunction at every
        call site.
        """
        return (
            self.analyzer
            or self.transformers_runtime
            or self.multi_vector
            or self.dense_embedding
        )


# Known families. Adding a new model is "drop the GGUF + pick (or add) a family".
_SNAPSHOT_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth"})


FAMILIES: dict[str, FamilyContract] = {
    "nuextract3": FamilyContract(
        name="nuextract3",
        template_delivery=TemplateDelivery.CHAT_TEMPLATE_KWARGS,
        response_format_style="nuextract3",
        prompt_profile="nuextract3",
        llama_server_args=("--jinja",),
        needs_mmproj=True,
        vision=True,
        default_temperature=0.2,
        # NuMind sizes long answers well past 4096 (their largest benchmark
        # ground truth is ~22k tokens); a CV's nested skill lists already come
        # within 150 tokens of 4096. llama-server stops at the slot's context
        # rather than rejecting a larger budget, so this only lifts the cap.
        default_max_tokens=16384,
        default_timeout_seconds=600.0,
        ollama_faithful=False,
    ),
    "nuextract_v1": FamilyContract(
        name="nuextract_v1",
        template_delivery=TemplateDelivery.INPROMPT_NUEXTRACT_V1,
        response_format_style="none",
        prompt_profile="nuextract_v1",
        stop_sequences=("<|end-output|>",),
        default_temperature=0.0,
        default_max_tokens=2000,
        default_timeout_seconds=300.0,
        ollama_faithful=True,
    ),
    "openai_chat": FamilyContract(
        name="openai_chat",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        response_format_style="openai_json_schema",
        prompt_profile="strict_extraction_v1",
        # --jinja: llama-server renders the GGUF's own chat template through
        # its jinja engine -- REQUIRED for native tool calling (without it a
        # `tools` request 200s with a plain completion, silently ignoring the
        # tools). Also the llama.cpp-recommended default for chat models.
        # Behavior note: this changes template rendering for existing
        # openai_chat deployments from the legacy path to the GGUF's own
        # template -- the honest rendering, but a change.
        llama_server_args=("--jinja",),
        tools=True,
        default_temperature=0.0,
        ollama_faithful=True,
    ),
    # LiquidAI LFM2.5 text (230M/350M/1.2B). A *standard* instruct model, so the
    # schema rides OpenAI `response_format` (openai_json_schema) — unlike
    # nuextract3 it needs NO chat_template_kwargs/extra_body. The one wrinkle: its
    # template is custom (`<|startoftext|>` + ChatML), which only renders
    # faithfully via the GGUF's embedded jinja template, hence `--jinja`. That
    # `--jinja` is the whole reason this is a distinct family from `openai_chat`.
    # A GBNF grammar (compiled from the json_schema) constrains the sampler, which
    # overrides LFM2.5's default Pythonic tool-call output and forces valid JSON.
    "lfm2": FamilyContract(
        name="lfm2",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        response_format_style="openai_json_schema",
        prompt_profile="strict_extraction_v1",
        llama_server_args=("--jinja",),
        tools=True,
        needs_mmproj=False,
        vision=False,
        default_temperature=0.0,  # extraction-deterministic (Liquid's 0.1 is a chat rec)
        default_max_tokens=4096,
        default_timeout_seconds=180.0,
        ollama_faithful=True,  # standard template; Ollama auto-extracts the embedded one
    ),
    # XHToken Spark-X2.5 (1.7B, arch_registry.py). A standard instruct model
    # from the extraction client's point of view -- response_format rides
    # openai_json_schema like lfm2, needing no chat_template_kwargs -- but
    # its GGUF chat template renders genuine tool-call XML (<tool_call>...),
    # unlike LFM2.5-1.2B-Thinking's baked template (which needed a manual
    # chat_template_file override, see #407), so tools=True should hold
    # out-of-the-box PROVIDED the template renders faithfully under
    # llama-server's --jinja (verify at deploy, not assumed here). Own family
    # rather than folded into "openai_chat" for the same reason as "lfm2":
    # its hybrid sliding-window/full-attention shape needs its own KV-cache
    # sizing constant (serving.resources.SPARK2_5_KV_CACHE_BYTES_PER_TOKEN).
    "spark2_5": FamilyContract(
        name="spark2_5",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        response_format_style="openai_json_schema",
        prompt_profile="strict_extraction_v1",
        llama_server_args=("--jinja",),
        tools=True,
        needs_mmproj=False,
        vision=False,
        default_temperature=0.0,
        default_max_tokens=4096,  # matches lfm2's own default (#435)
        default_timeout_seconds=180.0,
        ollama_faithful=True,
    ),
    # LiquidAI LFM3 (1.7B, arch "lfm3"). Same wire contract as "lfm2" -- its
    # own family so the sizing engine prices its hybrid conv/attention KV
    # cache (serving.resources.LFM3_KV_CACHE_BYTES_PER_TOKEN). Its template
    # thinks by default but honours enable_thinking=False, which extraction
    # always sends, so it is not an always-thinking checkpoint
    # (llm/reasoning.py).
    "lfm3": FamilyContract(
        name="lfm3",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        response_format_style="openai_json_schema",
        prompt_profile="strict_extraction_v1",
        llama_server_args=("--jinja",),
        tools=True,
        needs_mmproj=False,
        vision=False,
        default_temperature=0.0,  # Liquid's 0.1 + repetition_penalty 1.1 is a chat rec
        default_max_tokens=4096,
        default_timeout_seconds=180.0,
        ollama_faithful=True,
    ),
    # LiquidAI LFM2.5-VL (1.6B). SigLIP2 encoder + connector on the LFM2 backbone;
    # ships a projector, so `needs_mmproj=True` (family_launch_args appends
    # `--mmproj <path>` automatically -> ("--jinja", "--mmproj", <path>)). KEY diff
    # vs nuextract3: schema still rides `response_format` (openai_json_schema), so
    # it stays Ollama-faithful in template terms — but the TESTED VL runtime path is
    # llama-server (Ollama mmproj-via-ADAPTER support for lfm2-vl is unverified).
    "lfm2_vl": FamilyContract(
        name="lfm2_vl",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        response_format_style="openai_json_schema",
        prompt_profile="strict_extraction_v1",
        llama_server_args=("--jinja",),  # --mmproj appended by family_launch_args
        needs_mmproj=True,
        vision=True,
        default_temperature=0.1,  # slight temp helps VL; matches Liquid VL guidance
        default_max_tokens=4096,
        default_timeout_seconds=600.0,  # vision on CPU is slow (nuextract3 precedent)
        # Ollama mmproj/ADAPTER support for lfm2-vl is unverified; serve VL via
        # llama-server only (mirrors nuextract3 — refuses an Ollama Modelfile).
        ollama_faithful=False,
    ),
    # LiquidAI LFM3-VL (2B): the LFM3 backbone plus a projector. Its GGUF
    # reports "lfm3", promoted here by the mmproj (arch_registry
    # TEXT_ARCH_TO_VISION_FAMILY). Mirrors lfm2_vl.
    "lfm3_vl": FamilyContract(
        name="lfm3_vl",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        response_format_style="openai_json_schema",
        prompt_profile="strict_extraction_v1",
        llama_server_args=("--jinja",),
        needs_mmproj=True,
        vision=True,
        default_temperature=0.1,
        default_max_tokens=4096,
        default_timeout_seconds=600.0,
        ollama_faithful=False,
    ),
    # Vision OCR models: a vision GGUF whose job is FREE-TEXT OCR (image ->
    # text), not schema-driven extraction — e.g. sahilchachra/Unlimited-OCR-GGUF
    # (deepseek2-ocr). Shares the vision/mmproj plumbing with lfm2_vl/nuextract3
    # but with response_format_style="none": OCR output is plain text, and a
    # JSON grammar would corrupt it. The embedded (often trivial) chat template
    # is rendered by --jinja. Any vision OCR GGUF that ships an mmproj and emits
    # free text fits here regardless of base architecture — the contract, not
    # the arch, defines the family. Runtime support for the arch (e.g.
    # deepseek2-ocr) is a separate gate: a recent llama-server is required.
    "vision_ocr": FamilyContract(
        name="vision_ocr",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,  # unused: no schema sent
        response_format_style="none",
        prompt_profile="strict_extraction_v1",
        llama_server_args=("--jinja",),  # --mmproj appended by family_launch_args
        needs_mmproj=True,
        vision=True,
        default_temperature=0.0,
        default_max_tokens=4096,
        default_timeout_seconds=600.0,  # vision on CPU is slow
        ollama_faithful=False,
    ),
    # Embedding models (LFM2.5-Embedding-350M, or any GGUF embedding model).
    # llama-server exposes /v1/embeddings when launched with --embedding; mean
    # pooling is the standard sentence-embedding reduction. No template, no
    # schema — response_format is irrelevant on the embeddings path. Kept local
    # so a RAG/retrieval pipeline computes document vectors on THIS node and
    # they never leave the infra (the same data-boundary guarantee as the
    # security proxy, applied to indexing).
    "embedding": FamilyContract(
        name="embedding",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,  # unused on the embeddings path
        response_format_style="none",
        prompt_profile="strict_extraction_v1",
        llama_server_args=("--embedding", "--pooling", "mean"),
        embedding=True,
        ollama_faithful=True,
    ),
    "embeddinggemma2": FamilyContract(
        name="embeddinggemma2",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,  # unused for embeddings
        response_format_style="none",
        prompt_profile="strict_extraction_v1",
        embedding=True,
        dense_embedding=True,
        ollama_faithful=False,
    ),
    # Rerankers / late-interaction retrievers (LFM2.5-ColBERT-350M, or any GGUF
    # reranker). llama-server exposes /rerank (aliases /v1/rerank,
    # /v1/reranking) when launched with --reranking; llama.cpp additionally
    # requires --embedding --pooling rank alongside it (--reranking alone is not
    # enough — verified against the server's own --help and README). No
    # template, no schema — the family exists to type the deployment and carry
    # the launch flags, exactly like the embedding family above. Ollama has no
    # equivalent rerank pooling mode, so this is never Ollama-faithful.
    "reranker": FamilyContract(
        name="reranker",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,  # unused on the rerank path
        response_format_style="none",
        prompt_profile="strict_extraction_v1",
        llama_server_args=("--reranking", "--embedding", "--pooling", "rank"),
        reranker=True,
        ollama_faithful=False,
    ),
    # Multi-vector / late-interaction retrievers (ColBERT, PyLate checkpoints:
    # mixedbread-ai/mxbai-edge-colbert-*, lightonai/LateOn, colbert-ir/
    # colbertv2.0, ...) served by the MULTI-VECTOR runtime -- sentence-
    # transformers >= 6's MultiVectorEncoder (encode_query / encode_document /
    # similarity, MaxSim by default) -- NOT llama.cpp. Same /v1/rerank surface
    # as ``reranker`` above, but a safetensors SNAPSHOT (multi-file directory,
    # loaded via from_pretrained), not a GGUF: the two families exist precisely
    # because the same task ships in two incompatible artifact shapes. No
    # template/schema (a retriever, not an extractor); no llama-server args.
    "multi_vector": FamilyContract(
        name="multi_vector",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,  # unused on the rerank path
        response_format_style="none",
        prompt_profile="strict_extraction_v1",
        multi_vector=True,
        ollama_faithful=False,
    ),
    # Encoder analyzers (safetensors checkpoints served by the encoder runtime,
    # never llama.cpp): GLiNER zero-shot NER (urchade/gliner_*) and GLiNER2
    # guardrails (fastino/GLiNER2-*, PII + safety moderation in one checkpoint).
    # template_delivery/response_format are irrelevant on the encoder path; the
    # family exists to type the deployment and to carry the analyzer backend so
    # a store deploy needs no name-based guessing.
    "encoder_gliner": FamilyContract(
        name="encoder_gliner",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        response_format_style="none",
        prompt_profile="strict_extraction_v1",
        analyzer=True,
        encoder_backend="gliner",
        ollama_faithful=False,
    ),
    # GLiFormer: the analyzer surface of encoder_gliner plus a `structure` head
    # (see encoders/server.py). Served by the same encoder runtime.
    "encoder_gliformer": FamilyContract(
        name="encoder_gliformer",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        # Not "none": the schema on the request IS how this family extracts.
        # Without it the server has nothing to structure into and falls back to
        # entity recognition.
        response_format_style="openai_json_schema",
        # The schema travels in response_format, so the prompt carries the
        # document and nothing else — the same shape NuExtract3 uses. The
        # generic extraction prompt is instructions this model cannot follow and
        # spans it can mistake for the document.
        prompt_profile="document_only",
        analyzer=True,
        encoder_backend="gliformer",
        structured_extraction=True,
        ollama_faithful=False,
    ),
    # GLiClass: zero-shot CLASSIFIER, not an extractor. It scores candidate
    # labels for a document and locates nothing, so it has no structured
    # extraction head and no span output -- the encoder shim answers it on its
    # own branch. Pair it with an extractor family when a pipeline needs both
    # "what kind of document is this" and "what is in it".
    "encoder_gliclass": FamilyContract(
        name="encoder_gliclass",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        response_format_style="none",
        prompt_profile="document_only",
        analyzer=True,
        encoder_backend="gliclass",
        ollama_faithful=False,
    ),
    "encoder_gliner2": FamilyContract(
        name="encoder_gliner2",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        response_format_style="none",
        prompt_profile="strict_extraction_v1",
        analyzer=True,
        encoder_backend="gliner2",
        ollama_faithful=False,
    ),
    # Transformers / AutoModel LAST-RESORT family. A safetensors snapshot served
    # by the TRANSFORMERS runtime when no GGUF exists (or llama.cpp can't load
    # the arch). Chat + vision are handled generically by the repo's own chat
    # template — no per-model contract — at the cost of ~2-3x a GGUF's RAM and
    # slower CPU inference (surfaced as a memory warning at inspect/deploy).
    # template_delivery/response_format ride the standard OpenAI chat surface
    # the shim exposes; a schema (when sent) goes via response_format.
    "transformers": FamilyContract(
        name="transformers",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        # The shim free-generates (ignores response_format); rely on the prompt
        # for JSON. See ENGINE_DEFAULT_STYLE["transformers"] = "none".
        response_format_style="none",
        prompt_profile="strict_extraction_v1",
        transformers_runtime=True,
        trust_remote_code=False,
        vision=True,  # the shim auto-detects vision; the surface allows images
        default_max_tokens=1024,
        default_timeout_seconds=600.0,  # unquantized CPU inference is slow
        ollama_faithful=False,
    ),
    # Same runtime, opted into executing the repo's custom Python — required for
    # custom-code checkpoints (config.json auto_map, e.g. UnlimitedOCR/DeepSeek-
    # OCR). SECURITY: arbitrary code on the serving node, hence a DISTINCT family
    # so the trust is an explicit, auditable choice at deploy, never a default.
    "transformers_trust_remote_code": FamilyContract(
        name="transformers_trust_remote_code",
        template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,
        # The shim free-generates (ignores response_format); rely on the prompt
        # for JSON. See ENGINE_DEFAULT_STYLE["transformers"] = "none".
        response_format_style="none",
        prompt_profile="strict_extraction_v1",
        transformers_runtime=True,
        trust_remote_code=True,
        vision=True,
        default_max_tokens=1024,
        default_timeout_seconds=600.0,
        ollama_faithful=False,
    ),
}


def get_family(name: str) -> FamilyContract:
    try:
        return FAMILIES[name]
    except KeyError:
        known = ", ".join(sorted(FAMILIES))
        raise ValueError(f"Unknown model family {name!r}. Known families: {known}") from None


class ModelStoreError(RuntimeError):
    pass


# A store name is a directory under the store root, the ``model_store_entry``
# primary key, and the id a request names as ``store:<name>``. It arrives from
# an API body, so it is RESTRICTED rather than sanitized -- the same reasoning
# as ``node_identity.validate_node_id``: quietly rewriting a caller's name
# produces an entry they cannot address, while refusing it says what is wrong.
_MAX_STORE_NAME_LEN = 200  # ModelStoreEntry.name is String(200)
_STORE_NAME_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def validate_store_name(name: str) -> str:
    """Return ``name`` if it is usable as a store key and a directory, else raise.

    The pattern admits every name the system generates --
    ``hf_hub.default_store_name`` already emits ``[a-z0-9._-]`` with the ends
    stripped -- and excludes the separators that make a name a path: ``/``,
    ``\\`` and a leading dot, so ``..`` and ``../elsewhere`` cannot survive it.

    Callers that join a name into a filesystem path should ALSO contain the
    result (see ``_assert_within``). This is the earlier, cheaper gate: it
    rejects at the boundary, where the error can name the field, instead of
    deep inside a job that has already downloaded several gigabytes.
    """
    if not name or len(name) > _MAX_STORE_NAME_LEN:
        raise ModelStoreError(
            f"store name {name!r} must be 1-{_MAX_STORE_NAME_LEN} characters"
        )
    if not _STORE_NAME_PATTERN.match(name):
        raise ModelStoreError(
            f"store name {name!r} must start alphanumeric and contain only "
            f"letters, digits, '.', '_' or '-' (it becomes a directory and a "
            f"catalog key)"
        )
    return name


@dataclass(frozen=True)
class StoreEntry:
    name: str
    family: str
    model_path: Path
    mmproj_path: Path | None = None
    source: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "model_path": self.model_path.as_posix(),
            "mmproj_path": self.mmproj_path.as_posix() if self.mmproj_path else None,
            "source": self.source,
        }


def default_ollama_home() -> Path:
    """Resolve Ollama's models directory ($OLLAMA_MODELS or ~/.ollama/models)."""
    override = os.environ.get("OLLAMA_MODELS")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ollama" / "models"


def _ollama_manifest_path(ollama_home: Path, reference: str) -> Path:
    """Map an Ollama model reference to its on-disk manifest path.

    Examples:
        ``hf.co/numind/NuExtract3-GGUF:Q4_K_M``
            -> manifests/hf.co/numind/NuExtract3-GGUF/Q4_K_M
        ``nuextract:3.8b``
            -> manifests/registry.ollama.ai/library/nuextract/3.8b
    """
    name_part, _, tag = reference.partition(":")
    tag = tag or "latest"
    parts = [segment for segment in name_part.split("/") if segment]
    if not parts:
        raise ModelStoreError(f"Invalid Ollama reference: {reference!r}")
    if len(parts) == 1:  # bare library model, e.g. "nuextract"
        parts = ["registry.ollama.ai", "library", *parts]
    elif "." not in parts[0]:  # "namespace/model" without an explicit registry host
        parts = ["registry.ollama.ai", *parts]
    return ollama_home.joinpath("manifests", *parts, tag)


def _blob_path(ollama_home: Path, digest: str) -> Path:
    return ollama_home / "blobs" / digest.replace(":", "-")


@contextmanager
def _as_store_error() -> Iterator[None]:
    """Re-raise a backend failure as a ``ModelStoreError`` (#595 stage 1).

    The storage seam must not change what a store failure looks like from the
    outside: callers (and tests) have always seen ``ModelStoreError`` with the
    transfer's own message, so a ``StorageBackendError`` is translated rather
    than propagated. The message is preserved verbatim -- it names the model
    and the digests, which is the whole diagnostic value.

    A context manager rather than a try/except at each call site so the
    surrounding rollback blocks keep working unchanged: ``ModelStoreError`` is
    still an exception their ``except BaseException`` sees.
    """
    try:
        yield
    except StorageBackendError as exc:
        raise ModelStoreError(str(exc)) from exc


def _assert_within(path: Path, root: Path, *, label: str) -> Path:
    """Resolve ``path`` and require it to stay within ``root`` (block traversal).

    Both the Ollama ``reference`` (-> manifest path) and the store ``name`` (->
    destination dir) are attacker-influenced strings joined into filesystem
    paths. Legitimate references legitimately contain '/' and ':' (e.g.
    ``hf.co/numind/NuExtract3-GGUF:Q4_K_M``) and resolve *inside* the root, so we
    can't blanket-reject separators — instead resolve and enforce containment,
    which rejects only ``..``/absolute escapes.
    """
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ModelStoreError(
            f"Refusing {label}: resolved path escapes {root_resolved} "
            f"(possible path traversal)"
        )
    return resolved


class ModelStore:
    """A canonical on-disk GGUF store served by both llama.cpp and Ollama."""

    def __init__(self, root: str | Path, *, backend: StorageBackend | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        # Cross-PROCESS, because the two writers are two processes: a node's
        # seed job and an operator running `store add-gguf` / `store publish`,
        # which the fleet runbook asks for. Unlike the supervisor's state file
        # there is no single-writer contract to lean on. Singleton so the
        # nested acquire inside the read-modify-write reuses this instance's
        # counter instead of blocking on itself.
        self._lock = FileLock(self.root / INDEX_LOCK_NAME, timeout=60, is_singleton=True)
        # Hardcoded in stage 1 of #595: the seam exists, the choice does not.
        # The parameter is here so a test can inject a double and so stage 2
        # can wire a remote backend without touching every construction site.
        self._backend: StorageBackend = (
            backend if backend is not None else LocalDiskBackend(self.root)
        )

    # ------------------------------------------------------------------ seeding
    def seed_from_ollama(
        self,
        reference: str,
        *,
        name: str,
        family: str,
        ollama_home: Path | None = None,
        link: bool = True,
        mmproj_source: str | Path | None = None,
    ) -> StoreEntry:
        """Register a model already pulled by Ollama, without re-downloading.

        Reads Ollama's manifest for ``reference`` and hard-links (or copies) its
        weights — and vision projector, if present — into the store under stable
        ``*.gguf`` names.

        ``mmproj_source`` supplies a vision projector explicitly (an on-disk GGUF,
        e.g. one downloaded separately from Hugging Face). It takes precedence over
        the projector embedded in the Ollama manifest and is the ONLY way to make a
        ``needs_mmproj`` family (NuExtract3) deployable when the pulled GGUF ships
        no projector layer — the exact gap that otherwise leaves vision families
        un-servable via the seed path.
        """
        if not reference.strip() or not name.strip():
            raise ModelStoreError("seed requires a non-empty reference and name")
        contract = get_family(family)
        home = ollama_home or default_ollama_home()
        manifest_path = _ollama_manifest_path(home, reference)
        # Containment: a crafted reference ("../../..") must not read manifests
        # outside Ollama's manifests root, and a crafted store name must not write
        # blobs outside the store root.
        _assert_within(manifest_path, home / "manifests", label=f"Ollama reference {reference!r}")
        _assert_within(self.root / name, self.root, label=f"store name {name!r}")
        if not manifest_path.is_file():
            raise ModelStoreError(
                f"Ollama manifest not found for {reference!r} at {manifest_path}. "
                f"Pull it first (e.g. `ollama pull {reference}`)."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        layers = manifest.get("layers", [])
        model_digest = _select_layer(layers, "model")
        if model_digest is None:
            raise ModelStoreError(f"No model layer found in Ollama manifest for {reference!r}")
        projector_digest = _select_layer(layers, "projector")

        model_blob = _blob_path(home, model_digest)
        if not model_blob.is_file():
            raise ModelStoreError(f"Ollama model blob is missing: {model_blob}")

        # Resolve the vision projector: an explicit ``mmproj_source`` wins over the
        # manifest's projector layer (some GGUF pulls ship none). ``.is_file()``
        # rather than a mere digest presence check so a dangling projector digest
        # cannot masquerade as a usable projector for a ``needs_mmproj`` family.
        explicit_mmproj: Path | None = None
        if mmproj_source is not None:
            explicit_mmproj = Path(mmproj_source)
            if not explicit_mmproj.is_file():
                raise ModelStoreError(f"mmproj not found: {explicit_mmproj}")
        manifest_mmproj = _blob_path(home, projector_digest) if projector_digest else None
        if manifest_mmproj is not None and not manifest_mmproj.is_file():
            manifest_mmproj = None
        mmproj_blob = explicit_mmproj or manifest_mmproj
        if contract.needs_mmproj and mmproj_blob is None:
            raise ModelStoreError(
                f"Family {family!r} requires a vision projector but {reference!r} has none. "
                f"Pass mmproj_source=<projector.gguf> (or pull a GGUF that includes one)."
            )

        # An explicit mmproj_source is an HF download with no manifest digest, so
        # only copy-fidelity is checkable there; the manifest projector layer has
        # its canonical digest. Verify BOTH blobs before writing the index so a
        # canonical model.gguf / index entry only ever appears fully verified.
        mmproj_digest = None if explicit_mmproj is not None else projector_digest
        destination = self.root / name
        model_path = self._backend.resolve_local_path(f"{name}/model.gguf")
        mmproj_path: Path | None = None
        try:
            with _as_store_error():
                self._backend.write_verified(
                    model_blob,
                    f"{name}/model.gguf",
                    link=link,
                    expected_digest=model_digest,
                )
            if mmproj_blob is not None:
                mmproj_path = self._backend.resolve_local_path(f"{name}/mmproj.gguf")
                with _as_store_error():
                    self._backend.write_verified(
                        mmproj_blob,
                        f"{name}/mmproj.gguf",
                        link=link,
                        expected_digest=mmproj_digest,
                    )
        except BaseException:
            # A verify failure on either blob must leave zero partial state (no
            # stray model.gguf when mmproj fails). Bounded by ``name``.
            shutil.rmtree(destination, ignore_errors=True)
            raise

        entry = StoreEntry(
            name=name,
            family=family,
            model_path=model_path,
            mmproj_path=mmproj_path,
            source=f"ollama:{reference}",
        )
        self._write_entry(entry)
        return entry

    def add_gguf(
        self,
        *,
        name: str,
        family: str,
        model_gguf: str | Path,
        mmproj: str | Path | None = None,
        source: str | None = None,
        link: bool = True,
    ) -> StoreEntry:
        """Register a GGUF already on disk (e.g. downloaded from Hugging Face)."""
        contract = get_family(family)
        model_gguf = Path(model_gguf)
        if not model_gguf.is_file():
            raise ModelStoreError(f"GGUF not found: {model_gguf}")
        if contract.needs_mmproj and mmproj is None:
            raise ModelStoreError(f"Family {family!r} requires an mmproj (vision projector)")
        # The store NAME is attacker-influenced and joined into a filesystem
        # path, exactly as in seed_from_ollama / add_snapshot / remove_entry --
        # and unlike those three this path had no guard, so a name of
        # ``../elsewhere`` wrote outside the store root. The storage backend
        # refuses an escaping key too (defence in depth), but the refusal
        # belongs here so it reads as a store error like its siblings' do.
        _assert_within(self.root / name, self.root, label=f"store name {name!r}")

        # HF/on-disk GGUFs carry no canonical manifest digest, so only
        # copy-fidelity is verifiable (expected_digest=None) — best-effort, per
        # the HF trust boundary. Still routed through the backend's verified
        # write so no path publishes the canonical name un-verified.
        destination = self.root / name
        model_path = self._backend.resolve_local_path(f"{name}/model.gguf")
        mmproj_path: Path | None = None
        try:
            with _as_store_error():
                self._backend.write_verified(
                    model_gguf, f"{name}/model.gguf", link=link, expected_digest=None
                )
            if mmproj is not None:
                mmproj_path = self._backend.resolve_local_path(f"{name}/mmproj.gguf")
                with _as_store_error():
                    self._backend.write_verified(
                        Path(mmproj),
                        f"{name}/mmproj.gguf",
                        link=link,
                        expected_digest=None,
                    )
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        entry = StoreEntry(
            name=name,
            family=family,
            model_path=model_path,
            mmproj_path=mmproj_path,
            source=source,
        )
        self._write_entry(entry)
        return entry

    def add_snapshot(
        self,
        *,
        name: str,
        family: str,
        snapshot_dir: str | Path,
        source: str | None = None,
        link: bool = True,
    ) -> StoreEntry:
        """Register a safetensors checkpoint DIRECTORY (weights + config +
        tokenizer/processor) as a store entry.

        Every ``FamilyContract.snapshot`` family shares this path: analyzer
        (GLiNER/GLiNER2) checkpoints served by the encoder runtime, the
        LAST-RESORT transformers/AutoModel path, and multi-vector (ColBERT)
        retrievers served by the multi-vector runtime. All are multi-file
        safetensors, not single GGUFs, so ``model_path`` points at the snapshot
        DIRECTORY (``<root>/<name>/snapshot``) loaded via
        ``from_pretrained(<path>)`` — no network at deploy time. The whole tree
        is transferred into a ``.tmp`` sibling and renamed atomically, so a
        half-copied snapshot never appears under the canonical name.
        """
        contract = get_family(family)
        if not contract.snapshot:
            raise ModelStoreError(
                f"add_snapshot is for snapshot families (analyzer / transformers / "
                f"multi_vector / dense_embedding); {family!r} is not one"
            )
        src = Path(snapshot_dir)
        if not src.is_dir():
            raise ModelStoreError(f"snapshot directory not found: {src}")
        # Safetensors OR PyTorch weights: an encoder checkpoint that ships only
        # pytorch_model.bin loads the same way through ``from_pretrained``.
        if not any(
            p.suffix in _SNAPSHOT_WEIGHT_SUFFIXES for p in src.rglob("*") if p.is_file()
        ):
            raise ModelStoreError(
                f"snapshot {src} has no weights — not a servable checkpoint"
            )
        _assert_within(self.root / name, self.root, label=f"store name {name!r}")

        destination = self.root / name
        snapshot_path = self._backend.resolve_local_dir(f"{name}/snapshot")
        try:
            with _as_store_error():
                self._backend.write_tree(src, f"{name}/snapshot", link=link)
        except BaseException:
            # The backend cleans up its own staging; this rollback is store
            # policy it cannot see. A fresh seed with no prior index entry
            # leaves an orphan dir, and the index write below is what makes it
            # canonical, so drop it on failure.
            if name not in self._read_index():
                shutil.rmtree(destination, ignore_errors=True)
            raise

        entry = StoreEntry(name=name, family=family, model_path=snapshot_path, source=source)
        self._write_entry(entry)
        return entry

    # ------------------------------------------------------------------- query
    def entry(self, name: str) -> StoreEntry:
        index = self._read_index()
        if name not in index:
            raise ModelStoreError(f"Unknown model {name!r} in store {self.root}")
        return _entry_from_json(index[name])

    def list(self) -> list[StoreEntry]:
        return [_entry_from_json(value) for _, value in sorted(self._read_index().items())]

    # ----------------------------------------------------------------- serving
    def family_launch_args(self, name: str) -> tuple[str, ...]:
        """Family-specific llama-server flags for ``name`` (e.g. ``--jinja``, ``--mmproj``).

        This is the single source of truth for the template/vision flags a family
        needs; both ``llama_server_command`` and the background serving bridge
        (``docie up``) derive their invocation from it.
        """
        entry = self.entry(name)
        contract = get_family(entry.family)
        extra = list(contract.llama_server_args)
        if contract.needs_mmproj and entry.mmproj_path is not None:
            extra.extend(["--mmproj", entry.mmproj_path.as_posix()])
        return tuple(extra)

    def llama_server_command(
        self,
        name: str,
        *,
        executable: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 8088,
        context_length: int = DEFAULT_DEPLOY_CONTEXT_LENGTH,
    ) -> tuple[str, ...]:
        """Build the llama-server command that serves ``name`` per its family contract.

        Includes the family's required flags (e.g. ``--jinja``) and ``--mmproj``
        for vision families — exactly what the raw ``llama-server`` invocation in
        the docs needs, sourced from the canonical store.
        """
        entry = self.entry(name)
        command = [
            executable,
            "--model",
            entry.model_path.as_posix(),
            "--alias",
            name,
            "--host",
            host,
            "--port",
            str(port),
            "--ctx-size",
            str(context_length),
        ]
        command.extend(self.family_launch_args(name))
        return tuple(command)

    def ollama_modelfile(self, name: str) -> str:
        """Generate an Ollama Modelfile so Ollama can serve ``name`` from the store.

        Raises for families Ollama cannot serve faithfully (those whose template is
        delivered via ``chat_template_kwargs``), because the resulting deployment
        would silently ignore the template and produce wrong extractions.
        """
        entry = self.entry(name)
        contract = get_family(entry.family)
        if not contract.ollama_faithful:
            raise ModelStoreError(
                f"Family {entry.family!r} cannot be served faithfully by Ollama "
                f"(template is delivered via {contract.template_delivery}; Ollama drops it). "
                f"Serve {name!r} with `llama-server` instead — see "
                f"`ModelStore.llama_server_command`."
            )
        lines = [f'FROM {entry.model_path.as_posix()}']
        if entry.mmproj_path is not None:
            lines.append(f'ADAPTER {entry.mmproj_path.as_posix()}')
        for stop in contract.stop_sequences:
            lines.append(f'PARAMETER stop "{stop}"')
        lines.append(f"PARAMETER temperature {contract.default_temperature}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ private
    @contextmanager
    def _index_lock(self) -> Iterator[None]:
        try:
            with self._lock:
                yield
        except Timeout as exc:
            raise ModelStoreError(
                f"Timed out waiting for the store index lock "
                f"{self.root / INDEX_LOCK_NAME}. Another seed or `store` command "
                f"is holding it; wait for it to finish or stop it."
            ) from exc

    def _read_index(self) -> dict[str, Any]:
        with self._index_lock():
            if not self._index_path.is_file():
                return {}
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                # Raise, never fall back to {}. The index is the only record
                # that root/<name> holds a usable model, so reading a damaged
                # one as "this node holds nothing" would re-download every
                # model while the weights sit on disk unreferenced.
                raise ModelStoreError(
                    f"Corrupt store index: {self._index_path} ({exc}). The model "
                    f"directories beside it are untouched -- move the index aside "
                    f"and re-register them, or restore it from a backup."
                ) from exc
            if not isinstance(data, dict):
                raise ModelStoreError(f"Corrupt store index: {self._index_path}")
            return data

    def _write_entry(self, entry: StoreEntry) -> None:
        with self._index_lock():
            index = self._read_index()
            # A NEW name: the blob transfer just created root/name and this is the
            # first index reference to it. If the index write then crashes, that dir
            # is an orphan (no index key) -> clean it up so no partial state remains.
            # An EXISTING name (re-seed): the on-disk index already references
            # root/name; _write_index is atomic (temp + replace), so a failure leaves
            # the OLD index entry intact and still pointing at root/name. Deleting the
            # dir there would create a dangling reference — exactly the state this
            # store avoids — so only orphan-clean the fresh-seed case.
            is_new = entry.name not in index
            index[entry.name] = entry.to_json()
            try:
                self._write_index(index)
            except BaseException:
                if is_new:
                    destination = _assert_within(
                        self.root / entry.name, self.root, label=f"store name {entry.name!r}"
                    )
                    shutil.rmtree(destination, ignore_errors=True)
                raise

    def _write_index(self, index: dict[str, Any]) -> None:
        """Stage, fsync, rename -- as ``PersistentSupervisor._save`` does.

        ``os.replace`` is atomic with respect to the name, not the contents, so
        without the fsync a power loss can land the rename while the payload is
        still buffered. The per-write temp name keeps a second writer from
        staging onto the same path if it ever runs without the lock.
        """
        with self._index_lock():
            payload = json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            temporary = self._index_path.with_name(
                f".{self._index_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
            )
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._index_path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise

    def remove_entry(self, name: str) -> None:
        """Remove store entry ``name``: drop its index key, then delete its dir.

        Compensation for a seed whose catalog upsert failed, making the seed
        all-or-nothing. Ordering matters: rewrite the index WITHOUT the key first
        (atomic temp+replace) so a failure can never leave the index pointing at
        deleted files — an unreferenced directory is harmless, a dangling index
        reference is exactly the error-producing partial state this PR removes.
        Bounded by ``name`` (containment-checked) so it can only ever touch
        ``root/name`` and that one index key, never a pre-existing unrelated entry.
        """
        destination = _assert_within(self.root / name, self.root, label=f"store name {name!r}")
        with self._index_lock():
            index = self._read_index()
            if name in index:
                del index[name]
                self._write_index(index)
        shutil.rmtree(destination, ignore_errors=True)


def _select_layer(layers: list[dict[str, Any]], needle: str) -> str | None:
    for layer in layers:
        media_type = str(layer.get("mediaType", ""))
        if needle in media_type and layer.get("digest"):
            return str(layer["digest"])
    return None


def _entry_from_json(value: dict[str, Any]) -> StoreEntry:
    mmproj = value.get("mmproj_path")
    return StoreEntry(
        name=value["name"],
        family=value["family"],
        model_path=Path(value["model_path"]),
        mmproj_path=Path(mmproj) if mmproj else None,
        source=value.get("source"),
    )
