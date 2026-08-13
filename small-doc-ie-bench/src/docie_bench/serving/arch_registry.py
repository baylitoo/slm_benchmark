"""Architecture → family registry: pre-flight support detection.

The maintained mapping that replaces try-and-see onboarding. A model's
architecture (read from GGUF header metadata or ``config.json``, see
``hf_hub.inspect_repo``) resolves to a serving *family* — the contract that
tells the runtime how to serve it. This is NOT ``models.yaml`` (deployment
routing); it is the "do we support this architecture, and with which family"
question, answered from the repo's files without downloading the weights.

Granularity of the verdict:
  * ``supported``   — the architecture maps to a family; deploy now.
  * ``needs_family``— the architecture is detected but has no family contract
                      yet (a small PR to add one; whether llama.cpp can serve
                      it is verified at deploy, a separate runtime gate).
  * ``unsupported`` — nothing servable here (e.g. a non-GGUF, non-encoder repo).
"""

from __future__ import annotations

from dataclasses import dataclass

# Architecture (lowercase, as reported by GGUF `general.architecture` or a
# config's `model_type`) -> family name. Multimodal archs map to a vision
# family; the extraction-vs-OCR nuance among vision families is a user choice
# (both are offered), so this is a sensible DEFAULT, never a lock-in.
ARCH_TO_FAMILY: dict[str, str] = {
    # text chat / extraction
    "llama": "openai_chat",
    "qwen2": "openai_chat",
    "qwen2moe": "openai_chat",
    "qwen3": "openai_chat",
    "qwen3moe": "openai_chat",
    # Qwen3.5 backbone: TEXT by default. NuExtract3 shares this exact arch but
    # needs a different serving contract (template via chat_template_kwargs) — it
    # is disambiguated BY NAME in resolve_family, not by arch. A generic Qwen3.5
    # text model lands here (openai_chat); a generic Qwen3.5-VL gets upgraded to
    # a vision family by the mmproj signal below.
    "qwen35": "openai_chat",
    "mistral": "openai_chat",
    "mixtral": "openai_chat",
    "gemma": "openai_chat",
    "gemma2": "openai_chat",
    "gemma3": "openai_chat",
    "phi2": "openai_chat",
    "phi3": "openai_chat",
    "stablelm": "openai_chat",
    "lfm2": "lfm2",
    # vision (need an mmproj projector)
    "qwen2vl": "lfm2_vl",
    "qwen2_vl": "lfm2_vl",
    "qwen2.5vl": "lfm2_vl",
    "qwen3vl": "lfm2_vl",
    "qwen3_vl": "lfm2_vl",
    "lfm2-vl": "lfm2_vl",
    "lfm2_vl": "lfm2_vl",
    # OCR vision. Unlimited-OCR IS the DeepSeek-OCR architecture (SAM-ViT-B +
    # CLIP-L/14 vision tower → projector → DeepSeek-V2 MoE decoder). Its GGUF
    # `general.architecture` is "unlimited-ocr" (llama.cpp mtmd support merged
    # in ggml-org/llama.cpp#24969, 2026-06-24 — needs a llama-server built
    # after that). The deepseek-ocr lineage may report "deepseek-ocr" /
    # "deepseek2-ocr"; all map to the free-text OCR family.
    "unlimited-ocr": "vision_ocr",
    "deepseek-ocr": "vision_ocr",
    "deepseek2-ocr": "vision_ocr",
    # embeddings
    "bert": "embedding",
    "nomic-bert": "embedding",
    "xlm-roberta": "embedding",
    "roberta": "embedding",
}

# Non-exhaustive vision archs (used only to sanity-flag a vision/mmproj mismatch).
VISION_ARCHS = frozenset(
    {
        "qwen2vl",
        "qwen2_vl",
        "qwen2.5vl",
        "qwen3vl",
        "qwen3_vl",
        "lfm2-vl",
        "lfm2_vl",
        "unlimited-ocr",
        "deepseek-ocr",
        "deepseek2-ocr",
    }
)

# Families that are multimodal — a repo whose family resolves to one of these
# is already vision, no upgrade needed.
VISION_FAMILIES = frozenset({"lfm2_vl", "nuextract3", "vision_ocr"})

# When a repo ships an mmproj it is MULTIMODAL regardless of the reported base
# LM architecture: llama.cpp reports the backbone (e.g. "lfm2" for LFM2-VL,
# "qwen2" for a Qwen2-VL). The projector is a stronger modality signal than the
# arch string, so a text-family suggestion is upgraded to a vision family —
# per-backbone when known, else a sensible default (vision extraction; the user
# can switch to vision_ocr for OCR).
TEXT_ARCH_TO_VISION_FAMILY: dict[str, str] = {
    "lfm2": "lfm2_vl",
    "qwen2": "lfm2_vl",
    "qwen3": "lfm2_vl",
    "qwen35": "lfm2_vl",  # generic Qwen3.5-VL (a NuExtract3 is caught by name first)
    "llama": "lfm2_vl",
}
DEFAULT_VISION_FAMILY = "lfm2_vl"

# Archs whose SERVING support landed in llama.cpp only recently: the family
# contract exists, but the node's llama-server must be new enough to load the
# arch. This is a SEPARATE gate from "we have a family" — surfaced as a note so
# a "supported" verdict is honest ("rebuild the serving image if it won't
# load"). Keyed by arch → the note shown to the operator.
RUNTIME_NOTES: dict[str, str] = {
    "unlimited-ocr": (
        "serving needs a llama-server built after ggml-org/llama.cpp#24969 "
        "(2026-06-24) — rebuild the serving image if the deploy fails to load"
    ),
    "deepseek-ocr": (
        "serving needs a recent llama-server (deepseek-ocr mtmd support) — "
        "rebuild the serving image if the deploy fails to load"
    ),
    "deepseek2-ocr": (
        "serving needs a recent llama-server (deepseek-ocr mtmd support) — "
        "rebuild the serving image if the deploy fails to load"
    ),
}


# The LAST-RESORT transformers/AutoModel families (arch-agnostic: served from
# the repo's own chat template, no per-model contract). `transformers` is the
# safe default; `transformers_trust_remote_code` is the opt-in for custom-code
# checkpoints (config.json auto_map) that execute repo Python on the node.
TRANSFORMERS_FAMILY = "transformers"

# The memory disclaimer surfaced on any transformers verdict — carried on the
# verdict's ``runtime_note`` so the UI renders it in the SAME amber caveat as a
# runtime gate. States the preferred path plainly (the whole point of making
# this a last resort).
TRANSFORMERS_MEMORY_NOTE = (
    "served via transformers (AutoModel) — unquantized weights use ~2-3x the "
    "RAM of a GGUF Q4 and CPU inference is slower. Prefer a GGUF repo of this "
    "model if one exists; deploy here only as a last resort."
)


@dataclass(frozen=True)
class SupportVerdict:
    """The outcome of resolving an architecture to a family."""

    verdict: str  # "supported" | "needs_family" | "unsupported"
    family: str | None
    reason: str
    # A caveat about RUNTIME (not family) support — e.g. "needs a recent
    # llama-server". None when the runtime support is long-standing.
    runtime_note: str | None = None


def _is_reranker(repo_id: str | None) -> bool:
    """Rerankers / late-interaction retrievers are repurposed backbones —
    LFM2.5-ColBERT-350M reports arch ``lfm2`` (identical to the chat family),
    a BERT cross-encoder reports ``bert`` (identical to the embedding family).
    The arch alone can't tell a reranker apart from a chat/embedding model on
    the same backbone, so — same pattern as NuExtract3 — it is caught by name
    before the generic arch lookup runs."""
    if not repo_id:
        return False
    name = repo_id.lower()
    return any(kw in name for kw in ("colbert", "rerank", "cross-encoder", "crossencoder"))


def _is_nuextract3(arch: str, repo_id: str | None) -> bool:
    """NuExtract3 shares the qwen3.5-VL backbone (arch ``qwen35``) with generic
    Qwen3.5 text/VL models but needs a DIFFERENT serving contract (the extraction
    template rides ``chat_template_kwargs``, not ``response_format``). The arch
    alone can't tell them apart, so the flagship is detected by name; an
    unmatched ``qwen35`` falls through to the generic text/vision handling."""
    return arch == "qwen35" and bool(repo_id) and "nuextract" in repo_id.lower()


def resolve_family(
    architecture: str | None,
    *,
    has_gguf: bool,
    has_safetensors: bool,
    has_mmproj: bool,
    repo_id: str | None = None,
) -> SupportVerdict:
    """Map a detected architecture (+ repo shape) to a family and a verdict.

    The GGUF/llama.cpp path is always preferred. Only when the repo ships NO
    servable GGUF (safetensors-only) does the LAST-RESORT transformers family
    apply — the "no servable GGUF" hard gate (see ``_transformers_verdict``).

    ``repo_id`` disambiguates archs shared by models with different serving
    contracts (NuExtract3 vs a generic Qwen3.5-VL, both ``qwen35``).
    """
    arch = architecture.strip().lower() if architecture else ""
    runtime_note = RUNTIME_NOTES.get(arch)

    # Encoder analyzers are detected by the gliner marker, not a base arch
    # (GLiNER2's base is mdeberta but that is NOT what we serve it as). These
    # ARE safetensors-only, so this must precede the transformers gate below.
    if "gliner" in arch:
        family = "encoder_gliner2" if "2" in arch else "encoder_gliner"
        return SupportVerdict("supported", family, f"analyzer architecture {architecture!r}")

    if not architecture:
        if has_gguf:
            # A GGUF with no readable arch is still deployable as a plain chat
            # model — the family default the seed form already offers.
            return SupportVerdict(
                "needs_family",
                "openai_chat",
                "no architecture metadata found; defaulting to a plain chat family — "
                "confirm the family before deploying",
            )
        if has_safetensors:
            # No GGUF, no arch, but weights present — the transformers last
            # resort can still serve it from the repo's own chat template.
            return _transformers_verdict(architecture=None, has_mmproj=has_mmproj)
        return SupportVerdict(
            "unsupported",
            None,
            "no GGUF and no readable architecture — not a servable repo",
        )

    # No servable GGUF in THIS repo: even a llama.cpp-supported arch cannot be
    # served here via llama-server (it needs a .gguf file). If weights are
    # present, redirect to the transformers last resort (with the memory note
    # nudging the operator to a GGUF repo); otherwise it is not servable here.
    if not has_gguf:
        if has_safetensors:
            return _transformers_verdict(architecture=arch, has_mmproj=has_mmproj)
        return SupportVerdict(
            "unsupported",
            None,
            f"architecture {architecture!r} detected but the repo ships neither a "
            "GGUF nor safetensors weights — nothing servable here",
        )

    # NuExtract3 (qwen3.5-VL + chat_template_kwargs extraction contract), caught
    # by name BEFORE the generic qwen35 handling — which would otherwise serve it
    # as a plain vision model (lfm2_vl) and silently drop its template. It is
    # vision, so a missing projector is a needs_family, not a text mis-serve.
    if _is_nuextract3(arch, repo_id):
        if not has_mmproj:
            return SupportVerdict(
                "needs_family",
                "nuextract3",
                "NuExtract3 needs a vision projector but this repo ships no mmproj — "
                "pick a repo that includes one",
                runtime_note=runtime_note,
            )
        return SupportVerdict(
            "supported", "nuextract3", f"NuExtract3 ({architecture!r} extraction contract)"
        )

    # Reranker (LFM2.5-ColBERT-350M, a BERT cross-encoder, ...), caught by name
    # BEFORE the generic arch lookup — which would otherwise serve it as a plain
    # chat or embedding model on the shared backbone and silently drop its
    # actual task (query+documents -> relevance scores, not a chat/embed model).
    if _is_reranker(repo_id):
        return SupportVerdict(
            "supported",
            "reranker",
            f"repo name indicates a reranker/cross-encoder ({architecture!r} backbone) "
            "→ reranker",
            runtime_note=runtime_note,
        )

    family = ARCH_TO_FAMILY.get(arch)
    if family is not None:
        # A projector means multimodal even when the arch reports a text
        # backbone (LFM2-VL → "lfm2", Qwen2-VL → "qwen2"): upgrade to a vision
        # family so the deploy gets --mmproj + the right response contract.
        if has_mmproj and family not in VISION_FAMILIES:
            vision_family = TEXT_ARCH_TO_VISION_FAMILY.get(arch, DEFAULT_VISION_FAMILY)
            return SupportVerdict(
                "supported",
                vision_family,
                f"architecture {architecture!r} with an mmproj projector → "
                f"{vision_family} (vision) — switch to vision_ocr for OCR",
                runtime_note=runtime_note,
            )
        # Sanity: a vision-arch family with no projector will fail at load.
        if arch in VISION_ARCHS and not has_mmproj:
            return SupportVerdict(
                "needs_family",
                family,
                f"vision architecture {architecture!r} but the repo ships no mmproj "
                "projector — pick a repo that includes one",
                runtime_note=runtime_note,
            )
        return SupportVerdict(
            "supported",
            family,
            f"architecture {architecture!r} → {family}",
            runtime_note=runtime_note,
        )

    # Unknown arch but a projector present → still multimodal; suggest a vision
    # family rather than a blank needs_family.
    if has_mmproj:
        return SupportVerdict(
            "needs_family",
            DEFAULT_VISION_FAMILY,
            f"architecture {architecture!r} is not recognized but the repo ships an "
            "mmproj projector — try a vision family",
        )
    return SupportVerdict(
        "needs_family",
        None,
        f"architecture {architecture!r} is not recognized — pick a family manually "
        "before deploying",
    )


def _transformers_verdict(*, architecture: str | None, has_mmproj: bool) -> SupportVerdict:
    """The last-resort transformers verdict for a safetensors-only repo.

    ``supported`` — we CAN serve it — but the memory disclaimer rides
    ``runtime_note`` so the preferred (GGUF) path is stated loudly at the
    decision point. Custom-code detection (config.json ``auto_map``) is a
    deploy-time concern; the operator picks ``transformers_trust_remote_code``
    explicitly when the load needs it.
    """
    arch_label = f"architecture {architecture!r}" if architecture else "no readable architecture"
    return SupportVerdict(
        "supported",
        TRANSFORMERS_FAMILY,
        f"{arch_label}: no GGUF in this repo — {TRANSFORMERS_FAMILY} (last resort)",
        runtime_note=TRANSFORMERS_MEMORY_NOTE,
    )
