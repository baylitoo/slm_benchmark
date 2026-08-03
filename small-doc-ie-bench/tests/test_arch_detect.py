"""PR-1: architecture → family detection (registry + HF inspect, no download)."""

from __future__ import annotations

import httpx
import pytest

from docie_bench.serving.arch_registry import ARCH_TO_FAMILY, resolve_family
from docie_bench.serving.hf_hub import inspect_repo, search_models
from docie_bench.serving.model_store import FAMILIES

# ── registry / resolution (pure) ─────────────────────────────────────────────


def test_every_mapped_family_exists() -> None:
    for arch, family in ARCH_TO_FAMILY.items():
        assert family in FAMILIES, f"{arch} maps to unknown family {family}"


def test_resolve_chat_arch_supported() -> None:
    v = resolve_family("qwen2", has_gguf=True, has_safetensors=False, has_mmproj=False)
    assert v.verdict == "supported" and v.family == "openai_chat"


def test_resolve_vision_arch_needs_mmproj() -> None:
    ok = resolve_family("deepseek2-ocr", has_gguf=True, has_safetensors=False, has_mmproj=True)
    assert ok.verdict == "supported" and ok.family == "vision_ocr"
    missing = resolve_family(
        "deepseek2-ocr", has_gguf=True, has_safetensors=False, has_mmproj=False
    )
    assert missing.verdict == "needs_family" and "mmproj" in missing.reason


def test_mmproj_upgrades_text_backbone_to_vision_family() -> None:
    """LFM2-VL/Qwen2-VL GGUFs report the LM backbone arch ("lfm2"/"qwen2") but
    ship an mmproj — the projector is the modality signal, so the suggestion is
    upgraded to a vision family (not the text family)."""
    v = resolve_family("lfm2", has_gguf=True, has_safetensors=False, has_mmproj=True)
    assert v.verdict == "supported" and v.family == "lfm2_vl"
    # Without a projector, the same arch stays the text family.
    t = resolve_family("lfm2", has_gguf=True, has_safetensors=False, has_mmproj=False)
    assert t.family == "lfm2"


def test_unknown_arch_with_mmproj_suggests_vision() -> None:
    v = resolve_family("mystery-vl", has_gguf=True, has_safetensors=False, has_mmproj=True)
    assert v.verdict == "needs_family" and v.family == "lfm2_vl"


def test_resolve_gliner_marker() -> None:
    v = resolve_family("GLiNER2", has_gguf=False, has_safetensors=True, has_mmproj=False)
    assert v.verdict == "supported" and v.family == "encoder_gliner2"


def test_resolve_unknown_arch_needs_family() -> None:
    v = resolve_family("some-brand-new-moe", has_gguf=True, has_safetensors=False, has_mmproj=False)
    assert v.verdict == "needs_family" and v.family is None


def test_resolve_non_servable_repo() -> None:
    v = resolve_family(None, has_gguf=False, has_safetensors=False, has_mmproj=False)
    assert v.verdict == "unsupported"


# ── inspect_repo (HF metadata mocked) ────────────────────────────────────────


def _transport(payload: dict, *, config: dict | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/api/models/"):
            return httpx.Response(200, json=payload)
        if path.endswith("/config.json"):
            return httpx.Response(200, json=config or {})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_inspect_gguf_arch_from_gguf_block() -> None:
    payload = {
        "gguf": {"architecture": "deepseek2-ocr"},
        "siblings": [
            {"rfilename": "Unlimited-OCR-Q4_K_M.gguf", "size": 2_000_000_000},
            {"rfilename": "mmproj-Unlimited-OCR-F16.gguf", "size": 800_000_000},
        ],
    }
    async with httpx.AsyncClient(transport=_transport(payload)) as client:
        result = await inspect_repo("sahilchachra/Unlimited-OCR-GGUF", client=client)
    assert result["architecture"] == "deepseek2-ocr"
    assert result["verdict"] == "supported"
    assert result["family"] == "vision_ocr"
    assert result["has_mmproj"] is True
    assert result["quants"] == ["Q4_K_M"]


async def test_inspect_falls_back_to_config_json() -> None:
    # No gguf block, no config in model-info → raw config.json read.
    payload = {"siblings": [{"rfilename": "model.safetensors"}]}
    config = {"model_type": "gliner2"}
    async with httpx.AsyncClient(transport=_transport(payload, config=config)) as client:
        result = await inspect_repo("fastino/GLiNER2-Guardrails-PII-Multi", client=client)
    assert result["architecture"] == "gliner2"
    assert result["family"] == "encoder_gliner2"
    assert result["has_safetensors"] is True


async def test_inspect_unknown_arch_is_needs_family() -> None:
    payload = {
        "gguf": {"architecture": "exotic-new-arch"},
        "siblings": [{"rfilename": "m-Q4_K_M.gguf", "size": 1}],
    }
    async with httpx.AsyncClient(transport=_transport(payload)) as client:
        result = await inspect_repo("owner/exotic-GGUF", client=client)
    assert result["verdict"] == "needs_family"
    assert result["family"] is None


async def test_search_models_returns_light_cards() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/models"
        assert request.url.params.get("search") == "lfm2"
        assert request.url.params.get("filter") == "gguf"
        return httpx.Response(
            200,
            json=[
                {"id": "LiquidAI/LFM2.5-350M-GGUF", "downloads": 42000, "likes": 85,
                 "tags": ["gguf", "text-generation"]},
                {"modelId": "owner/via-modelId-GGUF", "downloads": 10},  # id fallback
                {"nope": 1},  # dropped (no id)
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cards = await search_models("lfm2", client=client, limit=10)
    ids = [c["id"] for c in cards]
    assert "LiquidAI/LFM2.5-350M-GGUF" in ids
    assert "owner/via-modelId-GGUF" in ids  # HF's modelId key handled
    assert cards[0]["downloads"] == 42000 and cards[0]["likes"] == 85
