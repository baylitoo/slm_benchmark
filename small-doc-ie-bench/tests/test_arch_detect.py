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


def test_resolve_unlimited_ocr_arch() -> None:
    # Unlimited-OCR's GGUF arch is "unlimited-ocr" (DeepSeek-OCR lineage).
    for arch in ("unlimited-ocr", "deepseek-ocr", "deepseek2-ocr"):
        v = resolve_family(arch, has_gguf=True, has_safetensors=False, has_mmproj=True)
        assert v.verdict == "supported" and v.family == "vision_ocr", arch
        # Honest about the separate runtime gate (recent llama-server needed).
        assert v.runtime_note and "llama-server" in v.runtime_note, arch


def test_common_arch_has_no_runtime_note() -> None:
    v = resolve_family("qwen2", has_gguf=True, has_safetensors=False, has_mmproj=False)
    assert v.runtime_note is None


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


def test_resolve_qwen35_is_nuextract3() -> None:
    # NuExtract3's GGUF backbone arch "qwen35" + mmproj → the nuextract3 schema
    # contract (not the generic json_schema vision family that would drop it).
    v = resolve_family("qwen35", has_gguf=True, has_safetensors=False, has_mmproj=True)
    assert v.verdict == "supported"
    assert v.family == "nuextract3"
    # Same arch without a projector fails the vision sanity check, not a silent
    # text mis-serve.
    t = resolve_family("qwen35", has_gguf=True, has_safetensors=False, has_mmproj=False)
    assert t.verdict == "needs_family"
    assert "mmproj" in t.reason


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


async def test_inspect_safetensors_only_falls_to_transformers() -> None:
    # A plain safetensors model with no GGUF → the transformers last resort,
    # with the memory disclaimer on runtime_note and no custom-code flag.
    payload = {"siblings": [{"rfilename": "model.safetensors", "size": 5_000_000_000}]}
    config = {"model_type": "some-new-lm"}
    async with httpx.AsyncClient(transport=_transport(payload, config=config)) as client:
        result = await inspect_repo("owner/new-lm", client=client)
    assert result["verdict"] == "supported"
    assert result["family"] == "transformers"
    assert result["has_gguf"] is False
    assert result["has_safetensors"] is True
    assert result["runtime_note"]
    assert "2-3x" in result["runtime_note"]
    assert result["needs_trust_remote_code"] is False


async def test_inspect_auto_map_flags_trust_remote_code() -> None:
    # A custom-code checkpoint (config.json auto_map) → the trust flag is set so
    # the Studio surfaces the security note + trust family.
    payload = {"siblings": [{"rfilename": "model.safetensors", "size": 3_000_000_000}]}
    config = {
        "model_type": "unlimited_ocr",
        "auto_map": {"AutoModel": "modeling_unlimited_ocr.UnlimitedOCRForCausalLM"},
    }
    async with httpx.AsyncClient(transport=_transport(payload, config=config)) as client:
        result = await inspect_repo("sahilchachra/Unlimited-OCR", client=client)
    assert result["family"] == "transformers"
    assert result["needs_trust_remote_code"] is True


async def test_inspect_gguf_never_flags_trust() -> None:
    # A GGUF is served by llama.cpp (no repo code) — never flag trust even if a
    # config with auto_map is also present in the repo.
    payload = {
        "gguf": {"architecture": "llama"},
        "siblings": [{"rfilename": "m-Q4_K_M.gguf", "size": 1}],
    }
    config = {"model_type": "llama", "auto_map": {"AutoModel": "x.Y"}}
    async with httpx.AsyncClient(transport=_transport(payload, config=config)) as client:
        result = await inspect_repo("owner/llama-GGUF", client=client)
    assert result["needs_trust_remote_code"] is False


async def test_search_non_gguf_drops_filter() -> None:
    # gguf_only=False must NOT send filter=gguf, so safetensors repos surface.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("filter") is None
        return httpx.Response(200, json=[{"id": "owner/some-safetensors"}])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cards = await search_models("ocr", client=client, gguf_only=False)
    assert cards[0]["id"] == "owner/some-safetensors"


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


async def test_search_enriches_card_with_family_and_params() -> None:
    # expand[] pulls arch + param count inline, so the card carries a PRELIM
    # verdict (resolve_family) and a size — no per-repo inspect round-trip.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("expand[]") is not None  # expand requested
        return httpx.Response(
            200,
            json=[
                {
                    "id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                    "downloads": 1000,
                    "downloadsAllTime": 5000,
                    "likes": 50,
                    "trendingScore": 9.9,
                    "tags": ["gguf", "text-generation"],
                    "pipeline_tag": "text-generation",
                    "library_name": "transformers",
                    "createdAt": "2024-09-18T00:00:00Z",
                    "lastModified": "2025-01-02T00:00:00Z",
                    "config": {"model_type": "qwen2"},
                    "gguf": {"architecture": "qwen2"},
                    "safetensors": {"total": 1_500_000_000},
                    "cardData": {"license": "apache-2.0"},
                }
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        (card,) = await search_models("qwen", client=client)
    assert card["architecture"] == "qwen2"
    assert card["verdict"] == "supported"
    assert card["family"] == "openai_chat"
    assert card["params"] == 1_500_000_000
    assert card["param_label"] == "1.5B"
    assert card["size_est_bytes"] == int(1_500_000_000 * 0.6)
    assert card["downloads_all_time"] == 5000
    assert card["trending_score"] == 9.9
    assert card["license"] == "apache-2.0"
    assert card["prelim"] is True


async def test_search_trending_empty_query_uses_trendingscore() -> None:
    # An empty query + sort=trending is the discovery feed: sort=trendingScore,
    # no `search` param.
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("sort") == "trendingScore"
        assert request.url.params.get("search") is None
        assert request.url.params.get("pipeline_tag") == "image-text-to-text"
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cards = await search_models(
            "", client=client, sort="trending", pipeline_tag="image-text-to-text"
        )
    assert cards == []


async def test_search_vision_prelim_assumes_projector() -> None:
    # The list can't see the mmproj file; a vision arch must still PRELIM-resolve
    # to a vision family, not be wrongly flagged "needs mmproj" (inspect corrects).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "id": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
                    "tags": ["gguf", "image-text-to-text"],
                    "pipeline_tag": "image-text-to-text",
                    "gguf": {"architecture": "qwen2.5vl"},
                }
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        (card,) = await search_models("qwen vl", client=client)
    assert card["verdict"] == "supported"
    assert card["family"] == "lfm2_vl"
