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


def test_qwen35_nuextract3_by_name() -> None:
    # NuExtract3 shares the qwen35 backbone but needs its own contract — detected
    # by NAME (not arch). With a projector → nuextract3; without → needs_family
    # (a missing projector, not a text mis-serve).
    v = resolve_family(
        "qwen35",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=True,
        repo_id="numind/NuExtract3-GGUF",
    )
    assert v.verdict == "supported"
    assert v.family == "nuextract3"
    t = resolve_family(
        "qwen35",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="numind/NuExtract3-GGUF",
    )
    assert t.verdict == "needs_family"
    assert t.family == "nuextract3"
    assert "mmproj" in t.reason


def test_qwen35_generic_text_is_chat_not_nuextract3() -> None:
    # A plain Qwen3.5 text GGUF (unsloth/Qwen3.5-0.8B-GGUF) must NOT default to
    # the nuextract3 vision contract — it is a text chat model.
    v = resolve_family(
        "qwen35",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="unsloth/Qwen3.5-0.8B-GGUF",
    )
    assert v.verdict == "supported"
    assert v.family == "openai_chat"


def test_qwen35_generic_vl_is_lfm2vl_not_nuextract3() -> None:
    # A generic Qwen3.5-VL (projector, non-NuExtract name) → the generic vision
    # family, NOT nuextract3.
    v = resolve_family(
        "qwen35",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=True,
        repo_id="unsloth/Qwen3.5-VL-4B-GGUF",
    )
    assert v.verdict == "supported"
    assert v.family == "lfm2_vl"


def test_resolve_gliner_marker() -> None:
    v = resolve_family("GLiNER2", has_gguf=False, has_safetensors=True, has_mmproj=False)
    assert v.verdict == "supported" and v.family == "encoder_gliner2"


def test_lfm2_colbert_reranker_by_name() -> None:
    # LFM2.5-ColBERT-350M reports arch "lfm2" — identical to the chat family's
    # backbone. Caught by NAME so it lands on "reranker", not "lfm2" (chat).
    v = resolve_family(
        "lfm2",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="LiquidAI/LFM2.5-ColBERT-350M-GGUF",
    )
    assert v.verdict == "supported"
    assert v.family == "reranker"


def test_bert_cross_encoder_reranker_by_name() -> None:
    # A BERT cross-encoder reports arch "bert" — identical to the embedding
    # family's backbone. Caught by NAME so it lands on "reranker", not
    # "embedding".
    v = resolve_family(
        "bert",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="mixedbread-ai/mxbai-rerank-base-v2-GGUF",
    )
    assert v.verdict == "supported"
    assert v.family == "reranker"


def test_lfm2_generic_chat_is_not_reranker() -> None:
    # A plain LFM2 chat GGUF, non-reranker name, must NOT default to reranker.
    v = resolve_family(
        "lfm2",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="LiquidAI/LFM2-1.2B-GGUF",
    )
    assert v.verdict == "supported"
    assert v.family == "lfm2"


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
    assert result["runtime"] == "llama.cpp"
    assert result["recommended_quant"] == "Q4_K_M"
    assert result["download_size_bytes"] == 2_800_000_000
    assert result["estimated_ram_bytes"] == 3_873_741_824
    assert [file["role"] for file in result["required_files"]] == [
        "model",
        "vision_projector",
    ]


async def test_inspect_quant_options_have_exact_fit_estimates() -> None:
    payload = {
        "gguf": {"architecture": "qwen2"},
        "siblings": [
            {"rfilename": "model-Q4_K_M.gguf", "size": 2_000_000_000},
            {"rfilename": "model-Q4_K_S.gguf", "size": 500_000_000},
            {"rfilename": "model-F16.gguf", "size": 4_000_000_000},
        ],
    }
    async with httpx.AsyncClient(transport=_transport(payload)) as client:
        result = await inspect_repo(
            "owner/model-GGUF",
            client=client,
            node_available_bytes=1_800_000_000,
        )

    assert result["recommended_quant"] == "Q4_K_M"
    assert result["fits_node"] is False
    assert result["readiness"] == "blocked"
    assert "insufficient_memory" in {item["code"] for item in result["blockers"]}
    by_quant = {option["quant"]: option for option in result["artifact_options"]}
    assert by_quant["Q4_K_M"]["fits_node"] is False
    assert by_quant["Q4_K_S"]["fits_node"] is True
    assert any("Q4_K_S" in text for text in result["recommendations"])


async def test_inspect_snapshot_prices_all_required_files() -> None:
    payload = {
        "siblings": [
            {"rfilename": "model-00001-of-00002.safetensors", "size": 900_000_000},
            {"rfilename": "model-00002-of-00002.safetensors", "size": 800_000_000},
            {"rfilename": "config.json", "size": 2_000},
            {"rfilename": "tokenizer.json", "size": 30_000},
            {"rfilename": "model.onnx", "size": 2_000_000_000},
        ],
    }
    config = {"model_type": "some-new-lm"}
    async with httpx.AsyncClient(transport=_transport(payload, config=config)) as client:
        result = await inspect_repo(
            "owner/new-lm",
            client=client,
            node_available_bytes=4_000_000_000,
        )

    assert result["runtime"] == "transformers"
    assert result["download_size_bytes"] == 1_700_032_000
    assert result["estimated_ram_bytes"] == 2_773_773_824
    assert result["fits_node"] is True
    assert result["readiness"] == "caution"  # last-resort runtime caveat
    assert "model.onnx" not in {file["filename"] for file in result["required_files"]}


async def test_inspect_multipart_only_repo_is_blocked_before_download() -> None:
    payload = {
        "gguf": {"architecture": "qwen2"},
        "siblings": [
            {"rfilename": "model-Q4_K_M-00001-of-00002.gguf", "size": 1_000_000_000},
            {"rfilename": "model-Q4_K_M-00002-of-00002.gguf", "size": 900_000_000},
        ],
    }
    async with httpx.AsyncClient(transport=_transport(payload)) as client:
        result = await inspect_repo("owner/multipart-GGUF", client=client)

    assert result["readiness"] == "blocked"
    assert result["artifact_options"] == []
    assert "no_servable_artifact" in {item["code"] for item in result["blockers"]}


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
    assert result["readiness"] == "blocked"
    assert "remote_code_approval_required" in {item["code"] for item in result["blockers"]}


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
                {
                    "id": "LiquidAI/LFM2.5-350M-GGUF",
                    "downloads": 42000,
                    "likes": 85,
                    "tags": ["gguf", "text-generation"],
                },
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


def test_annotate_fits_marks_fit_over_budget_and_unknown() -> None:
    from docie_bench.inngest.studio_api import _annotate_fits

    cards = [
        {"id": "a", "size_est_bytes": 1_000_000_000},  # fits
        {"id": "b", "size_est_bytes": 9_000_000_000},  # over budget
        {"id": "c", "size_est_bytes": None},  # unknown size
    ]
    out = _annotate_fits(cards, budget=3_000_000_000)
    fits = {c["id"]: c["fits_node"] for c in out}
    assert fits == {"a": True, "b": False, "c": None}
    assert all(c["node_available_bytes"] == 3_000_000_000 for c in out)


def test_annotate_fits_unknown_budget_is_none() -> None:
    from docie_bench.inngest.studio_api import _annotate_fits

    (card,) = _annotate_fits([{"id": "a", "size_est_bytes": 1_000_000_000}], budget=None)
    assert card["fits_node"] is None  # no snapshot -> never a false "fits"
    assert card["node_available_bytes"] is None


async def test_hf_inspect_route_passes_context_and_live_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docie_bench.inngest.studio_api import deploy
    from docie_bench.serving import hf_hub

    seen: dict[str, object] = {}

    async def fake_inspect(repo: str, **kwargs: object) -> dict[str, object]:
        seen.update({"repo": repo, **kwargs})
        return {"repo": repo, "readiness": "ready"}

    monkeypatch.setattr(deploy, "_node_available_bytes", lambda: 3_000_000_000)
    monkeypatch.setattr(hf_hub, "inspect_repo", fake_inspect)

    result = await deploy.hf_inspect("owner/model", context_length=16_384)

    assert result["readiness"] == "ready"
    assert seen["repo"] == "owner/model"
    assert seen["context_length"] == 16_384
    assert seen["node_available_bytes"] == 3_000_000_000
