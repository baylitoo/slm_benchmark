"""HF-direct seeding: hub client parsing, quant picking, and the seed job."""

from __future__ import annotations

import json

import httpx
import pytest

from docie_bench.inngest.functions import _run_seed_hf
from docie_bench.serving.hf_hub import (
    HfHubError,
    default_store_name,
    download_file,
    list_collection,
    list_repo_ggufs,
    parse_quant,
    pick_gguf,
    pick_mmproj,
)
from docie_bench.serving.model_store import ModelStore

REPO = "LiquidAI/LFM2.5-350M-Instruct-GGUF"

SIBLINGS = [
    {"rfilename": "README.md"},
    {"rfilename": "LFM2.5-350M-Instruct-Q4_K_M.gguf", "size": 230_000_000},
    {"rfilename": "LFM2.5-350M-Instruct-Q8_0.gguf", "size": 380_000_000},
    {"rfilename": "LFM2.5-350M-Instruct-F16.gguf", "size": 720_000_000},
    {"rfilename": "mmproj-LFM2.5-F16.gguf", "size": 90_000_000},
    {"rfilename": "big-00001-of-00003.gguf", "size": 1},
]

GGUF_BYTES = b"GGUF-fake-weights-" * 1024


def _hub_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/api/models/{REPO}":
            return httpx.Response(200, json={"siblings": SIBLINGS})
        if path == "/api/models/fastino/GLiNER2-Guardrails-PII-Multi":
            return httpx.Response(
                200,
                json={"siblings": [{"rfilename": "model.safetensors"}, {"rfilename": "config.json"}]},
            )
        if path == "/api/models/ghost/nope":
            return httpx.Response(404)
        if path == "/api/collections/LiquidAI/lfm25-abc123":
            return httpx.Response(
                200,
                json={
                    "title": "LFM2.5 collection",
                    "items": [
                        {"id": REPO, "type": "model"},
                        {"id": "LiquidAI/LFM2.5-1.2B-GGUF", "type": "model"},
                        {"id": "LiquidAI/some-paper", "type": "paper"},
                    ],
                },
            )
        if path.startswith(f"/{REPO}/resolve/main/"):
            return httpx.Response(
                200,
                headers={"content-length": str(len(GGUF_BYTES))},
                content=GGUF_BYTES,
            )
        return httpx.Response(500, json={"error": f"unexpected {path}"})

    return httpx.MockTransport(handler)


# ── parsing / picking (pure) ─────────────────────────────────────────────────


def test_parse_quant_variants() -> None:
    assert parse_quant("m-Q4_K_M.gguf") == "Q4_K_M"
    assert parse_quant("m.IQ4_XS.gguf") == "IQ4_XS"
    assert parse_quant("model-f16.gguf") == "F16"
    assert parse_quant("model.gguf") is None


def test_default_store_name_strips_gguf_suffix() -> None:
    assert default_store_name(REPO) == "lfm2.5-350m-instruct"
    assert default_store_name("a/B_C-GGUF", "Q4_K_M") == "b_c-q4_k_m"


async def test_list_repo_ggufs_and_pick() -> None:
    async with httpx.AsyncClient(transport=_hub_transport()) as client:
        files = await list_repo_ggufs(REPO, client=client)
    assert pick_gguf(files, "Q8_0").filename.endswith("Q8_0.gguf")
    assert pick_gguf(files, None).quant == "Q4_K_M"  # preferred default
    assert pick_mmproj(files).filename.startswith("mmproj")
    with pytest.raises(HfHubError, match="available"):
        pick_gguf(files, "Q2_K")


async def test_multipart_only_repo_is_refused() -> None:
    async with httpx.AsyncClient(transport=_hub_transport()) as client:
        files = await list_repo_ggufs(REPO, client=client)
    multiparts = [f for f in files if f.is_multipart or f.is_mmproj]
    with pytest.raises(HfHubError, match="multi-part"):
        pick_gguf(multiparts, None)


async def test_safetensors_repo_routes_to_encoder_runtime() -> None:
    """An encoder checkpoint must point at the encoder path, not a GGUF hunt."""
    async with httpx.AsyncClient(transport=_hub_transport()) as client:
        with pytest.raises(HfHubError, match="encoder"):
            await list_repo_ggufs("fastino/GLiNER2-Guardrails-PII-Multi", client=client)


async def test_unknown_repo_is_a_clear_error() -> None:
    async with httpx.AsyncClient(transport=_hub_transport()) as client:
        with pytest.raises(HfHubError, match="does not exist"):
            await list_repo_ggufs("ghost/nope", client=client)


async def test_list_collection_filters_models() -> None:
    async with httpx.AsyncClient(transport=_hub_transport()) as client:
        view = await list_collection(
            "https://huggingface.co/collections/LiquidAI/lfm25-abc123", client=client
        )
    assert view["title"] == "LFM2.5 collection"
    assert view["models"] == [REPO, "LiquidAI/LFM2.5-1.2B-GGUF"]


# ── download ─────────────────────────────────────────────────────────────────


async def test_download_streams_with_progress(tmp_path) -> None:
    seen: list[tuple[int, int | None]] = []

    async def progress(received: int, total: int | None) -> None:
        seen.append((received, total))

    dest = tmp_path / "model.gguf"
    async with httpx.AsyncClient(transport=_hub_transport()) as client:
        await download_file(
            REPO, "LFM2.5-350M-Instruct-Q4_K_M.gguf", dest, client=client, progress=progress
        )
    assert dest.read_bytes() == GGUF_BYTES
    assert not dest.with_suffix(".gguf.part").exists()
    assert seen[-1] == (len(GGUF_BYTES), len(GGUF_BYTES))


# ── the seed job end to end (store on disk, hub mocked, no catalog) ──────────


async def test_run_seed_hf_registers_store_entry(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    published: list[tuple[str, dict]] = []

    async def fake_publish(channel: str, topic: str, data: dict) -> None:
        published.append((topic, data))

    monkeypatch.setattr("docie_bench.inngest.functions.publish", fake_publish)

    result = await _run_seed_hf(
        {"repo": REPO, "quant": "Q4_K_M", "family": "lfm2"},
        "seed:test",
        transport=_hub_transport(),
    )
    # Store-only success (no DATABASE_URL in tests): entry on disk + honest flag.
    assert result["name"] == "lfm2.5-350m-instruct"
    assert result["family"] == "lfm2"
    assert result["source"] == f"hf:{REPO}:Q4_K_M"
    assert result["catalog_registered"] is False

    store = ModelStore(tmp_path / "models")
    entry = store.entry("lfm2.5-350m-instruct")
    assert entry.model_path.read_bytes() == GGUF_BYTES
    # The temp download dir is gone; the hard-linked canonical blob survives.
    assert not (store.root / ".hf-downloads" / "lfm2.5-350m-instruct").exists()

    topics = [topic for topic, _ in published]
    assert "progress" in topics
    final_progress = [d for t, d in published if t == "progress"][-1]
    assert final_progress["percent"] == 100.0
    assert json.dumps(final_progress)  # JSON-serializable payload


async def test_run_seed_hf_requires_repo(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    with pytest.raises(ValueError, match="repo"):
        await _run_seed_hf({}, "seed:test", transport=_hub_transport())
