"""PR-1: encoder families + analyzer flag + snapshot seed into the store."""

from __future__ import annotations

import httpx
import pytest

from docie_bench.inngest.functions import _entry_size_bytes, _run_seed_hf
from docie_bench.serving.catalog import available_backends
from docie_bench.serving.hf_hub import (
    HfHubError,
    _is_snapshot_file,
    list_snapshot_files,
)
from docie_bench.serving.model_store import FAMILIES, ModelStore, ModelStoreError, get_family

REPO = "fastino/GLiNER2-Guardrails-PII-Multi"

SNAPSHOT_SIBLINGS = [
    {"rfilename": "config.json", "size": 1200},
    {"rfilename": "model.safetensors", "size": 900},
    {"rfilename": "tokenizer.json", "size": 400},
    {"rfilename": "spm.model", "size": 300},
    {"rfilename": "pytorch_model.bin", "size": 900},  # skipped (alt weights)
    {"rfilename": "onnx/model.onnx", "size": 900},  # skipped (hardware export)
    {"rfilename": "README.md", "size": 50},
]


def _transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/api/models/{REPO}":
            return httpx.Response(200, json={"siblings": SNAPSHOT_SIBLINGS})
        if path == "/api/models/gguf/only":
            return httpx.Response(200, json={"siblings": [{"rfilename": "m.gguf"}]})
        if path.startswith(f"/{REPO}/resolve/main/"):
            fname = path.rsplit("/", 1)[-1]
            return httpx.Response(
                200, headers={"content-length": "10"}, content=f"data-{fname}".encode()[:10]
            )
        return httpx.Response(500, json={"error": f"unexpected {path}"})

    return httpx.MockTransport(handler)


# ── families + flag ──────────────────────────────────────────────────────────


def test_encoder_families_are_analyzers() -> None:
    g1 = get_family("encoder_gliner")
    g2 = get_family("encoder_gliner2")
    assert g1.analyzer and g1.encoder_backend == "gliner"
    assert g2.analyzer and g2.encoder_backend == "gliner2"
    assert not g1.embedding and not g1.vision


def test_available_backends_encoder_only_for_analyzers() -> None:
    assert available_backends("encoder_gliner2") == ["encoder"]
    assert "llama-server" in available_backends("lfm2")


# ── snapshot file selection (pure) ───────────────────────────────────────────


def test_snapshot_file_filter_skips_alt_formats() -> None:
    assert _is_snapshot_file("model.safetensors")
    assert _is_snapshot_file("tokenizer.json")
    assert _is_snapshot_file("spm.model")
    assert not _is_snapshot_file("pytorch_model.bin")
    assert not _is_snapshot_file("onnx/model.onnx")
    assert not _is_snapshot_file("model.gguf")


async def test_list_snapshot_files_keeps_safetensors_tree() -> None:
    async with httpx.AsyncClient(transport=_transport()) as client:
        files = await list_snapshot_files(REPO, client=client)
    names = {f.filename for f in files}
    assert "model.safetensors" in names
    assert "config.json" in names and "spm.model" in names
    assert "pytorch_model.bin" not in names and "onnx/model.onnx" not in names


async def test_list_snapshot_files_refuses_gguf_only_repo() -> None:
    async with httpx.AsyncClient(transport=_transport()) as client:
        with pytest.raises(HfHubError, match="no safetensors"):
            await list_snapshot_files("gguf/only", client=client)


# ── store.add_snapshot ───────────────────────────────────────────────────────


def test_add_snapshot_roundtrip(tmp_path) -> None:
    src = tmp_path / "dl"
    (src / "sub").mkdir(parents=True)
    (src / "model.safetensors").write_bytes(b"weights")
    (src / "config.json").write_text("{}")
    (src / "sub" / "tokenizer.json").write_text("{}")

    store = ModelStore(tmp_path / "store")
    entry = store.add_snapshot(
        name="guardrails-pii", family="encoder_gliner2", snapshot_dir=src, source="hf:x"
    )
    assert entry.model_path.is_dir()
    assert (entry.model_path / "model.safetensors").read_bytes() == b"weights"
    assert (entry.model_path / "sub" / "tokenizer.json").is_file()
    # Reloads from the index as a directory entry.
    assert store.entry("guardrails-pii").model_path.is_dir()
    assert _entry_size_bytes(entry) == len(b"weights") + len("{}") + len("{}")


def test_add_snapshot_rejects_non_analyzer_family(tmp_path) -> None:
    src = tmp_path / "dl"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"w")
    store = ModelStore(tmp_path / "store")
    with pytest.raises(ModelStoreError, match="analyzer"):
        store.add_snapshot(name="x", family="lfm2", snapshot_dir=src)


def test_add_snapshot_requires_safetensors(tmp_path) -> None:
    src = tmp_path / "dl"
    src.mkdir()
    (src / "config.json").write_text("{}")
    store = ModelStore(tmp_path / "store")
    with pytest.raises(ModelStoreError, match="safetensors"):
        store.add_snapshot(name="x", family="encoder_gliner", snapshot_dir=src)


# ── seed-hf job for an analyzer family (store on disk, hub mocked) ───────────


async def test_seed_hf_analyzer_downloads_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    published: list[tuple[str, dict]] = []

    async def fake_publish(channel: str, topic: str, data: dict) -> None:
        published.append((topic, data))

    monkeypatch.setattr("docie_bench.inngest.functions.publish", fake_publish)

    result = await _run_seed_hf(
        {"repo": REPO, "family": "encoder_gliner2", "name": "guardrails-pii"},
        "seed:test",
        transport=_transport(),
    )
    assert result["name"] == "guardrails-pii"
    assert result["family"] == "encoder_gliner2"
    assert result["source"] == f"hf:{REPO}"
    assert result["available_backends"] == ["encoder"]

    store = ModelStore(tmp_path / "models")
    entry = store.entry("guardrails-pii")
    assert entry.model_path.is_dir()
    assert (entry.model_path / "model.safetensors").is_file()
    assert not (entry.model_path / "pytorch_model.bin").exists()
    # download dir cleaned up
    assert not (store.root / ".hf-downloads" / "guardrails-pii").exists()
    assert "download-snapshot" in {d.get("stage") for _, d in published if _ == "progress"}
