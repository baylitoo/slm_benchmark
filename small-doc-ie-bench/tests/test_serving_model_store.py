from __future__ import annotations

import json
from pathlib import Path

import pytest

from docie_bench.serving.model_store import (
    FAMILIES,
    ModelStore,
    ModelStoreError,
    TemplateDelivery,
    get_family,
)


def _fake_ollama_home(
    tmp_path: Path,
    reference_parts: tuple[str, ...],
    tag: str,
    *,
    with_projector: bool = True,
) -> Path:
    """Build a minimal Ollama models dir: one manifest + content-addressed blobs."""
    home = tmp_path / "ollama"
    blobs = home / "blobs"
    blobs.mkdir(parents=True)

    model_digest = "sha256:" + "a" * 64
    (blobs / model_digest.replace(":", "-")).write_bytes(b"GGUF-model-weights")
    layers = [{"mediaType": "application/vnd.ollama.image.model", "digest": model_digest}]
    if with_projector:
        proj_digest = "sha256:" + "b" * 64
        (blobs / proj_digest.replace(":", "-")).write_bytes(b"GGUF-mmproj")
        layers.append(
            {"mediaType": "application/vnd.ollama.image.projector", "digest": proj_digest}
        )

    manifest_dir = home.joinpath("manifests", *reference_parts)
    manifest_dir.mkdir(parents=True)
    (manifest_dir / tag).write_text(json.dumps({"layers": layers}), encoding="utf-8")
    return home


def test_get_family_unknown_lists_known() -> None:
    with pytest.raises(ValueError, match="Unknown model family"):
        get_family("does-not-exist")


def test_nuextract3_contract_is_chat_template_kwargs_vision_and_not_ollama_faithful() -> None:
    contract = FAMILIES["nuextract3"]
    assert contract.template_delivery is TemplateDelivery.CHAT_TEMPLATE_KWARGS
    assert contract.needs_mmproj is True
    assert contract.vision is True
    assert "--jinja" in contract.llama_server_args
    assert contract.ollama_faithful is False


def test_seed_from_ollama_hardlinks_model_and_projector(tmp_path: Path) -> None:
    home = _fake_ollama_home(tmp_path, ("hf.co", "numind", "NuExtract3-GGUF"), "Q4_K_M")
    store = ModelStore(tmp_path / "models")

    entry = store.seed_from_ollama(
        "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
        name="nuextract3",
        family="nuextract3",
        ollama_home=home,
    )

    assert entry.model_path.is_file()
    assert entry.model_path.read_bytes() == b"GGUF-model-weights"
    assert entry.mmproj_path is not None
    assert entry.mmproj_path.is_file()
    assert entry.source == "ollama:hf.co/numind/NuExtract3-GGUF:Q4_K_M"
    # Round-trips through the persisted index.
    assert store.entry("nuextract3").model_path == entry.model_path
    assert [e.name for e in store.list()] == ["nuextract3"]


def test_seed_rejects_reference_path_traversal(tmp_path: Path) -> None:
    """A crafted reference must not read manifests outside the manifests root."""
    home = tmp_path / "ollama"
    (home / "manifests").mkdir(parents=True)
    store = ModelStore(tmp_path / "models")
    with pytest.raises(ModelStoreError, match="traversal|escapes"):
        store.seed_from_ollama(
            "../../../../../../etc/passwd",
            name="evil",
            family="openai_chat",
            ollama_home=home,
        )


def test_seed_rejects_store_name_path_traversal(tmp_path: Path) -> None:
    """A crafted store name must not write blobs outside the store root."""
    home = _fake_ollama_home(tmp_path, ("registry.ollama.ai", "library", "m"), "latest",
                             with_projector=False)
    store = ModelStore(tmp_path / "models")
    with pytest.raises(ModelStoreError, match="traversal|escapes"):
        store.seed_from_ollama(
            "m:latest",
            name="../../../../evil",
            family="openai_chat",
            ollama_home=home,
        )


def test_seed_allows_legit_hf_reference_with_slashes(tmp_path: Path) -> None:
    """Containment must NOT reject legitimate refs that contain '/' and ':'."""
    home = _fake_ollama_home(tmp_path, ("hf.co", "numind", "NuExtract3-GGUF"), "Q4_K_M")
    store = ModelStore(tmp_path / "models")
    entry = store.seed_from_ollama(
        "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
        name="nuextract3",
        family="nuextract3",
        ollama_home=home,
    )
    assert entry.name == "nuextract3"


def test_seed_missing_manifest_raises(tmp_path: Path) -> None:
    home = tmp_path / "ollama"
    (home / "blobs").mkdir(parents=True)
    store = ModelStore(tmp_path / "models")
    with pytest.raises(ModelStoreError, match="manifest not found"):
        store.seed_from_ollama(
            "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
            name="nuextract3",
            family="nuextract3",
            ollama_home=home,
        )


def test_seed_vision_family_without_projector_raises(tmp_path: Path) -> None:
    home = _fake_ollama_home(
        tmp_path, ("hf.co", "numind", "NuExtract3-GGUF"), "Q4_K_M", with_projector=False
    )
    store = ModelStore(tmp_path / "models")
    with pytest.raises(ModelStoreError, match="requires a vision projector"):
        store.seed_from_ollama(
            "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
            name="nuextract3",
            family="nuextract3",
            ollama_home=home,
        )


def test_llama_server_command_includes_jinja_and_mmproj(tmp_path: Path) -> None:
    home = _fake_ollama_home(tmp_path, ("hf.co", "numind", "NuExtract3-GGUF"), "Q4_K_M")
    store = ModelStore(tmp_path / "models")
    store.seed_from_ollama(
        "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
        name="nuextract3",
        family="nuextract3",
        ollama_home=home,
    )

    command = store.llama_server_command("nuextract3", port=8088)

    assert command[0] == "llama-server"
    assert "--jinja" in command
    assert "--mmproj" in command
    assert "--port" in command
    assert "8088" in command
    assert any(part.endswith("model.gguf") for part in command)


def test_ollama_modelfile_refuses_chat_template_kwargs_family(tmp_path: Path) -> None:
    home = _fake_ollama_home(tmp_path, ("hf.co", "numind", "NuExtract3-GGUF"), "Q4_K_M")
    store = ModelStore(tmp_path / "models")
    store.seed_from_ollama(
        "hf.co/numind/NuExtract3-GGUF:Q4_K_M",
        name="nuextract3",
        family="nuextract3",
        ollama_home=home,
    )
    with pytest.raises(ModelStoreError, match="cannot be served faithfully by Ollama"):
        store.ollama_modelfile("nuextract3")


def test_ollama_modelfile_for_faithful_family(tmp_path: Path) -> None:
    gguf = tmp_path / "src.gguf"
    gguf.write_bytes(b"weights")
    store = ModelStore(tmp_path / "models")
    store.add_gguf(name="legacy", family="nuextract_v1", model_gguf=gguf)

    modelfile = store.ollama_modelfile("legacy")

    assert modelfile.startswith("FROM ")
    assert "model.gguf" in modelfile
    assert 'PARAMETER stop "<|end-output|>"' in modelfile
    assert "PARAMETER temperature 0.0" in modelfile


def test_library_reference_resolves_to_registry_ollama_ai(tmp_path: Path) -> None:
    home = _fake_ollama_home(
        tmp_path, ("registry.ollama.ai", "library", "nuextract"), "3.8b", with_projector=False
    )
    store = ModelStore(tmp_path / "models")
    entry = store.seed_from_ollama(
        "nuextract:3.8b", name="nuextract-v1", family="nuextract_v1", ollama_home=home
    )
    assert entry.model_path.is_file()
    assert entry.mmproj_path is None
