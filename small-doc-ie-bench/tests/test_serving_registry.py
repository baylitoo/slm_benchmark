from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request

import pytest

from docie_bench.serving.registry import (
    ArtifactKind,
    ArtifactVerificationError,
    ModelConflictError,
    ModelManifest,
    ModelNotFoundError,
    ModelRegistry,
    TrustPolicy,
    sha256_file,
)


def _manifest(**overrides: object) -> ModelManifest:
    values: dict[str, object] = {
        "model_id": "Qwen/Qwen3-4B",
        "source": "huggingface",
        "revision": "pinned-sha",
        "license": "apache-2.0",
        "trust_policy": TrustPolicy.DENY_REMOTE_CODE,
        "aliases": ("invoice-fast",),
        "tags": ("extraction",),
        "quantization": "Q4_K_M",
        "required_memory_gb": 5,
        "context_length": 8192,
        "supported_tasks": ("generation", "structured_output"),
    }
    values.update(overrides)
    return ModelManifest(**values)


def test_import_get_list_alias_tag_and_remove_are_content_addressed(tmp_path: Path) -> None:
    source = tmp_path / "weights.gguf"
    source.write_bytes(b"model weights")
    registry = ModelRegistry(tmp_path / "registry")

    manifest = registry.import_model(
        _manifest(),
        {"weights.gguf": source},
        kinds={"weights.gguf": ArtifactKind.WEIGHTS},
    )

    assert registry.get("invoice-fast") == manifest
    assert registry.list_models(tag="extraction") == [manifest]
    artifact = manifest.artifacts[0]
    assert artifact.digest == sha256_file(source)
    assert registry.artifact_path(artifact.digest).read_bytes() == b"model weights"
    assert source.exists()
    assert "Qwen" not in str(registry._manifest_file(manifest.model_id))

    stored_path = registry.artifact_path(artifact.digest)
    removed = registry.remove("invoice-fast")
    assert removed == manifest
    assert not stored_path.exists()
    with pytest.raises(ModelNotFoundError):
        registry.get(manifest.model_id)


def test_shared_artifact_is_deduplicated_and_removed_after_last_reference(tmp_path: Path) -> None:
    source = tmp_path / "weights.gguf"
    source.write_bytes(b"same bytes")
    registry = ModelRegistry(tmp_path / "registry")

    first = registry.import_model(_manifest(), {"weights.gguf": source})
    second = registry.import_model(
        _manifest(model_id="Qwen/Qwen3-4B-copy", aliases=("copy",), tags=()),
        {"weights.gguf": source},
    )

    assert first.artifacts[0].digest == second.artifacts[0].digest
    artifact_path = registry.artifact_path(first.artifacts[0].digest)
    registry.remove(first.model_id)
    assert artifact_path.exists()
    registry.remove(second.model_id)
    assert not artifact_path.exists()


def test_checksum_verification_detects_import_and_stored_artifact_corruption(
    tmp_path: Path,
) -> None:
    source = tmp_path / "weights.bin"
    source.write_bytes(b"valid")
    registry = ModelRegistry(tmp_path / "registry")

    with pytest.raises(ArtifactVerificationError, match="checksum mismatch"):
        registry.import_artifact(source, expected_digest="0" * 64)

    manifest = registry.import_model(_manifest(), {"weights.bin": source})
    path = registry.artifact_path(manifest.artifacts[0].digest)
    path.write_bytes(b"corrupt")
    with pytest.raises(ArtifactVerificationError, match="checksum mismatch"):
        registry.get(manifest.model_id, verify=True)


def test_alias_conflicts_and_unsafe_names_do_not_escape_registry(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(_manifest())

    with pytest.raises(ModelConflictError, match="already refers"):
        registry.register(_manifest(model_id="../../other", revision="sha-2"))
    with pytest.raises(ValueError, match="plain file name"):
        registry.import_artifact(__file__, name="../../outside.bin")

    index = json.loads((tmp_path / "registry" / "index.json").read_text(encoding="utf-8"))
    assert index["aliases"] == {"invoice-fast": "Qwen/Qwen3-4B"}


def test_atomic_metadata_update_replaces_aliases_and_tags(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    registry.register(_manifest())
    updated = registry.register(
        _manifest(aliases=("production",), tags=("vision",), state="quarantined")
    )

    assert registry.get("production") == updated
    assert registry.list_models(tag="extraction") == []
    assert registry.list_models(tag="vision") == [updated]
    with pytest.raises(ModelNotFoundError):
        registry.get("invoice-fast")

    without_alias = registry.remove_alias("production", "production")
    without_labels = registry.remove_tag(without_alias.model_id, "vision")
    assert without_labels.aliases == ()
    assert without_labels.tags == ()


def test_pull_resumes_partial_http_download_and_verifies_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"complete model"
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    registry = ModelRegistry(tmp_path / "registry")
    url = "https://models.example/model.bin"
    download_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    partial = tmp_path / "registry" / "downloads" / f"{download_key}.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(content[:8])
    requests: list[Request] = []

    class Response(io.BytesIO):
        status = 206

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def urlopen(request: Request, *, timeout: float | None = None) -> Response:
        requests.append(request)
        return Response(content[8:])

    monkeypatch.setattr("docie_bench.serving.registry.urllib.request.urlopen", urlopen)
    expected = _manifest(
        artifacts=(
            {
                "name": "model.bin",
                "digest": digest,
                "size_bytes": len(content),
                "kind": "weights",
            },
        )
    )

    pulled = registry.pull(expected, {"model.bin": url})

    assert registry.artifact_path(pulled.artifacts[0].digest).read_bytes() == content
    assert requests[0].get_header("Range") == "bytes=8-"
    assert not partial.exists()
