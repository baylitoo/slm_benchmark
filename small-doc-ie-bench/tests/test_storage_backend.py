"""Tests for the pluggable storage backend (docs: storage-backend-abstraction).

``LocalDiskBackend`` must reproduce the exact behavior ``model_store.py``
had before the transfer logic was extracted into it (hard link, atomic
sha256-verified write, all-or-nothing tree copy). ``S3CompatibleBackend`` is
exercised against a mocked S3 (moto) — it has not been run against a live
bucket; that is deliberately follow-up work, not covered here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from docie_bench.serving.storage_backend import (
    LocalDiskBackend,
    S3CompatibleBackend,
    StorageBackendError,
    resolve_storage_backend,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# LocalDiskBackend
# --------------------------------------------------------------------------- #
def test_local_write_verified_hard_links_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    content = b"weights" * 100
    source.write_bytes(content)
    backend = LocalDiskBackend(tmp_path / "store")

    backend.write_verified(source, "model-a/model.gguf")

    dest = backend.resolve_local_path("model-a/model.gguf")
    assert dest.read_bytes() == content
    assert dest.stat().st_ino == source.stat().st_ino, "should hard-link, not copy"


def test_local_write_verified_rejects_wrong_canonical_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    source.write_bytes(b"weights")
    backend = LocalDiskBackend(tmp_path / "store")

    with pytest.raises(StorageBackendError, match="integrity check failed"):
        backend.write_verified(
            source, "model-a/model.gguf", expected_digest="sha256:" + "0" * 64
        )

    assert not backend.exists("model-a/model.gguf"), "no partial file on mismatch"
    assert not (backend.root / "model-a" / "model.gguf.tmp").exists()


def test_local_write_verified_accepts_correct_canonical_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    content = b"weights"
    source.write_bytes(content)
    backend = LocalDiskBackend(tmp_path / "store")

    backend.write_verified(
        source, "model-a/model.gguf", expected_digest=f"sha256:{_sha256(content)}"
    )

    assert backend.exists("model-a/model.gguf")


def test_local_write_tree_is_atomic_and_all_or_nothing(tmp_path: Path) -> None:
    src = tmp_path / "snapshot_src"
    (src / "sub").mkdir(parents=True)
    (src / "config.json").write_text("{}")
    (src / "sub" / "weights.safetensors").write_bytes(b"abc")
    backend = LocalDiskBackend(tmp_path / "store")

    backend.write_tree(src, "model-a/snapshot")

    resolved = backend.resolve_local_dir("model-a/snapshot")
    assert (resolved / "config.json").read_text() == "{}"
    assert (resolved / "sub" / "weights.safetensors").read_bytes() == b"abc"
    assert not (backend.root / "model-a" / "snapshot.tmp").exists()


def test_local_write_tree_replaces_existing_snapshot(tmp_path: Path) -> None:
    src = tmp_path / "snapshot_src"
    src.mkdir()
    (src / "a.txt").write_text("v1")
    backend = LocalDiskBackend(tmp_path / "store")
    backend.write_tree(src, "model-a/snapshot")

    (src / "a.txt").write_text("v2")
    (src / "b.txt").write_text("new")
    backend.write_tree(src, "model-a/snapshot")

    resolved = backend.resolve_local_dir("model-a/snapshot")
    assert (resolved / "a.txt").read_text() == "v2"
    assert (resolved / "b.txt").read_text() == "new"


def test_local_delete_removes_file_and_tree(tmp_path: Path) -> None:
    backend = LocalDiskBackend(tmp_path / "store")
    backend.write_verified(_touch(tmp_path / "s.gguf", b"x"), "m/model.gguf")
    backend.delete("m/model.gguf")
    assert not backend.exists("m/model.gguf")

    src = tmp_path / "snap"
    src.mkdir()
    (src / "f.txt").write_text("x")
    backend.write_tree(src, "m2/snapshot")
    backend.delete("m2/snapshot")
    assert not backend.exists("m2/snapshot")


def test_local_list_prefix(tmp_path: Path) -> None:
    backend = LocalDiskBackend(tmp_path / "store")
    src = tmp_path / "snap"
    (src / "sub").mkdir(parents=True)
    (src / "config.json").write_text("{}")
    (src / "sub" / "w.safetensors").write_bytes(b"x")
    backend.write_tree(src, "m/snapshot")

    keys = backend.list_prefix("m/snapshot")
    assert keys == ["m/snapshot/config.json", "m/snapshot/sub/w.safetensors"]


def _touch(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


# --------------------------------------------------------------------------- #
# S3CompatibleBackend (moto-mocked — no live bucket exercised)
# --------------------------------------------------------------------------- #
@pytest.fixture
def s3_backend(tmp_path: Path):
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="docie-models")
        yield S3CompatibleBackend(
            bucket="docie-models", cache_dir=tmp_path / "cache", client=client
        )


def test_s3_write_verified_and_materialize_roundtrip(tmp_path: Path, s3_backend) -> None:
    source = tmp_path / "model.gguf"
    content = b"weights" * 50
    source.write_bytes(content)

    s3_backend.write_verified(source, "model-a/model.gguf")
    assert s3_backend.exists("model-a/model.gguf")

    materialized = s3_backend.resolve_local_path("model-a/model.gguf")
    assert materialized.read_bytes() == content


def test_s3_write_verified_rejects_wrong_digest(tmp_path: Path, s3_backend) -> None:
    source = tmp_path / "model.gguf"
    source.write_bytes(b"weights")

    with pytest.raises(StorageBackendError, match="refusing to upload"):
        s3_backend.write_verified(
            source, "model-a/model.gguf", expected_digest="sha256:" + "0" * 64
        )
    assert not s3_backend.exists("model-a/model.gguf")


def test_s3_resolve_local_path_skips_redownload_when_cache_verifies(
    tmp_path: Path, s3_backend
) -> None:
    source = tmp_path / "model.gguf"
    source.write_bytes(b"weights")
    s3_backend.write_verified(source, "model-a/model.gguf")

    first = s3_backend.resolve_local_path("model-a/model.gguf")
    mtime_before = first.stat().st_mtime_ns

    second = s3_backend.resolve_local_path("model-a/model.gguf")
    assert second.stat().st_mtime_ns == mtime_before, "should reuse cached file, not redownload"


def test_s3_write_tree_and_resolve_local_dir(tmp_path: Path, s3_backend) -> None:
    src = tmp_path / "snapshot_src"
    (src / "sub").mkdir(parents=True)
    (src / "config.json").write_text("{}")
    (src / "sub" / "weights.safetensors").write_bytes(b"abc")

    s3_backend.write_tree(src, "model-a/snapshot")

    resolved = s3_backend.resolve_local_dir("model-a/snapshot")
    assert (resolved / "config.json").read_text() == "{}"
    assert (resolved / "sub" / "weights.safetensors").read_bytes() == b"abc"


def test_s3_delete_removes_prefix(tmp_path: Path, s3_backend) -> None:
    src = tmp_path / "snapshot_src"
    src.mkdir()
    (src / "a.txt").write_text("x")
    s3_backend.write_tree(src, "model-a/snapshot")

    s3_backend.delete("model-a/snapshot")

    assert s3_backend.list_prefix("model-a/snapshot") == []


def test_s3_list_prefix_strips_bucket_prefix(tmp_path: Path) -> None:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="docie-models")
        backend = S3CompatibleBackend(
            bucket="docie-models",
            cache_dir=tmp_path / "cache",
            prefix="envs/dev",
            client=client,
        )
        source = tmp_path / "model.gguf"
        source.write_bytes(b"weights")
        backend.write_verified(source, "model-a/model.gguf")

        assert backend.list_prefix("model-a") == ["model-a/model.gguf"]
        # confirm the object actually lives under the configured bucket prefix
        raw = client.list_objects_v2(Bucket="docie-models")["Contents"]
        assert raw[0]["Key"] == "envs/dev/model-a/model.gguf"


# --------------------------------------------------------------------------- #
# resolve_storage_backend (env-driven factory)
# --------------------------------------------------------------------------- #
def test_resolve_storage_backend_defaults_to_local(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DOCIE_STORAGE_BACKEND", raising=False)
    backend = resolve_storage_backend(tmp_path / "store")
    assert isinstance(backend, LocalDiskBackend)


def test_resolve_storage_backend_s3_requires_bucket(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOCIE_STORAGE_BACKEND", "s3")
    monkeypatch.delenv("DOCIE_S3_BUCKET", raising=False)
    with pytest.raises(StorageBackendError, match="DOCIE_S3_BUCKET"):
        resolve_storage_backend(tmp_path / "store")


def test_resolve_storage_backend_rejects_unknown_kind(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOCIE_STORAGE_BACKEND", "azure-blob")
    with pytest.raises(StorageBackendError, match="Unknown DOCIE_STORAGE_BACKEND"):
        resolve_storage_backend(tmp_path / "store")
