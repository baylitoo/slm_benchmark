"""The storage seam: local disk today, a bucket tomorrow (#595 stage 1).

Stage 1 changes no behaviour, so `test_serving_model_store.py` passing
unchanged is the real guard. These tests cover the two things that file cannot:
the backend's own contract, and the fact that `ModelStore` talks to it in
**keys** rather than paths.

That second one is not hypothetical. The first version of the delegation passed
an absolute `Path` where a key belongs, and every store test still passed --
because `root / <absolute path inside root>` returns that same path, so the
bytes landed exactly where they should. It worked by accident, and as an object
key in stage 2 it would have been badly wrong.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from docie_bench.serving.model_store import ModelStore, ModelStoreError
from docie_bench.serving.storage_backend import (
    LocalDiskBackend,
    StorageBackendError,
)

WEIGHTS = b"GGUF-model-weights"


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _source(tmp_path: Path, payload: bytes = WEIGHTS, name: str = "src.gguf") -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


# --- write_verified -------------------------------------------------------


def test_a_verified_write_publishes_the_canonical_name(tmp_path: Path) -> None:
    backend = LocalDiskBackend(tmp_path / "store")
    source = _source(tmp_path)

    backend.write_verified(
        source, "lfm2.5-350m/model.gguf", link=True, expected_digest=_digest(WEIGHTS)
    )

    published = backend.resolve_local_path("lfm2.5-350m/model.gguf")
    assert published.read_bytes() == WEIGHTS
    # And nothing half-written beside it.
    assert not published.with_name(published.name + ".tmp").exists()


def test_a_wrong_digest_leaves_no_canonical_file_behind(tmp_path: Path) -> None:
    """The property the whole verified-write dance exists for: a mislabeled
    blob must never appear under the name a runtime will mmap."""
    backend = LocalDiskBackend(tmp_path / "store")
    source = _source(tmp_path)

    with pytest.raises(StorageBackendError, match="integrity check failed"):
        backend.write_verified(
            source,
            "lfm2.5-350m/model.gguf",
            link=True,
            expected_digest=_digest(b"different bytes entirely"),
        )

    published = backend.resolve_local_path("lfm2.5-350m/model.gguf")
    assert not published.exists()
    assert not published.with_name(published.name + ".tmp").exists()


def test_a_hard_link_is_still_checked_against_the_canonical_digest(
    tmp_path: Path,
) -> None:
    """A same-inode tmp hashes to the source's real bytes, so a corrupt SOURCE
    is caught even though the "copy" is free. Copy-fidelity alone would pass
    this, which is why the canonical digest is checked when there is one."""
    backend = LocalDiskBackend(tmp_path / "store")
    corrupt = _source(tmp_path, b"corrupted-on-disk")

    with pytest.raises(StorageBackendError, match="manifest digest"):
        backend.write_verified(
            corrupt, "m/model.gguf", link=True, expected_digest=_digest(WEIGHTS)
        )


def test_without_a_canonical_digest_the_check_is_copy_fidelity(tmp_path: Path) -> None:
    """The HF trust boundary: there is no manifest to assert against, so the
    only checkable property is that the bytes arrived intact."""
    backend = LocalDiskBackend(tmp_path / "store")
    source = _source(tmp_path)

    backend.write_verified(source, "m/model.gguf", link=False, expected_digest=None)

    assert backend.resolve_local_path("m/model.gguf").read_bytes() == WEIGHTS


# --- the containment guard ------------------------------------------------


@pytest.mark.parametrize("key", ["../escape.gguf", "a/../../escape.gguf"])
def test_a_key_that_escapes_the_root_is_refused(tmp_path: Path, key: str) -> None:
    """Keys derive from store names, which are attacker-influenced. The store
    guards them too; this guard travels with the backend, so a future caller
    that forgets the store's check still cannot write outside the root."""
    backend = LocalDiskBackend(tmp_path / "store")
    source = _source(tmp_path)

    with pytest.raises(StorageBackendError, match="escapes|traversal"):
        backend.write_verified(source, key, link=True, expected_digest=None)


# --- write_tree -----------------------------------------------------------


def _snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snap"
    (root / "nested").mkdir(parents=True)
    (root / "model.safetensors").write_bytes(WEIGHTS)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "nested" / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


def test_a_tree_write_preserves_relative_paths(tmp_path: Path) -> None:
    """Snapshot families load a directory via from_pretrained, so the layout
    inside it is part of the artifact -- a flattened tree does not load."""
    backend = LocalDiskBackend(tmp_path / "store")

    backend.write_tree(_snapshot(tmp_path), "gliner/snapshot", link=True)

    published = backend.resolve_local_dir("gliner/snapshot")
    assert (published / "model.safetensors").read_bytes() == WEIGHTS
    assert (published / "nested" / "tokenizer.json").is_file()


def test_a_failed_tree_write_leaves_no_partial_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-copied checkpoint loads and then answers badly, which is worse
    than failing -- so the canonical prefix must never show one, and the
    staging directory must not survive either.

    The failure is injected partway through on purpose: the interesting state
    is "some files transferred, then it died", which is exactly what a dropped
    connection will look like once the backend is remote.
    """
    from docie_bench.serving import storage_backend

    backend = LocalDiskBackend(tmp_path / "store")
    source = _snapshot(tmp_path)
    published = backend.resolve_local_dir("gliner/snapshot")
    real_transfer = storage_backend._transfer
    calls: list[int] = []

    def failing(src: Path, dst: Path, *, link: bool) -> None:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("connection dropped mid-tree")
        real_transfer(src, dst, link=link)

    monkeypatch.setattr(storage_backend, "_transfer", failing)

    with pytest.raises(RuntimeError, match="connection dropped"):
        backend.write_tree(source, "gliner/snapshot", link=True)

    assert len(calls) == 2  # it really did fail partway, not before starting
    assert not published.exists()
    assert not published.with_name(published.name + ".tmp").exists()


def test_a_tree_write_replaces_an_existing_snapshot(tmp_path: Path) -> None:
    backend = LocalDiskBackend(tmp_path / "store")
    backend.write_tree(_snapshot(tmp_path), "gliner/snapshot", link=True)

    replacement = tmp_path / "snap2"
    replacement.mkdir()
    (replacement / "model.safetensors").write_bytes(b"new-weights")
    backend.write_tree(replacement, "gliner/snapshot", link=True)

    published = backend.resolve_local_dir("gliner/snapshot")
    assert (published / "model.safetensors").read_bytes() == b"new-weights"
    # The old tree is gone, not merged into the new one.
    assert not (published / "config.json").exists()


# --- the rest of the surface ----------------------------------------------


def test_exists_delete_and_list_prefix(tmp_path: Path) -> None:
    backend = LocalDiskBackend(tmp_path / "store")
    backend.write_verified(
        _source(tmp_path), "m/model.gguf", link=True, expected_digest=None
    )
    backend.write_tree(_snapshot(tmp_path), "g/snapshot", link=True)

    assert backend.exists("m/model.gguf") is True
    assert backend.exists("m/absent.gguf") is False
    assert backend.list_prefix("m") == ["m/model.gguf"]
    assert "g/snapshot/nested/tokenizer.json" in backend.list_prefix("g")

    backend.delete("m/model.gguf")
    backend.delete("g/snapshot")
    assert backend.exists("m/model.gguf") is False
    assert backend.exists("g/snapshot") is False
    backend.delete("m/model.gguf")  # missing is not an error


def test_list_prefix_of_a_missing_prefix_is_empty_not_an_error(tmp_path: Path) -> None:
    assert LocalDiskBackend(tmp_path / "store").list_prefix("nothing/here") == []


# --- what ModelStore sends it --------------------------------------------


class _RecordingBackend(LocalDiskBackend):
    """A real backend that remembers the keys it was given."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.keys: list[str] = []

    def write_verified(
        self,
        source: Path,
        key: str,
        *,
        link: bool,
        expected_digest: str | None,
    ) -> None:
        self.keys.append(key)
        super().write_verified(
            source, key, link=link, expected_digest=expected_digest
        )

    def write_tree(self, source_dir: Path, key_prefix: str, *, link: bool) -> None:
        self.keys.append(key_prefix)
        super().write_tree(source_dir, key_prefix, link=link)


def test_the_store_passes_relative_keys_not_absolute_paths(tmp_path: Path) -> None:
    """The accident this file's docstring describes. An absolute path happens
    to work on local disk, because it already points inside the root -- so
    nothing else notices, and a remote backend would receive a local filesystem
    path as an object key."""
    root = tmp_path / "store"
    backend = _RecordingBackend(root)
    store = ModelStore(root, backend=backend)
    gguf = _source(tmp_path)
    mmproj = _source(tmp_path, b"projector", name="mmproj.gguf")

    store.add_gguf(
        name="nuextract3", family="nuextract3", model_gguf=gguf, mmproj=mmproj
    )

    assert backend.keys == ["nuextract3/model.gguf", "nuextract3/mmproj.gguf"]
    for key in backend.keys:
        assert not Path(key).is_absolute()


def test_the_store_still_raises_its_own_error_type(tmp_path: Path) -> None:
    """A storage seam must not change what a store failure looks like from the
    outside. Callers and tests have always seen ModelStoreError with the
    transfer's own message."""
    root = tmp_path / "store"
    store = ModelStore(root)
    gguf = _source(tmp_path)

    # A name that escapes the root. add_gguf had NO such guard before the
    # seam, unlike seed_from_ollama / add_snapshot / remove_entry, so this
    # wrote outside the store root entirely.
    with pytest.raises(ModelStoreError, match="traversal|escapes"):
        store.add_gguf(name="../escape", family="openai_chat", model_gguf=gguf)

    assert not (root.parent / "escape").exists()


def test_a_backend_failure_surfaces_as_a_store_error_with_its_message(
    tmp_path: Path,
) -> None:
    class _Failing(LocalDiskBackend):
        def write_verified(
            self,
            source: Path,
            key: str,
            *,
            link: bool,
            expected_digest: str | None,
        ) -> None:
            raise StorageBackendError("blob integrity check failed for model.gguf")

    store = ModelStore(tmp_path / "store", backend=_Failing(tmp_path / "store"))

    with pytest.raises(ModelStoreError, match="integrity check failed"):
        store.add_gguf(
            name="m", family="openai_chat", model_gguf=_source(tmp_path)
        )

# --- replacing an existing tree -------------------------------------------


def test_a_failed_replace_leaves_the_existing_snapshot_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window this guards.

    The first version removed the destination and THEN replaced it, so a
    failure in between left the node with neither tree: the working model was
    already deleted, and the handler then removed the staged replacement too.
    Re-seeding a model a node was serving could destroy it.

    The replace can fail for ordinary reasons -- a cross-device rename, a
    permissions error, or on Windows a running encoder simply holding the
    snapshot directory open -- so this is not an exotic path.
    """
    import docie_bench.serving.storage_backend as module

    backend = LocalDiskBackend(tmp_path / "store")
    first = tmp_path / "v1"
    first.mkdir()
    (first / "model.safetensors").write_bytes(b"v1")
    backend.write_tree(first, "gliner/snapshot", link=True)

    second = tmp_path / "v2"
    second.mkdir()
    (second / "model.safetensors").write_bytes(b"v2")

    real = module.os.replace

    def fails_on_the_publish(src: object, dst: object) -> None:
        if str(dst).endswith("snapshot") and str(src).endswith(".tmp"):
            raise OSError("replace failed")
        return real(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(module.os, "replace", fails_on_the_publish)

    with pytest.raises(OSError, match="replace failed"):
        backend.write_tree(second, "gliner/snapshot", link=True)

    published = backend.resolve_local_dir("gliner/snapshot")
    assert published.is_dir(), "the working snapshot was destroyed"
    assert (published / "model.safetensors").read_bytes() == b"v1"


def test_a_failed_replace_leaves_no_debris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither the staging tree nor the set-aside copy may survive, or the
    next publish starts from someone else's leftovers."""
    import docie_bench.serving.storage_backend as module

    backend = LocalDiskBackend(tmp_path / "store")
    first = tmp_path / "v1"
    first.mkdir()
    (first / "model.safetensors").write_bytes(b"v1")
    backend.write_tree(first, "gliner/snapshot", link=True)

    second = tmp_path / "v2"
    second.mkdir()
    (second / "model.safetensors").write_bytes(b"v2")

    real = module.os.replace

    def fails_on_the_publish(src: object, dst: object) -> None:
        if str(dst).endswith("snapshot") and str(src).endswith(".tmp"):
            raise OSError("replace failed")
        return real(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(module.os, "replace", fails_on_the_publish)
    with pytest.raises(OSError, match="replace failed"):
        backend.write_tree(second, "gliner/snapshot", link=True)

    parent = backend.resolve_local_dir("gliner/snapshot").parent
    assert sorted(p.name for p in parent.iterdir()) == ["snapshot"]


def test_debris_from_an_earlier_crash_does_not_poison_the_next_publish(
    tmp_path: Path,
) -> None:
    """A process killed mid-publish can leave both temporaries behind. The
    next attempt must start clean rather than merging them in."""
    backend = LocalDiskBackend(tmp_path / "store")
    published = backend.resolve_local_dir("gliner/snapshot")
    published.parent.mkdir(parents=True, exist_ok=True)
    (published.parent / "snapshot.tmp").mkdir()
    (published.parent / "snapshot.tmp" / "stale.bin").write_bytes(b"old")
    (published.parent / "snapshot.superseded").mkdir()
    (published.parent / "snapshot.superseded" / "older.bin").write_bytes(b"older")

    source = tmp_path / "v1"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"v1")
    backend.write_tree(source, "gliner/snapshot", link=True)

    assert sorted(p.name for p in published.iterdir()) == ["model.safetensors"]
    assert sorted(p.name for p in published.parent.iterdir()) == ["snapshot"]


def test_a_successful_replace_still_swaps_the_tree_wholesale(
    tmp_path: Path,
) -> None:
    """The behaviour the fix must preserve: the new tree replaces the old one
    entirely rather than merging into it."""
    backend = LocalDiskBackend(tmp_path / "store")
    first = tmp_path / "v1"
    first.mkdir()
    (first / "model.safetensors").write_bytes(b"v1")
    (first / "config.json").write_bytes(b"{}")
    backend.write_tree(first, "gliner/snapshot", link=True)

    second = tmp_path / "v2"
    second.mkdir()
    (second / "model.safetensors").write_bytes(b"v2")
    backend.write_tree(second, "gliner/snapshot", link=True)

    published = backend.resolve_local_dir("gliner/snapshot")
    assert (published / "model.safetensors").read_bytes() == b"v2"
    assert not (published / "config.json").exists()
