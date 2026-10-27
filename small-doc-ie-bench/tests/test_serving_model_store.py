from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from unittest import mock

import pytest

from docie_bench.serving.model_store import (
    FAMILIES,
    INDEX_LOCK_NAME,
    ModelStore,
    ModelStoreError,
    TemplateDelivery,
    get_family,
)


def _write_blob(blobs: Path, content: bytes) -> str:
    """Write a content-addressed Ollama blob; return its canonical ``sha256:`` digest.

    Ollama names blobs ``blobs/sha256-<hex>`` by the sha256 of their content, and
    the manifest layer digest is that same canonical hash. The store now verifies
    transferred blobs against this digest, so a faithful fixture MUST use the real
    content hash (not a placeholder) for the transfer to pass.
    """
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    (blobs / digest.replace(":", "-")).write_bytes(content)
    return digest


def _fake_ollama_home(
    tmp_path: Path,
    reference_parts: tuple[str, ...],
    tag: str,
    *,
    with_projector: bool = True,
    model_content: bytes = b"GGUF-model-weights",
    model_digest_override: str | None = None,
) -> Path:
    """Build a minimal Ollama models dir: one manifest + content-addressed blobs.

    ``model_digest_override`` writes a manifest digest that does NOT match the
    blob content, simulating a corrupt/mislabeled source blob so the store's
    integrity check has something to reject.
    """
    home = tmp_path / "ollama"
    blobs = home / "blobs"
    blobs.mkdir(parents=True)

    if model_digest_override is not None:
        # Blob EXISTS at the manifest-referenced path but its content does not
        # hash to that digest — a corrupt/mislabeled source blob.
        model_digest = model_digest_override
        (blobs / model_digest.replace(":", "-")).write_bytes(model_content)
    else:
        model_digest = _write_blob(blobs, model_content)
    layers = [{"mediaType": "application/vnd.ollama.image.model", "digest": model_digest}]
    if with_projector:
        proj_digest = _write_blob(blobs, b"GGUF-mmproj")
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


def test_spark2_5_contract_is_a_standard_chat_family_own_kv_sizing() -> None:
    # Same wire contract as "lfm2" (response_format, --jinja, tool_calls) --
    # its own family purely so the sizing engine can price its hybrid
    # sliding-window/full-attention KV-cache cost separately (see
    # serving.resources.SPARK2_5_KV_CACHE_BYTES_PER_TOKEN).
    contract = FAMILIES["spark2_5"]
    assert contract.template_delivery is TemplateDelivery.OPENAI_JSON_SCHEMA
    assert contract.response_format_style == "openai_json_schema"
    assert "--jinja" in contract.llama_server_args
    assert contract.tools is True
    assert contract.needs_mmproj is False
    assert contract.vision is False
    assert contract.ollama_faithful is True


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


def test_seed_blob_mismatching_manifest_digest_fails_loud_no_partial_state(
    tmp_path: Path,
) -> None:
    """(a) Blob bytes != manifest digest -> ModelStoreError, zero partial state.

    Simulates a corrupt/mislabeled source blob: the manifest advertises a digest
    that is NOT the sha256 of the blob content. The seed must fail loud and leave
    NO model.gguf and NO index.json entry.
    """
    wrong_digest = "sha256:" + "c" * 64  # not the hash of b"GGUF-model-weights"
    home = _fake_ollama_home(
        tmp_path,
        ("registry.ollama.ai", "library", "m"),
        "latest",
        with_projector=False,
        model_digest_override=wrong_digest,
    )
    store = ModelStore(tmp_path / "models")
    with pytest.raises(ModelStoreError, match="integrity check failed"):
        store.seed_from_ollama("m:latest", name="corrupt", family="openai_chat", ollama_home=home)

    assert not (tmp_path / "models" / "corrupt").exists()  # no model.gguf, no dir
    assert store.list() == []  # no index entry
    with pytest.raises(ModelStoreError, match="Unknown model"):
        store.entry("corrupt")


def test_seed_write_entry_crash_leaves_no_index_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) Crash between the verified transfer and _write_entry -> no index entry.

    The canonical model.gguf is written+verified via a tmp->rename, so the blob
    may exist, but the entry is only visible once the index is written. A failure
    before that must leave the store un-indexed (never a half-registered entry).
    """
    home = _fake_ollama_home(
        tmp_path, ("registry.ollama.ai", "library", "m"), "latest", with_projector=False
    )
    store = ModelStore(tmp_path / "models")

    def _boom(_entry: object) -> None:
        raise RuntimeError("index write crashed")

    monkeypatch.setattr(store, "_write_entry", _boom)
    with pytest.raises(RuntimeError, match="index write crashed"):
        store.seed_from_ollama("m:latest", name="half", family="openai_chat", ollama_home=home)

    assert store.list() == []  # no index entry despite the transfer having run
    with pytest.raises(ModelStoreError, match="Unknown model"):
        store.entry("half")


def test_remove_entry_drops_index_key_and_dir_only_for_that_name(tmp_path: Path) -> None:
    """Compensation removes only ``root/name`` + its index key, never a sibling."""
    home = _fake_ollama_home(
        tmp_path, ("registry.ollama.ai", "library", "m"), "latest", with_projector=False
    )
    store = ModelStore(tmp_path / "models")
    store.seed_from_ollama("m:latest", name="keep", family="openai_chat", ollama_home=home)
    store.seed_from_ollama("m:latest", name="drop", family="openai_chat", ollama_home=home)

    store.remove_entry("drop")

    assert not (tmp_path / "models" / "drop").exists()
    assert [e.name for e in store.list()] == ["keep"]
    assert (tmp_path / "models" / "keep").is_dir()


def test_add_gguf_copy_is_verified_and_faithful(tmp_path: Path) -> None:
    """add_gguf routes through the verified transfer (copy-fidelity, link=False)."""
    gguf = tmp_path / "src.gguf"
    gguf.write_bytes(b"weights-abc")
    store = ModelStore(tmp_path / "models")
    entry = store.add_gguf(
        name="local", family="nuextract_v1", model_gguf=gguf, link=False
    )
    assert entry.model_path.read_bytes() == b"weights-abc"
    assert not entry.model_path.with_name("model.gguf.tmp").exists()

# --- the index is the only record that a model is here ---------------------


def _store_with_sources(tmp_path: Path, count: int) -> tuple[ModelStore, dict[str, Path]]:
    store = ModelStore(tmp_path / "models")
    sources = {}
    for i in range(count):
        name = f"model-{i}"
        source = tmp_path / f"{name}.gguf"
        source.write_bytes(b"weights-" + name.encode())
        sources[name] = source
    return store, sources


def _add_concurrently(
    store: ModelStore, sources: dict[str, Path], extra: list[threading.Thread] | None = None
) -> list[BaseException]:
    start = threading.Barrier(len(sources) + len(extra or []))
    errors: list[BaseException] = []

    def add(name: str) -> None:
        start.wait()
        try:
            store.add_gguf(
                name=name, family="nuextract_v1", model_gguf=sources[name], link=False
            )
        except BaseException as exc:  # noqa: BLE001 - asserted by the caller
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(n,)) for n in sources]
    for thread in threads + (extra or []):
        thread.start()
    for thread in threads + (extra or []):
        thread.join()
    return errors


def test_concurrent_seeds_keep_every_entry_and_leave_no_orphan(tmp_path: Path) -> None:
    """Two writers of one store root are the documented workflow: the runbook
    tells an operator to run ``store add-gguf`` on a node that is also seeding.

    Without a cross-process lock the read-modify-write loses the entry the
    loser wrote — ``add_gguf`` returns an entry and the CLI reports success,
    but the index never names it, so the weights on disk are invisible to
    every reader. Measured before the lock: 3 of 12 indexed, 6 orphan
    directories, 4 raised.
    """
    store, sources = _store_with_sources(tmp_path, 12)

    errors = _add_concurrently(store, sources)

    indexed = {entry.name for entry in store.list()}
    on_disk = {p.name for p in (tmp_path / "models").iterdir() if p.is_dir()}
    assert errors == [], f"concurrent seeds failed: {errors[:3]}"
    assert indexed == set(sources)
    assert on_disk - indexed == set(), "weights on disk the index cannot see"


def test_a_concurrent_reader_does_not_break_a_writer(tmp_path: Path) -> None:
    """A reader holding ``index.json`` open while a writer renames onto it is
    enough to fail the rename on Windows, so reads take the lock too."""
    store, sources = _store_with_sources(tmp_path, 6)
    read_errors: list[BaseException] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            try:
                store.list()
            except BaseException as exc:  # noqa: BLE001 - asserted below
                read_errors.append(exc)

    reader = threading.Thread(target=poll)
    reader.start()
    try:
        errors = _add_concurrently(store, sources)
    finally:
        stop.set()
        reader.join()

    assert errors == [], f"writes failed under a concurrent reader: {errors[:3]}"
    assert read_errors == [], f"reads failed under concurrent writes: {read_errors[:3]}"


def test_the_index_is_flushed_to_disk_before_the_rename(tmp_path: Path) -> None:
    """``os.replace`` is atomic for the name, not the contents: without the
    fsync a power loss can land the rename while the payload is buffered."""
    store, _ = _store_with_sources(tmp_path, 0)
    calls: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        calls.append("fsync")
        real_fsync(fd)

    def replace(src: object, dst: object) -> None:
        calls.append("replace")
        real_replace(src, dst)  # type: ignore[arg-type]

    with mock.patch.object(os, "fsync", fsync), mock.patch.object(os, "replace", replace):
        store._write_index({"lfm2.5-350m": {}})

    assert calls == ["fsync", "replace"]


def test_a_failed_write_leaves_no_staging_file_behind(tmp_path: Path) -> None:
    """One temp file per write means debris would otherwise accumulate per crash."""
    store, _ = _store_with_sources(tmp_path, 0)
    root = tmp_path / "models"

    with (
        mock.patch.object(os, "replace", side_effect=OSError("no space")),
        pytest.raises(OSError, match="no space"),
    ):
        store._write_index({"lfm2.5-350m": {}})

    assert [p.name for p in root.iterdir() if p.name.endswith(".tmp")] == []


def test_a_held_lock_times_out_with_an_actionable_error(tmp_path: Path) -> None:
    """A hung holder has to surface as a named error, not a CLI that never
    returns."""
    store, _ = _store_with_sources(tmp_path, 0)
    store._lock.timeout = 0.2
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with store._lock:
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    try:
        with pytest.raises(ModelStoreError, match=INDEX_LOCK_NAME):
            store.list()
    finally:
        release.set()
        holder.join()


def test_a_corrupt_index_is_an_actionable_error_not_a_json_traceback(
    tmp_path: Path,
) -> None:
    """And specifically not an empty store: treating a damaged index as "this
    node holds nothing" would re-download every model while the weights sit on
    disk unreferenced."""
    store, sources = _store_with_sources(tmp_path, 1)
    store.add_gguf(
        name="model-0", family="nuextract_v1", model_gguf=sources["model-0"], link=False
    )
    (tmp_path / "models" / "index.json").write_text('{"model-0": {"name": "mod', encoding="utf-8")

    with pytest.raises(ModelStoreError) as caught:
        store.list()

    message = str(caught.value)
    assert "index.json" in message
    assert "re-register" in message
