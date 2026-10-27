"""Publishing to and resolving from the shared model repository (#595 2b).

Run against a directory rather than a bucket, on purpose: these are the
repository's RULES -- publication order, deduplication, verification, what
counts as published -- and they should be provable without a network. The
MinIO integration test exists to prove the S3 adapter, not this logic.

The property most of these circle is the decided one:

    publish artifacts first and the manifest last, so an interrupted
    publication leaves garbage rather than a half-deployable model.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import pytest

from docie_bench.serving import manifest as manifest_mod
from docie_bench.serving.model_repository import (
    LocalObjectStore,
    ModelRepository,
    ModelRepositoryError,
    ObjectNotFoundError,
    iter_store_files,
    sha256_file,
)

WEIGHTS = b"GGUF-model-weights"
PROJECTOR = b"vision-projector"


@pytest.fixture
def local(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "store" / "nuextract3"
    source.mkdir(parents=True)
    model = source / "model.gguf"
    model.write_bytes(WEIGHTS)
    mmproj = source / "mmproj.gguf"
    mmproj.write_bytes(PROJECTOR)
    return model, mmproj


@pytest.fixture
def repo(tmp_path: Path) -> ModelRepository:
    return ModelRepository(LocalObjectStore(tmp_path / "bucket"), prefix="docie")


def _publish(repo: ModelRepository, local: tuple[Path, Path], **overrides: object):
    model, mmproj = local
    kwargs: dict = {
        "name": "nuextract3",
        "family": "nuextract3",
        "kind": "gguf",
        "files": [("model", "model.gguf", model), ("mmproj", "mmproj.gguf", mmproj)],
        "source": "hf:numind/NuExtract-2.0-2B-GGUF:Q4_K_M",
        "published_by": "ovh-cpu-01",
    }
    kwargs.update(overrides)
    return repo.publish(**kwargs)


# --- publication order ----------------------------------------------------


def test_a_published_model_resolves_to_a_complete_manifest(
    repo: ModelRepository, local: tuple[Path, Path]
) -> None:
    result = _publish(repo, local)

    digest = repo.resolve("nuextract3")
    manifest = repo.manifest("nuextract3", digest)

    assert digest == result.manifest_digest
    assert manifest.kind == "gguf"
    assert [f.path for f in manifest.files] == ["model.gguf", "mmproj.gguf"]
    assert manifest.total_bytes == len(WEIGHTS) + len(PROJECTOR)


def test_blobs_exist_before_the_pointer_does(
    tmp_path: Path, local: tuple[Path, Path]
) -> None:
    """The ordering property, observed rather than asserted about the code.

    A store that records the order of writes shows the pointer last -- which
    is what makes an interrupted publication leave no deployable model.
    """
    order: list[str] = []

    class Recording(LocalObjectStore):
        def put(self, key: str, payload: bytes) -> None:
            order.append(key)
            super().put(key, payload)

        def upload(self, source: Path, key: str) -> None:
            order.append(key)
            super().upload(source, key)

    repo = ModelRepository(Recording(tmp_path / "bucket"), prefix="docie")
    _publish(repo, local)

    assert order[-1].endswith("/current.json"), order
    assert all("blobs/sha256/" in key for key in order[:2]), order
    assert order[-2].endswith(".json")
    assert "current" not in order[-2]


def test_an_interrupted_publication_leaves_nothing_deployable(
    tmp_path: Path, local: tuple[Path, Path]
) -> None:
    """Garbage, never corruption. Blobs may be orphaned; the model simply does
    not exist, because nothing points at a manifest."""

    class FailsBeforePointer(LocalObjectStore):
        def put(self, key: str, payload: bytes) -> None:
            if key.endswith("current.json"):
                raise ModelRepositoryError("connection lost")
            super().put(key, payload)

    repo = ModelRepository(FailsBeforePointer(tmp_path / "bucket"), prefix="docie")

    with pytest.raises(ModelRepositoryError):
        _publish(repo, local)

    assert repo.published_names() == []
    with pytest.raises(ObjectNotFoundError):
        repo.resolve("nuextract3")


def test_orphaned_blobs_are_not_deleted_as_rollback(
    tmp_path: Path, local: tuple[Path, Path]
) -> None:
    """Deleting objects another publication may share is the dangerous move,
    so a failed publish cleans up nothing."""
    store = LocalObjectStore(tmp_path / "bucket")

    class FailsBeforePointer(LocalObjectStore):
        def put(self, key: str, payload: bytes) -> None:
            if key.endswith("current.json"):
                raise ModelRepositoryError("connection lost")
            super().put(key, payload)

    repo = ModelRepository(FailsBeforePointer(tmp_path / "bucket"), prefix="docie")
    with pytest.raises(ModelRepositoryError):
        _publish(repo, local)

    blobs = [k for k in store.list_prefix("docie/blobs") if k.endswith(("0", *"123456789abcdef"))]
    assert blobs, "the uploaded blobs should still be there"


# --- deduplication --------------------------------------------------------


def test_republishing_unchanged_weights_uploads_nothing(
    repo: ModelRepository, local: tuple[Path, Path]
) -> None:
    first = _publish(repo, local)
    second = _publish(repo, local)

    assert first.uploaded == ("model.gguf", "mmproj.gguf")
    assert second.uploaded == ()
    assert second.skipped == ("model.gguf", "mmproj.gguf")
    assert second.already_complete is True


def test_changing_one_file_uploads_only_that_file(
    repo: ModelRepository, local: tuple[Path, Path]
) -> None:
    """Why content-addressing earns its place: a new quant of a vision model
    re-uploads the weights, not the projector."""
    model, mmproj = local
    _publish(repo, local)
    model.write_bytes(b"GGUF-model-weights-v2")

    second = _publish(repo, (model, mmproj))

    assert second.uploaded == ("model.gguf",)
    assert second.skipped == ("mmproj.gguf",)


def test_two_models_sharing_a_projector_store_it_once(
    repo: ModelRepository, local: tuple[Path, Path]
) -> None:
    model, mmproj = local
    _publish(repo, local)

    second = _publish(repo, local, name="nuextract3-q8")

    assert second.skipped == ("model.gguf", "mmproj.gguf")
    assert sorted(repo.published_names()) == ["nuextract3", "nuextract3-q8"]


def test_republishing_moves_the_pointer_but_keeps_the_old_manifest(
    repo: ModelRepository, local: tuple[Path, Path]
) -> None:
    """Immutability is what lets a deployment pin a build: the manifest a
    running node resolved stays fetchable after someone republishes."""
    model, mmproj = local
    first = _publish(repo, local)
    model.write_bytes(b"GGUF-model-weights-v2")
    second = _publish(repo, (model, mmproj))

    assert first.manifest_digest != second.manifest_digest
    assert repo.resolve("nuextract3") == second.manifest_digest
    # The old build is still resolvable by digest.
    assert repo.manifest("nuextract3", first.manifest_digest).name == "nuextract3"


# --- what counts as published ---------------------------------------------


def test_a_blob_without_a_pointer_is_not_a_model(
    repo: ModelRepository, tmp_path: Path
) -> None:
    """The decided rule: arbitrary files uploaded to a bucket must not become
    deployable."""
    store = LocalObjectStore(tmp_path / "bucket")
    store.put("docie/blobs/sha256/" + "a" * 64, b"some bytes")

    assert repo.published_names() == []
    with pytest.raises(ObjectNotFoundError):
        repo.resolve("whatever")


def test_resolving_an_unpublished_model_says_so(repo: ModelRepository) -> None:
    with pytest.raises(ObjectNotFoundError, match="not published"):
        repo.resolve("never-seen")


def test_a_manifest_naming_a_different_model_is_refused(
    repo: ModelRepository, local: tuple[Path, Path], tmp_path: Path
) -> None:
    """A pointer and a manifest that disagree would serve one model under
    another's name."""
    result = _publish(repo, local)
    store = LocalObjectStore(tmp_path / "bucket")
    payload = store.get(result.manifest_key)
    store.put(
        "docie/models/other/" + result.manifest_digest.split(":")[1] + ".json", payload
    )

    with pytest.raises(ModelRepositoryError, match="names"):
        repo.manifest("other", result.manifest_digest)


# --- materialisation ------------------------------------------------------


def test_fetching_a_blob_verifies_it(
    repo: ModelRepository, local: tuple[Path, Path], tmp_path: Path
) -> None:
    _publish(repo, local)
    manifest = repo.manifest("nuextract3", repo.resolve("nuextract3"))
    destination = tmp_path / "cache" / "model.gguf"

    repo.fetch_blob(manifest.files[0], destination)

    assert destination.read_bytes() == WEIGHTS


def test_a_corrupted_object_never_lands_under_the_real_name(
    repo: ModelRepository, local: tuple[Path, Path], tmp_path: Path
) -> None:
    """A corrupt object that a runtime mmaps fails like a bad model rather
    than a bad download, which is why verification is not optional."""
    _publish(repo, local)
    manifest = repo.manifest("nuextract3", repo.resolve("nuextract3"))
    entry = manifest.files[0]
    # Corrupt the stored object in place.
    store = LocalObjectStore(tmp_path / "bucket")
    store.put("docie/" + entry.blob_key, b"not the weights you are looking for")
    destination = tmp_path / "cache" / "model.gguf"

    with pytest.raises(ModelRepositoryError, match="integrity check failed"):
        repo.fetch_blob(entry, destination)

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()


def test_an_interrupted_download_leaves_no_apparently_valid_entry(
    tmp_path: Path, local: tuple[Path, Path]
) -> None:
    class DropsMidTransfer(LocalObjectStore):
        def download(self, key: str, destination: Path) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(WEIGHTS[:4])  # a short, plausible file
            raise ModelRepositoryError("connection reset")

    repo = ModelRepository(LocalObjectStore(tmp_path / "bucket"), prefix="docie")
    _publish(repo, local)
    manifest = repo.manifest("nuextract3", repo.resolve("nuextract3"))
    broken = ModelRepository(DropsMidTransfer(tmp_path / "bucket"), prefix="docie")
    destination = tmp_path / "cache" / "model.gguf"

    with pytest.raises(ModelRepositoryError):
        broken.fetch_blob(manifest.files[0], destination)

    assert not destination.exists()
    assert list(destination.parent.glob("*.partial")) == []


def test_a_missing_object_is_reported_as_missing(
    repo: ModelRepository, local: tuple[Path, Path], tmp_path: Path
) -> None:
    """Distinct from a corrupt one: it calls for re-publishing, not retrying."""
    _publish(repo, local)
    manifest = repo.manifest("nuextract3", repo.resolve("nuextract3"))
    entry = manifest.files[0]
    (tmp_path / "bucket" / "docie" / entry.blob_key).unlink()

    with pytest.raises(ObjectNotFoundError):
        repo.fetch_blob(entry, tmp_path / "cache" / "model.gguf")


# --- the local-entry mapping ----------------------------------------------


def test_a_gguf_entry_maps_to_model_and_projector(local: tuple[Path, Path]) -> None:
    model, mmproj = local

    class Entry:
        model_path = model
        mmproj_path = mmproj

    assert [(r, p) for r, p, _ in iter_store_files(Entry())] == [
        ("model", "model.gguf"),
        ("mmproj", "mmproj.gguf"),
    ]


def test_a_snapshot_entry_maps_every_file_with_its_relative_path(
    tmp_path: Path,
) -> None:
    """`from_pretrained` loads a directory, so the layout inside it is part of
    the artifact."""
    root = tmp_path / "gliner" / "snapshot"
    (root / "tokenizer").mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"w")
    (root / "config.json").write_bytes(b"{}")
    (root / "tokenizer" / "tokenizer.json").write_bytes(b"{}")

    class Entry:
        model_path = root
        mmproj_path = None

    mapped = [(r, p) for r, p, _ in iter_store_files(Entry())]

    assert mapped == [
        ("snapshot", "config.json"),
        ("snapshot", "model.safetensors"),
        ("snapshot", "tokenizer/tokenizer.json"),
    ]


def test_a_snapshot_publishes_and_round_trips(
    repo: ModelRepository, tmp_path: Path
) -> None:
    root = tmp_path / "gliner" / "snapshot"
    (root / "tokenizer").mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"w")
    (root / "tokenizer" / "tokenizer.json").write_bytes(b"{}")

    class Entry:
        model_path = root
        mmproj_path = None

    repo.publish(
        name="gliner-multi-pii",
        family="encoder",
        kind="snapshot",
        files=list(iter_store_files(Entry())),
    )
    manifest = repo.manifest("gliner-multi-pii", repo.resolve("gliner-multi-pii"))

    assert [f.path for f in manifest.files] == [
        "model.safetensors",
        "tokenizer/tokenizer.json",
    ]


# --- helpers --------------------------------------------------------------


def test_sha256_file_reads_once_and_reports_both(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    path.write_bytes(WEIGHTS)

    digest, size = sha256_file(path)

    assert digest == "sha256:" + hashlib.sha256(WEIGHTS).hexdigest()
    assert size == len(WEIGHTS)


def test_a_prefix_namespaces_everything(
    tmp_path: Path, local: tuple[Path, Path]
) -> None:
    """So one bucket can hold several fleets without collision."""
    repo = ModelRepository(LocalObjectStore(tmp_path / "bucket"), prefix="fleet-a")
    _publish(repo, local)
    store = LocalObjectStore(tmp_path / "bucket")

    assert all(key.startswith("fleet-a/") for key in store.list_prefix(""))


def test_a_key_cannot_escape_the_local_store_root(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "bucket")

    with pytest.raises(ModelRepositoryError, match="escapes"):
        store.put("../outside.json", b"x")


def test_the_manifest_key_is_derived_from_its_own_bytes(
    repo: ModelRepository, local: tuple[Path, Path]
) -> None:
    result = _publish(repo, local)
    manifest = repo.manifest("nuextract3", result.manifest_digest)
    recomputed = "sha256:" + hashlib.sha256(manifest.to_json()).hexdigest()

    assert recomputed == result.manifest_digest
    assert result.manifest_key.endswith(
        manifest_mod.MODEL_PREFIX.split("/")[-1]
        + "/nuextract3/"
        + result.manifest_digest.split(":")[1]
        + ".json"
    )

# --- what the review found ------------------------------------------------


def test_a_substituted_manifest_is_caught_before_it_is_trusted(
    repo: ModelRepository, local: tuple[Path, Path], tmp_path: Path
) -> None:
    """The dangerous half of the integrity story.

    Blobs were verified from the start; manifests were not. `fetch_blob`
    verifies each blob against the manifest it was handed, so a substituted
    manifest passes every check underneath it and the node serves a different
    model than its pinned digest named. The digest is now checked on read.
    """
    result = _publish(repo, local)
    store = LocalObjectStore(tmp_path / "bucket")
    # Replace the object at the pinned key with a different, VALID manifest.
    other = _publish(repo, local, name="nuextract3", family="other-family")
    store.put(result.manifest_key, store.get(other.manifest_key))

    with pytest.raises(ModelRepositoryError, match="hashes to"):
        repo.manifest("nuextract3", result.manifest_digest)


def test_a_manifest_that_does_match_its_digest_is_accepted(
    repo: ModelRepository, local: tuple[Path, Path]
) -> None:
    """The negative half: the check must not refuse honest manifests."""
    result = _publish(repo, local)

    assert repo.manifest("nuextract3", result.manifest_digest).name == "nuextract3"


def test_a_malformed_pinned_digest_is_an_actionable_error(
    repo: ModelRepository,
) -> None:
    """A deployment intent carries this value. A truncated row or a digest
    written as bare hex used to die on split(':')[1] with a bare IndexError."""
    from docie_bench.serving.manifest import InvalidManifestError

    with pytest.raises((ModelRepositoryError, InvalidManifestError), match="canonical"):
        repo.manifest("nuextract3", "deadbeef")


def test_nothing_is_uploaded_when_the_manifest_would_be_invalid(
    tmp_path: Path, local: tuple[Path, Path]
) -> None:
    """Validation moved ahead of the upload loop.

    Building the manifest afterwards meant an unsafe path or a duplicate entry
    failed only once a multi-gigabyte transfer had finished — costing the whole
    upload and leaving orphaned blobs behind.
    """
    uploads: list[str] = []

    class Counting(LocalObjectStore):
        def upload(self, source: Path, key: str) -> None:
            uploads.append(key)
            super().upload(source, key)

    repo = ModelRepository(Counting(tmp_path / "bucket"), prefix="docie")
    model, _ = local

    with pytest.raises(Exception, match="safe relative path"):
        repo.publish(
            name="nuextract3",
            family="nuextract3",
            kind="gguf",
            files=[("model", "../escape.gguf", model)],
        )

    assert uploads == []


def test_a_retry_after_an_interrupted_publish_is_not_a_no_op(
    tmp_path: Path, local: tuple[Path, Path]
) -> None:
    """`already_complete` was `not uploaded`, so the retry that finally makes a
    model visible reported itself as having changed nothing — every blob was
    uploaded before the crash, so every blob is skipped."""

    class FailsBeforePointer(LocalObjectStore):
        fail = True

        def put(self, key: str, payload: bytes) -> None:
            if self.fail and key.endswith("current.json"):
                raise ModelRepositoryError("connection lost")
            super().put(key, payload)

    store = FailsBeforePointer(tmp_path / "bucket")
    repo = ModelRepository(store, prefix="docie")
    with pytest.raises(ModelRepositoryError):
        _publish(repo, local)

    store.fail = False
    retry = _publish(repo, local)

    assert retry.uploaded == ()  # they were all uploaded before the crash
    assert retry.already_complete is False  # but the model is new to the fleet
    assert repo.resolve("nuextract3") == retry.manifest_digest


def test_republishing_an_unchanged_model_really_is_a_no_op(
    repo: ModelRepository, local: tuple[Path, Path]
) -> None:
    """The other side of it, so the field still means what it says."""
    _publish(repo, local)

    second = _publish(repo, local)

    assert second.already_complete is True


def test_a_local_upload_that_dies_leaves_no_blob_under_the_digest(
    tmp_path: Path
) -> None:
    """Presence is taken as proof of content, so a truncated blob at a
    canonical key is permanent: every later publish skips it and every fetch
    fails its integrity check, with no way to self-heal."""
    store = LocalObjectStore(tmp_path / "bucket")
    source = tmp_path / "big.gguf"
    source.write_bytes(b"x" * 64)
    key = "blobs/sha256/" + "a" * 64

    import shutil as shutil_mod

    original = shutil_mod.copyfile

    def dies(src: object, dst: object, **kwargs: object) -> None:
        original(src, dst, **kwargs)  # type: ignore[arg-type]
        raise OSError("disk full")

    import docie_bench.serving.model_repository as repo_mod

    repo_mod.shutil.copyfile = dies  # type: ignore[assignment]
    try:
        with pytest.raises(OSError, match="disk full"):
            store.upload(source, key)
    finally:
        repo_mod.shutil.copyfile = original  # type: ignore[assignment]

    assert not store.exists(key)
    assert list((tmp_path / "bucket" / "blobs" / "sha256").glob("*")) == []


def test_concurrent_fetches_do_not_share_one_temp_file(
    repo: ModelRepository, local: tuple[Path, Path], tmp_path: Path
) -> None:
    """A fixed '.partial' name meant two pulls of the same model on one node
    wrote to the same file — each hashing bytes the other was still writing,
    and the first one's cleanup deleting the second's transfer."""
    _publish(repo, local)
    manifest = repo.manifest("nuextract3", repo.resolve("nuextract3"))
    entry = manifest.files[0]
    seen: list[str] = []

    class RecordingTemp(LocalObjectStore):
        def download(self, key: str, destination: Path) -> None:
            seen.append(destination.name)
            super().download(key, destination)

    watching = ModelRepository(RecordingTemp(tmp_path / "bucket"), prefix="docie")
    watching.fetch_blob(entry, tmp_path / "cache" / "model.gguf")
    watching.fetch_blob(entry, tmp_path / "cache2" / "model.gguf")

    assert len(set(seen)) == 2, f"both pulls used the same temp name: {seen}"

# --- the digest is a content address --------------------------------------


def _advancing_clock(monkeypatch: pytest.MonkeyPatch, *, step_s: int = 30) -> None:
    """Make every manifest built in this test carry a different ``created_at``.

    The real flake needed two publications to straddle a one-second boundary,
    which happens on roughly one run in N. Stepping the clock makes the same
    condition certain.
    """
    import docie_bench.serving.model_repository as repo_mod

    real_build = manifest_mod.build
    clock = [dt.datetime(2026, 10, 28, 12, 0, 0, tzinfo=dt.UTC)]

    def advancing(**kwargs: object):
        kwargs["now"] = clock[0]
        clock[0] = clock[0] + dt.timedelta(seconds=step_s)
        return real_build(**kwargs)

    monkeypatch.setattr(repo_mod.manifest_mod, "build", advancing)


def test_republishing_unchanged_weights_keeps_one_digest_across_a_clock_tick(
    repo: ModelRepository, local: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``created_at`` is in the manifest document, so it is in the manifest's
    digest. Building a fresh manifest for unchanged weights therefore minted a
    new digest, wrote a near-duplicate manifest and repointed current.json --
    purely because time passed. A digest that is not a function of content is
    not a content address.
    """
    _advancing_clock(monkeypatch)

    first = _publish(repo, local)
    second = _publish(repo, local)

    assert second.manifest_digest == first.manifest_digest
    assert second.already_complete is True
    assert repo.resolve("nuextract3") == first.manifest_digest


def test_one_manifest_object_is_written_however_often_it_is_republished(
    repo: ModelRepository, local: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Otherwise every republish litters ``models/<name>/`` with another
    near-identical manifest."""
    _advancing_clock(monkeypatch)
    for _ in range(3):
        _publish(repo, local)

    manifests = sorted(
        p.name for p in (tmp_path / "bucket" / "docie" / "models" / "nuextract3").iterdir()
    )

    assert len(manifests) == 2, manifests  # one manifest + current.json
    assert "current.json" in manifests


def test_two_nodes_publishing_the_same_weights_agree_on_one_digest(
    repo: ModelRepository, local: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance (``source``, ``published_by``) is deliberately not part of
    the comparison: identical bytes must resolve to one content address, and
    the manifest keeps the first publication's provenance."""
    _advancing_clock(monkeypatch)

    first = _publish(repo, local, published_by="ovh-cpu-01")
    second = _publish(repo, local, published_by="ovh-cpu-02")

    assert second.manifest_digest == first.manifest_digest
    assert repo.manifest("nuextract3", first.manifest_digest).published_by == "ovh-cpu-01"


def test_changed_weights_still_publish_a_new_digest(
    repo: ModelRepository, local: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reuse must be content-gated, not unconditional."""
    _advancing_clock(monkeypatch)
    first = _publish(repo, local)
    model, _mmproj = local
    model.write_bytes(b"GGUF-model-weights-v2")

    second = _publish(repo, local)

    assert second.manifest_digest != first.manifest_digest
    assert second.already_complete is False
    assert repo.resolve("nuextract3") == second.manifest_digest


def test_a_changed_family_publishes_a_new_digest(
    repo: ModelRepository, local: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Family drives how the model is served, so it is content, not provenance."""
    _advancing_clock(monkeypatch)
    first = _publish(repo, local)

    second = _publish(repo, local, family="openai_chat")

    assert second.manifest_digest != first.manifest_digest


def test_a_blob_deleted_under_a_published_model_is_restored_and_reported(
    repo: ModelRepository, local: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reuse must not short-circuit the blob loop: the manifest can be right
    while a blob has gone missing, and that republish is not a no-op."""
    _advancing_clock(monkeypatch)
    first = _publish(repo, local)
    manifest = repo.manifest("nuextract3", first.manifest_digest)
    missing = tmp_path / "bucket" / "docie" / manifest.files[0].blob_key
    missing.unlink()

    second = _publish(repo, local)

    assert second.manifest_digest == first.manifest_digest
    assert second.uploaded == (manifest.files[0].path,)
    assert second.already_complete is False  # it had to repair something
    assert missing.is_file()
