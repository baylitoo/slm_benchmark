"""What makes a model deployable from the shared repository (#595 stage 2).

The decided rule is that the manifest, not the presence of bytes, is what
makes a model exist — *"arbitrary files uploaded to a bucket must not
automatically become deployable models"*. So every rule about validity is a
rule about what will and will not be served, and belongs in a table test
rather than in a bucket.

Two properties carry most of the weight:

* a manifest that would **publish cleanly and fail to load** must be refused at
  construction, because its presence in the repository is a promise;
* serialisation must be **canonical**, since the digest of the bytes is what
  names the manifest — two publications of identical content landing under two
  keys would break deduplication and pinning at once.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest

from docie_bench.serving import manifest as m


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


MODEL_DIGEST = digest(b"weights")
PROJ_DIGEST = digest(b"projector")


def gguf(**overrides: object) -> m.Manifest:
    kwargs: dict = {
        "name": "nuextract3",
        "family": "nuextract3",
        "kind": "gguf",
        "files": [
            m.ManifestFile("model", "model.gguf", MODEL_DIGEST, 1690),
            m.ManifestFile("mmproj", "mmproj.gguf", PROJ_DIGEST, 104),
        ],
        "source": "hf:numind/NuExtract-2.0-2B-GGUF:Q4_K_M",
        "published_by": "ovh-cpu-01",
    }
    kwargs.update(overrides)
    return m.build(**kwargs)  # type: ignore[arg-type]


def snapshot(**overrides: object) -> m.Manifest:
    kwargs: dict = {
        "name": "gliner-multi-pii",
        "family": "encoder",
        "kind": "snapshot",
        "files": [
            m.ManifestFile("snapshot", "model.safetensors", MODEL_DIGEST, 900),
            m.ManifestFile("snapshot", "config.json", PROJ_DIGEST, 12),
        ],
    }
    kwargs.update(overrides)
    return m.build(**kwargs)  # type: ignore[arg-type]


# --- the layout -----------------------------------------------------------


def test_blobs_are_addressed_by_content_not_by_model() -> None:
    """Why a re-publication uploads only what changed, and why two models
    sharing a projector store it once."""
    shared = m.ManifestFile("mmproj", "mmproj.gguf", PROJ_DIGEST, 104)
    other = m.ManifestFile("mmproj", "vision.gguf", PROJ_DIGEST, 104)

    assert shared.blob_key == other.blob_key
    assert shared.blob_key.startswith("blobs/sha256/")
    assert "nuextract3" not in shared.blob_key


def test_a_manifest_is_keyed_by_its_own_digest() -> None:
    """Immutability: re-publishing a name writes a NEW object, so a pull in
    flight keeps resolving the bytes it started with."""
    manifest = gguf()
    key = manifest.manifest_key(digest(manifest.to_json()))

    assert key.startswith("models/nuextract3/")
    assert key.endswith(".json")
    assert "current" not in key


def test_the_pointer_is_the_only_mutable_object() -> None:
    assert m.pointer_key("nuextract3") == "models/nuextract3/current.json"


def test_the_pointer_names_a_digest_rather_than_describing_the_model() -> None:
    """What lets a deployment pin a build: the intent records the digest, so a
    later publication cannot silently change what a running node serves."""
    manifest = gguf()
    wanted = digest(manifest.to_json())

    assert m.parse_pointer(m.pointer_document(wanted)) == wanted


def test_a_pointer_from_a_newer_publisher_is_refused_not_guessed() -> None:
    document = json.dumps({"schema": m.SCHEMA_VERSION + 1, "manifest": MODEL_DIGEST})

    with pytest.raises(m.InvalidManifestError, match="schema"):
        m.parse_pointer(document)


def test_a_pointer_that_is_not_a_digest_is_refused() -> None:
    document = json.dumps({"schema": m.SCHEMA_VERSION, "manifest": "latest"})

    with pytest.raises(m.InvalidManifestError, match="digest"):
        m.parse_pointer(document)


# --- canonical bytes ------------------------------------------------------


def test_serialisation_is_canonical_so_identical_content_has_one_key() -> None:
    """The digest of the bytes NAMES the manifest. If field order could vary,
    the same model would land under two keys and deduplication would silently
    stop working."""
    fixed = dt.datetime(2026, 10, 27, 15, 4, 5, tzinfo=dt.UTC)
    first = gguf(now=fixed)
    second = gguf(now=fixed)

    assert first.to_json() == second.to_json()
    assert digest(first.to_json()) == digest(second.to_json())


def test_a_manifest_round_trips_without_loss() -> None:
    original = gguf()

    assert m.parse(original.to_json()).to_json() == original.to_json()


def test_fields_a_newer_publisher_added_survive_a_round_trip() -> None:
    """An older reader must not silently destroy what it does not understand,
    or a round-trip through one node would strip another's metadata."""
    document = json.loads(gguf().to_json())
    document["published_region"] = "eu-west-3"

    reparsed = m.parse(json.dumps(document))

    assert reparsed.extra["published_region"] == "eu-west-3"
    assert json.loads(reparsed.to_json())["published_region"] == "eu-west-3"


def test_total_bytes_is_written_as_well_as_derivable() -> None:
    """Redundant deliberately: the disk-headroom check runs before a download
    and should refuse without parsing every entry."""
    manifest = gguf()

    assert json.loads(manifest.to_json())["total_bytes"] == 1690 + 104
    assert manifest.total_bytes == 1794


# --- what is refused ------------------------------------------------------


def test_a_manifest_with_no_files_describes_nothing_servable() -> None:
    with pytest.raises(m.InvalidManifestError, match="no files|nothing"):
        gguf(files=[])


def test_a_gguf_manifest_without_its_weights_is_refused() -> None:
    """It would publish cleanly and fail to load — exactly the failure the
    manifest exists to prevent."""
    with pytest.raises(m.InvalidManifestError, match="no 'model' file"):
        gguf(files=[m.ManifestFile("mmproj", "mmproj.gguf", PROJ_DIGEST, 104)])


def test_a_snapshot_described_as_a_single_gguf_is_refused() -> None:
    with pytest.raises(m.InvalidManifestError, match="non-snapshot"):
        snapshot(files=[m.ManifestFile("model", "model.gguf", MODEL_DIGEST, 1)])


def test_a_gguf_manifest_listing_snapshot_files_is_refused() -> None:
    with pytest.raises(m.InvalidManifestError, match="snapshot files"):
        gguf(
            files=[
                m.ManifestFile("model", "model.gguf", MODEL_DIGEST, 1),
                m.ManifestFile("snapshot", "config.json", PROJ_DIGEST, 1),
            ]
        )


@pytest.mark.parametrize(
    "path",
    ["../escape.gguf", "a/../../b.gguf", "/absolute.gguf", ".hidden", "", "a\\b"],
)
def test_a_file_path_that_leaves_the_model_directory_is_refused(path: str) -> None:
    """The path is joined onto a local directory when materialising, so a
    manifest must not be able to write outside the model it describes — the
    same class as the store-name and log-path guards."""
    with pytest.raises(m.InvalidManifestError, match="safe relative path"):
        gguf(files=[m.ManifestFile("model", path, MODEL_DIGEST, 1)])


def test_a_nested_snapshot_path_is_allowed() -> None:
    """`from_pretrained` loads a directory and the layout inside it is part of
    the artifact, so subdirectories must survive."""
    manifest = snapshot(
        files=[
            m.ManifestFile("snapshot", "model.safetensors", MODEL_DIGEST, 1),
            m.ManifestFile("snapshot", "tokenizer/tokenizer.json", PROJ_DIGEST, 1),
        ]
    )

    assert manifest.files[1].path == "tokenizer/tokenizer.json"


@pytest.mark.parametrize(
    "bad", ["sha256:short", "md5:" + "0" * 32, "0" * 64, "sha256:" + "Z" * 64]
)
def test_a_file_without_a_sha256_digest_is_refused(bad: str) -> None:
    """The digest is the verified cache fill and the dedup key; a manifest
    that cannot be verified against is not a manifest."""
    with pytest.raises(m.InvalidManifestError, match="digest"):
        gguf(files=[m.ManifestFile("model", "model.gguf", bad, 1)])


def test_the_same_path_listed_twice_is_refused() -> None:
    with pytest.raises(m.InvalidManifestError, match="twice"):
        snapshot(
            files=[
                m.ManifestFile("snapshot", "config.json", MODEL_DIGEST, 1),
                m.ManifestFile("snapshot", "config.json", PROJ_DIGEST, 1),
            ]
        )


def test_an_unknown_role_is_refused() -> None:
    with pytest.raises(m.InvalidManifestError, match="role"):
        gguf(files=[m.ManifestFile("weights", "model.gguf", MODEL_DIGEST, 1)])


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(m.InvalidManifestError, match="kind"):
        gguf(kind="safetensors")


def test_a_manifest_without_a_family_is_refused() -> None:
    """The family is the launch contract. Without it a puller has bytes and no
    way to serve them."""
    with pytest.raises(m.InvalidManifestError, match="family"):
        gguf(family="")


def test_a_document_from_a_newer_schema_is_refused_not_guessed() -> None:
    document = json.loads(gguf().to_json())
    document["schema"] = m.SCHEMA_VERSION + 1

    with pytest.raises(m.InvalidManifestError, match="schema"):
        m.parse(json.dumps(document))


@pytest.mark.parametrize("raw", ["not json", "[]", '"a string"', "null"])
def test_a_document_that_is_not_an_object_is_refused(raw: str) -> None:
    with pytest.raises(m.InvalidManifestError):
        m.parse(raw)


def test_a_malformed_file_entry_names_itself() -> None:
    document = json.loads(gguf().to_json())
    document["files"] = [{"role": "model", "path": "model.gguf"}]  # no digest/size

    with pytest.raises(m.InvalidManifestError, match="malformed"):
        m.parse(json.dumps(document))


def test_created_at_is_utc_and_second_precision() -> None:
    """Stable across publications so a manifest's bytes depend on its content,
    not on how fast the clock was read."""
    manifest = gguf(now=dt.datetime(2026, 10, 27, 15, 4, 5, 123456, tzinfo=dt.UTC))

    assert manifest.created_at == "2026-10-27T15:04:05Z"
