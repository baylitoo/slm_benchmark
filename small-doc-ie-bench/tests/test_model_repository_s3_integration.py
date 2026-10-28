"""The S3 adapter against a REAL S3-compatible server (#595 2b).

``test_model_repository.py`` proves the repository's rules against a
directory. Those rules are the interesting part, but a directory cannot prove
the thing a bucket actually breaks on: how ``botocore`` reports a missing key
versus a denied one, whether a multipart upload round-trips byte-for-byte,
whether ``head_object`` is the right existence probe. Mocks assert what we
already believe; this asserts what the server does.

**Opt-in.** Skipped unless ``DOCIE_TEST_S3_ENDPOINT_URL`` is set, so CI and a
normal ``pytest`` run are unaffected. The variables are deliberately distinct
from the production ``DOCIE_MODEL_STORE_S3_*`` names: pointing this file at a
real bucket would have it create and delete objects there.

To run it::

    docker compose --profile minio up -d minio
    export DOCIE_TEST_S3_ENDPOINT_URL=http://localhost:9000
    export DOCIE_TEST_S3_BUCKET=docie-test
    export AWS_ACCESS_KEY_ID=docieminio
    export AWS_SECRET_ACCESS_KEY=docieminio
    export AWS_DEFAULT_REGION=us-east-1
    pytest tests/test_model_repository_s3_integration.py -v

No production bucket or credentials are needed; the compose profile brings up
a throwaway MinIO with those literal credentials.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from docie_bench.serving import manifest as manifest_mod
from docie_bench.serving.model_repository import (
    ModelRepository,
    ModelRepositoryError,
    ObjectNotFoundError,
    S3ObjectStore,
)

ENDPOINT = os.getenv("DOCIE_TEST_S3_ENDPOINT_URL", "")
BUCKET = os.getenv("DOCIE_TEST_S3_BUCKET", "docie-test")

pytestmark = pytest.mark.skipif(
    not ENDPOINT,
    reason="set DOCIE_TEST_S3_ENDPOINT_URL (see this module's docstring) to run",
)

WEIGHTS = b"GGUF-model-weights-" + b"x" * 4096
PROJECTOR = b"vision-projector-" + b"y" * 2048


def _client(**overrides: str):
    boto3 = pytest.importorskip("boto3", reason="the 's3' extra is not installed")
    kwargs = {
        "endpoint_url": ENDPOINT,
        "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }
    kwargs.update(overrides)
    return boto3.client("s3", **kwargs)


@pytest.fixture(scope="module")
def bucket() -> str:
    """Make sure the bucket exists; MinIO starts empty."""
    client = _client()
    try:
        client.head_bucket(Bucket=BUCKET)
    except Exception:  # noqa: BLE001 - absent is the normal first run
        client.create_bucket(Bucket=BUCKET)
    return BUCKET


@pytest.fixture
def repo(bucket: str) -> Iterator[ModelRepository]:
    """A repository under a prefix unique to this test, cleaned up after.

    Per-test prefix rather than per-test bucket: it keeps the tests independent
    without needing bucket-creation rights for each one, and it exercises the
    prefix handling the real deployment uses.
    """
    prefix = f"it-{uuid.uuid4().hex[:12]}"
    store = S3ObjectStore(bucket, endpoint_url=ENDPOINT)
    yield ModelRepository(store, prefix=prefix)
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    keys = [
        {"Key": item["Key"]}
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for item in page.get("Contents", [])
    ]
    for batch in (keys[i : i + 1000] for i in range(0, len(keys), 1000)):
        client.delete_objects(Bucket=bucket, Delete={"Objects": batch})


@pytest.fixture
def gguf(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "store" / "nuextract3"
    source.mkdir(parents=True)
    model = source / "model.gguf"
    model.write_bytes(WEIGHTS)
    mmproj = source / "mmproj.gguf"
    mmproj.write_bytes(PROJECTOR)
    return model, mmproj


def _publish_gguf(repo: ModelRepository, files: tuple[Path, Path]):
    model, mmproj = files
    return repo.publish(
        name="nuextract3",
        family="nuextract3",
        kind="gguf",
        files=[("model", "model.gguf", model), ("mmproj", "mmproj.gguf", mmproj)],
        source="hf:numind/NuExtract-2.0-2B-GGUF:Q4_K_M",
        published_by="integration-test",
    )


# --- publication round-trips through a real server -------------------------


def test_a_gguf_publishes_and_resolves(repo: ModelRepository, gguf: tuple[Path, Path]) -> None:
    result = _publish_gguf(repo, gguf)

    assert repo.resolve("nuextract3") == result.manifest_digest
    fetched = repo.manifest("nuextract3", result.manifest_digest)
    assert sorted(entry.path for entry in fetched.files) == ["mmproj.gguf", "model.gguf"]
    assert fetched.total_bytes == len(WEIGHTS) + len(PROJECTOR)


def test_blobs_round_trip_byte_for_byte(
    repo: ModelRepository, gguf: tuple[Path, Path], tmp_path: Path
) -> None:
    """What a directory cannot prove: the bytes survive a real upload and
    download, and ``fetch_blob``'s digest check passes against them."""
    result = _publish_gguf(repo, gguf)
    fetched = repo.manifest("nuextract3", result.manifest_digest)

    for entry in fetched.files:
        destination = tmp_path / "cache" / entry.path
        repo.fetch_blob(entry, destination)
        expected = WEIGHTS if entry.path == "model.gguf" else PROJECTOR
        assert destination.read_bytes() == expected


def test_a_multi_file_snapshot_publishes_with_its_tree(
    repo: ModelRepository, tmp_path: Path
) -> None:
    """Snapshot families are a DIRECTORY, including dotfiles, which is what
    broke the first path rule."""
    source = tmp_path / "snap"
    (source / "tokenizer").mkdir(parents=True)
    (source / "config.json").write_bytes(b'{"a": 1}')
    (source / "model.safetensors").write_bytes(b"s" * 2048)
    (source / "tokenizer" / "tokenizer.json").write_bytes(b'{"t": 2}')
    (source / ".gitattributes").write_bytes(b"* text=auto\n")
    files = [
        ("snapshot", str(path.relative_to(source)).replace(os.sep, "/"), path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]

    result = repo.publish(
        name="gliner-multi", family="gliner", kind="snapshot", files=files
    )

    fetched = repo.manifest("gliner-multi", result.manifest_digest)
    assert sorted(entry.path for entry in fetched.files) == [
        ".gitattributes",
        "config.json",
        "model.safetensors",
        "tokenizer/tokenizer.json",
    ]


def test_republishing_unchanged_weights_uploads_nothing(
    repo: ModelRepository, gguf: tuple[Path, Path]
) -> None:
    """Content addressing against a real server: existing blobs are skipped
    and the manifest digest does not move."""
    first = _publish_gguf(repo, gguf)

    second = _publish_gguf(repo, gguf)

    assert second.uploaded == ()
    assert sorted(second.skipped) == ["mmproj.gguf", "model.gguf"]
    assert second.manifest_digest == first.manifest_digest
    assert second.already_complete is True


# --- how the server reports failure ---------------------------------------


def test_a_missing_object_raises_object_not_found(repo: ModelRepository) -> None:
    """The mapping a directory store cannot exercise: botocore reports this as
    a NoSuchKey/404 ClientError, and the repository has to narrow it."""
    with pytest.raises(ObjectNotFoundError):
        repo.resolve("never-published")


def test_an_unpublished_name_is_absent_rather_than_an_error(
    repo: ModelRepository,
) -> None:
    """``exists`` must treat absence as an answer, via head_object's 404."""
    assert "never-published" not in repo.published_names()


def test_a_substituted_blob_fails_its_digest_check(
    repo: ModelRepository, gguf: tuple[Path, Path], tmp_path: Path, bucket: str
) -> None:
    """Integrity end to end: overwrite a blob at its content-addressed key and
    the fetch must refuse it rather than serve the wrong weights."""
    result = _publish_gguf(repo, gguf)
    fetched = repo.manifest("nuextract3", result.manifest_digest)
    entry = next(e for e in fetched.files if e.path == "model.gguf")
    # Locate the blob through the store's own listing rather than rebuilding
    # the key, so this test does not encode the prefix layout.
    hex_tail = manifest_mod.digest_hex(entry.digest)
    listing = S3ObjectStore(bucket, endpoint_url=ENDPOINT).list_prefix("")
    blobs = [key for key in listing if hex_tail in key]
    assert blobs, "the published blob should be listable"
    _client().put_object(Bucket=bucket, Key=blobs[0], Body=b"not-the-weights")

    with pytest.raises(ModelRepositoryError, match="hashes to"):
        repo.fetch_blob(entry, tmp_path / "cache" / "model.gguf")


def test_bad_credentials_surface_as_a_repository_error(bucket: str) -> None:
    """Denied access must not look like "not published"."""
    store = S3ObjectStore(
        bucket,
        client=_client(
            aws_access_key_id="wrong",
            aws_secret_access_key="also-wrong",  # noqa: S106 - wrong on purpose
        ),
    )
    repo = ModelRepository(store, prefix="it-denied")

    with pytest.raises(ModelRepositoryError):
        repo.published_names()
