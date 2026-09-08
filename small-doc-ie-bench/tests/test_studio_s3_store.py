"""S3-compatible Studio artifact store backend (``S3ArtifactBlobStore``).

Everything here runs against an in-memory fake of the five boto3 S3 client
calls the backend uses (``put_object`` / ``get_object`` / ``head_object`` /
``delete_object`` / ``get_paginator("list_objects_v2")``) — NO network, NO real
bucket, and boto3 itself is never required (the client is injected at the test
seam). The filesystem backend's behaviour is covered unchanged by
``test_studio_artifacts.py``; these tests prove the S3 backend honours the same
``BlobStoreBackend`` contract, including the GC mark-and-sweep semantics.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import sys
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import docie_bench.studio.store as store_module
from docie_bench.settings import Settings
from docie_bench.storage.db import Base
from docie_bench.studio.store import (
    ArtifactBlobStore,
    RunStore,
    S3ArtifactBlobStore,
    default_blob_store,
)

# ---------------------------------------------------------------------------
# In-memory stand-in for the boto3 S3 client surface the backend uses
# ---------------------------------------------------------------------------


class _FakeClientError(Exception):
    """Shape-compatible with botocore's ClientError (``.response["Error"]["Code"]``)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _FakePaginator:
    def __init__(self, client: FakeS3Client) -> None:
        self._client = client

    def paginate(self, *, Bucket: str, Prefix: str = ""):  # noqa: N803 - boto3 API shape
        contents = [
            {"Key": key, "LastModified": obj["last_modified"]}
            for (bucket, key), obj in sorted(self._client.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        if not contents:
            yield {}  # like real S3: an empty listing page has no "Contents" key
            return
        # Two entries per page so the pagination path is actually exercised.
        for start in range(0, len(contents), 2):
            yield {"Contents": contents[start : start + 2]}


class FakeS3Client:
    """Dict-backed fake of the S3 client calls ``S3ArtifactBlobStore`` makes."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}

    def put_object(  # noqa: N803 - boto3 API shape
        self, *, Bucket: str, Key: str, Body: bytes, ContentType: str = ""
    ) -> dict[str, Any]:
        self.objects[(Bucket, Key)] = {
            "body": bytes(Body),
            "content_type": ContentType,
            "last_modified": dt.datetime.now(dt.UTC),
        }
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        obj = self.objects.get((Bucket, Key))
        if obj is None:
            raise _FakeClientError("NoSuchKey")
        return {"Body": io.BytesIO(obj["body"])}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        obj = self.objects.get((Bucket, Key))
        if obj is None:
            raise _FakeClientError("404")
        return {"ContentLength": len(obj["body"]), "LastModified": obj["last_modified"]}

    def delete_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        self.objects.pop((Bucket, Key), None)
        return {}

    def get_paginator(self, operation_name: str) -> _FakePaginator:
        assert operation_name == "list_objects_v2"
        return _FakePaginator(self)

    # -- test helpers ------------------------------------------------------

    def age(self, bucket: str, key: str, *, hours: float) -> None:
        """Backdate an object's LastModified (the S3 analogue of ``os.utime``)."""
        obj = self.objects[(bucket, key)]
        obj["last_modified"] = obj["last_modified"] - dt.timedelta(hours=hours)


def _make_s3_run_store(
    db_path: Path, client: FakeS3Client, *, prefix: str = "studio"
) -> tuple[RunStore, sessionmaker[Session]]:
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    blobs = S3ArtifactBlobStore("artifacts", prefix=prefix, client=client)
    return RunStore(factory, blobs), factory


# ---------------------------------------------------------------------------
# Backend selection (default_blob_store)
# ---------------------------------------------------------------------------


def test_default_blob_store_is_filesystem_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(artifact_store_dir=Path("some/artifacts"))
    assert settings.artifact_store_backend == "filesystem"
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)
    store = default_blob_store()
    assert isinstance(store, ArtifactBlobStore)
    assert store.root == Path("some/artifacts")


def test_default_blob_store_selects_s3_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        artifact_store_backend="s3",
        artifact_store_s3_bucket="docie-artifacts",
        artifact_store_s3_endpoint_url="http://minio:9000",
        artifact_store_s3_prefix="/studio/",
    )
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)
    seen: dict[str, Any] = {}
    fake = FakeS3Client()

    def _fake_make_client(endpoint_url: str | None) -> FakeS3Client:
        seen["endpoint_url"] = endpoint_url
        return fake

    monkeypatch.setattr(store_module, "_make_s3_client", _fake_make_client)
    store = default_blob_store()
    assert isinstance(store, S3ArtifactBlobStore)
    assert store.bucket == "docie-artifacts"
    assert store.prefix == "studio"  # normalized: no stray slashes
    assert seen["endpoint_url"] == "http://minio:9000"
    # The store is live: a put lands under the configured bucket + prefix.
    blob = store.put(name="report.html", content=b"<ok/>", media_type="text/html")
    assert ("docie-artifacts", f"studio/{blob.relkey}") in fake.objects


def test_default_blob_store_s3_requires_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(artifact_store_backend="s3")
    monkeypatch.setattr(store_module, "get_settings", lambda: settings)
    with pytest.raises(RuntimeError, match="ARTIFACT_STORE_S3_BUCKET"):
        default_blob_store()


def test_s3_backend_without_boto3_names_the_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``None in sys.modules`` makes ``import boto3`` raise ImportError whether or
    # not boto3 is actually installed in this environment.
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(RuntimeError, match=r"pip install 'small-doc-ie-bench\[s3\]'"):
        S3ArtifactBlobStore("bucket")


# ---------------------------------------------------------------------------
# Blob contract: round-trip, dedup mtime refresh, key safety, prefix isolation
# ---------------------------------------------------------------------------


def test_s3_put_get_exists_delete_roundtrip() -> None:
    client = FakeS3Client()
    store = S3ArtifactBlobStore("bkt", prefix="studio", client=client)
    payload = b'{"row":1}\n'
    blob = store.put(name="predictions.jsonl", content=payload, media_type="application/x-ndjson")

    digest = hashlib.sha256(payload).hexdigest()
    assert blob.relkey == f"{digest[:2]}/{digest}/predictions.jsonl"  # same layout as filesystem
    assert blob.sha256 == digest
    assert blob.size_bytes == len(payload)

    assert store.exists(blob.relkey)
    assert store.read(blob.relkey) == payload
    assert store.modified_at(blob.relkey) is not None
    assert list(store.iter_keys()) == [blob.relkey]

    assert store.delete(blob.relkey) is True
    assert not store.exists(blob.relkey)
    assert store.delete(blob.relkey) is False  # idempotent
    assert store.modified_at(blob.relkey) is None
    with pytest.raises(FileNotFoundError):
        store.read(blob.relkey)


def test_s3_reput_of_existing_content_refreshes_last_modified() -> None:
    """Content-addressed dedup parity: a re-put must refresh the GC liveness signal."""
    client = FakeS3Client()
    store = S3ArtifactBlobStore("bkt", client=client)
    blob = store.put(name="metrics.json", content=b'{"m":1}')
    client.age("bkt", blob.relkey, hours=48)
    aged = store.modified_at(blob.relkey)
    assert aged is not None

    again = store.put(name="metrics.json", content=b'{"m":1}')
    assert again.relkey == blob.relkey  # identical bytes -> identical key
    refreshed = store.modified_at(blob.relkey)
    assert refreshed is not None
    assert refreshed > aged


def test_s3_rejects_traversal_and_malformed_keys() -> None:
    client = FakeS3Client()
    store = S3ArtifactBlobStore("bkt", prefix="studio", client=client)
    store.put(name="report.html", content=b"ok")
    for bad in ("../../etc/passwd", "/absolute", "a//b", "a/./b", "..", ""):
        assert store.exists(bad) is False
        assert store.delete(bad) is False
        assert store.modified_at(bad) is None
        with pytest.raises(ValueError, match="escapes the store root"):
            store.read(bad)
    with pytest.raises(ValueError, match="plain file name"):
        store.put(name="../evil.html", content=b"x")


def test_s3_prefix_isolation_within_a_shared_bucket() -> None:
    client = FakeS3Client()
    store_a = S3ArtifactBlobStore("bkt", prefix="tenant-a", client=client)
    store_b = S3ArtifactBlobStore("bkt", prefix="tenant-b", client=client)
    blob_a = store_a.put(name="report.html", content=b"<a/>")
    blob_b = store_b.put(name="report.html", content=b"<b/>")

    # Each store enumerates ONLY its own namespace...
    assert list(store_a.iter_keys()) == [blob_a.relkey]
    assert list(store_b.iter_keys()) == [blob_b.relkey]
    # ...and cannot see or reclaim the other's objects.
    assert not store_a.exists(blob_b.relkey)
    assert store_a.delete(blob_b.relkey) is False
    assert store_b.exists(blob_b.relkey)
    with pytest.raises(FileNotFoundError):
        store_a.read(blob_b.relkey)


def test_s3_iter_keys_paginates_and_skips_dot_leaves() -> None:
    client = FakeS3Client()
    store = S3ArtifactBlobStore("bkt", prefix="studio", client=client)
    blobs = {
        store.put(name=f"report-{i}.html", content=f"<r{i}/>".encode()).relkey
        for i in range(5)  # > one fake page (2 entries/page)
    }
    # A dot-leaf object (filesystem-temp-file parity) must not enumerate.
    client.put_object(Bucket="bkt", Key="studio/ab/abcd/.tmp-inflight", Body=b"x")
    assert set(store.iter_keys()) == blobs


# ---------------------------------------------------------------------------
# GC on the S3 backend: retention + orphan mark-and-sweep (parity with the
# filesystem-backed tests in test_studio_artifacts.py)
# ---------------------------------------------------------------------------


def _backdate_run(factory: sessionmaker[Session], event_id: str, when: dt.datetime) -> None:
    from docie_bench.studio.models import StudioRun

    with factory() as session:
        row = session.get(StudioRun, event_id)
        assert row is not None
        row.created_at = when
        session.commit()


def test_gc_retention_on_s3_deletes_orphans_keeps_shared_content(tmp_path: Path) -> None:
    client = FakeS3Client()
    store, factory = _make_s3_run_store(tmp_path / "s.db", client)
    now = dt.datetime.now(dt.UTC)

    shared_old = store.blobs.put(name="metrics.json", content=b'{"m":1}')
    old_unique = store.blobs.put(name="report.html", content=b"<old/>", media_type="text/html")
    store.claim(event_id="old", idempotency_key="ko", tenant_id="t")
    store.complete(
        event_id="old",
        metrics={},
        artifacts=[("metrics.json", shared_old), ("report.html", old_unique)],
    )
    _backdate_run(factory, "old", now - dt.timedelta(days=40))

    shared_new = store.blobs.put(name="metrics.json", content=b'{"m":1}')
    assert shared_new.relkey == shared_old.relkey  # identical content dedups
    store.claim(event_id="new", idempotency_key="kn", tenant_id="t")
    store.complete(event_id="new", metrics={}, artifacts=[("metrics.json", shared_new)])

    summary = store.gc(max_age_days=30, max_runs=1000, now=now, orphan_grace_hours=24)

    assert summary["deleted_runs"] == 1
    assert summary["deleted_blobs"] == 1  # only the old, unreferenced report.html
    assert store.get_run("old", tenant_id="t") is None
    assert store.get_run("new", tenant_id="t") is not None
    assert not store.blobs.exists(old_unique.relkey)  # orphan object removed
    assert store.blobs.exists(shared_old.relkey)  # still referenced by 'new'


def test_gc_orphan_sweep_on_s3_spares_inflight_and_referenced(tmp_path: Path) -> None:
    client = FakeS3Client()
    store, _ = _make_s3_run_store(tmp_path / "s.db", client)
    now = dt.datetime.now(dt.UTC)

    # Retained run with a committed artifact -> referenced, must survive.
    store.claim(event_id="kept", idempotency_key="kk", tenant_id="t")
    kept = store.blobs.put(name="report.html", content=b"<kept/>", media_type="text/html")
    store.complete(event_id="kept", metrics={}, artifacts=[("report.html", kept)])

    # Crash orphan: put() committed the object, the worker died before
    # complete() wrote its artifact row. Aged past grace via LastModified.
    store.claim(event_id="crashed", idempotency_key="kc", tenant_id="t")
    orphan = store.blobs.put(name="predictions.jsonl", content=b'{"row":1}\n')
    client.age("artifacts", f"studio/{orphan.relkey}", hours=48)

    # Freshly put, still-in-flight object: within grace, must NOT be swept.
    inflight = store.blobs.put(name="report.html", content=b"<inflight/>", media_type="text/html")

    summary = store.gc(max_age_days=3650, max_runs=1000, now=now, orphan_grace_hours=24)

    assert summary["deleted_runs"] == 0
    assert summary["deleted_blobs"] == 1  # only the aged crash orphan
    assert not store.blobs.exists(orphan.relkey)  # crash orphan reclaimed
    assert store.blobs.exists(inflight.relkey)  # in-flight put spared (within grace)
    assert store.blobs.exists(kept.relkey)  # still referenced by the retained run


def test_gc_grace_on_s3_spares_reput_content(tmp_path: Path) -> None:
    """A running job re-putting existing bytes refreshes LastModified, so the
    sweep cannot reclaim the identical content out from under the live job."""
    client = FakeS3Client()
    store, _ = _make_s3_run_store(tmp_path / "s.db", client)
    now = dt.datetime.now(dt.UTC)

    orphan = store.blobs.put(name="metrics.json", content=b'{"m":1}')
    client.age("artifacts", f"studio/{orphan.relkey}", hours=48)

    again = store.blobs.put(name="metrics.json", content=b'{"m":1}')
    assert again.relkey == orphan.relkey

    summary = store.gc(max_age_days=3650, max_runs=1000, now=now, orphan_grace_hours=1)
    assert summary["deleted_blobs"] == 0
    assert store.blobs.exists(orphan.relkey)  # spared: the re-put refreshed LastModified
