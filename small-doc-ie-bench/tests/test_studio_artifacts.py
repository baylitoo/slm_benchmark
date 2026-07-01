"""Durable Studio run/artifact store: path-independence, auth, idempotency, GC.

These run entirely against stubs — a sqlite index, an on-disk blob store, and a
mocked ``run_benchmark`` — so they prove PR-2's behaviour with NO live model
stack, NO Inngest server, and NO Docker. The live-only pieces (native Inngest
``idempotency`` dedup, the nightly cron trigger, cross-container volume mount)
are called out as deferred in the PR body.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import docie_bench.api as api
import docie_bench.inngest.studio_api as studio_api
from docie_bench.security import TenantQuotaManager
from docie_bench.storage.db import Base
from docie_bench.studio.models import StudioRun
from docie_bench.studio.store import ArtifactBlobStore, RunStore


def _make_store(db_path: Path, blob_root: Path) -> tuple[RunStore, sessionmaker[Session]]:
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    return RunStore(factory, ArtifactBlobStore(blob_root)), factory


# ---------------------------------------------------------------------------
# Path-independence: worker writes, a different replica reads by id
# ---------------------------------------------------------------------------


def test_artifact_reachable_from_second_replica_and_payload_has_no_local_paths(
    tmp_path: Path,
) -> None:
    db, blobs = tmp_path / "studio.db", tmp_path / "blobs"
    worker, _ = _make_store(db, blobs)

    worker.claim(event_id="ev1", idempotency_key="k1", tenant_id="t1", dataset="ds")
    report = worker.blobs.put(
        name="report.html", content=b"<html>ok</html>", media_type="text/html"
    )
    preds = worker.blobs.put(
        name="predictions.jsonl",
        content=b'{"id":1}\n{"id":2}\n',
        media_type="application/x-ndjson",
    )
    record = worker.complete(
        event_id="ev1",
        metrics={"summary": [{"f1": 0.9}]},
        artifacts=[("report.html", report), ("predictions.jsonl", preds)],
    )

    # The delivered payload is path-independent: it carries neither the legacy
    # run_dir/*_path keys nor the worker-local store root anywhere.
    serialized = json.dumps(record)
    for legacy in ("run_dir", "report_path", "predictions_path", "metrics_path"):
        assert legacy not in record
    assert str(tmp_path) not in serialized
    for art in record["artifacts"]:
        assert art["uri"].startswith("/v1/studio/artifacts/")

    # A SEPARATE reader (fresh engine, same shared store) — the api replica —
    # resolves bytes purely from artifact_id -> DB row -> store root.
    reader, _ = _make_store(db, blobs)
    art_id = next(a["id"] for a in record["artifacts"] if a["name"] == "report.html")
    resolved = reader.open_artifact(art_id, tenant_id="t1")
    assert resolved is not None
    meta, content = resolved
    assert content == b"<html>ok</html>"
    assert meta["media_type"] == "text/html"


def test_large_predictions_live_in_blob_store_not_the_db_row(tmp_path: Path) -> None:
    store, factory = _make_store(tmp_path / "s.db", tmp_path / "b")
    store.claim(event_id="ev1", idempotency_key="k1", tenant_id="t1")
    payload = b'{"row":0}\n' + b"x" * 5000
    preds = store.blobs.put(
        name="predictions.jsonl", content=payload, media_type="application/x-ndjson"
    )
    store.complete(
        event_id="ev1", metrics={"summary": [{"f1": 1.0}]}, artifacts=[("predictions.jsonl", preds)]
    )

    with factory() as session:
        row = session.get(StudioRun, "ev1")
        assert row is not None
        # Only the small metrics summary is in Postgres; the bulk bytes are not.
        assert row.metrics_json == {"summary": [{"f1": 1.0}]}
        assert b"x" * 5000 not in json.dumps(row.metrics_json).encode()
    # ...but the bytes ARE retrievable from the blob store.
    assert store.blobs.read(preds.relkey) == payload


# ---------------------------------------------------------------------------
# Idempotency: a double-fire does not double-run
# ---------------------------------------------------------------------------


def test_claim_dedupes_completed_redelivery_and_duplicate_triggers(tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")

    # (a) redelivery of a COMPLETED run (same event id) short-circuits.
    assert store.claim(event_id="ev1", idempotency_key="k1", tenant_id="t")[0] == "claimed"
    store.complete(event_id="ev1", metrics={}, artifacts=[])
    assert store.claim(event_id="ev1", idempotency_key="k1", tenant_id="t")[0] == "exists"

    # (b) duplicate trigger: different event id, same content key -> resolves to
    #     the original run without double-running.
    assert store.claim(event_id="evA", idempotency_key="dup", tenant_id="t")[0] == "claimed"
    outcome, record = store.claim(event_id="evB", idempotency_key="dup", tenant_id="t")
    assert outcome == "exists"
    assert record["event_id"] == "evA"


def test_claim_allows_retry_of_a_failed_or_running_run(tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")

    # A failed attempt under the same event id must be re-runnable (Inngest is
    # at-least-once; its automatic retry re-executes the step for this event id).
    assert store.claim(event_id="ev1", idempotency_key="k1", tenant_id="t")[0] == "claimed"
    store.fail(event_id="ev1", error="boom")
    outcome, record = store.claim(event_id="ev1", idempotency_key="k1", tenant_id="t")
    assert outcome == "claimed"
    assert record["status"] == "running"  # reset for the retry
    assert record["error"] is None


def test_benchmark_job_runs_once_on_double_fire(tmp_path: Path, monkeypatch) -> None:
    from docie_bench.benchmark import runner as runner_mod
    from docie_bench.inngest import functions as fns

    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    monkeypatch.setattr("docie_bench.studio.store.default_run_store", lambda: store)

    calls: list[int] = []

    async def fake_run_benchmark(**_kwargs):
        calls.append(1)
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps({"summary": [{"f1": 0.5}]}))
        (run_dir / "report.html").write_text("<html>report</html>")
        (run_dir / "predictions.jsonl").write_text('{"pred":1}\n')
        return runner_mod.BenchmarkResult(
            run_dir,
            run_dir / "predictions.jsonl",
            run_dir / "metrics.json",
            run_dir / "report.html",
            run_dir / "manifest.json",
        )

    monkeypatch.setattr(runner_mod, "run_benchmark", fake_run_benchmark)

    data = {"dataset": "ds", "tenant_id": "t1", "idempotency_key": "k1"}
    first = asyncio.run(fns._run_benchmark_job(dict(data), event_id="ev1"))
    second = asyncio.run(fns._run_benchmark_job(dict(data), event_id="ev1"))

    assert len(calls) == 1  # the second fire short-circuits
    assert first["status"] == "completed"
    assert second["event_id"] == "ev1"
    assert first["metrics"] == {"summary": [{"f1": 0.5}]}
    # No worker-local path leaks into the delivered record.
    assert str(tmp_path) not in json.dumps(first)
    names = {a["name"] for a in first["artifacts"]}
    assert {"metrics.json", "report.html", "predictions.jsonl"} <= names


def test_benchmark_job_records_failure_for_retry(tmp_path: Path, monkeypatch) -> None:
    from docie_bench.benchmark import runner as runner_mod
    from docie_bench.inngest import functions as fns

    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    monkeypatch.setattr("docie_bench.studio.store.default_run_store", lambda: store)

    async def boom(**_kwargs):
        raise RuntimeError("model unreachable")

    monkeypatch.setattr(runner_mod, "run_benchmark", boom)

    with pytest.raises(RuntimeError):
        asyncio.run(
            fns._run_benchmark_job(
                {"dataset": "ds", "tenant_id": "t1", "idempotency_key": "k1"}, event_id="ev1"
            )
        )
    record = store.get_run("ev1", tenant_id="t1")
    assert record is not None
    assert record["status"] == "failed"
    assert "model unreachable" in record["error"]


# ---------------------------------------------------------------------------
# Auth: download + run status are authenticated and tenant-scoped
# ---------------------------------------------------------------------------


def _seed_owned_run(store: RunStore, *, tenant: str, body: bytes) -> str:
    store.claim(event_id="ev1", idempotency_key="k1", tenant_id=tenant, dataset="ds")
    blob = store.blobs.put(name="report.html", content=body, media_type="text/html")
    record = store.complete(
        event_id="ev1", metrics={"summary": []}, artifacts=[("report.html", blob)]
    )
    return record["artifacts"][0]["id"]


def test_artifact_download_is_authenticated_and_tenant_scoped(tmp_path: Path, monkeypatch) -> None:
    from docie_bench import security

    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    art_id = _seed_owned_run(store, tenant="tenant-a", body=b"<html>secret-a</html>")
    monkeypatch.setattr(studio_api, "default_run_store", lambda: store)

    manager = TenantQuotaManager(
        api_keys={"secret-a": "tenant-a", "secret-b": "tenant-b"},
        auth_required=True,
        requests_per_window=100,
        window_seconds=60,
        max_concurrent=10,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)
    client = TestClient(api.app)

    # Unauthenticated -> 401 (fail closed).
    assert client.get(f"/v1/studio/artifacts/{art_id}").status_code == 401
    assert client.get("/v1/studio/runs/ev1").status_code == 401

    # Owner -> 200 + the actual bytes + download disposition.
    owned = client.get(f"/v1/studio/artifacts/{art_id}", headers={"X-API-Key": "secret-a"})
    assert owned.status_code == 200
    assert owned.content == b"<html>secret-a</html>"
    assert "attachment" in owned.headers.get("content-disposition", "")

    # Cross-tenant -> 404, never 403 (do not confirm another tenant's run exists).
    assert (
        client.get(f"/v1/studio/artifacts/{art_id}", headers={"X-API-Key": "secret-b"}).status_code
        == 404
    )
    assert client.get("/v1/studio/runs/ev1", headers={"X-API-Key": "secret-b"}).status_code == 404

    # Owner run status returns the durable record with addressable artifact URIs.
    record = client.get("/v1/studio/runs/ev1", headers={"X-API-Key": "secret-a"})
    assert record.status_code == 200
    body = record.json()
    assert body["status"] == "completed"
    assert body["artifacts"][0]["uri"] == f"/v1/studio/artifacts/{art_id}"

    # Listing is tenant-scoped too.
    mine = client.get("/v1/studio/runs", headers={"X-API-Key": "secret-a"}).json()
    assert [r["event_id"] for r in mine] == ["ev1"]
    assert client.get("/v1/studio/runs", headers={"X-API-Key": "secret-b"}).json() == []


def test_benchmark_idempotency_key_is_tenant_scoped(monkeypatch) -> None:
    """Two tenants firing an identical request must NOT collide on one key.

    A shared key would let native/DB dedup deny one tenant's run and return the
    other tenant's record — so the trigger namespaces the key by principal.
    """
    from docie_bench import security
    from docie_bench.inngest.client import inngest_client

    captured: list[dict] = []

    async def fake_send(event):
        captured.append(dict(event.data))
        return [f"ev-{len(captured)}"]

    monkeypatch.setattr(inngest_client, "send", fake_send)
    manager = TenantQuotaManager(
        api_keys={"secret-a": "tenant-a", "secret-b": "tenant-b"},
        auth_required=True,
        requests_per_window=100,
        window_seconds=60,
        max_concurrent=10,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)
    client = TestClient(api.app)

    body = {"dataset": "invoices"}
    resp_a = client.post("/v1/studio/benchmark", json=body, headers={"X-API-Key": "secret-a"})
    resp_b = client.post("/v1/studio/benchmark", json=body, headers={"X-API-Key": "secret-b"})
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    key_a, key_b = captured[0]["idempotency_key"], captured[1]["idempotency_key"]
    assert key_a != key_b
    assert key_a.startswith("tenant-a:")
    assert key_b.startswith("tenant-b:")
    assert captured[0]["tenant_id"] == "tenant-a"


# ---------------------------------------------------------------------------
# Retention / GC: bounded accumulation, orphan-blob cleanup, content-addressing
# ---------------------------------------------------------------------------


def _backdate(factory: sessionmaker[Session], event_id: str, when: dt.datetime) -> None:
    with factory() as session:
        row = session.get(StudioRun, event_id)
        assert row is not None
        row.created_at = when
        session.commit()


def test_gc_by_age_deletes_orphan_blobs_but_keeps_shared_content(tmp_path: Path) -> None:
    store, factory = _make_store(tmp_path / "s.db", tmp_path / "b")
    now = dt.datetime(2026, 7, 1, tzinfo=dt.UTC)

    # Shared, content-addressed blob referenced by BOTH an old and a new run.
    shared_old = store.blobs.put(
        name="metrics.json", content=b'{"m":1}', media_type="application/json"
    )
    old_unique = store.blobs.put(name="report.html", content=b"<old/>", media_type="text/html")
    store.claim(event_id="old", idempotency_key="ko", tenant_id="t")
    store.complete(
        event_id="old",
        metrics={},
        artifacts=[("metrics.json", shared_old), ("report.html", old_unique)],
    )
    _backdate(factory, "old", now - dt.timedelta(days=40))

    shared_new = store.blobs.put(
        name="metrics.json", content=b'{"m":1}', media_type="application/json"
    )
    assert shared_new.relkey == shared_old.relkey  # identical content dedups
    store.claim(event_id="new", idempotency_key="kn", tenant_id="t")
    store.complete(event_id="new", metrics={}, artifacts=[("metrics.json", shared_new)])

    summary = store.gc(max_age_days=30, max_runs=1000, now=now)

    assert summary["deleted_runs"] == 1
    assert summary["deleted_blobs"] == 1  # only the old, unreferenced report.html
    assert store.get_run("old", tenant_id="t") is None
    assert store.get_run("new", tenant_id="t") is not None
    assert not store.blobs.exists(old_unique.relkey)  # orphan removed
    assert store.blobs.exists(shared_old.relkey)  # still referenced by 'new'


def test_gc_by_count_trims_oldest_runs(tmp_path: Path) -> None:
    store, factory = _make_store(tmp_path / "s.db", tmp_path / "b")
    now = dt.datetime(2026, 7, 1, tzinfo=dt.UTC)
    for i in range(3):
        store.claim(event_id=f"ev{i}", idempotency_key=f"k{i}", tenant_id="t")
        store.complete(event_id=f"ev{i}", metrics={}, artifacts=[])
        _backdate(factory, f"ev{i}", now - dt.timedelta(hours=3 - i))  # ev0 oldest

    summary = store.gc(max_age_days=3650, max_runs=2, now=now)
    assert summary == {"deleted_runs": 1, "deleted_blobs": 0, "retained_runs": 2}
    assert store.get_run("ev0", tenant_id="t") is None
    assert store.get_run("ev2", tenant_id="t") is not None


def test_blob_store_rejects_path_traversal_keys(tmp_path: Path) -> None:
    blobs = ArtifactBlobStore(tmp_path / "b")
    blobs.put(name="report.html", content=b"ok")
    with pytest.raises(ValueError, match="escapes the store root"):
        blobs.path_for("../../etc/passwd")
    with pytest.raises(ValueError, match="plain file name"):
        blobs.put(name="../evil.html", content=b"x")


# ---------------------------------------------------------------------------
# Finding 1: a DB row whose blob is gone resolves to 404, never a 500
# ---------------------------------------------------------------------------


def test_open_artifact_with_missing_blob_returns_none(tmp_path: Path) -> None:
    import shutil

    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    art_id = _seed_owned_run(store, tenant="t1", body=b"<html>x</html>")
    # GC/backup skew or a future object-store backend: the row survives, the blob
    # file is gone. This must not raise FileNotFoundError out of open_artifact.
    shutil.rmtree(tmp_path / "b")
    assert store.open_artifact(art_id, tenant_id="t1") is None


def test_download_with_missing_blob_is_404_not_500(tmp_path: Path, monkeypatch) -> None:
    import shutil

    from docie_bench import security

    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    art_id = _seed_owned_run(store, tenant="tenant-a", body=b"<html>secret</html>")
    shutil.rmtree(tmp_path / "b")  # blob gone, row remains
    monkeypatch.setattr(studio_api, "default_run_store", lambda: store)

    manager = TenantQuotaManager(
        api_keys={"secret-a": "tenant-a"},
        auth_required=True,
        requests_per_window=100,
        window_seconds=60,
        max_concurrent=10,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)
    client = TestClient(api.app)

    resp = client.get(f"/v1/studio/artifacts/{art_id}", headers={"X-API-Key": "secret-a"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Finding 2: a superseded retry reclaims its orphan blobs (content-addressing kept)
# ---------------------------------------------------------------------------


def test_retry_reclaims_orphan_blob_but_keeps_shared_content(tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")

    # First attempt: a unique report + a shared, content-addressed metrics blob.
    store.claim(event_id="ev1", idempotency_key="k1", tenant_id="t")
    report_v1 = store.blobs.put(name="report.html", content=b"<v1/>", media_type="text/html")
    shared = store.blobs.put(name="metrics.json", content=b'{"m":1}', media_type="application/json")
    store.complete(
        event_id="ev1",
        metrics={},
        artifacts=[("report.html", report_v1), ("metrics.json", shared)],
    )

    # A DIFFERENT run references the SAME metrics blob (identical content dedups).
    store.claim(event_id="ev2", idempotency_key="k2", tenant_id="t")
    shared2 = store.blobs.put(
        name="metrics.json", content=b'{"m":1}', media_type="application/json"
    )
    assert shared2.relkey == shared.relkey
    store.complete(event_id="ev2", metrics={}, artifacts=[("metrics.json", shared2)])

    # Retry of ev1 supersedes report v1 with v2, orphaning v1's blob.
    report_v2 = store.blobs.put(name="report.html", content=b"<v2/>", media_type="text/html")
    store.complete(
        event_id="ev1",
        metrics={},
        artifacts=[("report.html", report_v2), ("metrics.json", shared)],
    )

    assert not store.blobs.exists(report_v1.relkey)  # orphan reclaimed on retry
    assert store.blobs.exists(report_v2.relkey)  # new content retained
    assert store.blobs.exists(shared.relkey)  # still referenced by ev1 + ev2


# ---------------------------------------------------------------------------
# Finding 3: the run-status proxy is tenant-scoped for extraction runs too
# ---------------------------------------------------------------------------


class _FakeResp:
    status_code = 200
    text = ""

    def json(self):
        return [{"status": "Completed"}]


class _FakeAsyncClient:
    def __init__(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        pass

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_args) -> bool:  # noqa: ANN002
        return False

    async def get(self, _url: str, headers: dict | None = None) -> _FakeResp:
        return _FakeResp()


def test_event_runs_proxy_is_tenant_scoped_for_extraction(tmp_path: Path, monkeypatch) -> None:
    from docie_bench import security

    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    # Extraction run: only an ownership row exists (no durable StudioRun).
    store.record_event_owner(event_id="evx", tenant_id="tenant-a")
    monkeypatch.setattr(studio_api, "default_run_store", lambda: store)
    monkeypatch.setattr(studio_api.httpx, "AsyncClient", _FakeAsyncClient)

    manager = TenantQuotaManager(
        api_keys={"secret-a": "tenant-a", "secret-b": "tenant-b"},
        auth_required=True,
        requests_per_window=100,
        window_seconds=60,
        max_concurrent=10,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)
    client = TestClient(api.app)

    # Cross-tenant id -> 404, never proxied (this is the leak the finding closes).
    assert client.get("/v1/studio/runs/evx", headers={"X-API-Key": "secret-b"}).status_code == 404
    # Unknown id with no recorded owner -> 404 (no unscoped proxy fallthrough).
    assert (
        client.get("/v1/studio/runs/unknown", headers={"X-API-Key": "secret-a"}).status_code == 404
    )
    # The legitimate owner still gets the proxied run status unchanged.
    owned = client.get("/v1/studio/runs/evx", headers={"X-API-Key": "secret-a"})
    assert owned.status_code == 200
    assert owned.json() == [{"status": "Completed"}]


# ---------------------------------------------------------------------------
# Finding 4: a terminally-failed run is re-runnable (key rotates); duplicates dedup
# ---------------------------------------------------------------------------


def test_terminal_failure_rotates_key_while_success_still_dedups(tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    base = "tenant-a:bench-abc"

    # No prior run -> emit the base key.
    assert store.effective_idempotency_key(base) == base

    # In-flight run -> a duplicate dedups to the same key.
    store.claim(event_id="ev1", idempotency_key=base, tenant_id="tenant-a")
    assert store.effective_idempotency_key(base) == base

    # Terminal failure -> rotate so a re-request is not locked out for the window.
    store.fail(event_id="ev1", error="model outage")
    rotated = store.effective_idempotency_key(base)
    assert rotated != base
    assert rotated.startswith(base)

    # The rotated run then succeeds -> duplicates dedup to it (no further rotation).
    store.claim(event_id="ev2", idempotency_key=rotated, tenant_id="tenant-a")
    store.complete(event_id="ev2", metrics={}, artifacts=[])
    assert store.effective_idempotency_key(base) == rotated


def test_effective_key_does_not_over_match_sibling_client_keys(tmp_path: Path) -> None:
    """A client-supplied key must not pull an unrelated key sharing its prefix.

    ``base`` terminally failed, but ``base + "bar"`` (a different request) is
    still running; resolving ``base`` must rotate off ``base`` — never dedup into
    the sibling run.
    """
    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    base = "tenant-a:foo"

    store.claim(event_id="ev1", idempotency_key=base, tenant_id="tenant-a")
    store.fail(event_id="ev1", error="boom")
    store.claim(event_id="ev2", idempotency_key=base + "bar", tenant_id="tenant-a")

    resolved = store.effective_idempotency_key(base)
    assert resolved != base + "bar"
    assert resolved.startswith(base + ":r")


def test_benchmark_trigger_rotates_key_after_terminal_failure(
    tmp_path: Path, monkeypatch
) -> None:
    from docie_bench import security
    from docie_bench.inngest.client import inngest_client

    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    monkeypatch.setattr(studio_api, "default_run_store", lambda: store)

    captured: list[dict] = []

    async def fake_send(event):
        captured.append(dict(event.data))
        return [f"ev-{len(captured)}"]

    monkeypatch.setattr(inngest_client, "send", fake_send)
    manager = TenantQuotaManager(
        api_keys={"secret-a": "tenant-a"},
        auth_required=True,
        requests_per_window=100,
        window_seconds=60,
        max_concurrent=10,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)
    client = TestClient(api.app)

    body = {"dataset": "invoices"}
    first = client.post("/v1/studio/benchmark", json=body, headers={"X-API-Key": "secret-a"})
    assert first.status_code == 200
    key1 = captured[0]["idempotency_key"]

    # Simulate the worker running that key and terminally failing (retries exhausted).
    store.claim(event_id="ev-1", idempotency_key=key1, tenant_id="tenant-a", dataset="invoices")
    store.fail(event_id="ev-1", error="model outage")

    # An identical re-request now emits a DIFFERENT key so it actually re-runs.
    second = client.post("/v1/studio/benchmark", json=body, headers={"X-API-Key": "secret-a"})
    assert second.status_code == 200
    key2 = captured[1]["idempotency_key"]
    assert key2 != key1
    assert key2.startswith(key1)


# ---------------------------------------------------------------------------
# Issue A (was F2): GC is a real mark-and-sweep — it reclaims a crash-orphaned
# blob (put() committed, run crashed before complete() wrote its artifact row),
# but spares an in-flight put (within grace) and any blob a retained run keeps.
# ---------------------------------------------------------------------------


def test_gc_sweeps_crash_orphan_blob_but_spares_inflight_and_referenced(tmp_path: Path) -> None:
    import os

    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    now = dt.datetime.now(dt.UTC)

    # Retained run with a committed artifact -> its blob is referenced, must survive.
    store.claim(event_id="kept", idempotency_key="kk", tenant_id="t")
    kept_blob = store.blobs.put(name="report.html", content=b"<kept/>", media_type="text/html")
    store.complete(event_id="kept", metrics={}, artifacts=[("report.html", kept_blob)])

    # The REAL orphan case the prior fix missed: claim() created a running row,
    # put() wrote the blob, then the worker crashed BEFORE complete() committed the
    # artifact row. No StudioRunArtifact references the blob and no run is doomed, so
    # the old row-driven GC could never reclaim it -> a permanent leak.
    store.claim(event_id="crashed", idempotency_key="kc", tenant_id="t")
    orphan = store.blobs.put(
        name="predictions.jsonl", content=b'{"row":1}\n', media_type="application/x-ndjson"
    )
    old_ts = (now - dt.timedelta(hours=48)).timestamp()  # aged past the default grace
    os.utime(store.blobs.path_for(orphan.relkey), (old_ts, old_ts))

    # A freshly put(), still-in-flight blob whose complete() has not run yet: within
    # the grace window, so it must NOT be swept out from under the running job.
    inflight = store.blobs.put(
        name="report.html", content=b"<inflight/>", media_type="text/html"
    )

    # No orphan_grace_hours override: relies on the 24h default, so this asserts the
    # sweep BEHAVIOUR (a row-driven-only GC leaves the crash orphan on disk forever).
    summary = store.gc(max_age_days=3650, max_runs=1000, now=now)

    assert summary["deleted_runs"] == 0  # nothing is time/count-doomed
    assert summary["deleted_blobs"] == 1  # only the aged crash orphan
    assert not store.blobs.exists(orphan.relkey)  # crash orphan reclaimed
    assert store.blobs.exists(inflight.relkey)  # in-flight put spared (within grace)
    assert store.blobs.exists(kept_blob.relkey)  # still referenced by the retained run


def test_gc_grace_refreshes_mtime_when_running_job_reputs_existing_content(
    tmp_path: Path,
) -> None:
    """A running job re-putting pre-existing (content-addressed) bytes refreshes the
    blob's mtime, so the sweep cannot reclaim it out from under the live job.
    """
    import os

    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    now = dt.datetime.now(dt.UTC)

    # An old, unreferenced orphan (aged well past grace).
    orphan = store.blobs.put(name="metrics.json", content=b'{"m":1}', media_type="application/json")
    old_ts = (now - dt.timedelta(hours=48)).timestamp()
    os.utime(store.blobs.path_for(orphan.relkey), (old_ts, old_ts))

    # A running job now re-puts the SAME content (dedup: no rewrite). This must
    # refresh the mtime so the identical bytes are treated as live, not orphaned.
    again = store.blobs.put(name="metrics.json", content=b'{"m":1}', media_type="application/json")
    assert again.relkey == orphan.relkey

    summary = store.gc(max_age_days=3650, max_runs=1000, now=now, orphan_grace_hours=1)
    assert summary["deleted_blobs"] == 0
    assert store.blobs.exists(orphan.relkey)  # spared: the re-put refreshed its mtime


# ---------------------------------------------------------------------------
# Issue B: deploy + seed run-status polling is owned + tenant-scoped (no 404
# regression for the triggering principal; cross-tenant/unknown still 404).
# ---------------------------------------------------------------------------


def test_deploy_and_seed_run_status_is_owned_and_tenant_scoped(
    tmp_path: Path, monkeypatch
) -> None:
    from docie_bench import security
    from docie_bench.inngest.client import inngest_client

    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    monkeypatch.setattr(studio_api, "default_run_store", lambda: store)
    monkeypatch.setattr(studio_api.httpx, "AsyncClient", _FakeAsyncClient)

    sent: list[str] = []

    async def fake_send(event):
        eid = f"ev-{len(sent)}"
        sent.append(eid)
        return [eid]

    monkeypatch.setattr(inngest_client, "send", fake_send)
    manager = TenantQuotaManager(
        api_keys={"secret-a": "tenant-a", "secret-b": "tenant-b"},
        auth_required=True,
        requests_per_window=100,
        window_seconds=60,
        max_concurrent=10,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)
    client = TestClient(api.app)

    # Both infra triggers now require auth (fail closed), like extract/benchmark.
    assert client.post("/v1/studio/deploy", json={"model": "m"}).status_code == 401
    assert (
        client.post("/v1/studio/seed-ollama", json={"reference": "r", "name": "n"}).status_code
        == 401
    )

    dep = client.post("/v1/studio/deploy", json={"model": "m"}, headers={"X-API-Key": "secret-a"})
    seed = client.post(
        "/v1/studio/seed-ollama",
        json={"reference": "qwen2.5:1.5b", "name": "q"},
        headers={"X-API-Key": "secret-a"},
    )
    assert dep.status_code == 200
    assert seed.status_code == 200
    dep_id = dep.json()["event_ids"][0]
    seed_id = seed.json()["event_ids"][0]

    for eid in (dep_id, seed_id):
        # The triggering principal polls run status -> proxied through (no 404).
        owned = client.get(f"/v1/studio/runs/{eid}", headers={"X-API-Key": "secret-a"})
        assert owned.status_code == 200
        assert owned.json() == [{"status": "Completed"}]
        # A different tenant is refused (404, never proxied).
        assert (
            client.get(f"/v1/studio/runs/{eid}", headers={"X-API-Key": "secret-b"}).status_code
            == 404
        )
    # Unknown id with no recorded owner -> 404 (no unscoped proxy fallthrough).
    assert client.get("/v1/studio/runs/nope", headers={"X-API-Key": "secret-a"}).status_code == 404
