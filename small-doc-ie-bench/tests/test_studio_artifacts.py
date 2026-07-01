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
