"""Batch extraction: the durable store, the trigger + read routes, the
per-document step, and results materialization/download.

The Inngest ``ctx``/step machinery is the framework's; what's tested here is
everything the batch feature owns: the store lifecycle (claim / per-item
record with no double-count / settle / redelivery / tenant scoping), the
trigger route (zip + inline documents, every rejection, DB-required 503,
documents stashed as blob KEYS in the event -- never bytes), the per-doc
step body (a bad document records ``failed`` and never raises), and the
JSONL/CSV materialization + the tenant-scoped download route.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import docie_bench.api as api
from docie_bench.inngest import functions
from docie_bench.storage.db import dispose_engine, init_engine
from docie_bench.studio import store as studio_store
from docie_bench.studio.batch_store import (
    BatchStoreUnavailableError,
    claim_batch_run,
    get_batch_run,
    list_batch_runs,
    record_batch_item,
    settle_batch_run,
)

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


@pytest.fixture(autouse=True)
def batch_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    init_engine(f"sqlite:///{tmp_path / 'batch.db'}")
    # Every blob-store user (route + worker) goes through default_blob_store();
    # point it at a temp dir so tests never touch the real artifacts/ tree.
    blobs = studio_store.ArtifactBlobStore(tmp_path / "artifacts")
    monkeypatch.setattr(studio_store, "default_blob_store", lambda: blobs)
    import docie_bench.inngest.studio_api.batch as batch_routes

    monkeypatch.setattr(batch_routes, "default_blob_store", lambda: blobs)
    yield blobs
    dispose_engine()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from docie_bench import security
    from docie_bench.security import TenantQuotaManager

    monkeypatch.setattr(
        security,
        "get_quota_manager",
        lambda: TenantQuotaManager(
            api_keys={"key-a": TENANT_A, "key-b": TENANT_B},
            auth_required=True,
            requests_per_window=100,
            window_seconds=60,
            max_concurrent=10,
        ),
    )
    return TestClient(api.app)


def _hdr(key: str = "key-a") -> dict[str, str]:
    return {"X-API-Key": key}


def _zip_b64(files: dict[str, bytes]) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# -- store -------------------------------------------------------------------


def test_store_claim_record_settle_lifecycle() -> None:
    outcome, run = claim_batch_run(
        event_id="e1",
        channel="batch:1",
        tenant_id=TENANT_A,
        name="invoices",
        schema_name="invoice",
        model_selector="m",
        filenames=["a.pdf", "b.pdf", "c.pdf"],
    )
    assert outcome == "claimed"
    assert run["total_items"] == 3
    assert [i["status"] for i in run["items"]] == ["pending"] * 3

    record_batch_item(event_id="e1", position=0, status="done", result={"x": 1}, latency_ms=5)
    record_batch_item(event_id="e1", position=1, status="failed", error="boom")
    got = get_batch_run("e1", tenant_id=TENANT_A)
    assert (got["done_items"], got["failed_items"]) == (1, 1)
    assert got["items"][1]["error"] == "boom"

    settled = settle_batch_run(
        event_id="e1", status="completed", artifacts=[{"name": "results.jsonl"}]
    )
    assert settled["status"] == "completed"
    assert settled["artifacts"] == [{"name": "results.jsonl"}]


def test_store_re_recording_an_item_never_double_counts() -> None:
    # A step retry re-runs _batch_extract_one -> record_batch_item twice for
    # the same position. Counters must reflect ONE terminal state per item.
    claim_batch_run(
        event_id="e2",
        channel="batch:2",
        tenant_id=TENANT_A,
        name="n",
        schema_name="invoice",
        model_selector=None,
        filenames=["a.pdf"],
    )
    record_batch_item(event_id="e2", position=0, status="failed", error="first")
    record_batch_item(event_id="e2", position=0, status="failed", error="retry")
    got = get_batch_run("e2", tenant_id=TENANT_A)
    assert got["failed_items"] == 1
    assert got["items"][0]["error"] == "retry"


def test_store_redelivery_of_a_completed_batch_is_exists_not_rerun() -> None:
    claim_batch_run(
        event_id="e3",
        channel="batch:3",
        tenant_id=TENANT_A,
        name="n",
        schema_name="invoice",
        model_selector=None,
        filenames=["a.pdf"],
    )
    settle_batch_run(event_id="e3", status="completed")
    outcome, _ = claim_batch_run(
        event_id="e3",
        channel="batch:3",
        tenant_id=TENANT_A,
        name="n",
        schema_name="invoice",
        model_selector=None,
        filenames=["a.pdf"],
    )
    assert outcome == "exists"


def test_store_retry_of_a_running_batch_keeps_item_progress() -> None:
    # A function-level retry re-claims; the items it already finished must
    # survive -- that's the whole point of per-item state.
    claim_batch_run(
        event_id="e4",
        channel="batch:4",
        tenant_id=TENANT_A,
        name="n",
        schema_name="invoice",
        model_selector=None,
        filenames=["a.pdf", "b.pdf"],
    )
    record_batch_item(event_id="e4", position=0, status="done", result={})
    outcome, run = claim_batch_run(
        event_id="e4",
        channel="batch:4",
        tenant_id=TENANT_A,
        name="n",
        schema_name="invoice",
        model_selector=None,
        filenames=["a.pdf", "b.pdf"],
    )
    assert outcome == "claimed"
    assert [i["status"] for i in run["items"]] == ["done", "pending"]


def test_store_is_tenant_scoped_and_lists_without_items() -> None:
    claim_batch_run(
        event_id="e5",
        channel="batch:5",
        tenant_id=TENANT_A,
        name="n",
        schema_name="invoice",
        model_selector=None,
        filenames=["a.pdf"],
    )
    assert get_batch_run("e5", tenant_id=TENANT_B) is None
    assert get_batch_run("e5", tenant_id=TENANT_A) is not None
    listed = list_batch_runs(tenant_id=TENANT_A)
    assert [r["event_id"] for r in listed] == ["e5"]
    assert "items" not in listed[0]
    assert list_batch_runs(tenant_id=TENANT_B) == []


def test_store_requires_a_database() -> None:
    dispose_engine()
    with pytest.raises(BatchStoreUnavailableError):
        claim_batch_run(
            event_id="x",
            channel="batch:x",
            tenant_id=TENANT_A,
            name="n",
            schema_name="invoice",
            model_selector=None,
            filenames=["a.pdf"],
        )


# -- trigger route -----------------------------------------------------------


@pytest.fixture
def captured_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    from docie_bench.inngest.client import inngest_client

    events: list[dict[str, Any]] = []

    async def fake_send(event):
        events.append({"name": event.name, "data": dict(event.data)})
        return [f"ev-{len(events)}"]

    monkeypatch.setattr(inngest_client, "send", fake_send)
    return events


def test_trigger_zip_stashes_documents_as_blob_keys_not_bytes(
    client: TestClient, captured_events: list[dict[str, Any]], batch_database
) -> None:
    payload = _zip_b64(
        {
            "inv-001.pdf": b"%PDF-1 fake one",
            "sub/inv-002.png": b"\x89PNG fake two",
            "__MACOSX/._inv-001.pdf": b"resource fork junk",  # skipped
            "README.txt": b"skip me? no -- .txt is supported",  # kept
            "notes.docx": b"unsupported type",  # skipped
        }
    )
    resp = client.post(
        "/v1/studio/extract/batch",
        json={"zip_b64": payload, "schema_name": "invoice", "deployment": "dep-1", "name": "Q3"},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["channel"].startswith("batch:")

    assert len(captured_events) == 1
    ev = captured_events[0]
    assert ev["name"] == "doc/batch.requested"
    data = ev["data"]
    assert data["tenant_id"] == TENANT_A
    assert data["name"] == "Q3"
    assert data["deployment"] == "dep-1"
    # Filenames are the leaf names, in archive order, junk skipped:
    assert [i["filename"] for i in data["inputs"]] == ["inv-001.pdf", "inv-002.png", "README.txt"]
    # The event carries KEYS into the shared blob store, never document bytes.
    for item in data["inputs"]:
        assert "relkey" in item
        assert "content_b64" not in item
        assert batch_database.exists(item["relkey"])
    assert batch_database.read(data["inputs"][0]["relkey"]) == b"%PDF-1 fake one"


def test_trigger_inline_documents(
    client: TestClient, captured_events: list[dict[str, Any]]
) -> None:
    resp = client.post(
        "/v1/studio/extract/batch",
        json={
            "documents": [
                {"filename": "a.pdf", "content_b64": base64.b64encode(b"AAA").decode()},
                {"filename": "dir/b.pdf", "content_b64": base64.b64encode(b"BBB").decode()},
            ],
            "model_profile": "cheap",
        },
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    data = captured_events[0]["data"]
    assert [i["filename"] for i in data["inputs"]] == ["a.pdf", "b.pdf"]
    assert data["model_profile"] == "cheap"
    assert data["name"] == "batch of 2"


@pytest.mark.parametrize(
    ("body", "status", "fragment"),
    [
        ({"schema_name": "invoice"}, 422, "exactly one of"),
        (
            {"zip_b64": "x", "documents": [{"filename": "a", "content_b64": "QQ=="}]},
            422,
            "exactly one of",
        ),
        ({"zip_b64": "!!not-b64!!"}, 422, "not valid base64"),
        ({"zip_b64": base64.b64encode(b"not a zip").decode()}, 422, "not a valid zip"),
        ({"documents": []}, 422, "exactly one of"),
        (
            {
                "deployment": "d",
                "model_profile": "m",
                "documents": [{"filename": "a.pdf", "content_b64": "QQ=="}],
            },
            400,
            "mutually exclusive",
        ),
    ],
)
def test_trigger_rejects_malformed_requests(
    client: TestClient,
    captured_events: list[dict[str, Any]],
    body: dict,
    status: int,
    fragment: str,
) -> None:
    resp = client.post("/v1/studio/extract/batch", json=body, headers=_hdr())
    assert resp.status_code == status, resp.text
    assert fragment in resp.json()["detail"]
    assert captured_events == []  # nothing enqueued


def test_trigger_rejects_a_zip_with_no_supported_documents(
    client: TestClient, captured_events: list[dict[str, Any]]
) -> None:
    resp = client.post(
        "/v1/studio/extract/batch",
        json={"zip_b64": _zip_b64({"notes.docx": b"x", "data.json": b"{}"})},
        headers=_hdr(),
    )
    assert resp.status_code == 422
    assert "no documents of a supported type" in resp.json()["detail"]
    assert captured_events == []


def test_trigger_requires_the_database_up_front(
    client: TestClient, captured_events: list[dict[str, Any]]
) -> None:
    # A batch's item results ARE the product: with no persistence, refuse NOW
    # (503) rather than start a job that can only end in "results lost".
    dispose_engine()
    resp = client.post(
        "/v1/studio/extract/batch",
        json={"documents": [{"filename": "a.pdf", "content_b64": "QQ=="}]},
        headers=_hdr(),
    )
    assert resp.status_code == 503
    assert captured_events == []


# -- read routes -------------------------------------------------------------


def test_list_and_get_are_tenant_scoped(client: TestClient) -> None:
    claim_batch_run(
        event_id="ev-a",
        channel="batch:a",
        tenant_id=TENANT_A,
        name="A's",
        schema_name="invoice",
        model_selector=None,
        filenames=["a.pdf"],
    )
    claim_batch_run(
        event_id="ev-b",
        channel="batch:b",
        tenant_id=TENANT_B,
        name="B's",
        schema_name="invoice",
        model_selector=None,
        filenames=["b.pdf"],
    )
    a_list = client.get("/v1/studio/batches", headers=_hdr("key-a")).json()
    assert [r["name"] for r in a_list] == ["A's"]
    assert client.get("/v1/studio/batches/ev-a", headers=_hdr("key-a")).status_code == 200
    # Foreign tenant: 404, never a 403 that confirms existence.
    assert client.get("/v1/studio/batches/ev-a", headers=_hdr("key-b")).status_code == 404
    assert client.get("/v1/studio/batches/nope", headers=_hdr("key-a")).status_code == 404


# -- the per-document step + results ------------------------------------------


class _FakeExtraction:
    """Stands in for _run_extraction: succeeds for most files, blows up for
    one -- proving a bad document is recorded and never propagates."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(data["filename"])
        if data["filename"] == "corrupt.pdf":
            raise RuntimeError("PDF has no pages")
        return {
            "request_id": f"r-{data['filename']}",
            "schema_name": data.get("schema_name", "invoice"),
            "model_profile": data.get("model_profile") or "studio_default",
            "result": {
                "invoice_number": {
                    "value": f"INV-{data['filename']}",
                    "confidence": 0.9,
                    "evidence_ids": [],
                },
                "total_ttc": {
                    "amount": "10.00",
                    "currency": "EUR",
                    "confidence": 0.9,
                    "evidence_ids": [],
                },
                "line_items": [{"description": {"value": "x"}}],
            },
        }


@pytest.mark.asyncio
async def test_per_document_step_records_success_and_never_raises_on_failure(
    monkeypatch: pytest.MonkeyPatch, batch_database
) -> None:
    fake = _FakeExtraction()
    monkeypatch.setattr(functions, "_run_extraction", fake)
    claim_batch_run(
        event_id="job-1",
        channel="batch:j1",
        tenant_id=TENANT_A,
        name="n",
        schema_name="invoice",
        model_selector="cheap",
        filenames=["ok.pdf", "corrupt.pdf"],
    )
    ok_key = batch_database.put(name="ok.pdf", content=b"OK", media_type="application/pdf").relkey
    bad_key = batch_database.put(
        name="corrupt.pdf", content=b"BAD", media_type="application/pdf"
    ).relkey
    data = {
        "event_id": "job-1",
        "schema_name": "invoice",
        "model_profile": "cheap",
        "tenant_id": TENANT_A,
    }

    good = await functions._batch_extract_one(data, {"filename": "ok.pdf", "relkey": ok_key}, 0)
    bad = await functions._batch_extract_one(
        data, {"filename": "corrupt.pdf", "relkey": bad_key}, 1
    )

    assert good["status"] == "done"
    assert good["result"]["result"]["invoice_number"]["value"] == "INV-ok.pdf"
    assert bad["status"] == "failed"
    assert "PDF has no pages" in bad["error"]
    # The per-document call carried the SHARED selectors + this doc's bytes,
    # never another document's, and the batch's tenant.
    assert fake.calls == ["ok.pdf", "corrupt.pdf"]
    # Persisted state matches the returned outcome, and the failure did not
    # stop the run from being settle-able as completed-with-failures.
    run = get_batch_run("job-1", tenant_id=TENANT_A)
    assert (run["done_items"], run["failed_items"]) == (1, 1)
    assert run["items"][1]["status"] == "failed"


def test_results_materialize_as_jsonl_and_flattened_csv(batch_database) -> None:
    outcomes = [
        {
            "position": 0,
            "filename": "a.pdf",
            "status": "done",
            "latency_ms": 5,
            "result": {
                "model_profile": "cheap",
                "result": {
                    "invoice_number": {"value": "INV-1"},
                    "total_ttc": {"amount": "10.00", "currency": "EUR"},
                    "line_items": [{"description": {"value": "x"}}],
                },
            },
        },
        {"position": 1, "filename": "b.pdf", "status": "failed", "error": "boom", "latency_ms": 2},
    ]
    artifacts = functions._batch_write_results(outcomes)
    assert [a["name"] for a in artifacts] == ["results.jsonl", "results.csv"]

    jsonl = batch_database.read(artifacts[0]["relkey"]).decode()
    lines = [json.loads(line) for line in jsonl.strip().splitlines()]
    assert [line["filename"] for line in lines] == ["a.pdf", "b.pdf"]
    assert lines[0]["result"]["result"]["total_ttc"]["amount"] == "10.00"  # lossless

    csv_text = batch_database.read(artifacts[1]["relkey"]).decode()
    header, row_a, row_b = csv_text.strip().splitlines()
    cols = header.split(",")
    # Wrapper fields flatten to field.value / field.amount+field.currency;
    # a list field becomes JSON text in one column.
    assert {"invoice_number.value", "total_ttc.amount", "total_ttc.currency", "line_items"} <= set(
        cols
    )
    assert "INV-1" in row_a
    assert "10.00" in row_a
    assert "EUR" in row_a
    assert "boom" in row_b


def test_download_results_is_tenant_scoped_and_409_before_settle(
    client: TestClient, batch_database
) -> None:
    claim_batch_run(
        event_id="dl-1",
        channel="batch:dl",
        tenant_id=TENANT_A,
        name="q3 invoices",
        schema_name="invoice",
        model_selector=None,
        filenames=["a.pdf"],
    )
    # Not settled yet: no artifacts -> 409, not a crash / empty file.
    assert client.get("/v1/studio/batches/dl-1/results.jsonl", headers=_hdr()).status_code == 409

    artifacts = functions._batch_write_results(
        [{"position": 0, "filename": "a.pdf", "status": "done", "result": {"result": {}}}]
    )
    settle_batch_run(event_id="dl-1", status="completed", artifacts=artifacts)

    ok = client.get("/v1/studio/batches/dl-1/results.jsonl", headers=_hdr())
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("application/x-ndjson")
    assert 'filename="q3 invoices.jsonl"' in ok.headers["content-disposition"]
    assert json.loads(ok.text.strip())["filename"] == "a.pdf"

    csv_resp = client.get("/v1/studio/batches/dl-1/results.csv", headers=_hdr())
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")

    # Foreign tenant cannot download; unknown format is a 404.
    assert (
        client.get("/v1/studio/batches/dl-1/results.jsonl", headers=_hdr("key-b")).status_code
        == 404
    )
    assert client.get("/v1/studio/batches/dl-1/results.xlsx", headers=_hdr()).status_code == 404


# ---------------------------------------------------------------------------
# P2 follow-ups: retry-failed, webhooks, routing_policy on batch, GC retain.
# ---------------------------------------------------------------------------


def test_migration_adds_new_columns_to_a_pre_upgrade_database(tmp_path: Path) -> None:
    # Simulate a database created by #232's shape (no input_relkey /
    # callback_url / selectors_json) via raw SQL, then run the migration:
    # CREATE TABLE IF NOT EXISTS alone would silently leave the columns
    # missing and the first claim would throw OperationalError.
    from sqlalchemy import create_engine, inspect, text

    from docie_bench.studio.batch_store import ensure_batch_tables

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE batch_runs (event_id VARCHAR(128) PRIMARY KEY, "
            "channel VARCHAR(128), tenant_id VARCHAR(128), name VARCHAR(200), "
            "schema_name VARCHAR(64), model_selector VARCHAR(200), status VARCHAR(32), "
            "total_items INTEGER, done_items INTEGER, failed_items INTEGER, "
            "error_text TEXT, artifacts_json JSON, created_at DATETIME, updated_at DATETIME)"
        ))
        conn.execute(text(
            "CREATE TABLE batch_items (id INTEGER PRIMARY KEY, run_event_id VARCHAR(128), "
            "position INTEGER, filename VARCHAR(300), status VARCHAR(32), "
            "result_json JSON, error_text TEXT, latency_ms INTEGER, updated_at DATETIME)"
        ))
    ensure_batch_tables(engine)
    runs_cols = {c["name"] for c in inspect(engine).get_columns("batch_runs")}
    items_cols = {c["name"] for c in inspect(engine).get_columns("batch_items")}
    assert {"callback_url", "selectors_json"} <= runs_cols
    assert "input_relkey" in items_cols
    # And it stays idempotent.
    ensure_batch_tables(engine)


def test_gc_never_reclaims_batch_inputs_or_results(batch_database) -> None:
    # THE data-loss bug this PR fixes: batch blobs live in the same store as
    # benchmark artifacts but are referenced only by batch tables, which the
    # orphan sweep ignored -- after the grace window every batch's inputs AND
    # results were deleted. grace=0 makes any unreferenced blob immediately
    # sweepable, so survival here proves the retain set now sees batch refs.
    from docie_bench.storage.db import get_session_factory
    from docie_bench.studio.store import RunStore

    input_blob = batch_database.put(name="doc.pdf", content=b"DOC", media_type="application/pdf")
    claim_batch_run(
        event_id="gc-1", channel="batch:gc", tenant_id=TENANT_A, name="n",
        schema_name="invoice", model_selector=None, filenames=["doc.pdf"],
        input_relkeys=[input_blob.relkey],
    )
    result_artifacts = functions._batch_write_results(
        [{"position": 0, "filename": "doc.pdf", "status": "done", "result": {"result": {}}}]
    )
    settle_batch_run(event_id="gc-1", status="completed", artifacts=result_artifacts)
    orphan = batch_database.put(
        name="orphan.bin", content=b"junk", media_type="application/octet-stream"
    )

    stats = RunStore(get_session_factory(), batch_database).gc(
        max_age_days=365, max_runs=10_000, orphan_grace_hours=0
    )

    assert batch_database.exists(input_blob.relkey), "batch INPUT swept as orphan"
    for artifact in result_artifacts:
        assert batch_database.exists(artifact["relkey"]), "batch RESULT swept as orphan"
    assert not batch_database.exists(orphan.relkey), "true orphan must still be swept"
    assert stats["deleted_blobs"] >= 1


def test_trigger_validates_and_forwards_routing_policy_and_callback(
    client: TestClient, captured_events: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "docie_bench.studio.routing_policies.get_routing_policy",
        lambda name: {"name": name, "policy": {}} if name == "cheap-then-strong" else None,
    )
    docs = [{"filename": "a.pdf", "content_b64": base64.b64encode(b"A").decode()}]
    # exclusivity now covers the policy too
    both = client.post(
        "/v1/studio/extract/batch",
        json={"documents": docs, "routing_policy": "cheap-then-strong", "deployment": "d"},
        headers=_hdr(),
    )
    assert both.status_code == 400
    assert "mutually exclusive" in both.json()["detail"]
    # unknown policy fails at the edge, nothing enqueued
    missing = client.post(
        "/v1/studio/extract/batch",
        json={"documents": docs, "routing_policy": "nope"},
        headers=_hdr(),
    )
    assert missing.status_code == 404
    # bad webhook scheme
    bad_url = client.post(
        "/v1/studio/extract/batch",
        json={"documents": docs, "callback_url": "ftp://x"},
        headers=_hdr(),
    )
    assert bad_url.status_code == 422
    assert captured_events == []
    # happy path forwards everything
    ok = client.post(
        "/v1/studio/extract/batch",
        json={
            "documents": docs,
            "routing_policy": "cheap-then-strong",
            "callback_url": "https://hooks.example/done",
            "callback_secret": "s3cret",
        },
        headers=_hdr(),
    )
    assert ok.status_code == 200, ok.text
    data = captured_events[0]["data"]
    assert data["routing_policy"] == "cheap-then-strong"
    assert data["callback_url"] == "https://hooks.example/done"
    assert data["callback_secret"] == "s3cret"  # noqa: S105 - test fixture value


@pytest.mark.asyncio
async def test_per_document_step_forwards_routing_policy(
    monkeypatch: pytest.MonkeyPatch, batch_database
) -> None:
    # _BATCH_PER_DOC_KEYS is the contract: routing_policy must reach
    # _run_extraction's data, where #231's _resolve_executor picks it up.
    seen: list[dict[str, Any]] = []

    async def fake_run(data: dict[str, Any]) -> dict[str, Any]:
        seen.append(data)
        return {
            "request_id": "r",
            "schema_name": "invoice",
            "model_profile": "strong",
            "result": {},
        }

    monkeypatch.setattr(functions, "_run_extraction", fake_run)
    claim_batch_run(
        event_id="rp-1", channel="batch:rp", tenant_id=TENANT_A, name="n",
        schema_name="invoice", model_selector="policy:x", filenames=["a.pdf"],
    )
    key = batch_database.put(name="a.pdf", content=b"A", media_type="application/pdf").relkey
    await functions._batch_extract_one(
        {"event_id": "rp-1", "schema_name": "invoice", "routing_policy": "x"},
        {"filename": "a.pdf", "relkey": key},
        0,
    )
    assert seen[0]["routing_policy"] == "x"
    assert "deployment" not in seen[0]


@pytest.mark.asyncio
async def test_webhook_signs_posts_and_retries_without_ever_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import hmac as hmac_mod

    calls: list[dict[str, Any]] = []
    responses = [500, 200]  # first attempt fails, retry succeeds

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, *, content, headers, timeout):
            calls.append({"url": url, "content": content, "headers": headers})
            import httpx as _httpx

            return _httpx.Response(responses[min(len(calls) - 1, len(responses) - 1)])

    monkeypatch.setattr(functions.httpx, "AsyncClient", _FakeClient)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(functions.asyncio, "sleep", no_sleep)

    record = {
        "event_id": "wh-1", "name": "n", "status": "completed",
        "total_items": 2, "done_items": 2, "failed_items": 0,
    }
    fake_secret = "not-a-real-secret"  # noqa: S105 - test fixture
    data = {"callback_url": "https://hooks.example/done", "callback_secret": fake_secret}
    await functions._deliver_batch_webhook(data, record)

    assert len(calls) == 2  # 500 then 200: retried, then stopped
    body = calls[0]["content"]
    expected_sig = hmac_mod.new(fake_secret.encode(), body, hashlib.sha256).hexdigest()
    assert calls[0]["headers"]["X-DocIE-Signature"] == expected_sig
    payload = json.loads(body)
    assert payload["event"] == "batch.settled"
    assert payload["results"]["jsonl"] == "/v1/studio/batches/wh-1/results.jsonl"

    # All attempts fail -> gives up silently, NEVER raises (results are durable;
    # the webhook is a courtesy).
    calls.clear()
    responses[:] = [500, 500, 500]
    await functions._deliver_batch_webhook(data, record)
    assert len(calls) == 3

    # No callback_url -> no traffic at all.
    calls.clear()
    await functions._deliver_batch_webhook({}, record)
    assert calls == []


def _seed_settled_batch_with_failures(blobs, *, event_id: str = "rt-1") -> str:
    ok_key = blobs.put(name="ok.pdf", content=b"OK", media_type="application/pdf").relkey
    bad_key = blobs.put(name="bad.pdf", content=b"BAD", media_type="application/pdf").relkey
    claim_batch_run(
        event_id=event_id, channel=f"batch:{event_id}", tenant_id=TENANT_A, name="q3",
        schema_name="invoice", model_selector="cheap", filenames=["ok.pdf", "bad.pdf"],
        input_relkeys=[ok_key, bad_key],
        selectors={"model_profile": "cheap", "language": "fr"},
    )
    record_batch_item(event_id=event_id, position=0, status="done", result={})
    record_batch_item(event_id=event_id, position=1, status="failed", error="boom")
    settle_batch_run(event_id=event_id, status="completed")
    return bad_key


def test_retry_failed_refires_only_failures_with_original_selectors(
    client: TestClient, captured_events: list[dict[str, Any]], batch_database
) -> None:
    bad_key = _seed_settled_batch_with_failures(batch_database)
    resp = client.post("/v1/studio/batches/rt-1/retry-failed", json={}, headers=_hdr())
    assert resp.status_code == 200, resp.text
    data = captured_events[0]["data"]
    assert data["name"] == "retry: q3"
    assert data["retry_of"] == "rt-1"
    assert data["inputs"] == [{"filename": "bad.pdf", "relkey": bad_key}]  # ONLY the failure
    assert data["model_profile"] == "cheap"  # original selectors carried
    assert data["language"] == "fr"


def test_retry_failed_supports_a_stronger_model_override(
    client: TestClient, captured_events: list[dict[str, Any]], batch_database
) -> None:
    _seed_settled_batch_with_failures(batch_database, event_id="rt-2")
    resp = client.post(
        "/v1/studio/batches/rt-2/retry-failed",
        json={"routing_policy": "cheap-then-strong"},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    data = captured_events[0]["data"]
    # The override REPLACES the original model selector entirely...
    assert data["routing_policy"] == "cheap-then-strong"
    assert "model_profile" not in data
    # ...but non-model selectors survive.
    assert data["language"] == "fr"


def test_retry_failed_guards(client: TestClient, captured_events, batch_database) -> None:
    # running -> 409
    claim_batch_run(
        event_id="rt-run", channel="batch:rt-run", tenant_id=TENANT_A, name="n",
        schema_name="invoice", model_selector=None, filenames=["a.pdf"], input_relkeys=["k"],
    )
    resp = client.post("/v1/studio/batches/rt-run/retry-failed", json={}, headers=_hdr())
    assert resp.status_code == 409
    # no failures -> 400
    claim_batch_run(
        event_id="rt-clean", channel="batch:rt-clean", tenant_id=TENANT_A, name="n",
        schema_name="invoice", model_selector=None, filenames=["a.pdf"], input_relkeys=["k"],
    )
    record_batch_item(event_id="rt-clean", position=0, status="done", result={})
    settle_batch_run(event_id="rt-clean", status="completed")
    resp = client.post("/v1/studio/batches/rt-clean/retry-failed", json={}, headers=_hdr())
    assert resp.status_code == 400
    # pre-upgrade rows (no input_relkey) -> 409 naming the files
    claim_batch_run(
        event_id="rt-old", channel="batch:rt-old", tenant_id=TENANT_A, name="n",
        schema_name="invoice", model_selector=None, filenames=["legacy.pdf"],
    )
    record_batch_item(event_id="rt-old", position=0, status="failed", error="x")
    settle_batch_run(event_id="rt-old", status="completed")
    old = client.post("/v1/studio/batches/rt-old/retry-failed", json={}, headers=_hdr())
    assert old.status_code == 409
    assert "legacy.pdf" in old.json()["detail"]
    # blob deleted since -> 410
    gone_key = _seed_settled_batch_with_failures(batch_database, event_id="rt-gone")
    batch_database.delete(gone_key)
    resp = client.post("/v1/studio/batches/rt-gone/retry-failed", json={}, headers=_hdr())
    assert resp.status_code == 410
    # foreign tenant -> 404
    _seed_settled_batch_with_failures(batch_database, event_id="rt-t")
    resp = client.post("/v1/studio/batches/rt-t/retry-failed", json={}, headers=_hdr("key-b"))
    assert resp.status_code == 404
    assert captured_events == []  # every guard fired before any event
