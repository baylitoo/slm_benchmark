"""Batch schedules: the durable store, the CRUD + run-now routes, and the
cron tick that turns due schedules into ordinary batch events.

The Inngest cron machinery is the framework's; what's tested here is
everything the schedule feature owns: interval validation, the store
lifecycle (create / list / patch-with-reschedule / delete, tenant-scoped,
degrade-to-empty on the cron side), the create route's source-batch
re-materialization contract (404 unknown, 409 pre-durable-inputs, selector
override exclusivity), run-now (fires the exact cron event without touching
the cadence; 410 when the source's blobs are gone), and the tick itself
(due -> event fired + next_run_at advanced; disabled/not-due skipped;
missing blobs -> last_error once per interval; a dead send leaves the
schedule due so the next tick retries).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import docie_bench.api as api
from docie_bench.inngest import functions
from docie_bench.storage.db import dispose_engine, init_engine, session_scope
from docie_bench.studio import store as studio_store
from docie_bench.studio.batch_store import claim_batch_run, settle_batch_run
from docie_bench.studio.models import BatchSchedule, utcnow
from docie_bench.studio.schedule_store import (
    ScheduleStoreUnavailableError,
    ScheduleValidationError,
    create_schedule,
    delete_schedule,
    due_schedules,
    get_schedule,
    interval_delta,
    list_schedules,
    mark_fired,
    update_schedule,
)

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


@pytest.fixture(autouse=True)
def schedule_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    init_engine(f"sqlite:///{tmp_path / 'schedules.db'}")
    # Every blob-store user goes through default_blob_store(); point it at a
    # temp dir. schedule_store reads it via studio_store attribute access, so
    # the one patch covers store, routes, and worker helpers alike.
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


@pytest.fixture
def captured_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    from docie_bench.inngest.client import inngest_client

    events: list[dict[str, Any]] = []

    async def fake_send(event):
        events.append({"name": event.name, "data": dict(event.data)})
        return [f"ev-{len(events)}"]

    monkeypatch.setattr(inngest_client, "send", fake_send)
    return events


def _hdr(key: str = "key-a") -> dict[str, str]:
    return {"X-API-Key": key}


def _seed_source_batch(
    blobs,
    *,
    event_id: str = "src-1",
    tenant_id: str = TENANT_A,
    selectors: dict[str, Any] | None = None,
) -> list[str]:
    """A settled batch with durably stored inputs -- what schedules re-run."""
    keys = [
        blobs.put(name=fn, content=fn.encode(), media_type="application/pdf").relkey
        for fn in ("a.pdf", "b.pdf")
    ]
    claim_batch_run(
        event_id=event_id,
        channel=f"batch:{event_id}",
        tenant_id=tenant_id,
        name="q3 invoices",
        schema_name="invoice",
        model_selector="lfm2.5-350m",
        filenames=["a.pdf", "b.pdf"],
        input_relkeys=keys,
        selectors=selectors
        if selectors is not None
        else {"model_profile": "lfm2.5-350m", "language": "fr"},
    )
    settle_batch_run(event_id=event_id, status="completed")
    return keys


def _make_due(schedule_id: str, *, minutes_ago: int = 5) -> None:
    with session_scope() as session:
        assert session is not None
        row = session.get(BatchSchedule, schedule_id)
        assert row is not None
        row.next_run_at = utcnow() - dt.timedelta(minutes=minutes_ago)


# -- interval validation ------------------------------------------------------


@pytest.mark.parametrize(
    ("interval", "n", "expected_minutes"),
    [
        ("hourly", None, 60),
        ("daily", None, 1440),
        ("weekly", None, 10080),
        ("every_n_minutes", 15, 15),
    ],
)
def test_interval_delta_safe_set(interval: str, n: int | None, expected_minutes: int) -> None:
    assert interval_delta(interval, n) == dt.timedelta(minutes=expected_minutes)


@pytest.mark.parametrize(
    ("interval", "n"),
    [
        ("every_5_seconds", None),
        ("every_n_minutes", None),
        ("every_n_minutes", 14),
        ("every_n_minutes", 10081),
    ],
)
def test_interval_delta_rejects_unsafe_intervals(interval: str, n: int | None) -> None:
    with pytest.raises(ScheduleValidationError):
        interval_delta(interval, n)


# -- store -------------------------------------------------------------------


def test_store_create_list_get_delete_lifecycle() -> None:
    created = create_schedule(
        tenant_id=TENANT_A,
        name="nightly",
        source_event_id="src-1",
        schema_name="invoice",
        selectors={"model_profile": "nuextract3"},
        interval="daily",
        every_n_minutes=None,
    )
    assert created["enabled"] is True
    assert created["last_run_at"] is None
    # next_run_at starts one interval out, never "right now".
    next_at = dt.datetime.fromisoformat(created["next_run_at"])
    assert next_at > dt.datetime.now(dt.UTC) + dt.timedelta(hours=23)

    assert [s["id"] for s in list_schedules(tenant_id=TENANT_A)] == [created["id"]]
    assert list_schedules(tenant_id=TENANT_B) == []
    assert get_schedule(created["id"], tenant_id=TENANT_B) is None  # foreign: not-found
    assert get_schedule(created["id"], tenant_id=TENANT_A) is not None

    assert delete_schedule(created["id"], tenant_id=TENANT_B) is False  # survives foreign delete
    assert get_schedule(created["id"], tenant_id=TENANT_A) is not None
    assert delete_schedule(created["id"], tenant_id=TENANT_A) is True
    assert list_schedules(tenant_id=TENANT_A) == []


def test_store_update_reschedules_on_interval_change_and_reenable() -> None:
    created = create_schedule(
        tenant_id=TENANT_A,
        name="n",
        source_event_id="src-1",
        schema_name="invoice",
        selectors=None,
        interval="weekly",
        every_n_minutes=None,
    )
    sid = created["id"]
    # Interval change recomputes next_run_at from now + the NEW interval.
    updated = update_schedule(
        sid, tenant_id=TENANT_A, interval="every_n_minutes", every_n_minutes=30
    )
    assert updated is not None
    assert (updated["interval"], updated["every_n_minutes"]) == ("every_n_minutes", 30)
    next_at = dt.datetime.fromisoformat(updated["next_run_at"])
    assert next_at < dt.datetime.now(dt.UTC) + dt.timedelta(minutes=31)

    # Disable, drift into the past, re-enable: next_run_at jumps forward so a
    # long-disabled schedule never fires a catch-up the second it's back on.
    update_schedule(sid, tenant_id=TENANT_A, enabled=False)
    _make_due(sid)
    reenabled = update_schedule(sid, tenant_id=TENANT_A, enabled=True)
    assert reenabled is not None
    assert reenabled["enabled"] is True
    assert dt.datetime.fromisoformat(reenabled["next_run_at"]) > dt.datetime.now(dt.UTC)

    # A bad interval patch changes nothing.
    with pytest.raises(ScheduleValidationError):
        update_schedule(sid, tenant_id=TENANT_A, interval="every_n_minutes", every_n_minutes=1)
    assert update_schedule("nope", tenant_id=TENANT_A, enabled=False) is None


def test_store_requires_a_database_for_crud_but_scan_degrades() -> None:
    dispose_engine()
    with pytest.raises(ScheduleStoreUnavailableError):
        create_schedule(
            tenant_id=TENANT_A,
            name="n",
            source_event_id="s",
            schema_name="invoice",
            selectors=None,
            interval="daily",
            every_n_minutes=None,
        )
    with pytest.raises(ScheduleStoreUnavailableError):
        list_schedules(tenant_id=TENANT_A)
    # The cron side must NEVER raise: no database reads as "nothing due".
    assert due_schedules() == []
    assert mark_fired("any", event_id="e") is None


# -- create route ------------------------------------------------------------


def test_create_route_defaults_from_the_source_batch(
    client: TestClient, schedule_database
) -> None:
    _seed_source_batch(schedule_database)
    resp = client.post(
        "/v1/studio/batch-schedules",
        json={"source_event_id": "src-1", "interval": "daily"},
        headers=_hdr(),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "re-run: q3 invoices"
    assert body["schema_name"] == "invoice"
    assert body["source_event_id"] == "src-1"
    # Selectors carried from the source submission verbatim.
    assert body["selectors"] == {"model_profile": "lfm2.5-350m", "language": "fr"}
    assert body["enabled"] is True


def test_create_route_model_override_replaces_the_source_selector(
    client: TestClient, schedule_database
) -> None:
    _seed_source_batch(schedule_database)
    resp = client.post(
        "/v1/studio/batch-schedules",
        json={
            "source_event_id": "src-1",
            "interval": "hourly",
            "routing_policy": "cheap-then-strong",
            "name": "hourly rerun",
        },
        headers=_hdr(),
    )
    assert resp.status_code == 201, resp.text
    selectors = resp.json()["selectors"]
    assert selectors["routing_policy"] == "cheap-then-strong"
    assert "model_profile" not in selectors  # replaced, not stacked
    assert selectors["language"] == "fr"  # non-model selectors survive


def test_create_route_guards(client: TestClient, schedule_database) -> None:
    _seed_source_batch(schedule_database)
    # unknown source batch -> 404; foreign tenant's batch -> 404 too
    assert (
        client.post(
            "/v1/studio/batch-schedules",
            json={"source_event_id": "nope", "interval": "daily"},
            headers=_hdr(),
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/v1/studio/batch-schedules",
            json={"source_event_id": "src-1", "interval": "daily"},
            headers=_hdr("key-b"),
        ).status_code
        == 404
    )
    # bad interval / sub-floor cadence -> 422
    for body in (
        {"source_event_id": "src-1", "interval": "every_second"},
        {"source_event_id": "src-1", "interval": "every_n_minutes", "every_n_minutes": 5},
        {"source_event_id": "src-1", "interval": "every_n_minutes"},
    ):
        resp = client.post("/v1/studio/batch-schedules", json=body, headers=_hdr())
        assert resp.status_code == 422, resp.text
    # model override exclusivity -> 400
    resp = client.post(
        "/v1/studio/batch-schedules",
        json={
            "source_event_id": "src-1",
            "interval": "daily",
            "deployment": "d",
            "model_profile": "m",
        },
        headers=_hdr(),
    )
    assert resp.status_code == 400
    assert "mutually exclusive" in resp.json()["detail"]
    # a batch whose items predate durable input storage -> 409
    claim_batch_run(
        event_id="src-legacy",
        channel="batch:legacy",
        tenant_id=TENANT_A,
        name="legacy",
        schema_name="invoice",
        model_selector=None,
        filenames=["old.pdf"],
    )
    settle_batch_run(event_id="src-legacy", status="completed")
    resp = client.post(
        "/v1/studio/batch-schedules",
        json={"source_event_id": "src-legacy", "interval": "daily"},
        headers=_hdr(),
    )
    assert resp.status_code == 409
    assert "old.pdf" in resp.json()["detail"]
    assert client.get("/v1/studio/batch-schedules", headers=_hdr()).json() == []


def test_create_route_requires_the_database(client: TestClient) -> None:
    dispose_engine()
    resp = client.post(
        "/v1/studio/batch-schedules",
        json={"source_event_id": "src-1", "interval": "daily"},
        headers=_hdr(),
    )
    assert resp.status_code == 503


# -- list / patch / delete routes --------------------------------------------


def test_list_patch_delete_routes_are_tenant_scoped(
    client: TestClient, schedule_database
) -> None:
    _seed_source_batch(schedule_database)
    _seed_source_batch(schedule_database, event_id="src-b", tenant_id=TENANT_B)
    a = client.post(
        "/v1/studio/batch-schedules",
        json={"source_event_id": "src-1", "interval": "daily"},
        headers=_hdr(),
    ).json()
    client.post(
        "/v1/studio/batch-schedules",
        json={"source_event_id": "src-b", "interval": "weekly"},
        headers=_hdr("key-b"),
    )
    assert [s["id"] for s in client.get("/v1/studio/batch-schedules", headers=_hdr()).json()] == [
        a["id"]
    ]

    # PATCH: disable, then change the interval; foreign tenant reads not-found.
    off = client.patch(
        f"/v1/studio/batch-schedules/{a['id']}", json={"enabled": False}, headers=_hdr()
    )
    assert off.status_code == 200
    assert off.json()["enabled"] is False
    changed = client.patch(
        f"/v1/studio/batch-schedules/{a['id']}",
        json={"interval": "every_n_minutes", "every_n_minutes": 45},
        headers=_hdr(),
    )
    assert changed.status_code == 200
    assert (changed.json()["interval"], changed.json()["every_n_minutes"]) == (
        "every_n_minutes",
        45,
    )
    assert (
        client.patch(
            f"/v1/studio/batch-schedules/{a['id']}", json={"enabled": True}, headers=_hdr("key-b")
        ).status_code
        == 404
    )
    bad = client.patch(
        f"/v1/studio/batch-schedules/{a['id']}",
        json={"every_n_minutes": 3},
        headers=_hdr(),
    )
    assert bad.status_code == 422
    empty = client.patch(f"/v1/studio/batch-schedules/{a['id']}", json={}, headers=_hdr())
    assert empty.status_code == 422

    # DELETE: foreign tenant 404s and the row survives; the owner deletes it.
    assert (
        client.delete(f"/v1/studio/batch-schedules/{a['id']}", headers=_hdr("key-b")).status_code
        == 404
    )
    assert client.delete(f"/v1/studio/batch-schedules/{a['id']}", headers=_hdr()).status_code == 200
    assert client.get("/v1/studio/batch-schedules", headers=_hdr()).json() == []
    assert client.delete(f"/v1/studio/batch-schedules/{a['id']}", headers=_hdr()).status_code == 404


# -- run-now -----------------------------------------------------------------


def test_run_now_fires_the_cron_event_without_touching_the_cadence(
    client: TestClient, captured_events: list[dict[str, Any]], schedule_database
) -> None:
    keys = _seed_source_batch(schedule_database)
    created = client.post(
        "/v1/studio/batch-schedules",
        json={"source_event_id": "src-1", "interval": "daily", "name": "nightly"},
        headers=_hdr(),
    ).json()

    resp = client.post(
        f"/v1/studio/batch-schedules/{created['id']}/run-now", headers=_hdr()
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["channel"].startswith("batch:")
    assert len(captured_events) == 1
    ev = captured_events[0]
    assert ev["name"] == "doc/batch.requested"
    data = ev["data"]
    assert data["tenant_id"] == TENANT_A
    assert data["name"] == "scheduled: nightly"
    assert data["scheduled_by"] == created["id"]
    assert data["scheduled_from"] == "src-1"
    assert data["model_profile"] == "lfm2.5-350m"
    assert [i["relkey"] for i in data["inputs"]] == keys  # re-read, never re-uploaded

    after = client.get("/v1/studio/batch-schedules", headers=_hdr()).json()[0]
    assert after["last_event_id"] == "ev-1"
    assert after["last_run_at"] is not None
    assert after["next_run_at"] == created["next_run_at"]  # cadence untouched


def test_run_now_guards(
    client: TestClient, captured_events: list[dict[str, Any]], schedule_database
) -> None:
    keys = _seed_source_batch(schedule_database)
    created = client.post(
        "/v1/studio/batch-schedules",
        json={"source_event_id": "src-1", "interval": "daily"},
        headers=_hdr(),
    ).json()
    # foreign tenant / unknown id -> 404
    assert (
        client.post(
            f"/v1/studio/batch-schedules/{created['id']}/run-now", headers=_hdr("key-b")
        ).status_code
        == 404
    )
    assert (
        client.post("/v1/studio/batch-schedules/nope/run-now", headers=_hdr()).status_code == 404
    )
    # a source blob swept since -> 410, nothing enqueued
    schedule_database.delete(keys[0])
    gone = client.post(f"/v1/studio/batch-schedules/{created['id']}/run-now", headers=_hdr())
    assert gone.status_code == 410
    assert "a.pdf" in gone.json()["detail"]
    assert captured_events == []


# -- the cron tick -----------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_fires_due_schedules_and_advances_next_run_at(
    captured_events: list[dict[str, Any]], schedule_database
) -> None:
    keys = _seed_source_batch(schedule_database)
    due = create_schedule(
        tenant_id=TENANT_A,
        name="nightly",
        source_event_id="src-1",
        schema_name="invoice",
        selectors={"model_profile": "nuextract3"},
        interval="every_n_minutes",
        every_n_minutes=15,
    )
    not_due = create_schedule(
        tenant_id=TENANT_A,
        name="later",
        source_event_id="src-1",
        schema_name="invoice",
        selectors=None,
        interval="daily",
        every_n_minutes=None,
    )
    disabled = create_schedule(
        tenant_id=TENANT_A,
        name="off",
        source_event_id="src-1",
        schema_name="invoice",
        selectors=None,
        interval="hourly",
        every_n_minutes=None,
        enabled=False,
    )
    _make_due(due["id"])
    _make_due(disabled["id"])

    summary = await functions._run_batch_schedule_tick()

    assert summary == {"due": 1, "fired": 1, "skipped": 0, "send_errors": 0}
    assert len(captured_events) == 1
    data = captured_events[0]["data"]
    assert captured_events[0]["name"] == "doc/batch.requested"
    assert data["scheduled_by"] == due["id"]
    assert data["model_profile"] == "nuextract3"
    assert [i["relkey"] for i in data["inputs"]] == keys

    fired = get_schedule(due["id"], tenant_id=TENANT_A)
    assert fired is not None
    assert fired["last_event_id"] == "ev-1"
    assert fired["last_error"] is None
    next_at = dt.datetime.fromisoformat(fired["next_run_at"])
    now = dt.datetime.now(dt.UTC)
    assert now < next_at <= now + dt.timedelta(minutes=15)
    # The others were never touched.
    untouched = get_schedule(not_due["id"], tenant_id=TENANT_A)
    assert untouched is not None
    assert untouched["last_event_id"] is None
    still_off = get_schedule(disabled["id"], tenant_id=TENANT_A)
    assert still_off is not None
    assert still_off["last_event_id"] is None

    # A second immediate tick sees nothing due: advancing IS the idempotency.
    assert (await functions._run_batch_schedule_tick())["due"] == 0
    assert len(captured_events) == 1


@pytest.mark.asyncio
async def test_tick_records_last_error_once_per_interval_when_blobs_are_gone(
    captured_events: list[dict[str, Any]], schedule_database
) -> None:
    keys = _seed_source_batch(schedule_database)
    sched = create_schedule(
        tenant_id=TENANT_A,
        name="broken",
        source_event_id="src-1",
        schema_name="invoice",
        selectors=None,
        interval="hourly",
        every_n_minutes=None,
    )
    _make_due(sched["id"])
    schedule_database.delete(keys[1])

    summary = await functions._run_batch_schedule_tick()

    assert summary == {"due": 1, "fired": 0, "skipped": 1, "send_errors": 0}
    assert captured_events == []
    after = get_schedule(sched["id"], tenant_id=TENANT_A)
    assert after is not None
    assert after["last_error"] is not None
    assert "b.pdf" in after["last_error"]
    assert after["last_event_id"] is None  # nothing actually ran
    # next_run_at still advanced: the error surfaces once per interval, not
    # once per minute.
    assert dt.datetime.fromisoformat(after["next_run_at"]) > dt.datetime.now(dt.UTC)
    assert (await functions._run_batch_schedule_tick())["due"] == 0


@pytest.mark.asyncio
async def test_tick_leaves_a_schedule_due_when_the_send_fails(
    monkeypatch: pytest.MonkeyPatch, schedule_database
) -> None:
    from docie_bench.inngest.client import inngest_client

    _seed_source_batch(schedule_database)
    sched = create_schedule(
        tenant_id=TENANT_A,
        name="n",
        source_event_id="src-1",
        schema_name="invoice",
        selectors=None,
        interval="daily",
        every_n_minutes=None,
    )
    _make_due(sched["id"])

    async def dead_send(event):
        raise RuntimeError("inngest server unreachable")

    monkeypatch.setattr(inngest_client, "send", dead_send)
    summary = await functions._run_batch_schedule_tick()
    assert summary == {"due": 1, "fired": 0, "skipped": 0, "send_errors": 1}
    after = get_schedule(sched["id"], tenant_id=TENANT_A)
    assert after is not None
    assert after["last_event_id"] is None
    # Still due: the NEXT tick retries once the server is back.
    assert dt.datetime.fromisoformat(after["next_run_at"]) < dt.datetime.now(dt.UTC)


@pytest.mark.asyncio
async def test_tick_records_event_owner_for_the_scheduled_run(
    captured_events: list[dict[str, Any]], schedule_database
) -> None:
    # The tenant-scoped /runs/{event_id} proxy consults StudioEventOwner; a
    # cron-fired batch must be readable by the schedule's own tenant.
    from docie_bench.studio.models import StudioEventOwner

    _seed_source_batch(schedule_database)
    sched = create_schedule(
        tenant_id=TENANT_A,
        name="n",
        source_event_id="src-1",
        schema_name="invoice",
        selectors=None,
        interval="daily",
        every_n_minutes=None,
    )
    _make_due(sched["id"])
    await functions._run_batch_schedule_tick()
    with session_scope() as session:
        assert session is not None
        owner = session.get(StudioEventOwner, "ev-1")
        assert owner is not None
        assert owner.tenant_id == TENANT_A
