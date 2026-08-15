"""GET /v1/studio/seeds -- the Downloads tab's durable listing route."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import docie_bench.api as api
from docie_bench.storage.db import dispose_engine, init_engine
from docie_bench.studio.seed_store import claim_seed_run, complete_seed_run, fail_seed_run


@pytest.fixture(autouse=True)
def seed_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'seeds.db'}")
    yield
    dispose_engine()


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def test_lists_seed_runs_for_the_authenticated_tenant(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docie_bench import security
    from docie_bench.security import TenantQuotaManager

    monkeypatch.setattr(
        security,
        "get_quota_manager",
        lambda: TenantQuotaManager(
            api_keys={"secret-a": "tenant-a"},
            auth_required=True,
            requests_per_window=100,
            window_seconds=60,
            max_concurrent=10,
        ),
    )

    claim_seed_run(
        event_id="ev1", channel="seed:a", tenant_id="tenant-a", kind="ollama",
        name="qwen", reference="qwen2.5:1.5b",
    )
    complete_seed_run(event_id="ev1", result={"name": "qwen", "size_bytes": 10})
    claim_seed_run(
        event_id="ev2", channel="seed:b", tenant_id="tenant-a", kind="hf",
        name="", reference="LiquidAI/LFM2.5-350M-Instruct-GGUF",
    )
    fail_seed_run(event_id="ev2", error="404 repo not found")

    resp = client.get("/v1/studio/seeds", headers={"X-API-Key": "secret-a"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    by_id = {r["event_id"]: r for r in body}
    assert by_id["ev1"]["status"] == "completed"
    assert by_id["ev1"]["result"]["size_bytes"] == 10
    assert by_id["ev2"]["status"] == "failed"
    assert by_id["ev2"]["error"] == "404 repo not found"


def test_no_database_returns_empty_list_not_500(client: TestClient) -> None:
    dispose_engine()

    resp = client.get("/v1/studio/seeds")

    assert resp.status_code == 200
    assert resp.json() == []


def test_seeds_are_not_visible_across_tenants(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docie_bench import security
    from docie_bench.security import TenantQuotaManager

    claim_seed_run(
        event_id="ev1", channel="seed:a", tenant_id="tenant-a", kind="ollama", name="a"
    )

    manager = TenantQuotaManager(
        api_keys={"secret-a": "tenant-a", "secret-b": "tenant-b"},
        auth_required=True,
        requests_per_window=100,
        window_seconds=60,
        max_concurrent=10,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)

    mine = client.get("/v1/studio/seeds", headers={"X-API-Key": "secret-a"}).json()
    theirs = client.get("/v1/studio/seeds", headers={"X-API-Key": "secret-b"}).json()

    assert [r["event_id"] for r in mine] == ["ev1"]
    assert theirs == []


def test_seed_triggers_carry_tenant_id_so_the_worker_can_scope_its_seedrun(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seed-hf/seed-ollama trigger routes are the one place that used to
    NOT put tenant_id in the event payload (unlike extract/benchmark) --
    _record_event_owners recorded ownership separately, but the worker job
    itself had no tenant to scope a SeedRun row by. Confirms both routes now
    include it, end to end through the real event payload the worker reads."""
    from docie_bench import security
    from docie_bench.inngest.client import inngest_client
    from docie_bench.security import TenantQuotaManager

    monkeypatch.setattr(
        security,
        "get_quota_manager",
        lambda: TenantQuotaManager(
            api_keys={"secret-a": "tenant-a"},
            auth_required=True,
            requests_per_window=100,
            window_seconds=60,
            max_concurrent=10,
        ),
    )
    captured: list[dict] = []

    async def fake_send(event):
        captured.append(dict(event.data))
        return [f"ev-{len(captured)}"]

    monkeypatch.setattr(inngest_client, "send", fake_send)

    client.post(
        "/v1/studio/seed-ollama",
        json={"reference": "qwen2.5:1.5b", "name": "q"},
        headers={"X-API-Key": "secret-a"},
    )
    client.post(
        "/v1/studio/seed-hf",
        json={"repo": "LiquidAI/LFM2.5-350M-Instruct-GGUF"},
        headers={"X-API-Key": "secret-a"},
    )

    assert len(captured) == 2
    assert all(data["tenant_id"] == "tenant-a" for data in captured)
