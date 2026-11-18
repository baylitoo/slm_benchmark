"""Lifecycle actions on a fleet go to the node that owns the deployment (#590)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.settings import get_settings

GIB = 1 << 30
NAME = "lfm3-1.7b-f09ae4"


@pytest.fixture
def central(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """The api process: no local record for NAME, events captured."""
    from docie_bench.inngest import serving_api
    from docie_bench.serving.node_identity import resolve_node_id

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    (tmp_path / "serving").mkdir()
    monkeypatch.setenv("DOCIE_SERVING_NODE_ID", "central")
    resolve_node_id.cache_clear()
    get_settings.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    sent: list[Any] = []

    async def fake_send(client: object, event: Any) -> list[str]:
        sent.append(event)
        return ["evt"]

    monkeypatch.setattr(serving_api, "send_or_503", fake_send)
    try:
        yield sent
    finally:
        db.dispose_engine()
        resolve_node_id.cache_clear()
        get_settings.cache_clear()


def _node(node_id: str) -> None:
    ModelCatalog().publish_node_snapshot(
        node_id=node_id, total_bytes=8 * GIB, free_bytes=4 * GIB, source="cgroup", sum_rss_bytes=0
    )


def _place(name: str, node_id: str) -> None:
    ModelCatalog().record_placement(
        name,
        model_name="lfm3-1.7b",
        engine="llama-server",
        endpoint=f"http://{node_id}:8088/v1",
        state="ready",
        node_id=node_id,
    )


def test_on_a_fleet_a_pin_is_queued_for_the_owner_not_sent_as_an_event(
    central: list[Any],
) -> None:
    from docie_bench.inngest.serving_api import PinRequest, pin_deployment

    _node("central")
    _node("rocky-llm")
    _place(NAME, "rocky-llm")

    result = asyncio.run(pin_deployment(NAME, PinRequest(pinned=True), tenant=None))  # type: ignore[arg-type]

    assert central == []
    assert result["node_id"] == "rocky-llm"
    (request,) = ModelCatalog().pending_node_requests("rocky-llm")
    assert (request["deployment"], request["verb"]) == (NAME, "pin")
    assert ModelCatalog().pending_node_requests("central") == []


def test_a_single_node_keeps_the_event_path(central: list[Any]) -> None:
    from docie_bench.inngest.serving_api import _route_to_owner

    _node("rocky-llm")
    _place(NAME, "rocky-llm")

    routed = _route_to_owner(NAME, event="serving/pin.requested", extra={}, remote_action=None)
    assert routed is None


def test_a_delete_also_removes_the_desired_row_so_it_cannot_come_back(
    central: list[Any],
) -> None:
    from docie_bench.inngest.serving_api import delete_deployment

    _node("central")
    _node("rocky-llm")
    _place(NAME, "rocky-llm")
    catalog = ModelCatalog()
    catalog.set_desired_deployment(
        "rocky-llm", NAME, desired_state="running", spec={"model": "lfm3-1.7b"}
    )

    asyncio.run(delete_deployment(NAME, tenant=None))  # type: ignore[arg-type]

    assert catalog.desired_snapshot("rocky-llm")["deployments"] == []
    (request,) = catalog.pending_node_requests("rocky-llm")
    assert request["verb"] == "delete"


def test_the_owner_runs_its_own_requests_and_records_the_outcome(central: list[Any]) -> None:
    from docie_bench.serving.reconciler import ServingReconciler

    catalog = ModelCatalog()
    ok = catalog.enqueue_node_request("central", "nuextract3", "pin")
    bad = catalog.enqueue_node_request("central", "lfm2.5-350m", "repair", {"port": 8090})
    other = catalog.enqueue_node_request("rocky-llm", NAME, "unload")
    ran: list[tuple[str, str, dict[str, Any]]] = []

    def execute(verb: str, name: str, args: dict[str, Any]) -> None:
        ran.append((verb, name, args))
        if verb == "repair":
            raise RuntimeError("port 8090 is in use")

    ServingReconciler._run_node_requests(SimpleNamespace(_request_executor=execute))  # type: ignore[arg-type]

    assert ran == [("pin", "nuextract3", {}), ("repair", "lfm2.5-350m", {"port": 8090})]
    latest = catalog.latest_node_requests(["nuextract3", "lfm2.5-350m", NAME])
    assert latest["nuextract3"]["status"] == "done"
    assert latest["lfm2.5-350m"]["status"] == "failed"
    assert latest["lfm2.5-350m"]["error"] == "port 8090 is in use"
    assert latest[NAME]["id"] == other["id"]
    assert latest[NAME]["status"] == "pending"
    assert ok["id"] != bad["id"]


def test_a_request_no_node_picks_up_expires(central: list[Any]) -> None:
    catalog = ModelCatalog()
    catalog.enqueue_node_request("rocky-llm", NAME, "pin")

    assert catalog.expire_node_requests(-1) == 1
    assert catalog.latest_node_requests([NAME])[NAME]["status"] == "expired"


def test_the_listing_shows_a_pending_request_and_the_owners_pin(central: list[Any]) -> None:
    from docie_bench.inngest.serving_api import list_deployments

    _node("central")
    _node("rocky-llm")
    ModelCatalog().publish_observed(
        NAME,
        node_id="rocky-llm",
        engine="llama-server",
        state="ready",
        endpoint="http://rocky-llm:8088/v1",
        phase="hot",
        pid=None,
        pid_create_time=None,
        rss_bytes=0,
        health_ok=True,
        last_error=None,
        context_length=8192,
        pinned=True,
    )
    ModelCatalog().enqueue_node_request("rocky-llm", NAME, "unpin")

    rows = {r["spec"]["name"]: r for r in asyncio.run(list_deployments())}

    assert rows[NAME]["pinned"] is True
    assert rows[NAME]["spec"]["launch"]["context_length"] == 8192
    assert rows[NAME]["node_request"]["verb"] == "unpin"
    assert rows[NAME]["node_request"]["status"] == "pending"


def test_a_deleted_deployment_is_forgotten_by_convergence(central: list[Any]) -> None:
    from docie_bench.serving.applied_state import (
        AppliedState,
        read_applied_state,
        write_applied_state,
    )
    from docie_bench.serving.convergence import plan_convergence
    from docie_bench.serving.reconciler import ServingReconciler

    write_applied_state(AppliedState(generation=3, names=frozenset({NAME}), loads={NAME: "r1"}))
    ModelCatalog().enqueue_node_request("central", NAME, "delete")

    ServingReconciler._run_node_requests(  # type: ignore[arg-type]
        SimpleNamespace(_request_executor=lambda verb, name, args: None)
    )

    applied = read_applied_state()
    assert NAME not in applied.names
    assert NAME not in applied.loads
    # Two later cycles with no desired row and no local record: nothing to do.
    for _ in range(2):
        plan = plan_convergence(desired={}, local_names=[], previously_applied=applied.names)
        assert not plan.to_start
        assert not plan.to_remove
        assert not plan.to_stop


@pytest.mark.parametrize("route", ["delete", "drain"])
def test_a_client_tier_key_cannot_change_nodes_or_deployments(
    central: list[Any], route: str
) -> None:
    from fastapi import HTTPException

    from docie_bench.inngest.serving_api import NodeDrainRequest, delete_deployment, drain_node
    from docie_bench.security import TenantContext

    client_key = TenantContext(tenant_id="adbi-app", authenticated=True, api_key_id="k1")
    call = (
        delete_deployment(NAME, tenant=client_key)
        if route == "delete"
        else drain_node("rocky-llm", NodeDrainRequest(), tenant=client_key)
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(call)
    assert exc.value.status_code == 403


def test_a_studio_sign_in_may_still_act(central: list[Any]) -> None:
    from docie_bench.inngest.serving_api import _require_fleet_manager
    from docie_bench.security import TenantContext

    _require_fleet_manager(TenantContext(tenant_id="ops", authenticated=True, studio_user=True))
    _require_fleet_manager(TenantContext(tenant_id="ops", authenticated=True, operator=True))
    _require_fleet_manager(None)


def test_activity_does_not_count_a_replica_on_a_drained_node_as_live(central: list[Any]) -> None:
    catalog = ModelCatalog()
    _node("central")
    _node("rocky-llm")
    _place(NAME, "rocky-llm")
    catalog.record_activity("lfm3-1.7b")
    assert catalog.list_activity()[0]["live_replica_count"] == 1

    catalog.set_node_drain("rocky-llm", drained=True)

    assert catalog.list_activity()[0]["live_replica_count"] == 0


def test_usage_is_attributed_to_the_node_that_served_it(central: list[Any]) -> None:
    from docie_bench.studio.usage_store import record_usage, usage_by_node

    _place(NAME, "rocky-llm")
    record_usage(deployment=NAME, surface="chat", tenant_id="ops", latency_ms=120)
    record_usage(
        deployment="nuextract3",
        surface="chat",
        tenant_id="ops",
        latency_ms=80,
        node_id="central",
        status="error",
    )
    record_usage(deployment="store:lfm3-1.7b", surface="chat", tenant_id="ops", latency_ms=50)

    rows = {r["node_id"]: r for r in usage_by_node(tenant_id="ops", window_hours=24)}

    assert rows["rocky-llm"]["requests"] == 1
    assert rows["central"]["errors"] == 1
    assert rows[None]["requests"] == 1
