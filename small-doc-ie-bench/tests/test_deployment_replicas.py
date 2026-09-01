"""Deployment replicas: N instances of one store model, load-balanced.

Covers the four seams the feature adds on top of the existing scale plumbing:

* deploy fan-out — ``POST /v1/studio/deploy`` with ``replicas: N`` fans out one
  ordinary deploy event per missing replica (``<model>``, ``<model>-2``, …);
* RAM admission — N x the per-instance footprint is checked against the SAME
  sizing budget the Sizing tab prices with, and a provable deficit is a 422
  BEFORE any event is queued (fail-open when the budget is unknowable);
* round-robin routing — ``store:<name>`` resolution rotates across the live
  replica endpoints with a process-local counter, skipping dead replicas;
* scale down — ``POST /store/{name}/scale`` below the current count drains the
  highest-suffix replicas through the real delete job, never the bare base;
* replica counts — ``GET /v1/serving/deployments`` annotates each record with
  its ``replica_group`` and group size.

No live processes anywhere: deployments are seeded through a fake adapter
(the test_serving_api pattern) and events are captured on a stubbed client.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.serving.control_plane import (
    replica_names_to_add,
    replica_names_to_remove,
)
from docie_bench.serving.runtime import RuntimeKind, RuntimeLaunchSpec
from docie_bench.serving.supervisor import DeploymentSpec, PersistentSupervisor
from docie_bench.settings import get_settings

GIB = 1024**3
MODEL = "lfm2.5-350m"


class _FakeAdapter:
    def __init__(self) -> None:
        self.next_pid = 500

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> object:
        del log_path
        from docie_bench.serving.runtime import RuntimeProcess

        self.next_pid += 1
        return RuntimeProcess(spec.runtime, f"http://{spec.host}:{spec.port}/v1", self.next_pid)

    def is_running(self, pid: int | None) -> bool:
        return pid is not None

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del pid, timeout

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> object:
        del spec, timeout
        from docie_bench.serving.runtime import HealthResult

        return HealthResult(True, 200)


def _seed_replicas(home: Path, records: dict[str, tuple[str, int]]) -> None:
    """Seed deployment records ``name -> (alias, port)``. Replicas of one store
    model share the launch ``--alias`` (the base name) while their RECORD names
    differ — the exact shape ``serve_store_model`` writes for a scaled model."""
    supervisor = PersistentSupervisor(
        home / "deployments.json", adapters={RuntimeKind.LLAMACPP: _FakeAdapter()}
    )
    for name, (alias, port) in records.items():
        supervisor.deploy(
            DeploymentSpec(
                name=name,
                launch=RuntimeLaunchSpec(
                    runtime=RuntimeKind.LLAMACPP,
                    model=f"/models/{alias}.gguf",
                    alias=alias,
                    port=port,
                ),
            )
        )


@pytest.fixture
def serving_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "serving"
    home.mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_START", "8088")
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_END", "8188")
    get_settings.cache_clear()
    try:
        yield home
    finally:
        get_settings.cache_clear()


@pytest.fixture
def captured_events(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Capture every Inngest event the api would enqueue (shared client)."""
    from docie_bench.inngest.client import inngest_client

    sent: list[Any] = []

    async def fake_send(event: Any) -> list[str]:
        sent.append(event)
        return [f"evt-{len(sent)}"]

    monkeypatch.setattr(inngest_client, "send", fake_send)
    return sent


def _no_sizing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the admission gate fail-open: no catalog, no node snapshot."""
    from docie_bench.inngest import serving_api

    monkeypatch.setattr(serving_api, "_sizing_inputs", lambda: ([], None, [], None))


def _scale(name: str, replicas: int, **extra: Any) -> dict[str, Any]:
    from docie_bench.inngest.serving_api import ScaleRequest, scale_store_model

    return asyncio.run(
        scale_store_model(name, ScaleRequest(replicas=replicas, **extra), tenant=None)
    )


# ── deploy fan-out: replicas on the deploy trigger ──────────────────────────


def _deploy(payload_kwargs: dict[str, Any]) -> Any:
    from docie_bench.inngest.studio_api.deploy import DeployRequest, trigger_deploy

    tenant = SimpleNamespace(tenant_id="tenant-test")
    return asyncio.run(trigger_deploy(DeployRequest(**payload_kwargs), tenant=tenant))


def test_deploy_replicas_fans_out_one_event_per_instance(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sizing(monkeypatch)

    response = _deploy({"model": MODEL, "replicas": 3})

    names = [e.data["deployment_name"] for e in captured_events]
    assert names == [MODEL, f"{MODEL}-2", f"{MODEL}-3"]
    assert all(e.name == "serving/deploy.requested" for e in captured_events)
    assert all(e.data["model"] == MODEL for e in captured_events)
    # One shared progress channel; one event id per replica.
    channels = {e.data["channel"] for e in captured_events}
    assert channels == {response.channel}
    assert response.channel.startswith("deploy:")
    assert len(response.event_ids) == 3


def test_deploy_replicas_tops_up_existing_instances(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``replicas`` is a TARGET total (idempotent with the scale endpoint):
    two instances already running + replicas=3 launches exactly one more."""
    _no_sizing(monkeypatch)
    _seed_replicas(serving_home, {MODEL: (MODEL, 8088), f"{MODEL}-2": (MODEL, 8089)})

    _deploy({"model": MODEL, "replicas": 3})

    assert [e.data["deployment_name"] for e in captured_events] == [f"{MODEL}-3"]


def test_deploy_replicas_rejects_explicit_runtime(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sizing(monkeypatch)

    with pytest.raises(HTTPException) as excinfo:
        _deploy({"model": MODEL, "replicas": 2, "runtime": "llamacpp"})
    assert excinfo.value.status_code == 422
    assert "store-entry" in excinfo.value.detail
    assert captured_events == []


def test_deploy_replicas_rejects_explicit_name_and_port(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sizing(monkeypatch)

    for extra in ({"name": "my-alias"}, {"port": 8090}):
        with pytest.raises(HTTPException) as excinfo:
            _deploy({"model": MODEL, "replicas": 2, **extra})
        assert excinfo.value.status_code == 422
        assert "auto-allocated ports" in excinfo.value.detail
    assert captured_events == []


def test_deploy_single_replica_path_is_unchanged(
    serving_home: Path, captured_events: list[Any]
) -> None:
    """replicas=1 (the default) keeps the proven single-event deploy shape —
    no deployment_name, no admission detour."""
    response = _deploy({"model": MODEL})

    assert len(captured_events) == 1
    event = captured_events[0]
    assert event.data["model"] == MODEL
    assert "deployment_name" not in event.data
    assert len(response.event_ids) == 1


def test_deploy_single_instance_threads_n_parallel_and_cache_reuse(
    serving_home: Path, captured_events: list[Any]
) -> None:
    """The non-replicated path model_dumps the whole DeployRequest, so
    n_parallel/cache_reuse ride along to the worker automatically (#248/#321)."""
    _deploy({"model": MODEL, "n_parallel": 4, "cache_reuse": 256})

    assert len(captured_events) == 1
    event = captured_events[0]
    assert event.data["n_parallel"] == 4
    assert event.data["cache_reuse"] == 256


def test_deploy_replicas_thread_n_parallel_into_each_event(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replicated path builds each event dict by hand — n_parallel/
    cache_reuse must be threaded in the same way context_length already is."""
    _no_sizing(monkeypatch)

    _deploy({"model": MODEL, "replicas": 2, "n_parallel": 4, "cache_reuse": 256})

    assert len(captured_events) == 2
    assert all(e.data["n_parallel"] == 4 for e in captured_events)
    assert all(e.data["cache_reuse"] == 256 for e in captured_events)


def test_deploy_replicas_omits_n_parallel_when_default(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """n_parallel=1 (the default) is not sent -- mirrors how max_tokens is only
    included when set, keeping the event shape unchanged for ordinary deploys."""
    _no_sizing(monkeypatch)

    _deploy({"model": MODEL, "replicas": 2})

    assert all("n_parallel" not in e.data for e in captured_events)
    assert all("cache_reuse" not in e.data for e in captured_events)


# ── RAM admission: N x footprint against the sizing budget ─────────────────


def _sized_inputs(
    *, size_gib: float, total_gib: float, free_gib: float
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], None]:
    models = [
        {
            "name": MODEL,
            "family": "openai_chat",
            "size_bytes": int(size_gib * GIB),
            "model_path": None,
            "mmproj_path": None,
        }
    ]
    snapshot = {
        "total_bytes": int(total_gib * GIB),
        "free_bytes": int(free_gib * GIB),
        "source": "cgroup",
    }
    return models, snapshot, [], None


def test_scale_up_rejected_when_replicas_exceed_ram_budget(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """2 more instances of a ~4.5 GiB-footprint model against ~1.2 GiB of
    effective free RAM must 422 with the explicit deficit — and queue nothing."""
    from docie_bench.inngest import serving_api

    monkeypatch.setattr(
        serving_api,
        "_sizing_inputs",
        lambda: _sized_inputs(size_gib=4.0, total_gib=8.0, free_gib=2.0),
    )

    with pytest.raises(HTTPException) as excinfo:
        _scale(MODEL, 2)
    assert excinfo.value.status_code == 422
    detail = excinfo.value.detail
    assert "not enough RAM" in detail
    assert MODEL in detail
    assert "short by" in detail
    assert captured_events == []


def test_deploy_replicas_rejected_when_ram_budget_exceeded(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deploy trigger runs the SAME admission gate as the scale endpoint."""
    from docie_bench.inngest import serving_api

    monkeypatch.setattr(
        serving_api,
        "_sizing_inputs",
        lambda: _sized_inputs(size_gib=4.0, total_gib=8.0, free_gib=2.0),
    )

    with pytest.raises(HTTPException) as excinfo:
        _deploy({"model": MODEL, "replicas": 3})
    assert excinfo.value.status_code == 422
    assert "not enough RAM" in excinfo.value.detail
    assert captured_events == []


def test_deploy_replicas_ram_gate_scales_with_n_parallel(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RAM admission gate must see n_parallel too (#248/#321): a plan that
    fits at n_parallel=1 can legitimately not fit at n_parallel=2, since the
    KV-cache term (and therefore the priced footprint) scales with it."""
    from docie_bench.inngest import serving_api

    # weights=0.5 GiB -> footprint(n_parallel=1)=1.5 GiB, footprint(n_parallel=2)
    # =2.0 GiB (ctx=8192 default). free_effective ~= 1.75 GiB sits between them.
    monkeypatch.setattr(
        serving_api,
        "_sizing_inputs",
        lambda: _sized_inputs(size_gib=0.5, total_gib=8.0, free_gib=2.55),
    )
    _seed_replicas(serving_home, {MODEL: (MODEL, 8088)})

    _deploy({"model": MODEL, "replicas": 2, "n_parallel": 1})
    assert len(captured_events) == 1

    with pytest.raises(HTTPException) as excinfo:
        _deploy({"model": MODEL, "replicas": 2, "n_parallel": 2})
    assert excinfo.value.status_code == 422
    assert "not enough RAM" in excinfo.value.detail


def test_scale_up_admits_when_budget_fits(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from docie_bench.inngest import serving_api

    # ~0.85 GiB footprint x 2 fits 16 GiB free minus the margin comfortably.
    monkeypatch.setattr(
        serving_api,
        "_sizing_inputs",
        lambda: _sized_inputs(size_gib=0.35, total_gib=32.0, free_gib=16.0),
    )

    result = _scale(MODEL, 2)
    assert result["adding"] == [MODEL, f"{MODEL}-2"]
    assert len(captured_events) == 2


def test_scale_up_ram_gate_scales_with_n_parallel(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same n_parallel-aware admission as the deploy trigger, via ScaleRequest."""
    from docie_bench.inngest import serving_api

    monkeypatch.setattr(
        serving_api,
        "_sizing_inputs",
        lambda: _sized_inputs(size_gib=0.5, total_gib=8.0, free_gib=2.55),
    )

    result = _scale(MODEL, 1, n_parallel=1)
    assert result["adding"] == [MODEL]

    with pytest.raises(HTTPException) as excinfo:
        _scale(MODEL, 1, n_parallel=2)
    assert excinfo.value.status_code == 422


def test_scale_admission_fails_open_without_snapshot(
    serving_home: Path, captured_events: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No capacity measurement => admit (the serving container's live fit gate
    is the backstop) — never a false 'does not fit' from a blind api."""
    _no_sizing(monkeypatch)

    result = _scale(MODEL, 2)
    assert result["adding"] == [MODEL, f"{MODEL}-2"]
    assert len(captured_events) == 2


# ── scale down: drain surplus replicas via the real delete job ─────────────


def test_scale_down_fires_delete_per_surplus_replica(
    serving_home: Path, captured_events: list[Any]
) -> None:
    _seed_replicas(
        serving_home,
        {
            MODEL: (MODEL, 8088),
            f"{MODEL}-2": (MODEL, 8089),
            f"{MODEL}-3": (MODEL, 8090),
        },
    )

    result = _scale(MODEL, 1)

    # Highest suffix drains first; the bare base record always survives.
    assert result["removing"] == [f"{MODEL}-3", f"{MODEL}-2"]
    assert result["adding"] == []
    assert result["current"] == 3
    assert result["channel"].startswith("scale:")
    assert [e.name for e in captured_events] == ["serving/delete.requested"] * 2
    assert [e.data["name"] for e in captured_events] == [f"{MODEL}-3", f"{MODEL}-2"]


def test_scale_at_target_is_a_no_op_in_both_directions(
    serving_home: Path, captured_events: list[Any]
) -> None:
    _seed_replicas(serving_home, {MODEL: (MODEL, 8088), f"{MODEL}-2": (MODEL, 8089)})

    result = _scale(MODEL, 2)

    assert result["adding"] == []
    assert result["removing"] == []
    assert result["event_ids"] == []
    assert result["channel"] is None
    assert captured_events == []


def test_replica_names_to_remove_orders_and_bounds() -> None:
    existing = [MODEL, f"{MODEL}-2", f"{MODEL}-5", "nuextract3"]
    # Gap-tolerant: highest suffix first, unrelated names untouched.
    assert replica_names_to_remove(MODEL, existing, 2) == [f"{MODEL}-5"]
    assert replica_names_to_remove(MODEL, existing, 1) == [f"{MODEL}-5", f"{MODEL}-2"]
    # At/above target: nothing to remove (idempotent).
    assert replica_names_to_remove(MODEL, existing, 3) == []
    assert replica_names_to_remove(MODEL, existing, 16) == []
    # Mirror sanity: add/remove agree on what a replica of MODEL is (three
    # exist — MODEL, -2, -5 — so target 4 fills the first free slot).
    assert replica_names_to_add(MODEL, existing, 4) == [f"{MODEL}-3"]


# ── round-robin routing across live replicas ────────────────────────────────


@pytest.fixture
def sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


@pytest.fixture(autouse=True)
def _fresh_round_robin() -> Iterator[None]:
    from docie_bench.serving import placement_resolver

    placement_resolver._ROUND_ROBIN_COUNTERS.clear()
    placement_resolver._SESSION_AFFINITY.clear()
    try:
        yield
    finally:
        placement_resolver._ROUND_ROBIN_COUNTERS.clear()
        placement_resolver._SESSION_AFFINITY.clear()


def _seed_store(name: str) -> None:
    from docie_bench.serving.model_store import StoreEntry

    ModelCatalog().upsert(
        StoreEntry(name=name, family="openai_chat", model_path=Path(f"/models/{name}/model.gguf"))
    )


def _place(record_name: str, model_name: str, *, port: int, state: str = "ready") -> None:
    ModelCatalog().record_placement(
        record_name,
        model_name=model_name,
        engine="llama-server",
        endpoint=f"http://worker:{port}/v1",
        state=state,
    )


def test_round_robin_rotates_across_replica_endpoints(sqlite_catalog: None) -> None:
    from docie_bench.serving.placement_resolver import resolve_store_profile

    _seed_store(MODEL)
    _place(MODEL, MODEL, port=8091)
    _place(f"{MODEL}-2", MODEL, port=8092)
    _place(f"{MODEL}-3", MODEL, port=8093)

    urls = [resolve_store_profile(MODEL).base_url for _ in range(6)]

    # Deterministic rotation (record-name order), wrapping after one full lap.
    lap = [
        "http://worker:8091/v1",
        "http://worker:8092/v1",
        "http://worker:8093/v1",
    ]
    assert urls == lap + lap
    # Every replica answers to the shared base model id.
    assert resolve_store_profile(MODEL).model == MODEL


def test_round_robin_skips_dead_replicas(sqlite_catalog: None) -> None:
    """A replica whose placement is no longer ready (the reconciler republishes
    observed state every cycle) leaves the rotation — requests only ever hit
    live endpoints."""
    from docie_bench.serving.placement_resolver import resolve_store_profile

    _seed_store(MODEL)
    _place(MODEL, MODEL, port=8091)
    _place(f"{MODEL}-2", MODEL, port=8092, state="failed")
    _place(f"{MODEL}-3", MODEL, port=8093)

    urls = [resolve_store_profile(MODEL).base_url for _ in range(4)]

    assert urls == [
        "http://worker:8091/v1",
        "http://worker:8093/v1",
        "http://worker:8091/v1",
        "http://worker:8093/v1",
    ]


def test_round_robin_counters_are_per_model(sqlite_catalog: None) -> None:
    from docie_bench.serving.placement_resolver import resolve_store_profile

    _seed_store(MODEL)
    _seed_store("nuextract3")
    _place(MODEL, MODEL, port=8091)
    _place(f"{MODEL}-2", MODEL, port=8092)
    _place("nuextract3", "nuextract3", port=8095)
    _place("nuextract3-2", "nuextract3", port=8096)

    # Interleaved resolution: each model rotates on its own counter.
    assert resolve_store_profile(MODEL).base_url == "http://worker:8091/v1"
    assert resolve_store_profile("nuextract3").base_url == "http://worker:8095/v1"
    assert resolve_store_profile(MODEL).base_url == "http://worker:8092/v1"
    assert resolve_store_profile("nuextract3").base_url == "http://worker:8096/v1"


# ── session affinity: pin a conversation to one replica (#337) ─────────────


def test_session_affinity_pins_same_replica_across_calls(sqlite_catalog: None) -> None:
    """Repeat calls with the SAME session_id must land on the SAME replica —
    the whole point being to keep llama-server's prefix-KV cache warm across
    turns, instead of round-robining every top-level chat completion."""
    from docie_bench.serving.placement_resolver import resolve_store_profile

    _seed_store(MODEL)
    _place(MODEL, MODEL, port=8091)
    _place(f"{MODEL}-2", MODEL, port=8092)
    _place(f"{MODEL}-3", MODEL, port=8093)

    urls = [
        resolve_store_profile(MODEL, session_id="conv-abc").base_url for _ in range(5)
    ]
    assert len(set(urls)) == 1


def test_session_affinity_distributes_across_different_sessions(
    sqlite_catalog: None,
) -> None:
    """Different session_ids (or no session_id at all) must still spread load
    across the live replicas — affinity must not collapse load balancing."""
    from docie_bench.serving.placement_resolver import resolve_store_profile

    _seed_store(MODEL)
    _place(MODEL, MODEL, port=8091)
    _place(f"{MODEL}-2", MODEL, port=8092)
    _place(f"{MODEL}-3", MODEL, port=8093)

    urls = {
        resolve_store_profile(MODEL, session_id=f"conv-{i}").base_url for i in range(6)
    }
    assert urls == {
        "http://worker:8091/v1",
        "http://worker:8092/v1",
        "http://worker:8093/v1",
    }

    # No session_id given: unchanged round-robin behavior too.
    no_session_urls = [resolve_store_profile(MODEL).base_url for _ in range(3)]
    assert set(no_session_urls) == urls


def test_session_affinity_falls_back_and_repins_when_replica_dies(
    sqlite_catalog: None,
) -> None:
    """A pinned replica that's no longer live must fall back to a normal pick
    for that request AND update the pin, so the NEXT call for the same
    session_id follows the replacement instead of reverting.

    Three replicas are seeded (not two) so that after the pinned one dies,
    two replicas remain live -- resolve_store_profile's single-live-replica
    shortcut is deliberately NOT triggered, and the fallback/re-pin path
    inside session_affinity_choice actually runs."""
    from docie_bench.serving.placement_resolver import resolve_store_profile

    _seed_store(MODEL)
    _place(MODEL, MODEL, port=8091)
    _place(f"{MODEL}-2", MODEL, port=8092)
    _place(f"{MODEL}-3", MODEL, port=8093)

    first = resolve_store_profile(MODEL, session_id="conv-xyz")
    assert first.base_url == "http://worker:8091/v1"  # first pin: round-robin index 0

    # Kill the replica the session got pinned to; two replicas remain live.
    ModelCatalog().record_placement(
        MODEL,
        model_name=MODEL,
        engine="llama-server",
        endpoint="http://worker:8091/v1",
        state="failed",
    )

    second = resolve_store_profile(MODEL, session_id="conv-xyz")
    assert second.base_url != first.base_url

    # The pin follows the replacement on the next call.
    third = resolve_store_profile(MODEL, session_id="conv-xyz")
    assert third.base_url == second.base_url


def test_no_session_id_reproduces_prior_round_robin_behavior(
    sqlite_catalog: None,
) -> None:
    """Omitting session_id entirely (the existing call shape) must behave
    exactly as before this change — plain round-robin, no pinning."""
    from docie_bench.serving.placement_resolver import resolve_store_profile

    _seed_store(MODEL)
    _place(MODEL, MODEL, port=8091)
    _place(f"{MODEL}-2", MODEL, port=8092)
    _place(f"{MODEL}-3", MODEL, port=8093)

    urls = [resolve_store_profile(MODEL).base_url for _ in range(6)]
    lap = [
        "http://worker:8091/v1",
        "http://worker:8092/v1",
        "http://worker:8093/v1",
    ]
    assert urls == lap + lap


# ── replica counts on the deployments view ──────────────────────────────────


def test_list_deployments_annotates_replica_group_and_count(
    serving_home: Path,
) -> None:
    from docie_bench.inngest.serving_api import list_deployments

    _seed_replicas(
        serving_home,
        {
            MODEL: (MODEL, 8088),
            f"{MODEL}-2": (MODEL, 8089),
            "nuextract3": ("nuextract3", 8090),
        },
    )

    records = asyncio.run(list_deployments())

    by_name = {r["spec"]["name"]: r for r in records}
    assert by_name[MODEL]["replicas"] == 2
    assert by_name[MODEL]["replica_group"] == MODEL
    assert by_name[f"{MODEL}-2"]["replicas"] == 2
    assert by_name[f"{MODEL}-2"]["replica_group"] == MODEL
    assert by_name["nuextract3"]["replicas"] == 1
    assert by_name["nuextract3"]["replica_group"] == "nuextract3"
