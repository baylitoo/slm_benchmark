"""PR-4: dynamic load/unload lifecycle (design doc §4).

Stub-tested end to end: scripted adapters (no real processes), a scripted
clock (no real waiting), sqlite where a placement row is asserted, and an
injected fit gate (no psutil/cgroup reads). What each block proves:

* ``unload`` is DISTINCT from ``stop`` (fix #3): the record, its port
  reservation and its placement row all survive an unload (row UPDATEd to
  ``evicted``), and ``activation=managed`` marks it auto-reloadable — where a
  manual Stop stays ``manual``/cold and is never auto-reloaded.
* Idle-TTL unload: a hot deployment idle past the TTL (scripted clock) is
  unloaded by the reconciler; pinned deployments and just-loaded (min-hot)
  deployments survive.
* Cold-start pileup: N concurrent ``LoadCoordinator.load`` calls produce ONE
  spawn; an already-hot load is an idempotent no-op (what makes a step-retried
  load event harmless).
* Eviction policy: LRU by ``last_served`` among hot unpinned deployments,
  min-hot-time guard, per-attempt rate limit, and fit-before-evict — when the
  allowed victims cannot cover the deficit NOTHING is evicted.
* The worker's autoload gate: evicted+managed => load-and-wait; manual cold
  => no autoload (the resolver's honest refusal stands).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import docie_bench.storage.db as db
from docie_bench.serving import recency
from docie_bench.serving.lifecycle import (
    FitDecision,
    LoadCoordinator,
    LoadError,
    load_timeout_s,
    select_victims,
)
from docie_bench.serving.reconciler import ObservedDeployment, ServingReconciler
from docie_bench.serving.runtime import (
    HealthResult,
    LifecycleState,
    RuntimeKind,
    RuntimeLaunchSpec,
    RuntimeProcess,
)
from docie_bench.serving.supervisor import (
    Activation,
    DeploymentRecord,
    DeploymentSpec,
    DesiredState,
    PersistentSupervisor,
)

SPAWN_CREATE_TIME = 1000.0


class ScriptedAdapter:
    """A fake runtime: scripted health, crashable processes, optional slow start."""

    def __init__(self) -> None:
        self.next_pid = 100
        self.running: set[int] = set()
        self.starts = 0
        self.stops: list[int | None] = []
        self.healthy: dict[str, bool] = {}
        self.start_delay_s = 0.0

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> RuntimeProcess:
        del log_path
        if self.start_delay_s:
            time.sleep(self.start_delay_s)
        self.starts += 1
        self.next_pid += 1
        self.running.add(self.next_pid)
        return RuntimeProcess(spec.runtime, f"http://{spec.host}:{spec.port}/v1", self.next_pid)

    def is_running(self, pid: int | None) -> bool:
        return pid in self.running

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del timeout
        self.stops.append(pid)
        if pid is not None:
            self.running.discard(pid)

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        del timeout
        if self.healthy.get(spec.alias, True):
            return HealthResult(True, 200)
        return HealthResult(False, detail="connection refused")


def _spec(name: str = "invoice", *, port: int = 8090, **overrides: Any) -> DeploymentSpec:
    values: dict[str, Any] = {
        "name": name,
        "launch": RuntimeLaunchSpec(
            runtime=RuntimeKind.LLAMACPP,
            model=f"/models/{name}.gguf",
            alias=name,
            port=port,
        ),
    }
    values.update(overrides)
    return DeploymentSpec(**values)


def _supervisor(
    tmp_path: Path, adapter: ScriptedAdapter, clock: Any = time.time
) -> PersistentSupervisor:
    return PersistentSupervisor(
        tmp_path / "deployments.json",
        adapters={RuntimeKind.LLAMACPP: adapter},
        clock=clock,
        create_time=lambda pid: SPAWN_CREATE_TIME,
    )


def _fits() -> FitDecision:
    return FitDecision(True, None, None, 0, "")


def _no_fit(*, needed: int, free: int, margin: int = 0) -> FitDecision:
    return FitDecision(False, needed, free, margin, "does not fit (scripted)")


# ------------------------------------------------------- unload vs stop (#3)


def test_unload_keeps_record_port_and_flips_activation_managed(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    record = supervisor.deploy(_spec())
    spawned_pid = record.pid

    unloaded = supervisor.unload("invoice")

    assert unloaded.state == LifecycleState.STOPPED
    assert unloaded.spec.desired_state == DesiredState.STOPPED
    assert unloaded.activation == Activation.MANAGED  # evicted: auto-reloadable
    assert adapter.stops == [spawned_pid]  # process actually killed
    # Record + port reservation SURVIVE (only remove() frees them).
    kept = supervisor.get("invoice")
    assert kept.spec.launch.port == 8090
    # And the lifecycle metadata survives the JSON roundtrip.
    reloaded = PersistentSupervisor(
        tmp_path / "deployments.json", adapters={RuntimeKind.LLAMACPP: adapter}
    )
    assert reloaded.get("invoice").activation == Activation.MANAGED


def test_manual_stop_stays_manual_and_is_never_auto_reloaded(tmp_path: Path) -> None:
    """A user Stop => activation=manual => phase 'cold'; cycles never respawn it."""
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    supervisor.deploy(_spec())
    supervisor.stop("invoice")
    assert supervisor.get("invoice").activation == Activation.MANUAL

    reconciler = ServingReconciler(
        supervisor,
        fit_check=lambda record: (True, ""),
        rss_reader=lambda pid: 0,
        create_time=lambda pid: SPAWN_CREATE_TIME,
        publisher=lambda observations: None,
        recency_home=tmp_path,
        snapshot_publisher=lambda snapshot: None,
        idle_ttl_s=1.0,
    )
    for _ in range(3):
        observations = reconciler.run_cycle()
    observed = [obs for obs in observations if obs.name == "invoice"][0]
    assert observed.phase == "cold"  # manual => cold, NOT evicted
    assert adapter.starts == 1  # never auto-reloaded


# ------------------------------------------------------------ idle-TTL unload


def _idle_reconciler(
    supervisor: PersistentSupervisor,
    tmp_path: Path,
    now: list[float],
    *,
    idle_ttl_s: float = 300.0,
    min_hot_s: float = 0.0,
) -> tuple[ServingReconciler, list[list[ObservedDeployment]]]:
    published: list[list[ObservedDeployment]] = []
    reconciler = ServingReconciler(
        supervisor,
        fit_check=lambda record: (True, ""),
        rss_reader=lambda pid: 42,
        create_time=lambda pid: SPAWN_CREATE_TIME,
        publisher=published.append,
        recency_home=tmp_path,
        snapshot_publisher=lambda snapshot: None,
        idle_ttl_s=idle_ttl_s,
        min_hot_s=min_hot_s,
        clock=lambda: now[0],
    )
    return reconciler, published


def test_idle_ttl_unloads_a_hot_deployment_after_the_ttl(tmp_path: Path) -> None:
    """Scripted clock: served at t=1000, TTL=300 => still hot at t=1200,
    evicted at t=1400. Record + port kept; activation=managed."""
    now = [1000.0]
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter, clock=lambda: now[0])
    supervisor.deploy(_spec())
    recency.stamp("invoice", timestamp=1000.0, home=tmp_path)
    reconciler, _ = _idle_reconciler(supervisor, tmp_path, now)

    now[0] = 1200.0  # idle 200s < 300s TTL
    observations = reconciler.run_cycle()
    assert observations[0].phase == "hot"

    now[0] = 1400.0  # idle 400s > TTL
    observations = reconciler.run_cycle()
    assert observations[0].phase == "evicted"
    assert observations[0].endpoint == ""
    record = supervisor.get("invoice")  # record + port KEPT
    assert record.state == LifecycleState.STOPPED
    assert record.activation == Activation.MANAGED
    assert record.spec.launch.port == 8090

    # Stays evicted on later cycles: the reconciler never auto-reloads.
    observations = reconciler.run_cycle()
    assert observations[0].phase == "evicted"
    assert adapter.starts == 1


def test_fresh_traffic_resets_the_idle_countdown(tmp_path: Path) -> None:
    now = [1000.0]
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter, clock=lambda: now[0])
    supervisor.deploy(_spec())
    recency.stamp("invoice", timestamp=1000.0, home=tmp_path)
    reconciler, _ = _idle_reconciler(supervisor, tmp_path, now)

    now[0] = 1290.0
    recency.stamp("invoice", timestamp=1290.0, home=tmp_path)  # a request lands
    now[0] = 1400.0  # 110s since traffic: NOT idle even though 400s since load
    observations = reconciler.run_cycle()
    assert observations[0].phase == "hot"


def test_pinned_deployment_survives_the_idle_ttl(tmp_path: Path) -> None:
    now = [1000.0]
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter, clock=lambda: now[0])
    supervisor.deploy(_spec())
    supervisor.pin("invoice", pinned=True)
    recency.stamp("invoice", timestamp=1000.0, home=tmp_path)
    reconciler, _ = _idle_reconciler(supervisor, tmp_path, now)

    now[0] = 99_999.0
    observations = reconciler.run_cycle()
    assert observations[0].phase == "hot"  # pinned: never idle-unloaded
    assert supervisor.get("invoice").state == LifecycleState.READY


def test_min_hot_time_shields_a_just_loaded_deployment(tmp_path: Path) -> None:
    """A stale last_served must not evict a model that JUST spawned (thrash
    guard): min_hot_s wins over the idle verdict until it elapses."""
    now = [5000.0]
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter, clock=lambda: now[0])
    supervisor.deploy(_spec())  # loaded_at = 5000
    recency.stamp("invoice", timestamp=1.0, home=tmp_path)  # ancient traffic stamp
    reconciler, _ = _idle_reconciler(
        supervisor, tmp_path, now, idle_ttl_s=300.0, min_hot_s=600.0
    )

    now[0] = 5100.0  # hot for only 100s < 600s min-hot
    observations = reconciler.run_cycle()
    assert observations[0].phase == "hot"

    now[0] = 5700.0  # min-hot elapsed AND idle (last activity 5000 < now-300)
    observations = reconciler.run_cycle()
    assert observations[0].phase == "evicted"


def test_never_served_deployment_idles_out_from_its_load_time(tmp_path: Path) -> None:
    """No recency sidecar ever written: loaded_at is the honest age source."""
    now = [1000.0]
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter, clock=lambda: now[0])
    supervisor.deploy(_spec())
    reconciler, _ = _idle_reconciler(supervisor, tmp_path, now, idle_ttl_s=300.0)

    now[0] = 1400.0
    observations = reconciler.run_cycle()
    assert observations[0].phase == "evicted"


# ------------------------------------------------- load lock / idempotency


def test_concurrent_loads_trigger_exactly_one_spawn(tmp_path: Path) -> None:
    """The cold-start pileup lock: N concurrent requests to one evicted
    deployment => ONE spawn; every caller gets the READY record."""
    adapter = ScriptedAdapter()
    adapter.start_delay_s = 0.05  # widen the race window
    supervisor = _supervisor(tmp_path, adapter)
    supervisor.deploy(_spec())
    supervisor.unload("invoice")
    assert adapter.starts == 1

    coordinator = LoadCoordinator(
        supervisor, assess=lambda record: _fits(), sleep=lambda seconds: None
    )
    results: list[DeploymentRecord] = []
    errors: list[Exception] = []

    def call() -> None:
        try:
            results.append(coordinator.load("invoice"))
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert adapter.starts == 2  # initial deploy + exactly ONE reload
    assert len(results) == 6
    assert all(record.state == LifecycleState.READY for record in results)


def test_load_on_a_hot_deployment_is_an_idempotent_noop(tmp_path: Path) -> None:
    """What makes a re-fired/step-retried load event harmless (fix #7)."""
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    supervisor.deploy(_spec())

    coordinator = LoadCoordinator(
        supervisor, assess=lambda record: _fits(), sleep=lambda seconds: None
    )
    record = coordinator.load("invoice")

    assert record.state == LifecycleState.READY
    assert adapter.starts == 1  # no second spawn


def test_load_fails_honestly_after_the_size_aware_timeout(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    adapter.healthy["invoice"] = False  # never becomes ready
    supervisor = _supervisor(tmp_path, adapter)
    supervisor.deploy(_spec(health_failure_threshold=60))
    supervisor.unload("invoice")

    coordinator = LoadCoordinator(
        supervisor,
        assess=lambda record: _fits(),
        sleep=lambda seconds: None,
        timeout_for=lambda model: 0.0,  # expire immediately: no real waiting
    )
    with pytest.raises(LoadError, match="size-aware load budget"):
        coordinator.load("invoice")


def test_load_works_on_a_manual_cold_deployment(tmp_path: Path) -> None:
    """The Load button IS the explicit Start: manual-cold loads fine — only
    the AUTO path (worker autoload) refuses manual deployments."""
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    supervisor.deploy(_spec())
    supervisor.stop("invoice")

    coordinator = LoadCoordinator(
        supervisor, assess=lambda record: _fits(), sleep=lambda seconds: None
    )
    record = coordinator.load("invoice")
    assert record.state == LifecycleState.READY
    assert adapter.starts == 2


def test_load_timeout_scales_with_weights_and_respects_floor_and_ceiling(
    tmp_path: Path,
) -> None:
    small = tmp_path / "small.gguf"
    small.write_bytes(b"g" * 10)
    assert load_timeout_s(str(small)) == pytest.approx(120.0)  # floor
    assert load_timeout_s(None) == pytest.approx(120.0)  # unknowable => floor
    assert load_timeout_s(str(tmp_path / "missing.gguf")) == pytest.approx(120.0)
    four_gb = 4 * 1024**3
    scaled = load_timeout_s(None, size_bytes=four_gb)
    assert scaled > 120.0  # a big GGUF gets a real budget
    assert scaled == pytest.approx(four_gb / (24 * 1024 * 1024) + 60.0)
    assert load_timeout_s(None, size_bytes=10**15) == pytest.approx(1800.0)  # ceiling


# ------------------------------------------------------------------ eviction


def _hot_record(
    name: str,
    *,
    last_served: float | None,
    loaded_at: float | None = 100.0,
    pinned: bool = False,
) -> DeploymentRecord:
    return DeploymentRecord(
        spec=_spec(name),
        state=LifecycleState.READY,
        endpoint="http://serving:8090/v1",
        pid=1,
        pinned=pinned,
        last_served=last_served,
        loaded_at=loaded_at,
    )


def test_select_victims_picks_lru_unpinned_and_never_pinned() -> None:
    records = [
        _hot_record("newest", last_served=900.0),
        _hot_record("oldest", last_served=100.0),
        _hot_record("pinned-oldest", last_served=50.0, pinned=True),
        _hot_record("middle", last_served=500.0),
    ]
    victims = select_victims(
        records,
        deficit_bytes=150,
        now=10_000.0,
        min_hot_s=0.0,
        max_evictions=10,
        price=lambda record: 100,
    )
    # LRU among UNPINNED: oldest (100) then middle (500); pinned-oldest never.
    assert victims == ["oldest", "middle"]


def test_select_victims_honors_min_hot_time() -> None:
    records = [
        _hot_record("just-loaded", last_served=None, loaded_at=9_990.0),
        _hot_record("aged", last_served=500.0, loaded_at=100.0),
    ]
    victims = select_victims(
        records,
        deficit_bytes=50,
        now=10_000.0,
        min_hot_s=60.0,  # just-loaded is only 10s hot
        max_evictions=10,
        price=lambda record: 100,
    )
    assert victims == ["aged"]


def test_select_victims_rate_limit_makes_it_all_or_nothing() -> None:
    """Storm guard: deficit needs two victims but the per-cycle limit is one
    => evict NOTHING (never a pointless partial eviction)."""
    records = [
        _hot_record("a", last_served=100.0),
        _hot_record("b", last_served=200.0),
    ]
    victims = select_victims(
        records,
        deficit_bytes=150,
        now=10_000.0,
        min_hot_s=0.0,
        max_evictions=1,
        price=lambda record: 100,
    )
    assert victims is None


def test_select_victims_never_evicts_to_not_fit() -> None:
    """Fit-before-evict: even evicting EVERYTHING cannot cover the deficit
    => None, so the caller evicts nothing and fails the load honestly."""
    records = [
        _hot_record("a", last_served=100.0),
        _hot_record("b", last_served=200.0),
    ]
    victims = select_victims(
        records,
        deficit_bytes=10_000,
        now=10_000.0,
        min_hot_s=0.0,
        max_evictions=10,
        price=lambda record: 100,
    )
    assert victims is None


def test_select_victims_skips_unpriceable_and_non_hot_records() -> None:
    records = [
        _hot_record("unpriceable", last_served=1.0),  # price 0: unknown gain
        DeploymentRecord(spec=_spec("stopped"), state=LifecycleState.STOPPED),
        _hot_record("priced", last_served=500.0),
    ]
    victims = select_victims(
        records,
        deficit_bytes=50,
        now=10_000.0,
        min_hot_s=0.0,
        max_evictions=10,
        price=lambda record: 0 if record.spec.name == "unpriceable" else 100,
    )
    assert victims == ["priced"]


def test_load_evicts_lru_victim_then_spawns_and_pinned_survives(tmp_path: Path) -> None:
    """Through the coordinator: a no-fit load evicts exactly the LRU unpinned
    victim via the DISTINCT unload path (activation=managed), then spawns."""
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    # Two hot deployments with real (tiny) weights so victim pricing works.
    for name, port, served in (("alpha", 8091, 100.0), ("beta", 8092, 900.0)):
        gguf = tmp_path / f"{name}.gguf"
        gguf.write_bytes(b"g" * 1024)
        supervisor.deploy(
            _spec(
                name,
                port=port,
                launch=RuntimeLaunchSpec(
                    runtime=RuntimeKind.LLAMACPP,
                    model=str(gguf),
                    alias=name,
                    port=port,
                ),
            )
        )
        supervisor.fold_recency({name: served})
    # Pin the newer one too: even if selection somehow skipped the LRU, the
    # pinned deployment must remain untouchable.
    supervisor.pin("beta", pinned=True)
    # The cold candidate.
    supervisor.deploy(_spec("gamma", port=8093))
    supervisor.unload("gamma")
    starts_before = adapter.starts

    calls: list[str] = []
    decisions = iter(
        [_no_fit(needed=100, free=90, margin=0)]  # deficit 10: one victim covers
    )

    def assess(record: DeploymentRecord) -> FitDecision:
        calls.append(record.spec.name)
        return next(decisions, _fits())

    unloaded: list[str] = []

    def unload(name: str) -> object:
        unloaded.append(name)
        return supervisor.unload(name)

    coordinator = LoadCoordinator(
        supervisor,
        assess=assess,
        unload=unload,
        min_hot_s=0.0,
        max_evictions=2,
        sleep=lambda seconds: None,
    )
    record = coordinator.load("gamma")

    assert record.state == LifecycleState.READY
    assert calls == ["gamma"]
    assert unloaded == ["alpha"]  # LRU (last_served=100) — beta untouched
    assert supervisor.get("alpha").activation == Activation.MANAGED
    assert supervisor.get("alpha").state == LifecycleState.STOPPED
    assert supervisor.get("beta").state == LifecycleState.READY
    assert adapter.starts == starts_before + 1


def test_load_that_cannot_fit_evicts_nothing_and_raises(tmp_path: Path) -> None:
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    gguf = tmp_path / "alpha.gguf"
    gguf.write_bytes(b"g" * 1024)
    supervisor.deploy(
        _spec(
            "alpha",
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.LLAMACPP, model=str(gguf), alias="alpha", port=8091
            ),
        )
    )
    supervisor.deploy(_spec("gamma", port=8093))
    supervisor.unload("gamma")
    starts_before = adapter.starts

    coordinator = LoadCoordinator(
        supervisor,
        # Deficit astronomically larger than anything alpha can release.
        assess=lambda record: _no_fit(needed=10**18, free=0, margin=0),
        min_hot_s=0.0,
        max_evictions=10,
        sleep=lambda seconds: None,
    )
    with pytest.raises(LoadError, match="never evict-to-not-fit"):
        coordinator.load("gamma")

    assert supervisor.get("alpha").state == LifecycleState.READY  # NOT evicted
    assert adapter.starts == starts_before  # nothing spawned


# ------------------------------------------- placement row survives (sqlite)


@pytest.fixture
def _sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


@pytest.mark.usefixtures("_sqlite_catalog")
def test_control_plane_unload_updates_row_to_evicted_and_load_restores_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop()-clears-row trap (fix #3), asserted at the control-plane seam:
    unload UPDATEs the row (evicted, endpoint "") and NEVER deletes it; a
    later load flips it back to ready/hot with the live endpoint."""
    from docie_bench.serving.catalog import ModelCatalog
    from docie_bench.serving.control_plane import ControlPlane, _DefaultSupervisor

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    record = supervisor.deploy(_spec())
    ModelCatalog().record_placement(
        "invoice",
        model_name="invoice",
        engine="llama-server",
        endpoint=str(record.endpoint),
        state="ready",
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None)
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    asyncio.run(plane.unload("invoice"))

    row = ModelCatalog().get_placement("invoice")
    assert row is not None  # the row SURVIVES an unload
    assert row["phase"] == "evicted"
    assert row["endpoint"] == ""
    assert row["state"] == "stopped"
    kept = supervisor.get("invoice")
    assert kept.activation == Activation.MANAGED
    assert wrapper._reserved_ports() == {8090}  # port still reserved

    # Load through the facade: idempotency + row restored to hot/ready.
    result = asyncio.run(plane.load("invoice"))
    assert isinstance(result, dict)
    assert result["state"] == "ready"
    row = ModelCatalog().get_placement("invoice")
    assert row is not None
    assert row["phase"] == "hot"
    assert row["state"] == "ready"
    assert row["endpoint"].startswith("http://")


@pytest.mark.usefixtures("_sqlite_catalog")
def test_control_plane_stop_marks_cold_not_evicted(tmp_path: Path) -> None:
    from docie_bench.serving.catalog import ModelCatalog
    from docie_bench.serving.control_plane import _DefaultSupervisor

    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    record = supervisor.deploy(_spec())
    ModelCatalog().record_placement(
        "invoice",
        model_name="invoice",
        engine="llama-server",
        endpoint=str(record.endpoint),
        state="ready",
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None)

    wrapper.stop("invoice")

    row = ModelCatalog().get_placement("invoice")
    assert row is not None
    assert row["phase"] == "cold"  # manual stop: cold, NOT evicted
    assert supervisor.get("invoice").activation == Activation.MANUAL


def test_control_plane_pin_roundtrip(tmp_path: Path) -> None:
    from docie_bench.serving.control_plane import ControlPlane, _DefaultSupervisor

    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    supervisor.deploy(_spec())
    wrapper = _DefaultSupervisor(supervisor, planner=None)
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    result = asyncio.run(plane.pin("invoice", pinned=True))
    assert isinstance(result, dict)
    assert result["pinned"] is True
    assert supervisor.get("invoice").pinned is True

    asyncio.run(plane.pin("invoice", pinned=False))
    assert supervisor.get("invoice").pinned is False


# ----------------------------------------------- worker autoload gate + wait


@pytest.fixture
def serving_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "serving"
    home.mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    return home


def test_autoload_target_fires_only_for_managed_or_inflight(serving_home: Path) -> None:
    from docie_bench.inngest.functions import _autoload_target

    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)

    # Live deployment: no autoload needed.
    supervisor.deploy(_spec("hot-one", port=8090))
    assert _autoload_target({"deployment": "hot-one"}) is None

    # Evicted + managed: load-and-wait.
    supervisor.deploy(_spec("evicted-one", port=8091))
    supervisor.unload("evicted-one")
    target = _autoload_target({"deployment": "evicted-one"})
    assert target is not None
    name, budget = target
    assert name == "evicted-one"
    assert budget >= 120.0  # the size-aware floor

    # Manual cold: NEVER auto-loaded (the resolver's refusal stands).
    supervisor.deploy(_spec("manual-one", port=8092))
    supervisor.stop("manual-one")
    assert _autoload_target({"deployment": "manual-one"}) is None

    # Load in flight (desired running, not READY yet): wait, never 502.
    adapter.healthy["loading-one"] = False
    supervisor.deploy(_spec("loading-one", port=8093))
    assert supervisor.get("loading-one").state == LifecycleState.STARTING
    target = _autoload_target({"deployment": "loading-one"})
    assert target is not None
    assert target[0] == "loading-one"

    # model_profile naming a deployment record works like the explicit selector.
    target = _autoload_target({"model_profile": "evicted-one"})
    assert target is not None
    assert target[0] == "evicted-one"

    # Unknown names / plain yaml profiles: not a deployment, no autoload.
    assert _autoload_target({"model_profile": "studio_default"}) is None
    assert _autoload_target({}) is None


async def test_await_deployment_live_returns_when_ready(serving_home: Path) -> None:
    from docie_bench.inngest.functions import _await_deployment_live

    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)
    supervisor.deploy(_spec())

    result = await _await_deployment_live("invoice", timeout_s=1.0, interval_s=0.01)
    assert result["state"] == "ready"
    assert str(result["endpoint"]).startswith("http://")


async def test_await_deployment_live_times_out_honestly(serving_home: Path) -> None:
    from docie_bench.inngest.functions import _await_deployment_live

    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)
    supervisor.deploy(_spec())
    supervisor.unload("invoice")  # nothing will ever load it in this test

    with pytest.raises(TimeoutError, match="size-aware load budget"):
        await _await_deployment_live("invoice", timeout_s=0.05, interval_s=0.01)


# -------------------------------------------------------------- api endpoints


class _FakeInngestClient:
    def __init__(self) -> None:
        self.sent: list[Any] = []

    async def send(self, event: Any) -> list[str]:
        self.sent.append(event)
        return [f"evt_{len(self.sent)}"]


@pytest.fixture
def api_env(
    serving_home: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, _FakeInngestClient]:
    from docie_bench.inngest import serving_api

    fake = _FakeInngestClient()
    monkeypatch.setattr(serving_api, "inngest_client", fake)
    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)
    supervisor.deploy(_spec())
    return serving_api, fake


def test_load_unload_pin_endpoints_fire_the_lifecycle_events(
    api_env: tuple[Any, _FakeInngestClient],
) -> None:
    serving_api, fake = api_env

    result = asyncio.run(serving_api.load_deployment("invoice", tenant="t"))
    assert result["name"] == "invoice"
    assert result["channel"].startswith("load:")
    assert result["event_ids"] == ["evt_1"]

    result = asyncio.run(serving_api.unload_deployment("invoice", tenant="t"))
    assert result["channel"].startswith("unload:")

    result = asyncio.run(
        serving_api.pin_deployment(
            "invoice", serving_api.PinRequest(pinned=False), tenant="t"
        )
    )
    assert result["channel"].startswith("pin:")

    events = [(event.name, dict(event.data)) for event in fake.sent]
    assert events[0][0] == "serving/load.requested"
    assert events[0][1]["name"] == "invoice"
    assert events[1][0] == "serving/unload.requested"
    assert events[2][0] == "serving/pin.requested"
    assert events[2][1]["pinned"] is False


def test_lifecycle_endpoints_404_an_unknown_deployment(
    api_env: tuple[Any, _FakeInngestClient],
) -> None:
    from fastapi import HTTPException

    serving_api, fake = api_env
    for call in (
        lambda: serving_api.load_deployment("nope", tenant="t"),
        lambda: serving_api.unload_deployment("nope", tenant="t"),
        lambda: serving_api.pin_deployment(
            "nope", serving_api.PinRequest(pinned=True), tenant="t"
        ),
    ):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(call())
        assert excinfo.value.status_code == 404
    assert fake.sent == []  # a typo never queues a job
