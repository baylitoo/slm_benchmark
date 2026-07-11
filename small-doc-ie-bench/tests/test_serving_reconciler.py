"""PR-1: the in-serving reconciler + observed-state publish + real delete.

Stub-tested with a scripted adapter (no real processes, sockets, or Postgres —
sqlite where a database is needed). Honest limits, stated plainly: these stubs
CANNOT reach the cross-container failure modes — N replicas clobbering
``deployments.json`` or the round-robin advertise host probing the wrong
replica. Those are *designed out* by the P1 topology (single-replica
``serving`` service owns every mutation), so there is nothing left to stub;
what CAN race inside one process (reconciler cycle vs a concurrent handler
mutation) is covered by the lock-serialization test below.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text

import docie_bench.storage.db as db
from docie_bench.serving import recency
from docie_bench.serving.catalog import (
    ModelCatalog,
    ensure_placement_observed_columns,
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
    DeploymentSpec,
    PersistentSupervisor,
)

SPAWN_CREATE_TIME = 1000.0


class ScriptedAdapter:
    """A fake runtime whose processes can be crashed and whose health is scripted."""

    def __init__(self) -> None:
        self.next_pid = 100
        self.running: set[int] = set()
        self.starts = 0
        self.stops: list[int | None] = []
        # alias -> healthy? (default True). The reconciler probes by launch spec.
        self.healthy: dict[str, bool] = {}
        self.probe_log: list[str] = []
        self.probe_hook: Any = None  # optional callable(alias) for concurrency tests

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> RuntimeProcess:
        del log_path
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

    def crash(self, pid: int | None) -> None:
        """Simulate the runtime dying on its own (no shutdown bookkeeping)."""
        if pid is not None:
            self.running.discard(pid)

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        del timeout
        self.probe_log.append(spec.alias)
        if self.probe_hook is not None:
            self.probe_hook(spec.alias)
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


_Built = tuple[
    PersistentSupervisor, ScriptedAdapter, ServingReconciler, list[list[ObservedDeployment]]
]


def _build(
    tmp_path: Path,
    *,
    fit: tuple[bool, str] = (True, ""),
    ready_miss_threshold: int = 2,
    create_time_seen: float | None = SPAWN_CREATE_TIME,
) -> _Built:
    adapter = ScriptedAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "deployments.json",
        adapters={RuntimeKind.LLAMACPP: adapter},
        create_time=lambda pid: SPAWN_CREATE_TIME,  # captured at spawn
    )
    published: list[list[ObservedDeployment]] = []
    reconciler = ServingReconciler(
        supervisor,
        interval_s=0.01,
        ready_miss_threshold=ready_miss_threshold,
        fit_check=lambda record: fit,
        rss_reader=lambda pid: 123_000_000,
        create_time=lambda pid: create_time_seen,  # observed on later cycles
        publisher=published.append,
        recency_home=tmp_path,
    )
    return supervisor, adapter, reconciler, published


def _only(observations: list[ObservedDeployment], name: str) -> ObservedDeployment:
    matches = [obs for obs in observations if obs.name == name]
    assert len(matches) == 1
    return matches[0]


# --------------------------------------------------------------------- cycles


def test_healthy_ready_deployment_publishes_hot_with_rss(tmp_path: Path) -> None:
    supervisor, adapter, reconciler, published = _build(tmp_path)
    record = supervisor.deploy(_spec())
    assert record.state == LifecycleState.READY

    observations = reconciler.run_cycle()

    observed = _only(observations, "invoice")
    assert observed.phase == "hot"
    assert observed.health_ok is True
    assert observed.state == "ready"
    assert observed.endpoint == record.endpoint
    assert observed.rss_bytes == 123_000_000
    assert observed.pid == record.pid
    assert observed.pid_create_time == SPAWN_CREATE_TIME
    assert published == [observations]


def test_crashed_ready_deployment_fails_fast_without_respawn_when_fit_denies(
    tmp_path: Path,
) -> None:
    """READY -> crash -> FAILED within ONE cycle (dead pid is definitive), and
    the gated restart withholds the respawn when the fit-check says no —
    the anti-respawn-storm behavior naive reconcile_all() lacks."""
    supervisor, adapter, reconciler, _ = _build(tmp_path, fit=(False, "would OOM"))
    record = supervisor.deploy(_spec())
    reconciler.run_cycle()  # observe hot

    adapter.crash(record.pid)
    observations = reconciler.run_cycle()

    observed = _only(observations, "invoice")
    assert observed.phase == "failed"
    assert observed.health_ok is False
    assert "restart withheld" in (observed.last_error or "")
    assert adapter.starts == 1  # never respawned
    assert supervisor.get("invoice").state == LifecycleState.FAILED

    # Subsequent cycles do not churn: still failed, still no respawn.
    for _ in range(3):
        observations = reconciler.run_cycle()
    assert adapter.starts == 1
    assert _only(observations, "invoice").phase == "failed"


def test_crashed_deployment_respawns_when_budget_and_fit_allow(tmp_path: Path) -> None:
    supervisor, adapter, reconciler, _ = _build(tmp_path)
    record = supervisor.deploy(_spec())
    reconciler.run_cycle()

    adapter.crash(record.pid)
    observations = reconciler.run_cycle()

    assert adapter.starts == 2  # gated restart approved
    observed = _only(observations, "invoice")
    assert observed.phase == "hot"  # scripted health passes immediately post-spawn
    repaired = supervisor.get("invoice")
    assert repaired.state == LifecycleState.READY
    assert repaired.restart_count == 1


def test_restart_budget_exhausted_blocks_respawn_even_when_fit_allows(
    tmp_path: Path,
) -> None:
    supervisor, adapter, reconciler, _ = _build(tmp_path)
    record = supervisor.deploy(_spec(max_restarts=0))
    reconciler.run_cycle()

    adapter.crash(record.pid)
    observations = reconciler.run_cycle()

    assert adapter.starts == 1
    observed = _only(observations, "invoice")
    assert observed.phase == "failed"
    assert "restart budget exhausted" in (observed.last_error or "")


def test_starting_cold_load_is_never_killed_by_the_reconciler(tmp_path: Path) -> None:
    """A never-READY (cold-loading) deployment keeps the HIGH tolerance: many
    unhealthy cycles observe 'loading' and never shutdown/mark-failed it —
    the fast-death threshold applies only to the READY->unreachable path."""
    supervisor, adapter, reconciler, _ = _build(tmp_path, ready_miss_threshold=2)
    adapter.healthy["invoice"] = False  # model still loading: health refused
    record = supervisor.deploy(_spec())
    assert record.state == LifecycleState.STARTING

    for _ in range(10):
        observations = reconciler.run_cycle()

    observed = _only(observations, "invoice")
    assert observed.phase == "loading"
    assert observed.health_ok is False
    assert adapter.stops == []  # never killed
    live = supervisor.get("invoice")
    assert live.state == LifecycleState.STARTING
    assert live.pid is not None
    assert adapter.is_running(live.pid)


def test_ready_then_unresponsive_hits_fast_death_within_threshold(tmp_path: Path) -> None:
    """A previously-READY but hung (alive, unhealthy) runtime is declared
    failed after ready_miss_threshold misses — not after the deploy-time
    health_failure_threshold=60 (~10 minutes)."""
    supervisor, adapter, reconciler, _ = _build(
        tmp_path, fit=(False, "no memory"), ready_miss_threshold=2
    )
    record = supervisor.deploy(_spec())
    spawned_pid = record.pid
    reconciler.run_cycle()  # hot

    adapter.healthy["invoice"] = False  # hangs: alive but unreachable
    first_miss = reconciler.run_cycle()
    observed = _only(first_miss, "invoice")
    assert observed.phase == "loading"  # below threshold: observed, not killed
    assert adapter.stops == []

    second_miss = reconciler.run_cycle()
    observed = _only(second_miss, "invoice")
    assert observed.phase == "failed"
    assert adapter.stops == [spawned_pid]  # fast-death tears the hung process down
    assert "unresponsive after READY" in (supervisor.get("invoice").last_error or "")


def test_create_time_mismatch_is_treated_as_dead_without_killing_the_impostor(
    tmp_path: Path,
) -> None:
    """PID-reuse guard: the pid is 'running' but its create_time does not match
    what was captured at spawn => not our process => dead. Crucially the
    impostor process is NOT shutdown (that would SIGTERM an unrelated pid)."""
    supervisor, adapter, reconciler, _ = _build(
        tmp_path, fit=(False, "hold"), create_time_seen=SPAWN_CREATE_TIME + 500.0
    )
    supervisor.deploy(_spec())

    observations = reconciler.run_cycle()

    observed = _only(observations, "invoice")
    assert observed.phase == "failed"
    assert adapter.stops == []  # never killed the recycled pid
    assert "create_time mismatch" in (supervisor.get("invoice").last_error or "")


def test_vanished_process_create_time_none_is_treated_as_dead(tmp_path: Path) -> None:
    """psutil.NoSuchProcess maps to create_time=None => dead, never an exception."""
    supervisor, adapter, reconciler, _ = _build(
        tmp_path, fit=(False, "hold"), create_time_seen=None
    )
    supervisor.deploy(_spec())

    observations = reconciler.run_cycle()

    assert _only(observations, "invoice").phase == "failed"
    assert supervisor.get("invoice").state == LifecycleState.FAILED


def test_stopped_deployment_publishes_cold_with_empty_endpoint(tmp_path: Path) -> None:
    supervisor, adapter, reconciler, _ = _build(tmp_path)
    supervisor.deploy(_spec())
    supervisor.stop("invoice")

    observations = reconciler.run_cycle()

    observed = _only(observations, "invoice")
    assert observed.phase == "cold"  # manual stop => cold, not evicted
    assert observed.state == "stopped"
    assert observed.endpoint == ""  # readers treat "" as "no live endpoint"
    assert observed.health_ok is False
    assert observed.rss_bytes == 0


# -------------------------------------------------------------------- recency


def test_recency_sidecars_fold_into_last_served_monotonically(tmp_path: Path) -> None:
    supervisor, _, reconciler, _ = _build(tmp_path)
    supervisor.deploy(_spec())

    recency.stamp("invoice", timestamp=1000.5, home=tmp_path)
    reconciler.run_cycle()
    assert supervisor.get("invoice").last_served == pytest.approx(1000.5)

    # An older stamp never regresses the fold (monotonic max semantics).
    recency.stamp("invoice", timestamp=999.0, home=tmp_path)
    reconciler.run_cycle()
    assert supervisor.get("invoice").last_served == pytest.approx(1000.5)

    recency.stamp("invoice", timestamp=2000.0, home=tmp_path)
    reconciler.run_cycle()
    assert supervisor.get("invoice").last_served == pytest.approx(2000.0)


def test_lifecycle_fields_survive_the_json_roundtrip(tmp_path: Path) -> None:
    supervisor, adapter, reconciler, _ = _build(tmp_path)
    supervisor.deploy(_spec())
    recency.stamp("invoice", timestamp=1234.0, home=tmp_path)
    reconciler.run_cycle()

    reloaded = PersistentSupervisor(
        tmp_path / "deployments.json", adapters={RuntimeKind.LLAMACPP: adapter}
    )
    record = reloaded.get("invoice")
    assert record.last_served == pytest.approx(1234.0)
    assert record.activation.value == "manual"
    assert record.pinned is False
    assert record.pid_create_time == SPAWN_CREATE_TIME


# ---------------------------------------------------------------- concurrency


def test_reconcile_cycle_and_handler_mutation_are_lock_serialized(tmp_path: Path) -> None:
    """The intra-process race PR-1 closes (design fix #4): a handler mutation
    fired mid-cycle blocks on the supervisor lock until the whole cycle ends —
    no interleaved read-modify-save, no lost write. Deterministic: the health
    probe hook signals the main thread to call stop(), then sleeps; if the
    lock did not serialize, 'stop' would land between the two probes."""
    supervisor, adapter, reconciler, _ = _build(tmp_path)
    supervisor.deploy(_spec("alpha", port=8091))
    supervisor.deploy(_spec("beta", port=8092))

    events: list[str] = []
    in_first_probe = threading.Event()

    def probe_hook(alias: str) -> None:
        events.append(f"probe:{alias}")
        if alias == "alpha":
            in_first_probe.set()
            time.sleep(0.3)  # give the stop() thread time to contend for the lock

    adapter.probe_hook = probe_hook

    def stop_beta() -> None:
        in_first_probe.wait(timeout=5)
        supervisor.stop("beta")
        events.append("stop:beta")

    stopper = threading.Thread(target=stop_beta)
    stopper.start()
    reconciler.run_cycle()
    stopper.join(timeout=10)
    assert not stopper.is_alive()

    # The stop() fired DURING the cycle must have waited for the whole cycle:
    # both probes strictly precede it.
    assert events == ["probe:alpha", "probe:beta", "stop:beta"]
    assert supervisor.get("beta").state == LifecycleState.STOPPED
    # And nothing was lost: both records still present and consistent on disk.
    reloaded = PersistentSupervisor(
        tmp_path / "deployments.json", adapters={RuntimeKind.LLAMACPP: adapter}
    )
    assert {record.spec.name for record in reloaded.list()} == {"alpha", "beta"}


# ------------------------------------------------------------------- database


@pytest.fixture
def _sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


def test_no_database_cycle_still_repairs_and_skips_publish(tmp_path: Path) -> None:
    """Postgres is NOT required to kill staleness (fix #8): with no
    DATABASE_URL the default publisher degrades to a no-op and the repair
    (gated respawn) still happens + deployments.json is _save()d."""
    from docie_bench.serving.reconciler import _publish_via_catalog

    db.dispose_engine()
    adapter = ScriptedAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "deployments.json",
        adapters={RuntimeKind.LLAMACPP: adapter},
        create_time=lambda pid: SPAWN_CREATE_TIME,
    )
    reconciler = ServingReconciler(
        supervisor,
        create_time=lambda pid: SPAWN_CREATE_TIME,
        fit_check=lambda record: (True, ""),
        publisher=_publish_via_catalog,  # the real one: must swallow no-DB
        recency_home=tmp_path,
    )
    record = supervisor.deploy(_spec())
    adapter.crash(record.pid)

    observations = reconciler.run_cycle()  # must not raise

    assert adapter.starts == 2  # repaired
    assert _only(observations, "invoice").phase == "hot"


@pytest.mark.usefixtures("_sqlite_catalog")
def test_publish_observed_updates_and_creates_rows() -> None:
    catalog = ModelCatalog()
    view = catalog.publish_observed(
        "invoice",
        engine="llama-server",
        state="ready",
        endpoint="http://serving:8090/v1",
        phase="hot",
        pid=4242,
        pid_create_time=111.0,
        rss_bytes=2_000_000_000,
        health_ok=True,
        last_error=None,
    )
    assert view["phase"] == "hot"
    assert view["rss_bytes"] == 2_000_000_000
    assert view["health_ok"] is True
    assert view["last_probe_at"] is not None

    view = catalog.publish_observed(
        "invoice",
        engine="llama-server",
        state="failed",
        endpoint="",
        phase="failed",
        pid=None,
        pid_create_time=None,
        rss_bytes=0,
        health_ok=False,
        last_error="runtime process exited",
    )
    assert view["phase"] == "failed"
    assert view["endpoint"] == ""
    assert view["last_error"] == "runtime process exited"
    assert len(catalog.list_placements()) == 1  # updated, not duplicated


def test_migration_adds_observed_columns_to_a_legacy_table(tmp_path: Path) -> None:
    """A model_placement created BEFORE PR-1 (no observed columns) gains
    exactly the missing columns via the explicit ALTER TABLE migration —
    create_all alone would silently leave them absent (the size_bytes hazard)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(
            sa_text(
                """
                CREATE TABLE model_placement (
                    name VARCHAR(200) PRIMARY KEY,
                    model_name VARCHAR(200),
                    engine VARCHAR(32) NOT NULL,
                    endpoint TEXT NOT NULL,
                    state VARCHAR(32) NOT NULL,
                    negotiated_style VARCHAR(64),
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
                """
            )
        )

    added = ensure_placement_observed_columns(engine)

    assert added == [
        "phase",
        "pid",
        "pid_create_time",
        "rss_bytes",
        "health_ok",
        "last_probe_at",
        "last_error",
    ]
    columns = {column["name"] for column in sa_inspect(engine).get_columns("model_placement")}
    assert {"phase", "pid", "rss_bytes", "health_ok", "last_probe_at", "last_error"} <= columns
    # Idempotent: a second run adds nothing.
    assert ensure_placement_observed_columns(engine) == []
    # And the full init path (migration + create_all) works against it.
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    try:
        ModelCatalog().publish_observed(
            "legacy",
            engine="llama-server",
            state="ready",
            endpoint="http://serving:8088/v1",
            phase="hot",
            pid=1,
            pid_create_time=1.0,
            rss_bytes=1,
            health_ok=True,
            last_error=None,
        )
        assert ModelCatalog().get_placement("legacy")["phase"] == "hot"
    finally:
        db.dispose_engine()


# ------------------------------------------------------------------ deletions


@pytest.mark.usefixtures("_sqlite_catalog")
def test_control_plane_remove_kills_record_frees_port_and_deletes_row(tmp_path: Path) -> None:
    """The real delete (PR-1): process shutdown + record gone (port freed —
    _reserved_ports scans records) + placement row DELETEd. The only path
    that deletes a row."""
    from docie_bench.serving.control_plane import ControlPlane, _DefaultSupervisor

    adapter = ScriptedAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "deployments.json", adapters={RuntimeKind.LLAMACPP: adapter}
    )
    record = supervisor.deploy(_spec(port=8095))
    ModelCatalog().record_placement(
        "invoice",
        model_name="invoice",
        engine="llama-server",
        endpoint=str(record.endpoint),
        state="ready",
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None)
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    result = asyncio.run(plane.remove("invoice"))

    assert result == {"name": "invoice", "removed": True}
    assert adapter.stops == [record.pid]  # process actually killed
    with pytest.raises(KeyError):
        supervisor.get("invoice")
    assert ModelCatalog().get_placement("invoice") is None  # row DELETEd
    assert wrapper._reserved_ports() == set()  # port freed


def test_control_plane_remove_unknown_deployment_raises(tmp_path: Path) -> None:
    from docie_bench.serving.control_plane import ControlPlane, _DefaultSupervisor

    supervisor = PersistentSupervisor(
        tmp_path / "deployments.json", adapters={RuntimeKind.LLAMACPP: ScriptedAdapter()}
    )
    plane = ControlPlane(
        None, None, _DefaultSupervisor(supervisor, planner=None), None  # type: ignore[arg-type]
    )
    with pytest.raises(KeyError):
        asyncio.run(plane.remove("no-such-deployment"))


# ----------------------------------------------------------------- topology


def test_role_partition_is_disjoint_and_complete() -> None:
    """P1 routing is load-bearing: lifecycle functions register ONLY on the
    serving service, extraction/benchmark ONLY on workers, no overlap."""
    from docie_bench.inngest import functions as fn
    from docie_bench.inngest.client import APP_ID

    def ids(items: list[Any]) -> set[str]:
        # Function.id is app-prefixed ("{APP_ID}-{fn_id}"); compare local ids.
        return {f.id.removeprefix(f"{APP_ID}-") for f in items}

    serving_ids = ids(fn.serving_functions)
    worker_ids = ids(fn.worker_functions)
    assert serving_ids == {"serving-deploy", "serving-seed-ollama", "serving-delete"}
    assert worker_ids == {"doc-extract", "benchmark-run", "studio-runs-gc"}
    assert serving_ids.isdisjoint(worker_ids)
    assert ids(fn.functions) == serving_ids | worker_ids

    assert ids(fn.functions_for_role("serving")) == serving_ids
    assert ids(fn.functions_for_role("worker")) == worker_ids
    assert ids(fn.functions_for_role(None)) == serving_ids | worker_ids
    with pytest.raises(ValueError, match="DOCIE_WORKER_ROLE"):
        fn.functions_for_role("both")


def test_reconciler_enabled_only_for_the_serving_role(monkeypatch: pytest.MonkeyPatch) -> None:
    from docie_bench.inngest.worker import _reconciler_enabled

    monkeypatch.delenv("DOCIE_SERVING_RECONCILE", raising=False)
    assert _reconciler_enabled("serving") is True
    assert _reconciler_enabled("worker") is False  # NEVER on a scaled worker
    assert _reconciler_enabled("all") is False  # dev default: opt-in

    monkeypatch.setenv("DOCIE_SERVING_RECONCILE", "1")
    assert _reconciler_enabled("all") is True
    assert _reconciler_enabled("worker") is False  # not even by opt-in

    monkeypatch.setenv("DOCIE_SERVING_RECONCILE", "0")
    assert _reconciler_enabled("serving") is False  # explicit kill switch
