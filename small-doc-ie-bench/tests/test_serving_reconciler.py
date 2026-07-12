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


def test_postgres_migration_uses_race_safe_add_column_if_not_exists() -> None:
    """On PostgreSQL every ADD uses the design's `ADD COLUMN IF NOT EXISTS`
    form (race-safe under concurrent api/serving/worker startups), not the
    inspect-then-plain-ALTER pattern (which aborts with DuplicateColumn when
    two processes race the inspector snapshot)."""
    from sqlalchemy.dialects import postgresql

    from docie_bench.serving.catalog import _OBSERVED_COLUMNS, _postgres_add_column_ddl

    dialect = postgresql.dialect()
    for name, column_type in _OBSERVED_COLUMNS:
        ddl = _postgres_add_column_ddl(name, column_type, dialect)
        assert ddl.startswith(
            f"ALTER TABLE model_placement ADD COLUMN IF NOT EXISTS {name} "
        )


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
    from docie_bench.inngest.client import APP_ID, SERVING_APP_ID

    def ids(items: list[Any]) -> set[str]:
        # Function.id is app-prefixed ("{app_id}-{fn_id}"); compare local ids.
        result = set()
        for f in items:
            local = f.id
            for prefix in (f"{SERVING_APP_ID}-", f"{APP_ID}-"):
                if local.startswith(prefix):
                    local = local.removeprefix(prefix)
                    break
            result.add(local)
        return result

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


def test_roles_register_separate_inngest_apps() -> None:
    """The Inngest Connect app-id contract: a worker sync registers its function
    list as the app's AUTHORITATIVE set, so serving and worker fleets MUST NOT
    share one app id with disjoint function subsets (their registrations would
    overwrite each other on every reconnect). Each role registers its own app,
    each app id always carries its full function set, and the dev `all` role
    registers both apps over the one (multi-app) connection."""
    from docie_bench.inngest import functions as fn
    from docie_bench.inngest.client import (
        APP_ID,
        SERVING_APP_ID,
        inngest_client,
        serving_client,
    )

    # Two distinct apps, two distinct clients.
    assert APP_ID != SERVING_APP_ID
    assert inngest_client.app_id == APP_ID
    assert serving_client.app_id == SERVING_APP_ID

    # Every function is bound to exactly its role's app (Function.id is
    # "{app_id}-{fn_id}", stamped by the client that created it).
    assert all(f.id.startswith(f"{SERVING_APP_ID}-") for f in fn.serving_functions)
    assert all(f.id.startswith(f"{APP_ID}-") for f in fn.worker_functions)

    # apps_for_role: the (client, functions) registrations connect() consumes.
    serving_apps = fn.apps_for_role("serving")
    worker_apps = fn.apps_for_role("worker")
    all_apps = fn.apps_for_role("all")
    assert [(c.app_id, [f.id for f in fns]) for c, fns in serving_apps] == [
        (SERVING_APP_ID, [f.id for f in fn.serving_functions])
    ]
    assert [(c.app_id, [f.id for f in fns]) for c, fns in worker_apps] == [
        (APP_ID, [f.id for f in fn.worker_functions])
    ]
    # `all` registers BOTH apps, each with its FULL set — never a subset of
    # either app (a subset sync is exactly the flapping failure mode).
    assert {c.app_id for c, _ in all_apps} == {APP_ID, SERVING_APP_ID}
    by_app = {c.app_id: [f.id for f in fns] for c, fns in all_apps}
    assert by_app[APP_ID] == [f.id for f in fn.worker_functions]
    assert by_app[SERVING_APP_ID] == [f.id for f in fn.serving_functions]
    with pytest.raises(ValueError, match="DOCIE_WORKER_ROLE"):
        fn.apps_for_role("both")


def test_serving_role_warns_when_advertise_host_names_a_scaled_service(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Upgrade-trap guard: a pre-split .env carries
    DOCIE_SERVING_ADVERTISE_HOST=worker; the serving role warns loudly at
    startup instead of silently recording round-robin endpoints."""
    import logging

    import docie_bench.inngest.worker as worker_mod

    class _Settings:
        serving_advertise_host = "worker"

    monkeypatch.setattr(worker_mod, "get_settings", lambda: _Settings())
    with caplog.at_level(logging.WARNING, logger="docie_bench.inngest.worker"):
        worker_mod._warn_legacy_advertise_host("serving")
    assert any("SCALED compose service" in record.getMessage() for record in caplog.records)

    # The correct single-replica name stays silent; the worker role never warns.
    caplog.clear()
    _Settings.serving_advertise_host = "serving"
    with caplog.at_level(logging.WARNING, logger="docie_bench.inngest.worker"):
        worker_mod._warn_legacy_advertise_host("serving")
    _Settings.serving_advertise_host = "worker"
    with caplog.at_level(logging.WARNING, logger="docie_bench.inngest.worker"):
        worker_mod._warn_legacy_advertise_host("worker")
    assert caplog.records == []


# -------------------------------------------------- restart-budget forgiveness


def test_restart_budget_resets_after_sustained_healthy_streak(tmp_path: Path) -> None:
    """The budget bounds crash STORMS, not lifetime crashes: after
    healthy_reset_threshold consecutive healthy cycles restart_count returns
    to 0, so a deployment that crashed twice over weeks of healthy uptime is
    still repairable. Pre-fix, restart_count only ever ratcheted up."""
    adapter = ScriptedAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "deployments.json",
        adapters={RuntimeKind.LLAMACPP: adapter},
        create_time=lambda pid: SPAWN_CREATE_TIME,
    )
    reconciler = ServingReconciler(
        supervisor,
        ready_miss_threshold=2,
        healthy_reset_threshold=3,
        fit_check=lambda record: (True, ""),
        rss_reader=lambda pid: 0,
        create_time=lambda pid: SPAWN_CREATE_TIME,
        publisher=lambda observations: None,
        recency_home=tmp_path,
    )
    record = supervisor.deploy(_spec(max_restarts=2))

    # Two crash->respawn rounds: budget nearly exhausted.
    for expected in (1, 2):
        adapter.crash(supervisor.get("invoice").pid)
        reconciler.run_cycle()
        assert supervisor.get("invoice").restart_count == expected

    # A sustained healthy streak forgives the budget...
    for _ in range(3):
        reconciler.run_cycle()
    assert supervisor.get("invoice").restart_count == 0
    # ...and the reset survives the JSON roundtrip (it is _save()d).
    reloaded = PersistentSupervisor(
        tmp_path / "deployments.json", adapters={RuntimeKind.LLAMACPP: adapter}
    )
    assert reloaded.get("invoice").restart_count == 0

    # So a LATER crash is repairable again instead of stuck at the cap.
    adapter.crash(supervisor.get("invoice").pid)
    observations = reconciler.run_cycle()
    assert _only(observations, "invoice").phase == "hot"
    assert supervisor.get("invoice").restart_count == 1
    del record


def test_crash_storm_never_earns_the_budget_reset(tmp_path: Path) -> None:
    """Every miss/death zeroes the healthy streak, so rapid crash->respawn
    loops still exhaust the budget exactly as before."""
    adapter = ScriptedAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "deployments.json",
        adapters={RuntimeKind.LLAMACPP: adapter},
        create_time=lambda pid: SPAWN_CREATE_TIME,
    )
    reconciler = ServingReconciler(
        supervisor,
        healthy_reset_threshold=2,
        fit_check=lambda record: (True, ""),
        rss_reader=lambda pid: 0,
        create_time=lambda pid: SPAWN_CREATE_TIME,
        publisher=lambda observations: None,
        recency_home=tmp_path,
    )
    supervisor.deploy(_spec(max_restarts=2))

    # crash / one-healthy-cycle / crash / ... — the single healthy cycle
    # between crashes never reaches the 2-cycle streak, so no forgiveness.
    for _ in range(2):
        adapter.crash(supervisor.get("invoice").pid)
        reconciler.run_cycle()  # respawn (healthy streak restarts at 1)
    adapter.crash(supervisor.get("invoice").pid)
    observations = reconciler.run_cycle()
    assert _only(observations, "invoice").phase == "failed"
    assert "restart budget exhausted" in (_only(observations, "invoice").last_error or "")


# ------------------------------------------- fast-death survives a restart


def test_fast_death_memory_survives_a_serving_restart(tmp_path: Path) -> None:
    """Finding: _was_ready was process-local, so restarting the serving
    service reset death-detection for a HUNG (alive-but-unreachable) runtime
    back to the ~10-minute slow-load tolerance. Was-readiness is now derived
    from the PERSISTED record state (READY survives the deployments.json
    roundtrip), so a FRESH reconciler still fast-deaths within threshold."""
    supervisor, adapter, reconciler, _ = _build(tmp_path, ready_miss_threshold=2)
    record = supervisor.deploy(_spec())
    spawned_pid = record.pid
    reconciler.run_cycle()  # observed hot; record.state persisted as READY

    adapter.healthy["invoice"] = False  # runtime hangs: alive but unreachable

    # Simulate a serving-service restart: reload the supervisor from disk and
    # build a BRAND-NEW reconciler with empty in-memory _was_ready/_misses.
    restarted_supervisor = PersistentSupervisor(
        tmp_path / "deployments.json",
        adapters={RuntimeKind.LLAMACPP: adapter},
        create_time=lambda pid: SPAWN_CREATE_TIME,
    )
    fresh = ServingReconciler(
        restarted_supervisor,
        ready_miss_threshold=2,
        fit_check=lambda rec: (False, "hold"),
        rss_reader=lambda pid: 0,
        create_time=lambda pid: SPAWN_CREATE_TIME,
        publisher=lambda observations: None,
        recency_home=tmp_path,
    )

    first = fresh.run_cycle()
    assert _only(first, "invoice").phase == "loading"  # miss 1 of 2
    second = fresh.run_cycle()
    observed = _only(second, "invoice")
    assert observed.phase == "failed"  # fast death, NOT slow-load tolerance
    assert adapter.stops == [spawned_pid]  # hung process torn down
    assert "unresponsive after READY" in (
        restarted_supervisor.get("invoice").last_error or ""
    )


# ------------------------------------------------------------ singleton lease


def test_reconciler_lease_refuses_a_second_live_instance(tmp_path: Path) -> None:
    """--scale serving=2 containment: replica B sees A's fresh heartbeat on the
    shared volume and refuses to start its reconciler."""
    from docie_bench.serving.reconciler import ReconcilerLease, ReconcilerSingletonError

    lease_path = tmp_path / "reconciler-lease.json"
    now = [1000.0]
    lease_a = ReconcilerLease(lease_path, "replica-a", stale_after_s=60.0, clock=lambda: now[0])
    lease_b = ReconcilerLease(lease_path, "replica-b", stale_after_s=60.0, clock=lambda: now[0])

    lease_a.claim()
    now[0] += 10.0  # A heartbeated 10s ago: live
    with pytest.raises(ReconcilerSingletonError, match="replica-a"):
        lease_b.claim()

    # A stale lease (crashed replica) is claimable...
    now[0] += 120.0
    lease_b.claim()
    # ...and re-claiming one's own fresh lease is an idempotent restart.
    lease_b.claim()

    # Release drops the file only when it is still ours.
    lease_a.release()  # not the holder: no-op
    assert lease_path.exists()
    lease_b.release()
    assert not lease_path.exists()


def test_reconciler_cycle_heartbeats_the_lease(tmp_path: Path) -> None:
    import json as json_lib

    from docie_bench.serving.reconciler import ReconcilerLease

    lease_path = tmp_path / "reconciler-lease.json"
    now = [500.0]
    lease = ReconcilerLease(lease_path, "replica-a", stale_after_s=60.0, clock=lambda: now[0])
    supervisor, _, _, _ = _build(tmp_path)
    reconciler = ServingReconciler(
        supervisor,
        fit_check=lambda record: (True, ""),
        rss_reader=lambda pid: 0,
        create_time=lambda pid: SPAWN_CREATE_TIME,
        publisher=lambda observations: None,
        recency_home=tmp_path,
        lease=lease,
    )
    reconciler.claim_singleton()
    now[0] = 555.0
    reconciler.run_cycle()
    payload = json_lib.loads(lease_path.read_text(encoding="utf-8"))
    assert payload == {"instance": "replica-a", "timestamp": 555.0}


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
