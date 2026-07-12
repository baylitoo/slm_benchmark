"""PR-2: resource tracker + node snapshot publish.

Stub-tested end to end: injected cgroup files (tmp dirs standing in for
/sys/fs/cgroup) and fake VM readers — no real container, no Postgres (sqlite
where a database is needed). Honest limits: these stubs cannot observe the
real mmap ramp of a llama-server loading a multi-GB GGUF, nor compare the
published numbers against `docker stats`/htop in the WSL2 VM — both are
live-verification items called out on the PR.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select

import docie_bench.storage.db as db
from docie_bench.serving import resources
from docie_bench.serving.catalog import (
    ModelCatalog,
    ServingNode,
    ensure_serving_node_table,
)
from docie_bench.serving.resources import (
    DEFAULT_CONTEXT_LENGTH,
    KV_CACHE_BYTES_PER_TOKEN,
    RUNTIME_OVERHEAD_BYTES,
    FootprintStore,
    NodeMemory,
    NodeSnapshot,
    ResourceTracker,
    footprint_bytes,
    footprint_key,
    predict_footprint_bytes,
    predicted_footprint_for_model,
    publish_snapshot_via_catalog,
    read_node_memory,
)

GIB = 1024**3


def _write_cgroup(root: Path, *, limit: str | None, current: str | None) -> None:
    if limit is not None:
        (root / "memory.max").write_text(limit, encoding="ascii")
    if current is not None:
        (root / "memory.current").write_text(current, encoding="ascii")


def _vm_reader() -> NodeMemory:
    return NodeMemory(total_bytes=32 * GIB, free_bytes=20 * GIB, source="vm")


# ---------------------------------------------------------------- node memory


def test_cgroup_preferred_when_limited(tmp_path: Path) -> None:
    """A real memory.max limit is authoritative: total=limit,
    free=limit-current, flagged source=cgroup — the VM reader is not consulted."""
    _write_cgroup(tmp_path, limit=f"{8 * GIB}\n", current=f"{2 * GIB}\n")

    def vm_must_not_be_called() -> NodeMemory:
        raise AssertionError("VM fallback must not be consulted when cgroup is limited")

    reading = read_node_memory(cgroup_root=tmp_path, vm_reader=vm_must_not_be_called)

    assert reading.source == "cgroup"
    assert reading.total_bytes == 8 * GIB
    assert reading.free_bytes == 6 * GIB


def test_cgroup_max_sentinel_falls_back_to_vm_flagged(tmp_path: Path) -> None:
    """memory.max == 'max' (compose set no limit): the cgroup ceiling is
    meaningless for sizing, so the reading degrades to the VM view and SAYS so
    via source=vm (the UI's soft-number badge input)."""
    _write_cgroup(tmp_path, limit="max\n", current=f"{2 * GIB}\n")

    reading = read_node_memory(cgroup_root=tmp_path, vm_reader=_vm_reader)

    assert reading.source == "vm"
    assert reading.total_bytes == 32 * GIB
    assert reading.free_bytes == 20 * GIB


def test_missing_or_corrupt_cgroup_falls_back_to_vm(tmp_path: Path) -> None:
    # No cgroup files at all (not a container / cgroup v1).
    assert read_node_memory(cgroup_root=tmp_path, vm_reader=_vm_reader).source == "vm"
    # Corrupt limit.
    _write_cgroup(tmp_path, limit="not-a-number", current=None)
    assert read_node_memory(cgroup_root=tmp_path, vm_reader=_vm_reader).source == "vm"


def test_cgroup_missing_current_still_authoritative(tmp_path: Path) -> None:
    """memory.current unreadable degrades to used=0, not to the VM fallback —
    an authoritative LIMIT is still the honest denominator."""
    _write_cgroup(tmp_path, limit=str(8 * GIB), current=None)

    reading = read_node_memory(cgroup_root=tmp_path, vm_reader=_vm_reader)

    assert reading.source == "cgroup"
    assert reading.total_bytes == reading.free_bytes == 8 * GIB


# ---------------------------------------------------------- predicted footprint


def test_predicted_footprint_is_the_planner_formula() -> None:
    weights = 4 * GIB
    predicted = predict_footprint_bytes(weights, context_length=4096, n_parallel=1)
    assert predicted == weights + KV_CACHE_BYTES_PER_TOKEN * 4096 + RUNTIME_OVERHEAD_BYTES

    # KV scales with context and parallel slots; quant factor scales weights;
    # mmproj is additive (vision families).
    assert predict_footprint_bytes(weights, context_length=8192) - predict_footprint_bytes(
        weights, context_length=4096
    ) == KV_CACHE_BYTES_PER_TOKEN * 4096
    assert predict_footprint_bytes(
        weights, context_length=4096, n_parallel=4
    ) - predict_footprint_bytes(weights, context_length=4096) == (
        3 * KV_CACHE_BYTES_PER_TOKEN * 4096
    )
    assert predict_footprint_bytes(weights, quant_factor=0.5, context_length=4096) == (
        weights // 2 + KV_CACHE_BYTES_PER_TOKEN * 4096 + RUNTIME_OVERHEAD_BYTES
    )
    assert predict_footprint_bytes(
        weights, context_length=4096, mmproj_bytes=100
    ) == predict_footprint_bytes(weights, context_length=4096) + 100

    # No context on the launch spec => llama-server's default ctx.
    assert predict_footprint_bytes(weights) == predict_footprint_bytes(
        weights, context_length=DEFAULT_CONTEXT_LENGTH
    )


def test_footprint_is_max_of_observed_and_predicted() -> None:
    """The design's calibration rule: trust the measurement once there is one,
    fall back to the formula for models never yet run — never the minimum."""
    assert footprint_bytes(3 * GIB, None) == 3 * GIB  # never run: formula
    assert footprint_bytes(3 * GIB, 2 * GIB) == 3 * GIB  # fresh mmap'd RSS below prediction
    assert footprint_bytes(3 * GIB, 5 * GIB) == 5 * GIB  # measurement wins


def test_weights_come_from_store_size_or_stat_never_the_registry(tmp_path: Path) -> None:
    """PR-2 weights contract (design fix #6): ModelStoreEntry.size_bytes when
    given, on-disk stat as fallback, None when unknowable — and the resources
    module never imports the registry (whose plan() raises on store models)."""
    # size_bytes preferred: no file needs to exist.
    predicted = predicted_footprint_for_model(
        size_bytes=4 * GIB, model_path="/nonexistent/x.gguf", context_length=4096
    )
    assert predicted == predict_footprint_bytes(4 * GIB, context_length=4096)

    # Fallback: stat of the GGUF on disk.
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"g" * 1024)
    assert predicted_footprint_for_model(
        size_bytes=None, model_path=str(gguf), context_length=4096
    ) == predict_footprint_bytes(1024, context_length=4096)

    # Unknowable => None (unpriceable, not a fake number).
    assert (
        predicted_footprint_for_model(size_bytes=None, model_path="/nonexistent/x.gguf") is None
    )
    assert predicted_footprint_for_model(size_bytes=None, model_path=None) is None

    # Structural guard: no registry import anywhere in resources.py.
    tree = ast.parse(Path(resources.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "registry" not in (node.module or "")
        if isinstance(node, ast.Import):
            assert all("registry" not in alias.name for alias in node.names)


# ------------------------------------------------------------- footprint store


def test_footprint_store_persists_max_of_steady_samples(tmp_path: Path) -> None:
    store = FootprintStore(home=tmp_path)
    assert store.get("model.gguf") is None

    store.record_steady("model.gguf", 2 * GIB)
    assert store.get("model.gguf") == 2 * GIB
    store.record_steady("model.gguf", GIB)  # lower sample never regresses
    assert store.get("model.gguf") == 2 * GIB
    store.record_steady("model.gguf", 3 * GIB)
    assert store.get("model.gguf") == 3 * GIB

    # Survives a restart (sidecar on the serving volume, not process memory).
    assert FootprintStore(home=tmp_path).get("model.gguf") == 3 * GIB


def test_footprint_key_is_per_model_file_and_traversal_safe(tmp_path: Path) -> None:
    # Two deployments of the same GGUF share one calibration key.
    assert footprint_key("/models/invoice.gguf") == footprint_key("/other/dir/invoice.gguf")
    # Unsafe names map to a digest that stays inside the footprints dir.
    hostile = footprint_key("..\\..\\escape:me")
    assert hostile.startswith("sha256-")
    store = FootprintStore(home=tmp_path)
    store.record_steady("..\\..\\escape:me", GIB)
    assert store.get("..\\..\\escape:me") == GIB
    assert all(p.parent == store.directory for p in store.directory.iterdir())


# ------------------------------------------------------------------ calibration


@dataclass(frozen=True)
class _Obs:
    name: str
    phase: str
    rss_bytes: int
    model: str = "/models/invoice.gguf"


def _tracker(tmp_path: Path) -> ResourceTracker:
    return ResourceTracker(
        memory_reader=lambda: NodeMemory(8 * GIB, 6 * GIB, "cgroup"),
        footprints=FootprintStore(home=tmp_path),
        steady_hot_cycles=3,
        stability_fraction=0.02,
    )


def test_calibration_ignores_loading_phase_samples(tmp_path: Path) -> None:
    """The mmap ramp exclusion, part 1: 'loading' RSS (climbing as pages fault
    in) is NEVER a calibration sample, however many cycles it lasts."""
    tracker = _tracker(tmp_path)
    for rss in (GIB, 2 * GIB, 3 * GIB, 3 * GIB, 3 * GIB, 3 * GIB):
        tracker.observe_cycle([_Obs("invoice", "loading", rss)])
    assert tracker.footprints.get("invoice.gguf") is None


def test_calibration_waits_for_steady_state_hot_rss(tmp_path: Path) -> None:
    """The mmap ramp exclusion, part 2: early-hot cycles still ramping (health
    passes long before all pages are resident) are skipped; only a hot streak
    whose RSS has stopped moving is recorded."""
    tracker = _tracker(tmp_path)
    ramp = [1 * GIB, int(1.5 * GIB), 2 * GIB]  # deltas >2%: still faulting in
    for rss in ramp:
        tracker.observe_cycle([_Obs("invoice", "hot", rss)])
    assert tracker.footprints.get("invoice.gguf") is None  # streak long enough, not stable

    steady = 2 * GIB + 1024  # <2% delta vs previous: steady state
    tracker.observe_cycle([_Obs("invoice", "hot", steady)])
    assert tracker.footprints.get("invoice.gguf") == steady

    # A phase interruption (crash/unload) resets the streak: the next hot
    # sample alone is not trusted again.
    tracker.observe_cycle([_Obs("invoice", "cold", 0)])
    tracker.observe_cycle([_Obs("invoice", "hot", 3 * GIB)])
    tracker.observe_cycle([_Obs("invoice", "hot", 3 * GIB)])
    assert tracker.footprints.get("invoice.gguf") == steady  # streak of 2 < 3


def test_tracker_footprint_for_applies_the_max_rule(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    predicted = predict_footprint_bytes(2 * GIB, context_length=4096)
    assert tracker.footprint_for("/models/invoice.gguf", predicted) == predicted
    tracker.footprints.record_steady(footprint_key("/models/invoice.gguf"), 5 * GIB)
    assert tracker.footprint_for("/models/invoice.gguf", predicted) == 5 * GIB


def test_snapshot_sums_only_live_rss(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    snapshot = tracker.observe_cycle(
        [
            _Obs("a", "hot", 2 * GIB),
            _Obs("b", "loading", GIB),  # resident (ramping) => counted
            _Obs("c", "cold", 0),
            _Obs("d", "failed", 0),
        ]
    )
    assert snapshot == NodeSnapshot(
        total_bytes=8 * GIB, free_bytes=6 * GIB, source="cgroup", sum_rss_bytes=3 * GIB
    )


# --------------------------------------------------------------------- database


@pytest.fixture
def _sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


@pytest.mark.usefixtures("_sqlite_catalog")
def test_publish_node_snapshot_is_a_single_row_upsert() -> None:
    catalog = ModelCatalog()
    catalog.publish_node_snapshot(
        total_bytes=8 * GIB, free_bytes=6 * GIB, source="cgroup", sum_rss_bytes=GIB
    )
    catalog.publish_node_snapshot(
        total_bytes=8 * GIB, free_bytes=5 * GIB, source="cgroup", sum_rss_bytes=2 * GIB
    )

    view = catalog.get_node_snapshot()
    assert view is not None
    assert view["free_bytes"] == 5 * GIB
    assert view["sum_rss_bytes"] == 2 * GIB
    assert view["source"] == "cgroup"
    assert view["updated_at"] is not None

    with db.session_scope() as session:
        assert session is not None
        assert len(session.scalars(select(ServingNode)).all()) == 1  # one row, ever


def test_publish_skipped_without_database() -> None:
    """No DATABASE_URL: the default snapshot publisher degrades to a no-op —
    the repair cycle must never depend on Postgres (design fix #8)."""
    db.dispose_engine()
    publish_snapshot_via_catalog(
        NodeSnapshot(total_bytes=8 * GIB, free_bytes=6 * GIB, source="vm", sum_rss_bytes=0)
    )  # must not raise
    with pytest.raises(Exception, match="DATABASE_URL"):
        ModelCatalog().get_node_snapshot()


def test_ensure_serving_node_table_is_race_safe_and_idempotent(tmp_path: Path) -> None:
    """Mirrors the PR-1 migration pattern: CREATE TABLE IF NOT EXISTS (advisory
    lock on PostgreSQL), so concurrently starting processes cannot abort each
    other; a second run is a no-op."""
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    assert ensure_serving_node_table(engine) is True
    assert ensure_serving_node_table(engine) is False  # already there: no-op

    # The DDL itself is the race-safe form on PostgreSQL.
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    ddl = str(CreateTable(ServingNode.__table__, if_not_exists=True).compile(
        dialect=postgresql.dialect()
    ))
    assert "CREATE TABLE IF NOT EXISTS serving_node" in ddl


# -------------------------------------------------------- reconciler integration


def test_reconciler_publishes_node_snapshot_each_cycle(tmp_path: Path) -> None:
    """The reconciler is the sole snapshot writer: every cycle folds the
    observations through the tracker and hands the snapshot to the publisher
    (sum of live RSS + cgroup-first node numbers + source flag)."""
    from test_serving_reconciler import SPAWN_CREATE_TIME, ScriptedAdapter, _spec

    from docie_bench.serving.reconciler import ServingReconciler
    from docie_bench.serving.runtime import RuntimeKind
    from docie_bench.serving.supervisor import PersistentSupervisor

    adapter = ScriptedAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "deployments.json",
        adapters={RuntimeKind.LLAMACPP: adapter},
        create_time=lambda pid: SPAWN_CREATE_TIME,
    )
    snapshots: list[NodeSnapshot] = []
    reconciler = ServingReconciler(
        supervisor,
        fit_check=lambda record: (True, ""),
        rss_reader=lambda pid: 2 * GIB,
        create_time=lambda pid: SPAWN_CREATE_TIME,
        publisher=lambda observations: None,
        recency_home=tmp_path,
        tracker=_tracker(tmp_path),
        snapshot_publisher=snapshots.append,
    )
    supervisor.deploy(_spec())

    reconciler.run_cycle()
    reconciler.run_cycle()

    assert len(snapshots) == 2
    assert snapshots[-1] == NodeSnapshot(
        total_bytes=8 * GIB, free_bytes=6 * GIB, source="cgroup", sum_rss_bytes=2 * GIB
    )


def test_reconciler_cycle_survives_snapshot_failure(tmp_path: Path) -> None:
    """A measurement hiccup (unreadable node) must never fail the repair cycle."""
    from test_serving_reconciler import SPAWN_CREATE_TIME, ScriptedAdapter, _spec

    from docie_bench.serving.reconciler import ServingReconciler
    from docie_bench.serving.runtime import RuntimeKind
    from docie_bench.serving.supervisor import PersistentSupervisor

    def broken_reader() -> NodeMemory:
        raise OSError("cgroup went away")

    adapter = ScriptedAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "deployments.json",
        adapters={RuntimeKind.LLAMACPP: adapter},
        create_time=lambda pid: SPAWN_CREATE_TIME,
    )
    published: list[list[object]] = []
    snapshots: list[NodeSnapshot] = []
    reconciler = ServingReconciler(
        supervisor,
        fit_check=lambda record: (True, ""),
        rss_reader=lambda pid: GIB,
        create_time=lambda pid: SPAWN_CREATE_TIME,
        publisher=lambda observations: published.append(list(observations)),
        recency_home=tmp_path,
        tracker=ResourceTracker(
            memory_reader=broken_reader, footprints=FootprintStore(home=tmp_path)
        ),
        snapshot_publisher=snapshots.append,
    )
    supervisor.deploy(_spec())

    observations = reconciler.run_cycle()  # must not raise

    assert [obs.phase for obs in observations] == ["hot"]  # repair/observe intact
    assert published  # observed rows still published
    assert snapshots == []  # snapshot honestly absent, not fabricated


# ---------------------------------------------------------------------- endpoint


@pytest.mark.usefixtures("_sqlite_catalog")
def test_resources_endpoint_serves_snapshot_and_source_flag() -> None:
    from docie_bench.inngest.serving_api import serving_resources

    catalog = ModelCatalog()
    catalog.publish_node_snapshot(
        total_bytes=8 * GIB, free_bytes=6 * GIB, source="cgroup", sum_rss_bytes=2 * GIB
    )
    catalog.publish_observed(
        "invoice",
        engine="llama-server",
        state="ready",
        endpoint="http://serving:8090/v1",
        phase="hot",
        pid=42,
        pid_create_time=1.0,
        rss_bytes=2 * GIB,
        health_ok=True,
        last_error=None,
    )

    payload = asyncio.run(serving_resources())

    assert payload["observed_available"] is True
    assert payload["source"] == "cgroup"
    assert payload["node"]["total_bytes"] == 8 * GIB
    assert payload["node"]["free_bytes"] == 6 * GIB
    assert payload["node"]["sum_rss_bytes"] == 2 * GIB
    assert payload["deployments"] == [
        {"name": "invoice", "rss_bytes": 2 * GIB, "phase": "hot"}
    ]
    assert payload["detail"] is None


@pytest.mark.usefixtures("_sqlite_catalog")
def test_resources_endpoint_honest_when_snapshot_never_published() -> None:
    """DB up but the reconciler never published: 'observed unavailable', with a
    reason — never a locally-measured or fabricated number."""
    from docie_bench.inngest.serving_api import serving_resources

    payload = asyncio.run(serving_resources())

    assert payload["observed_available"] is False
    assert payload["node"] is None
    assert payload["source"] is None
    assert "no node snapshot published yet" in payload["detail"]


def test_resources_endpoint_honest_when_database_down() -> None:
    from docie_bench.inngest.serving_api import serving_resources

    db.dispose_engine()
    payload = asyncio.run(serving_resources())

    assert payload["observed_available"] is False
    assert payload["node"] is None
    assert payload["deployments"] == []
    assert "DATABASE_URL" in payload["detail"]
