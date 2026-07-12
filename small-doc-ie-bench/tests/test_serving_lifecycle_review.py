"""PR-4 review fixes, each block encoding one finding. All failed pre-fix.

* BLOCKER — ``store:<name>`` selectors bypassed autoload AND recency stamping
  because the prefix was never stripped: with the idle TTL on by default a
  store-routed deployment was evicted while actively serving and every later
  ``store:`` extraction hard-failed with no auto-reload.
* HIGH — ``last_served`` was stamped only by the Inngest extract path; the
  direct API endpoints and the benchmark runner never stamped, so deployments
  serving those surfaces read as idle forever (first eviction victims).
* MEDIUM — fit-before-evict was TOCTOU: a recovered-pid eviction did not wait
  for the victim to die, and two concurrent loads of DIFFERENT deployments
  both passed ``assess_fit`` against the same ``free_bytes``.
* MEDIUM — ``_DefaultSupervisor._coordinator()`` lazy-init was an
  unsynchronized check-then-set: two concurrent first loads could each build
  a LoadCoordinator with its own lock map.
* MEDIUM — the worker's await of a serving-side load blocked in ONE step for
  the whole (up to 1800s) size-aware budget, betting on an unpinned Inngest
  server step/lease ceiling; it must be chunked into bounded steps.
* LOW — the autoload decision was evaluated OUTSIDE any step, so a
  function-level retry could replay a different step graph than the memoized
  attempt.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from docie_bench.serving import recency
from docie_bench.serving.lifecycle import FitDecision, LoadCoordinator, LoadError
from docie_bench.serving.runtime import (
    HealthResult,
    LifecycleState,
    RuntimeKind,
    RuntimeLaunchSpec,
    RuntimeProcess,
)
from docie_bench.serving.supervisor import (
    DeploymentRecord,
    DeploymentSpec,
    PersistentSupervisor,
)

SPAWN_CREATE_TIME = 1000.0


class ScriptedAdapter:
    """A fake runtime: scripted health, crashable processes (mirror of the
    fixture in test_serving_lifecycle.py)."""

    def __init__(self) -> None:
        self.next_pid = 100
        self.running: set[int] = set()
        self.starts = 0
        self.stops: list[int | None] = []
        self.healthy: dict[str, bool] = {}

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


def _supervisor(home: Path, adapter: ScriptedAdapter) -> PersistentSupervisor:
    return PersistentSupervisor(
        home / "deployments.json",
        adapters={RuntimeKind.LLAMACPP: adapter},
        create_time=lambda pid: SPAWN_CREATE_TIME,
    )


@pytest.fixture
def serving_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "serving"
    home.mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    return home


# ---------------------------------------- blocker: store:<name> selectors


def test_autoload_target_resolves_store_prefixed_selectors(serving_home: Path) -> None:
    """``store:<name>`` routes via the placement of deployment record
    ``<name>`` (serve_store_model names the record after the store entry), so
    the prefix must be stripped before the record lookup — otherwise a
    store-routed deployment evicted by the idle TTL never auto-reloads."""
    from docie_bench.inngest.functions import _autoload_target

    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)

    # Live store deployment: no autoload needed, prefix or not.
    supervisor.deploy(_spec("hot-store", port=8095))
    assert _autoload_target({"model_profile": "store:hot-store"}) is None

    # Evicted + managed via the store: selector => load-and-wait on the RECORD.
    supervisor.deploy(_spec("invoice", port=8096))
    supervisor.unload("invoice")
    target = _autoload_target({"model_profile": "store:invoice"})
    assert target is not None
    name, budget = target
    assert name == "invoice"  # the record name, NOT "store:invoice"
    assert budget >= 120.0

    # A store: ref that names no record stays a no-op (placement-only path).
    assert _autoload_target({"model_profile": "store:unknown-model"}) is None


def test_recency_stamp_covers_store_prefixed_profiles(serving_home: Path) -> None:
    """A store-routed profile is named ``store:<name>`` (placement_resolver),
    so the recency stamp must strip the prefix to hit the record's sidecar —
    or store-routed traffic never counts as activity and the deployment is
    evicted mid-use."""
    from docie_bench.inngest.functions import _stamp_deployment_recency

    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)
    supervisor.deploy(_spec())

    _stamp_deployment_recency(explicit=None, profile_name="store:invoice")
    assert "invoice" in recency.read_for(["invoice"], home=serving_home)

    # A plain models.yaml profile is not a deployment: no sidecar.
    _stamp_deployment_recency(explicit=None, profile_name="studio_default")
    assert "studio_default" not in recency.read_for(
        ["studio_default"], home=serving_home
    )


# ------------------------- high: every serving surface stamps last_served


def test_api_extract_surface_stamps_recency(
    serving_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direct API extract endpoints serve traffic too: finalize_response
    must stamp the deployment's recency sidecar or API-driven deployments
    read as idle forever (first idle-TTL/LRU eviction victims mid-use)."""
    import docie_bench.api as api_module

    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)
    supervisor.deploy(_spec())

    monkeypatch.setattr(api_module, "record_extraction", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_module.settings, "redacted_response_fields", "")
    response = SimpleNamespace(model_profile="invoice", result={})

    api_module.finalize_response(response, tenant_id="tenant-a")  # type: ignore[arg-type]

    assert "invoice" in recency.read_for(["invoice"], home=serving_home)


def test_api_extract_surface_stamps_store_profiles_too(
    serving_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import docie_bench.api as api_module

    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)
    supervisor.deploy(_spec())

    monkeypatch.setattr(api_module, "record_extraction", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_module.settings, "redacted_response_fields", "")
    response = SimpleNamespace(model_profile="store:invoice", result={})

    api_module.finalize_response(response, tenant_id="tenant-a")  # type: ignore[arg-type]

    assert "invoice" in recency.read_for(["invoice"], home=serving_home)


def test_benchmark_runner_stamps_recency(
    serving_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A benchmark drives sustained load through a deployment — that IS served
    traffic. Without stamping, a longer-than-idle-TTL run's deployment is
    unloaded mid-benchmark by the reconciler."""
    from docie_bench.benchmark.runner import run_benchmark

    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)
    supervisor.deploy(_spec("extractor", port=8097))

    document = tmp_path / "doc-0.txt"
    document.write_text("Invoice INV-0", encoding="utf-8")
    dataset = tmp_path / "manifest.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "doc_id": "doc-0",
                "file_path": document.name,
                "schema_name": "invoice",
                "ground_truth": {"invoice_number": "INV-0"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    models = tmp_path / "models.yaml"
    models.write_text(
        "profiles:\n"
        "  extractor:\n"
        "    model: deterministic-model\n"
        "    base_url: http://extractor/v1\n"
        "    temperature: 0\n",
        encoding="utf-8",
    )

    class FakeExtractionService:
        def __init__(self, profile: Any) -> None:
            self.profile = profile

        async def extract_from_file(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                schema_name="invoice",
                dynamic_schema=None,
                latency_ms=5,
                validation=SimpleNamespace(model_dump=lambda: {"valid": True}),
                result={"invoice_number": {"value": "INV-0", "evidence_ids": ["b1"]}},
            )

    monkeypatch.setattr(
        "docie_bench.benchmark.runner.get_settings",
        lambda: SimpleNamespace(default_ocr_backend="pdf_text", runs_dir=tmp_path / "runs"),
    )
    monkeypatch.setattr(
        "docie_bench.benchmark.runner.ExtractionService", FakeExtractionService
    )

    asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            model_profile="extractor",
            output_dir=tmp_path / "run",
            concurrency=1,
        )
    )

    assert "extractor" in recency.read_for(["extractor"], home=serving_home)


# --------------- medium: concurrent admission shares ONE free-RAM budget


def test_concurrent_loads_of_different_deployments_share_one_fit_budget(
    tmp_path: Path,
) -> None:
    """The per-deployment locks serialize same-name loads only; admission for
    DIFFERENT names must not double-spend the same measured free_bytes. The
    first admitted load holds an in-flight reservation until READY; the
    second sees free minus the reservation and fails honestly."""
    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    for name, port in (("alpha", 8091), ("beta", 8092)):
        supervisor.deploy(_spec(name, port=port))
        supervisor.unload(name)
    adapter.healthy["alpha"] = False  # alpha stays loading until released

    # Node has 150 free; each model needs 100: only ONE may be admitted.
    def assess(record: DeploymentRecord) -> FitDecision:
        return FitDecision(True, 100, 150, 0, "")

    coordinator = LoadCoordinator(
        supervisor, assess=assess, sleep=lambda seconds: time.sleep(0.005)
    )
    errors: list[Exception] = []

    def load_alpha() -> None:
        try:
            coordinator.load("alpha")
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    thread = threading.Thread(target=load_alpha)
    thread.start()
    try:
        deadline = time.time() + 5.0
        while "alpha" not in coordinator._inflight and time.time() < deadline:
            time.sleep(0.005)
        assert "alpha" in coordinator._inflight  # admitted, reservation held

        with pytest.raises(LoadError, match="never evict-to-not-fit"):
            coordinator.load("beta")
    finally:
        adapter.healthy["alpha"] = True
        thread.join(timeout=30)

    assert errors == []
    assert supervisor.get("alpha").state == LifecycleState.READY
    assert "alpha" not in coordinator._inflight  # reservation released on READY

    # With alpha's reservation released, the SAME beta load is admitted (the
    # scripted gate still reports 150 free) and completes.
    record = coordinator.load("beta")
    assert record.state == LifecycleState.READY


def test_recovered_pid_shutdown_waits_for_the_victim_to_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit-before-evict is only sound if eviction WAITS: after a serving
    restart there is no Popen handle, and a fire-and-forget SIGTERM would let
    the next assess_fit price against RAM the dying victim still holds."""
    from docie_bench.serving import runtime as runtime_module

    events: list[str] = []

    class FakeNoSuchProcess(Exception):  # noqa: N818 - mirrors psutil's name
        pass

    class FakeTimeoutExpired(Exception):  # noqa: N818 - mirrors psutil's name
        pass

    class FakeExternalProcess:
        def __init__(self, pid: int) -> None:
            events.append("lookup")

        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, timeout: float | None = None) -> None:
            events.append("wait")
            if events.count("wait") == 1:
                raise FakeTimeoutExpired()  # SIGTERM ignored: must escalate

        def kill(self) -> None:
            events.append("kill")

    class FakePsutil:
        NoSuchProcess = FakeNoSuchProcess
        TimeoutExpired = FakeTimeoutExpired
        Process = FakeExternalProcess

        @staticmethod
        def pid_exists(pid: int) -> bool:
            return True

    monkeypatch.setattr(runtime_module, "psutil", FakePsutil)
    from docie_bench.serving.runtime import LlamaCppRuntime

    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    adapter.shutdown(424242, timeout=0.01)  # recovered pid: not in _processes

    assert events == ["lookup", "terminate", "wait", "kill", "wait"]


# ------------- medium: the shared LoadCoordinator is built exactly once


def test_shared_load_coordinator_is_built_exactly_once_under_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent FIRST loads (cold-start pileup — exactly when the
    coordinator's lock map matters) must never each construct their own
    LoadCoordinator via the lazy check-then-set."""
    from docie_bench.serving import lifecycle as lifecycle_module
    from docie_bench.serving.control_plane import _DefaultSupervisor

    adapter = ScriptedAdapter()
    supervisor = _supervisor(tmp_path, adapter)
    wrapper = _DefaultSupervisor(supervisor, planner=None)

    barrier = threading.Barrier(2)
    created: list[Any] = []

    class RacyCoordinator:
        def __init__(self, backend: Any, **kwargs: Any) -> None:
            # Reachable only past the None check. If TWO threads get here the
            # barrier releases instantly and two instances are built (the
            # race); with the init lock the second thread never enters, the
            # barrier times out for the first, and exactly one is built.
            with contextlib.suppress(threading.BrokenBarrierError):
                barrier.wait(timeout=0.3)
            created.append(self)

    monkeypatch.setattr(lifecycle_module, "LoadCoordinator", RacyCoordinator)

    results: list[Any] = []
    threads = [
        threading.Thread(target=lambda: results.append(wrapper._coordinator()))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(created) == 1
    assert len(results) == 2
    assert results[0] is results[1]


# -------- medium: worker await is chunked; low: the plan is a durable step


class FakeStep:
    """Minimal durable-step engine: memoizes by step id, JSON-round-trips
    results (like the real engine), and records the PRESENTED step sequence
    so tests can assert graph determinism across simulated retries."""

    def __init__(self) -> None:
        self.memo: dict[str, Any] = {}
        self.calls: list[str] = []
        self.sent: list[Any] = []
        self.after: dict[str, Any] = {}

    async def run(self, step_id: str, fn: Any) -> Any:
        self.calls.append(step_id)
        if step_id in self.memo:
            return self.memo[step_id]
        value = fn()
        if inspect.isawaitable(value):
            value = await value
        value = json.loads(json.dumps(value))
        self.memo[step_id] = value
        callback = self.after.pop(step_id, None)
        if callback is not None:
            callback()
        return value

    async def send_event(self, step_id: str, event: Any) -> list[str]:
        self.calls.append(step_id)
        memoized = self.memo.get(step_id)
        if memoized is not None:
            return list(memoized)
        self.sent.append(event)
        ids = [f"evt_{len(self.sent)}"]
        self.memo[step_id] = ids
        return ids


async def test_autoload_await_is_chunked_into_bounded_steps(serving_home: Path) -> None:
    """No single worker step may block for the whole (up to 1800s) size-aware
    budget — the Inngest server's step/lease ceiling is not pinned. The wait
    must be split into bounded await-deployment-ready:{i} chunks and fail
    honestly only after the WHOLE budget."""
    from docie_bench.inngest import functions as functions_module

    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)
    supervisor.deploy(_spec())
    supervisor.unload("invoice")  # evicted + managed; nothing will reload it

    step = FakeStep()
    # Memoized plan (as a real attempt would have recorded it): budget 0.06s
    # against a 0.02s per-step ceiling => exactly 3 bounded chunks.
    step.memo["plan-autoload"] = ["invoice", 0.06]

    with pytest.raises(TimeoutError, match="size-aware load budget"):
        await functions_module._ensure_deployment_live(
            step,
            {"deployment": "invoice"},
            "chan",
            step_budget_s=0.02,
            poll_interval_s=0.005,
        )

    awaits = [c for c in step.calls if c.startswith("await-deployment-ready:")]
    assert awaits == [
        "await-deployment-ready:0",
        "await-deployment-ready:1",
        "await-deployment-ready:2",
    ]


async def test_autoload_plan_is_durable_and_replays_an_identical_step_graph(
    serving_home: Path,
) -> None:
    """The autoload DECISION must be a memoized step: a function-level retry
    after the deployment went live must replay the same request-load/await
    steps, not silently skip them (non-deterministic step graph)."""
    from docie_bench.inngest import functions as functions_module

    adapter = ScriptedAdapter()
    supervisor = _supervisor(serving_home, adapter)
    supervisor.deploy(_spec())
    supervisor.unload("invoice")

    step = FakeStep()
    # After the first (not-ready) chunk executes, the serving side "finishes
    # the load": the record goes live and the next chunk observes it.
    step.after["await-deployment-ready:0"] = lambda: supervisor.deploy(_spec())

    target = await functions_module._ensure_deployment_live(
        step,
        {"deployment": "invoice"},
        "chan",
        step_budget_s=0.02,
        poll_interval_s=0.005,
    )
    assert target == "invoice"
    assert step.sent[0].name == functions_module.LOAD_EVENT
    assert step.sent[0].data == {"name": "invoice"}
    first_graph = list(step.calls)
    assert first_graph == [
        "plan-autoload",
        "request-load",
        "await-deployment-ready:0",
        "await-deployment-ready:1",
    ]

    # Simulated function-level retry: memo retained, deployment NOW live. The
    # replay must present the identical step sequence from the memoized plan
    # (pre-fix, a fresh _autoload_target read would skip the load steps).
    step.calls.clear()
    target = await functions_module._ensure_deployment_live(
        step,
        {"deployment": "invoice"},
        "chan",
        step_budget_s=0.02,
        poll_interval_s=0.005,
    )
    assert target == "invoice"
    assert step.calls == first_graph
    assert len(step.sent) == 1  # the memoized send_event did not re-fire
