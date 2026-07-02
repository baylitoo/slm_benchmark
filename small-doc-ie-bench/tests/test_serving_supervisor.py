from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from docie_bench.serving.runtime import (
    HealthResult,
    LifecycleState,
    RemoteRuntime,
    RuntimeKind,
    RuntimeLaunchSpec,
    RuntimeProcess,
)
from docie_bench.serving.supervisor import (
    DeploymentSpec,
    DesiredState,
    PersistentSupervisor,
    RestartPolicy,
    SupervisorStateError,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.next_pid = 100
        self.running: set[int] = set()
        self.health_results: list[HealthResult] = []
        self.starts = 0
        self.stops: list[int | None] = []
        self.log_paths: list[Path | None] = []

    def start(
        self,
        spec: RuntimeLaunchSpec,
        *,
        log_path: Path | None = None,
    ) -> RuntimeProcess:
        self.starts += 1
        self.next_pid += 1
        self.running.add(self.next_pid)
        self.log_paths.append(log_path)
        return RuntimeProcess(spec.runtime, f"http://{spec.host}:{spec.port}/v1", self.next_pid)

    def is_running(self, pid: int | None) -> bool:
        return pid in self.running

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del timeout
        self.stops.append(pid)
        if pid is not None:
            self.running.discard(pid)

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        del spec, timeout
        return self.health_results.pop(0) if self.health_results else HealthResult(True, 200)


def _deployment(**overrides: Any) -> DeploymentSpec:
    values: dict[str, Any] = {
        "name": "invoice",
        "launch": RuntimeLaunchSpec(
            runtime=RuntimeKind.VLLM,
            model="org/model",
            alias="invoice",
        ),
    }
    values.update(overrides)
    return DeploymentSpec(**values)


def test_deploy_persists_and_restart_reuses_observed_process(tmp_path: Path) -> None:
    state_path = tmp_path / "supervisor.json"
    adapter = FakeAdapter()
    supervisor = PersistentSupervisor(state_path, adapters={RuntimeKind.VLLM: adapter})

    first = supervisor.deploy(_deployment())
    restarted = PersistentSupervisor(state_path, adapters={RuntimeKind.VLLM: adapter})
    second = restarted.reconcile("invoice")

    assert first.state == LifecycleState.READY
    assert first.pid == second.pid
    assert adapter.starts == 1
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["deployments"]["invoice"]["spec"]["launch"]["runtime"] == "vllm"


def test_processless_remote_deployment_records_endpoint(tmp_path: Path) -> None:
    adapter = RemoteRuntime(
        health_get=lambda url, timeout, headers: HealthResult(True, 200),
    )
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.REMOTE: adapter},
    )
    deployment = _deployment(
        launch=RuntimeLaunchSpec(
            runtime=RuntimeKind.REMOTE,
            model="remote-model",
            alias="invoice",
            endpoint="https://models.example/v1",
        )
    )

    record = supervisor.deploy(deployment)

    assert record.state == LifecycleState.READY
    assert record.pid is None
    assert record.endpoint == "https://models.example/v1"


def test_dead_process_is_recovered_with_restart_count(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.VLLM: adapter},
    )
    first = supervisor.deploy(_deployment(max_restarts=2))
    assert first.pid is not None
    first_pid = first.pid
    adapter.running.remove(first_pid)

    recovered = supervisor.reconcile("invoice")

    assert recovered.state == LifecycleState.READY
    assert recovered.pid != first_pid
    assert recovered.restart_count == 1
    assert adapter.starts == 2


def test_health_failures_degrade_then_restart(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    adapter.health_results = [
        HealthResult(True, 200),
        HealthResult(False, 503, "loading failed"),
        HealthResult(False, 503, "loading failed"),
        HealthResult(True, 200),
    ]
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.VLLM: adapter},
    )
    initial = supervisor.deploy(_deployment(health_failure_threshold=2))
    initial_pid = initial.pid

    assert supervisor.reconcile("invoice").state == LifecycleState.STARTING
    degraded = supervisor.reconcile("invoice")
    assert degraded.state == LifecycleState.DEGRADED
    assert degraded.pid is None
    assert adapter.stops == [initial_pid]
    recovered = supervisor.reconcile("invoice")
    assert recovered.state == LifecycleState.READY
    assert recovered.restart_count == 1


def test_await_ready_polls_until_healthy(tmp_path: Path) -> None:
    # A model that is still loading refuses connections on the first probes and
    # only becomes healthy once loaded. deploy() spawns + probes ONCE (fail);
    # await_ready must keep re-probing until READY instead of freezing at
    # STARTING.
    adapter = FakeAdapter()
    adapter.health_results = [
        HealthResult(False, None, "Connection refused"),  # deploy-time probe
        HealthResult(False, None, "Connection refused"),  # await_ready poll 1
        HealthResult(True, 200),  # await_ready poll 2 -> loaded
    ]
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.VLLM: adapter},
    )
    # High threshold so the readiness window cannot trip degrade-and-kill.
    started = supervisor.deploy(_deployment(health_failure_threshold=10))
    assert started.state == LifecycleState.STARTING
    assert started.consecutive_health_failures == 1

    ready = supervisor.await_ready(
        "invoice", timeout_s=30, interval_s=0.01, sleep=lambda _: None
    )

    assert ready.state == LifecycleState.READY
    assert ready.consecutive_health_failures == 0
    assert adapter.starts == 1  # never killed/restarted while loading


def test_await_ready_returns_last_state_on_timeout(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    # Never becomes healthy: every probe refuses the connection.
    adapter.health = lambda spec, *, timeout=2: HealthResult(  # type: ignore[method-assign]
        False, None, "Connection refused"
    )
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.VLLM: adapter},
    )
    supervisor.deploy(_deployment(health_failure_threshold=10))

    calls: list[float] = []
    record = supervisor.await_ready(
        "invoice", timeout_s=0.0, interval_s=0.01, sleep=calls.append
    )

    assert record.state != LifecycleState.READY
    assert calls == []  # timeout_s=0 -> no polling loop, honest early return


def test_never_restart_policy_leaves_failed_process_stopped(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.VLLM: adapter},
    )
    record = supervisor.deploy(_deployment(restart_policy=RestartPolicy.NEVER))
    assert record.pid is not None
    adapter.running.remove(record.pid)

    failed = supervisor.reconcile("invoice")

    assert failed.state == LifecycleState.FAILED
    assert failed.pid is None
    assert adapter.starts == 1


def test_stop_and_remove_are_persistent(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    adapter = FakeAdapter()
    supervisor = PersistentSupervisor(state_path, adapters={RuntimeKind.VLLM: adapter})
    deployed = supervisor.deploy(_deployment())
    deployed_pid = deployed.pid

    stopped = supervisor.stop("invoice")
    assert stopped.state == LifecycleState.STOPPED
    assert stopped.spec.desired_state == DesiredState.STOPPED
    assert adapter.stops == [deployed_pid]

    supervisor.remove("invoice")
    assert supervisor.list() == ()
    assert json.loads(state_path.read_text(encoding="utf-8"))["deployments"] == {}


def test_launch_failure_is_persisted_without_crashing_reconcile(tmp_path: Path) -> None:
    adapter = FakeAdapter()

    def fail_start(*args: Any, **kwargs: Any) -> RuntimeProcess:
        raise RuntimeError("runtime unavailable")

    adapter.start = fail_start  # type: ignore[method-assign]
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.VLLM: adapter},
    )

    record = supervisor.deploy(_deployment())

    assert record.state == LifecycleState.FAILED
    assert record.last_error == "runtime unavailable"


def test_corrupt_state_is_rejected(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(SupervisorStateError, match="Invalid supervisor state"):
        PersistentSupervisor(state_path, adapters={RuntimeKind.VLLM: FakeAdapter()})
