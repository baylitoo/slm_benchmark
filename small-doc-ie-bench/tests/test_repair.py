"""ControlPlane.repair: recover a deployment on a (re)allocated port."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from docie_bench.serving.control_plane import ControlPlane, _DefaultSupervisor
from docie_bench.serving.runtime import HealthResult, RuntimeKind, RuntimeLaunchSpec, RuntimeProcess
from docie_bench.serving.supervisor import DeploymentSpec, PersistentSupervisor


class _CapturingAdapter:
    """Fake adapter capturing every launched spec; port 8090 is 'already bound'
    to model the orphan-holds-the-port bind loop the repair steps around."""

    def __init__(self, blocked_ports: set[int] | None = None) -> None:
        self.next_pid = 300
        self.running: set[int] = set()
        self.specs: list[RuntimeLaunchSpec] = []
        self.blocked_ports = blocked_ports or set()
        self._pid_port: dict[int, int] = {}
        self.rejected_contexts: set[int] = set()

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> Any:
        del log_path
        self.specs.append(spec)
        self.next_pid += 1
        self._pid_port[self.next_pid] = spec.port
        # A bind collision (EADDRINUSE) exits the process immediately — so a pid
        # started on a blocked port is NOT running (the reallocation signature).
        if (
            spec.port not in self.blocked_ports
            and (spec.context_length or 0) not in self.rejected_contexts
        ):
            self.running.add(self.next_pid)
        return RuntimeProcess(spec.runtime, f"http://{spec.host}:{spec.port}/v1", self.next_pid)

    def is_running(self, pid: int | None) -> bool:
        return pid in self.running

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del timeout
        if pid is not None:
            self.running.discard(pid)

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        del timeout
        healthy = (
            spec.port not in self.blocked_ports
            and (spec.context_length or 0) not in self.rejected_contexts
        )
        return HealthResult(healthy, 200 if healthy else 500)


def _plane(tmp_path: Path, adapter: _CapturingAdapter) -> tuple[Any, PersistentSupervisor]:
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.LLAMACPP: adapter}
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None)
    return ControlPlane(None, None, wrapper, None), supervisor  # type: ignore[arg-type]


def _seed_failed(
    supervisor: PersistentSupervisor,
    name: str,
    port: int,
    *,
    max_restarts: int = 5,
) -> None:
    supervisor.deploy(
        DeploymentSpec(
            name=name,
            max_restarts=max_restarts,
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.LLAMACPP,
                model=f"/store/{name}/model.gguf",
                alias=name,
                port=port,
            ),
        )
    )


def test_repair_reallocates_port_around_a_blocked_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_START", "8090")
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_END", "8188")
    # 8090 is held by an orphan (bind fails there); repair must pick another.
    adapter = _CapturingAdapter(blocked_ports={8090})
    plane, supervisor = _plane(tmp_path, adapter)
    _seed_failed(supervisor, "guardrails-pii", 8090)

    record = asyncio.run(plane.repair("guardrails-pii"))

    new_port = record["spec"]["launch"]["port"]
    assert new_port != 8090
    assert record["state"] == "ready"
    # Preserved the launch (same model), only the port changed.
    assert adapter.specs[-1].model == "/store/guardrails-pii/model.gguf"


def test_repair_honors_explicit_port(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_START", "8090")
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_END", "8188")
    adapter = _CapturingAdapter()
    plane, supervisor = _plane(tmp_path, adapter)
    _seed_failed(supervisor, "svc", 8090)

    record = asyncio.run(plane.repair("svc", port=8137))

    assert record["spec"]["launch"]["port"] == 8137
    assert record["state"] == "ready"


def test_repair_resets_restart_budget(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_START", "8090")
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_END", "8188")
    adapter = _CapturingAdapter()
    plane, supervisor = _plane(tmp_path, adapter)
    _seed_failed(supervisor, "svc", 8090)
    # Simulate an exhausted budget on the existing record.
    rec = supervisor.get("svc")
    rec.restart_count = 5
    rec.consecutive_health_failures = 9

    asyncio.run(plane.repair("svc", port=8100))

    after = supervisor.get("svc")
    assert after.restart_count == 0
    assert after.consecutive_health_failures == 0


def test_reconfigure_restarts_hot_deployment_on_same_port(tmp_path: Path) -> None:
    adapter = _CapturingAdapter()
    plane, supervisor = _plane(tmp_path, adapter)
    _seed_failed(supervisor, "svc", 8090, max_restarts=0)
    launches_before = len(adapter.specs)

    record = asyncio.run(
        plane.reconfigure("svc", context_length=32768, max_tokens=4096)
    )

    launch = record["spec"]["launch"]
    assert launch["port"] == 8090
    assert launch["context_length"] == 32768
    assert launch["max_tokens"] == 4096
    assert record["state"] == "ready"
    assert len(adapter.specs) == launches_before + 1


def test_reconfigure_keeps_offloaded_deployment_offloaded_and_pinned(tmp_path: Path) -> None:
    adapter = _CapturingAdapter()
    plane, supervisor = _plane(tmp_path, adapter)
    _seed_failed(supervisor, "svc", 8090)
    supervisor.pin("svc", pinned=True)
    supervisor.unload("svc")
    launches_before = len(adapter.specs)

    record = asyncio.run(
        plane.reconfigure("svc", context_length=16384, max_tokens=None)
    )

    assert record["spec"]["launch"]["context_length"] == 16384
    assert record["spec"]["launch"]["max_tokens"] is None
    assert record["spec"]["desired_state"] == "stopped"
    assert record["state"] == "stopped"
    assert record["activation"] == "managed"
    assert record["pinned"] is True
    assert len(adapter.specs) == launches_before


def test_reconfigure_rolls_back_when_runtime_rejects_context(tmp_path: Path) -> None:
    adapter = _CapturingAdapter()
    plane, supervisor = _plane(tmp_path, adapter)
    _seed_failed(supervisor, "svc", 8090, max_restarts=0)
    old_launch = supervisor.get("svc").spec.launch
    adapter.rejected_contexts.add(65536)

    try:
        asyncio.run(plane.reconfigure("svc", context_length=65536, max_tokens=4096))
    except RuntimeError:
        pass
    else:
        raise AssertionError("the rejected context should fail reconfiguration")

    restored = supervisor.get("svc")
    assert restored.spec.launch == old_launch
    assert restored.state.value == "ready"
