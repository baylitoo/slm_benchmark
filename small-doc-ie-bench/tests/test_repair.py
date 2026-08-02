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

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> Any:
        del log_path
        self.specs.append(spec)
        self.next_pid += 1
        self._pid_port[self.next_pid] = spec.port
        # A bind collision (EADDRINUSE) exits the process immediately — so a pid
        # started on a blocked port is NOT running (the reallocation signature).
        if spec.port not in self.blocked_ports:
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
        return HealthResult(spec.port not in self.blocked_ports, 200)


def _plane(tmp_path: Path, adapter: _CapturingAdapter) -> tuple[Any, PersistentSupervisor]:
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.LLAMACPP: adapter}
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None)
    return ControlPlane(None, None, wrapper, None), supervisor  # type: ignore[arg-type]


def _seed_failed(supervisor: PersistentSupervisor, name: str, port: int) -> None:
    supervisor.deploy(
        DeploymentSpec(
            name=name,
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
