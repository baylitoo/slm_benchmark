"""ControlPlane.resize_store_model: zero-downtime context-length resize.

Mirrors ``test_repair.py``'s fake-adapter pattern (no real llama-server is
ever launched). ``_CapturingAdapter`` additionally records a ``timeline`` of
start/shutdown events by pid so the drain-and-relaunch sequencing itself is
asserted, not just the end state: the new instance's ``start`` must precede
the old instance's ``shutdown``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from docie_bench.serving.control_plane import (
    ControlPlane,
    ResizeAdmissionError,
    ResizeUnsupportedError,
    _DefaultSupervisor,
)
from docie_bench.serving.resources import NodeMemory
from docie_bench.serving.runtime import (
    HealthResult,
    RuntimeKind,
    RuntimeLaunchSpec,
    RuntimeProcess,
)
from docie_bench.serving.supervisor import DeploymentSpec, PersistentSupervisor


class _CapturingAdapter:
    """Fake adapter tracking every launched spec AND a start/shutdown
    timeline (keyed by pid), so a resize's old+new instances can coexist in
    the test the same way they briefly do in production."""

    def __init__(self, rejected_contexts: set[int] | None = None) -> None:
        self.next_pid = 500
        self.running: set[int] = set()
        self.specs: list[RuntimeLaunchSpec] = []
        self.timeline: list[tuple[str, int]] = []
        self.rejected_contexts = rejected_contexts or set()
        # pid -> the launch it was started with, so find_processes can prove
        # it matches ONLY a same-port+same-model orphan -- never the NEW
        # instance's pid, which always binds a DIFFERENT port during a resize.
        self._launch_by_pid: dict[int, RuntimeLaunchSpec] = {}

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> Any:
        del log_path
        self.specs.append(spec)
        self.next_pid += 1
        pid = self.next_pid
        self._launch_by_pid[pid] = spec
        if (spec.context_length or 0) not in self.rejected_contexts:
            self.running.add(pid)
        self.timeline.append(("start", pid))
        return RuntimeProcess(spec.runtime, f"http://{spec.host}:{spec.port}/v1", pid)

    def is_running(self, pid: int | None) -> bool:
        return pid in self.running

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del timeout
        if pid is not None:
            self.running.discard(pid)
            self.timeline.append(("shutdown", pid))

    def find_processes(self, spec: RuntimeLaunchSpec) -> tuple[int, ...]:
        # Mirrors the real adapter's orphan-reap match: same reserved port AND
        # same model path, restricted to still-running pids.
        return tuple(
            pid
            for pid in self.running
            if (launch := self._launch_by_pid.get(pid)) is not None
            and launch.port == spec.port
            and launch.model == spec.model
        )

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        del timeout
        healthy = (spec.context_length or 0) not in self.rejected_contexts
        return HealthResult(healthy, 200 if healthy else 500)


def _plane(tmp_path: Path, adapter: _CapturingAdapter) -> tuple[Any, PersistentSupervisor]:
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.LLAMACPP: adapter, RuntimeKind.ENCODER: adapter},
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None)
    return ControlPlane(None, None, wrapper, None), supervisor  # type: ignore[arg-type]


def _seed_llamacpp(
    supervisor: PersistentSupervisor,
    name: str,
    port: int,
    model_path: Path,
    *,
    context_length: int = 8192,
) -> None:
    supervisor.deploy(
        DeploymentSpec(
            name=name,
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.LLAMACPP,
                model=str(model_path),
                alias=name,
                port=port,
                context_length=context_length,
            ),
        )
    )


def _weights_file(tmp_path: Path, name: str, size_bytes: int = 10 * 1024 * 1024) -> Path:
    path = tmp_path / name / "model.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size_bytes)
    return path


def _set_port_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_START", "8090")
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_END", "8188")


def test_resize_drains_new_instance_before_stopping_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_port_range(monkeypatch)
    model_path = _weights_file(tmp_path, "svc")
    adapter = _CapturingAdapter()
    plane, supervisor = _plane(tmp_path, adapter)
    _seed_llamacpp(supervisor, "svc", 8090, model_path, context_length=8192)
    old_pid = supervisor.get("svc").pid
    assert old_pid in adapter.running

    from docie_bench.serving import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "read_node_memory",
        lambda: NodeMemory(total_bytes=100 * 1024**3, free_bytes=50 * 1024**3, source="vm"),
    )

    record = asyncio.run(plane.resize_store_model("svc", context_length=32768))

    # Identity preserved: same deployment name, new launch settings.
    assert record["spec"]["name"] == "svc"
    assert record["spec"]["launch"]["context_length"] == 32768
    new_port = record["spec"]["launch"]["port"]
    assert new_port != 8090
    new_pid = record["pid"]
    assert new_pid != old_pid
    assert record["state"] == "ready"

    # Zero-downtime sequencing: the new instance started before the old one
    # was stopped -- routing was never pointed at nothing.
    start_index = adapter.timeline.index(("start", new_pid))
    stop_index = adapter.timeline.index(("shutdown", old_pid))
    assert start_index < stop_index

    # Old process reaped -- including via the orphan-sweep (find_processes),
    # which matches on port+model and must NOT catch the new instance (it
    # binds a DIFFERENT port): the new pid survives the old one's reap.
    assert old_pid not in adapter.running
    assert new_pid in adapter.running
    assert [r.spec.name for r in supervisor.list()] == ["svc"]


def test_resize_admission_rejects_when_ram_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_port_range(monkeypatch)
    model_path = _weights_file(tmp_path, "svc", size_bytes=50 * 1024 * 1024)
    adapter = _CapturingAdapter()
    plane, supervisor = _plane(tmp_path, adapter)
    _seed_llamacpp(supervisor, "svc", 8090, model_path, context_length=8192)
    old_pid = supervisor.get("svc").pid
    specs_before = list(adapter.specs)

    from docie_bench.serving import lifecycle

    monkeypatch.setattr(
        lifecycle,
        "read_node_memory",
        lambda: NodeMemory(total_bytes=1_000_000_000, free_bytes=1_000, source="vm"),
    )

    with pytest.raises(ResizeAdmissionError):
        asyncio.run(plane.resize_store_model("svc", context_length=65536))

    # Nothing touched: no shadow was ever launched, old process still hot.
    assert adapter.specs == specs_before
    after = supervisor.get("svc")
    assert after.pid == old_pid
    assert after.spec.launch.context_length == 8192
    assert old_pid in adapter.running
    assert [r.spec.name for r in supervisor.list()] == ["svc"]


def test_resize_rejects_non_llamacpp_runtime(tmp_path: Path) -> None:
    adapter = _CapturingAdapter()
    plane, supervisor = _plane(tmp_path, adapter)
    supervisor.deploy(
        DeploymentSpec(
            name="guard",
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.ENCODER,
                model="org/gliner-model",
                alias="guard",
                port=8090,
            ),
        )
    )
    specs_before = list(adapter.specs)

    with pytest.raises(ResizeUnsupportedError, match="does not accept a context-length"):
        asyncio.run(plane.resize_store_model("guard", context_length=4096))

    assert adapter.specs == specs_before


def test_resize_edits_a_stopped_deployment_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No process to drain -- same in-place edit reconfigure uses."""
    _set_port_range(monkeypatch)
    model_path = _weights_file(tmp_path, "svc")
    adapter = _CapturingAdapter()
    plane, supervisor = _plane(tmp_path, adapter)
    _seed_llamacpp(supervisor, "svc", 8090, model_path)
    supervisor.stop("svc")
    specs_before = list(adapter.specs)

    record = asyncio.run(plane.resize_store_model("svc", context_length=16384))

    assert record["spec"]["launch"]["context_length"] == 16384
    assert record["spec"]["desired_state"] == "stopped"
    assert record["state"] == "stopped"
    assert adapter.specs == specs_before  # nothing spawned
