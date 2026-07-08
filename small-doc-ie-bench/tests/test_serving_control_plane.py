from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest
from test_serving_supervisor import FakeAdapter

from docie_bench.serving.control_plane import (
    ControlPlane,
    PortAllocator,
    _DefaultSupervisor,
    to_data,
)
from docie_bench.serving.model_store import ModelStore, ModelStoreError
from docie_bench.serving.runtime import LifecycleState, RuntimeKind, RuntimeLaunchSpec
from docie_bench.serving.supervisor import DeploymentRecord, DeploymentSpec, PersistentSupervisor


class State(Enum):
    READY = "ready"


@dataclass
class Model:
    name: str
    state: State
    path: Path


class FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def list_models(self) -> list[Model]:
        return [Model("tiny", State.READY, Path("models/tiny"))]

    async def get_model(self, name: str) -> dict[str, str]:
        self.calls.append(("get", name))
        return {"name": name}

    async def pull_model(
        self,
        name: str,
        *,
        runtime: str | None,
        revision: str | None,
        trust_remote_code: bool,
    ) -> dict[str, object]:
        self.calls.append(("pull", name, runtime, revision, trust_remote_code))
        return {"name": name, "revision": revision}

    def remove_model(self, name: str) -> dict[str, str]:
        self.calls.append(("remove", name))
        return {"removed": name}


class FakeRuntimes:
    def list_runtimes(self) -> list[dict[str, object]]:
        return [{"name": "llamacpp", "available": True}]

    async def probe_runtime(self, name: str) -> dict[str, str]:
        return {"name": name, "version": "1.2.3"}


class FakeSupervisor:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def list_deployments(self) -> list[dict[str, str]]:
        return [{"name": "invoice", "state": "running"}]

    def deployment_status(self, name: str) -> dict[str, str]:
        self.calls.append(("status", name))
        return {"name": name, "state": "running"}

    async def serve(
        self,
        model: str,
        *,
        name: str | None,
        runtime: str | None,
        replicas: int,
    ) -> dict[str, object]:
        self.calls.append(("serve", model, name, runtime, replicas))
        return {"name": name, "replicas": replicas}

    def start(self, name: str) -> dict[str, str]:
        self.calls.append(("start", name))
        return {"name": name, "state": "running"}

    def stop(self, name: str) -> dict[str, str]:
        self.calls.append(("stop", name))
        return {"name": name, "state": "stopped"}


class FakePlanner:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def plan(self, model: str, *, runtime: str | None, replicas: int) -> dict[str, object]:
        self.calls.append((model, runtime, replicas))
        return {"compatible": True, "model": model, "replicas": replicas, "runtime": runtime}


def make_plane() -> tuple[ControlPlane, FakeRegistry, FakeSupervisor, FakePlanner]:
    registry = FakeRegistry()
    supervisor = FakeSupervisor()
    planner = FakePlanner()
    return (
        ControlPlane(registry, FakeRuntimes(), supervisor, planner),
        registry,
        supervisor,
        planner,
    )


def test_control_plane_normalizes_results_and_supports_sync_and_async_contracts() -> None:
    plane, registry, supervisor, planner = make_plane()

    assert asyncio.run(plane.list_models()) == [
        {"name": "tiny", "path": "models/tiny", "state": "ready"}
    ]
    assert asyncio.run(plane.show_model(" tiny ")) == {"name": "tiny"}
    assert asyncio.run(
        plane.pull_model(
            "org/model",
            runtime=" vllm ",
            revision=" sha ",
            trust_remote_code=True,
        )
    ) == {"name": "org/model", "revision": "sha"}
    served = asyncio.run(plane.serve("org/model", name=" api ", runtime=" llamacpp ", replicas=2))
    assert served == {
        "name": "api",
        "replicas": 2,
    }
    assert asyncio.run(plane.plan("org/model", runtime=" ", replicas=3))["compatible"] is True

    assert registry.calls[-1] == ("pull", "org/model", "vllm", "sha", True)
    assert supervisor.calls[-1] == ("serve", "org/model", "api", "llamacpp", 2)
    assert planner.calls[-1] == ("org/model", None, 3)


def test_control_plane_exposes_all_observation_and_lifecycle_operations() -> None:
    plane, registry, supervisor, _ = make_plane()

    assert asyncio.run(plane.remove_model("tiny")) == {"removed": "tiny"}
    assert asyncio.run(plane.list_runtimes())[0]["name"] == "llamacpp"
    assert asyncio.run(plane.probe_runtime("vllm"))["version"] == "1.2.3"
    assert asyncio.run(plane.list_deployments())[0]["state"] == "running"
    assert asyncio.run(plane.deployment_status("invoice"))["name"] == "invoice"
    assert asyncio.run(plane.start("invoice"))["state"] == "running"
    assert asyncio.run(plane.stop("invoice"))["state"] == "stopped"

    assert registry.calls[-1] == ("remove", "tiny")
    assert supervisor.calls[-1] == ("stop", "invoice")


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda plane: plane.show_model("  "), "model must not be empty"),
        (lambda plane: plane.probe_runtime(""), "runtime must not be empty"),
        (lambda plane: plane.serve("tiny", replicas=0), "replicas must be at least 1"),
        (lambda plane: plane.plan("tiny", replicas=-1), "replicas must be at least 1"),
    ],
)
def test_control_plane_rejects_invalid_operations(operation, message: str) -> None:
    plane, _, _, _ = make_plane()

    with pytest.raises(ValueError, match=message):
        asyncio.run(operation(plane))


def _seed_nuextract3_store(root: Path) -> ModelStore:
    model_gguf = root.parent / "model.gguf"
    mmproj_gguf = root.parent / "mmproj.gguf"
    model_gguf.write_bytes(b"GGUF-weights")
    mmproj_gguf.write_bytes(b"GGUF-mmproj")
    store = ModelStore(root)
    store.add_gguf(
        name="nuextract3",
        family="nuextract3",
        model_gguf=model_gguf,
        mmproj=mmproj_gguf,
    )
    return store


def test_up_bridges_store_entry_to_a_correct_llama_server_launch_spec(tmp_path: Path) -> None:
    # FakeAdapter.start() returns a RuntimeProcess without spawning a process, so this
    # validates the constructed launch spec/command -- not real STARTING->READY readiness.
    root = tmp_path / "models"
    store = _seed_nuextract3_store(root)
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.LLAMACPP: FakeAdapter()},
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None, model_store_root=root)
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    asyncio.run(plane.up("nuextract3", port=8088))

    launch = supervisor.get("nuextract3").spec.launch
    assert str(launch.runtime) == "llamacpp"
    assert launch.model.endswith("model.gguf")
    assert launch.alias == "nuextract3"
    assert launch.port == 8088
    assert launch.context_length == 8192
    mmproj_posix = store.entry("nuextract3").mmproj_path.as_posix()
    assert launch.extra_args == ("--jinja", "--mmproj", mmproj_posix)
    # The bridge derives its flags from the same source as llama_server_command, so the
    # family flags appear verbatim as a contiguous suffix of the full command.
    assert store.family_launch_args("nuextract3") == ("--jinja", "--mmproj", mmproj_posix)
    command = store.llama_server_command("nuextract3", port=8088)
    assert command[-3:] == ("--jinja", "--mmproj", mmproj_posix)


def test_up_missing_entry_raises_with_store_root_and_seeding_pointer(tmp_path: Path) -> None:
    root = tmp_path / "models"
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.LLAMACPP: FakeAdapter()},
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None, model_store_root=root)
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    with pytest.raises(ModelStoreError) as excinfo:
        asyncio.run(plane.up("ghost"))

    message = str(excinfo.value)
    assert str(root.resolve()) in message
    assert "Seed it first" in message


# ── PR-B: per-deploy port allocation ────────────────────────────────────────────


def _always_free(_host: str, _port: int) -> bool:
    return True


def _loopback_hosts() -> dict[str, object]:
    # Advertise 127.0.0.1 so the deterministic-addressing guard is a no-op and no
    # resolver/DNS is touched (mirrors the local-CLI path).
    return {"advertise_host": "127.0.0.1", "bind_host": "127.0.0.1"}


def _seed_openai_store(root: Path, name: str) -> ModelStore:
    model_gguf = root.parent / f"{name}.gguf"
    model_gguf.write_bytes(b"GGUF-weights")
    store = ModelStore(root)
    store.add_gguf(name=name, family="openai_chat", model_gguf=model_gguf)
    return store


def test_allocator_recommend_is_deterministic_and_skips_reserved() -> None:
    allocator = PortAllocator(range_start=8088, range_end=8188)
    # Pure function of (range, reserved): 8088/8089 held by records -> 8090.
    first = allocator.recommend(bind_host="127.0.0.1", reserved={8088, 8089})
    second = allocator.recommend(bind_host="127.0.0.1", reserved={8088, 8089})
    assert first == second == 8090


def test_allocator_reserved_honored_across_states_including_stopped(tmp_path: Path) -> None:
    # A STOPPED record still reserves its port (it can be start()-ed later); only
    # remove() frees it. Proven through _DefaultSupervisor._reserved_ports.
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.LLAMACPP: FakeAdapter()}
    )
    supervisor.deploy(
        DeploymentSpec(
            name="held",
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.LLAMACPP, model="/m.gguf", alias="held", port=8090
            ),
        )
    )
    supervisor.stop("held")  # STOPPED, but must keep 8090 reserved
    wrapper = _DefaultSupervisor(supervisor, planner=None)
    assert 8090 in wrapper._reserved_ports()
    assert 8090 not in wrapper._reserved_ports(exclude_name="held")


def test_allocator_skips_bound_socket() -> None:
    # Even with no record holding a port, a probe that reports it bound is skipped.
    def probe(_host: str, port: int) -> bool:
        return port != 8088  # 8088 "bound" by something the record-scan can't see

    allocator = PortAllocator(range_start=8088, range_end=8188, probe=probe)
    assert allocator.allocate(bind_host="127.0.0.1", reserved=set()) == 8089


def test_range_exhaustion_raises_clear_error() -> None:
    allocator = PortAllocator(range_start=8088, range_end=8089)
    with pytest.raises(RuntimeError, match="no free port in range 8088-8089"):
        allocator.recommend(bind_host="127.0.0.1", reserved={8088, 8089})


def test_absent_port_no_longer_pins_8088(tmp_path: Path) -> None:
    # Guards against the dead-code regression: up(port=None) with 8088 already held
    # by another record must NOT land on 8088.
    root = tmp_path / "models"
    _seed_openai_store(root, "a")
    _seed_openai_store(root, "b")
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.LLAMACPP: FakeAdapter()}
    )
    wrapper = _DefaultSupervisor(
        supervisor,
        planner=None,
        model_store_root=root,
        port_range=(8088, 8188),
        port_probe=_always_free,
        **_loopback_hosts(),
    )
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    asyncio.run(plane.up("a", port=8088))  # explicit -> pins 8088
    asyncio.run(plane.up("b", port=None))  # absent -> must allocate elsewhere

    assert supervisor.get("a").spec.launch.port == 8088
    assert supervisor.get("b").spec.launch.port != 8088
    assert supervisor.get("b").spec.launch.port == 8089


def test_explicit_override_is_honored_without_probing(tmp_path: Path) -> None:
    root = tmp_path / "models"
    _seed_openai_store(root, "a")
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.LLAMACPP: FakeAdapter()}
    )

    def _boom(_host: str, _port: int) -> bool:
        raise AssertionError("prober must not be consulted for an explicit override")

    wrapper = _DefaultSupervisor(
        supervisor,
        planner=None,
        model_store_root=root,
        port_range=(8088, 8188),
        port_probe=_boom,
        **_loopback_hosts(),
    )
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    asyncio.run(plane.up("a", port=9001))  # outside the range on purpose
    assert supervisor.get("a").spec.launch.port == 9001


class _ReallocBackend:
    """Backend whose await_ready FAILS (started-then-exited) on ``bad_ports``.

    Lets the reallocation loop be driven deterministically with no real sockets,
    sleeps, or restart churn: the first (bad) port returns FAILED+exited, the next
    returns READY. Records the deploy order so the test can prove the loop advanced.
    """

    def __init__(self, bad_ports: set[int]) -> None:
        self.bad_ports = bad_ports
        self.records: dict[str, DeploymentRecord] = {}
        self.deployed_ports: list[int] = []

    def list(self) -> tuple[DeploymentRecord, ...]:
        return tuple(self.records.values())

    def deploy(self, spec: DeploymentSpec) -> DeploymentRecord:
        record = DeploymentRecord(spec=spec, state=LifecycleState.STOPPED)
        self.records[spec.name] = record
        self.deployed_ports.append(spec.launch.port)
        return record

    def await_ready(self, name: str) -> DeploymentRecord:
        record = self.records[name]
        if record.spec.launch.port in self.bad_ports:
            record.state = LifecycleState.FAILED
            record.exited_after_start = True
            record.last_error = "bind: address already in use"
            record.pid = None
        else:
            record.state = LifecycleState.READY
            record.endpoint = record.spec.launch.endpoint
            record.pid = 4242
        return record


def test_reallocates_on_immediate_exit_and_rebuilds_endpoint(tmp_path: Path) -> None:
    root = tmp_path / "models"
    _seed_openai_store(root, "a")
    backend = _ReallocBackend(bad_ports={8088})
    wrapper = _DefaultSupervisor(
        backend,
        planner=None,
        model_store_root=root,
        port_range=(8088, 8188),
        port_probe=_always_free,
        advertise_host="serving",  # non-loopback so the endpoint embeds the port
        bind_host="0.0.0.0",  # noqa: S104 - the in-container all-interfaces bind
        resolve_host=lambda _host: ("10.0.0.2",),
    )
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    record = to_data(asyncio.run(plane.up("a", port=None)))
    assert isinstance(record, dict)

    # First port collided (8088), loop advanced to 8089 and it came up READY.
    assert backend.deployed_ports == [8088, 8089]
    assert record["state"] == "ready"
    # The endpoint was rebuilt THROUGH reachable_launch on the second attempt, so it
    # embeds the reallocated port — not the stale first one.
    assert record["spec"]["launch"]["port"] == 8089
    assert record["endpoint"] == "http://serving:8089/v1"


class _CaptureBackend:
    def __init__(self) -> None:
        self.specs: list[DeploymentSpec] = []

    def list(self) -> tuple[DeploymentRecord, ...]:
        return ()

    def deploy(self, spec: DeploymentSpec) -> DeploymentRecord:
        self.specs.append(spec)
        return DeploymentRecord(spec=spec, state=LifecycleState.READY)


def test_serve_allocates_and_skips_remote(tmp_path: Path) -> None:
    del tmp_path
    backend = _CaptureBackend()
    probe_calls: list[tuple[str, int]] = []

    def probe(host: str, port: int) -> bool:
        probe_calls.append((host, port))
        return True

    wrapper = _DefaultSupervisor(
        backend,
        planner=None,
        port_range=(8088, 8188),
        port_probe=probe,
        **_loopback_hosts(),
    )
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    asyncio.run(plane.serve("org/model", name="vllm-dep", runtime="vllm", replicas=1))
    assert backend.specs[-1].launch.port == 8088  # allocated from the window
    assert probe_calls == [("127.0.0.1", 8088)]

    probe_calls.clear()
    asyncio.run(
        plane.serve("m", name="remote-dep", runtime="remote", replicas=1)
    )
    # REMOTE binds nothing locally -> no allocation, no probe, default port kept.
    assert backend.specs[-1].launch.port == 8000
    assert probe_calls == []


def test_to_data_sorts_mapping_keys_and_hides_private_attributes() -> None:
    class Result:
        def __init__(self) -> None:
            self.z = 1
            self._private = "hidden"
            self.a = State.READY

    assert to_data(Result()) == {"a": "ready", "z": 1}
    assert to_data(frozenset({"vision", "batching"})) == ["batching", "vision"]
