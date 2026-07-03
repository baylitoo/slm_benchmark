from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest
from test_serving_supervisor import FakeAdapter

from docie_bench.serving.control_plane import ControlPlane, _DefaultSupervisor, to_data
from docie_bench.serving.model_store import ModelStore, ModelStoreError
from docie_bench.serving.runtime import RuntimeKind
from docie_bench.serving.supervisor import PersistentSupervisor


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


def _up_plane(tmp_path: Path) -> tuple[ControlPlane, PersistentSupervisor]:
    root = tmp_path / "models"
    _seed_nuextract3_store(root)
    supervisor = PersistentSupervisor(
        tmp_path / "state.json",
        adapters={RuntimeKind.LLAMACPP: FakeAdapter()},
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None, model_store_root=root)
    return ControlPlane(None, None, wrapper, None), supervisor  # type: ignore[arg-type]


def test_serve_store_model_binds_loopback_by_default(tmp_path: Path, monkeypatch) -> None:
    # The default (host-native CLI, no DOCIE_ADVERTISE_HOST) advertises loopback,
    # so the auth-less llama-server must also BIND loopback — a laptop `docie up`
    # never exposes the model on LAN/Wi-Fi interfaces.
    monkeypatch.delenv("DOCIE_ADVERTISE_HOST", raising=False)
    plane, supervisor = _up_plane(tmp_path)

    asyncio.run(plane.up("nuextract3", port=8088))

    assert supervisor.get("nuextract3").spec.launch.host == "127.0.0.1"


@pytest.mark.parametrize("advertise", ["localhost", "::1"])
def test_serve_store_model_binds_loopback_for_loopback_aliases(
    tmp_path: Path, monkeypatch, advertise: str
) -> None:
    monkeypatch.setenv("DOCIE_ADVERTISE_HOST", advertise)
    plane, supervisor = _up_plane(tmp_path)

    asyncio.run(plane.up("nuextract3", port=8088))

    assert supervisor.get("nuextract3").spec.launch.host == "127.0.0.1"


def test_serve_store_model_binds_all_interfaces_when_advertised_beyond_loopback(
    tmp_path: Path, monkeypatch
) -> None:
    # Only a non-loopback advertise (compose sets the worker service name so
    # api/bench containers dial in) justifies the 0.0.0.0 bind of the auth-less
    # llama-server (trusted private networks only — see the SECURITY note in
    # control_plane.serve_store_model).
    monkeypatch.setenv("DOCIE_ADVERTISE_HOST", "worker")
    plane, supervisor = _up_plane(tmp_path)

    asyncio.run(plane.up("nuextract3", port=8088))

    assert supervisor.get("nuextract3").spec.launch.host == "0.0.0.0"


def test_advertise_host_default_127(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DOCIE_ADVERTISE_HOST", raising=False)
    plane, supervisor = _up_plane(tmp_path)

    record = asyncio.run(plane.up("nuextract3", port=8088))

    # Advertised URL (record endpoint + launch spec endpoint) is loopback by default.
    assert supervisor.get("nuextract3").spec.launch.endpoint == "http://127.0.0.1:8088/v1"
    assert record["endpoint"] == "http://127.0.0.1:8088/v1"


def test_advertise_host_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOCIE_ADVERTISE_HOST", "worker")
    plane, supervisor = _up_plane(tmp_path)

    record = asyncio.run(plane.up("nuextract3", port=8090))

    assert supervisor.get("nuextract3").spec.launch.endpoint == "http://worker:8090/v1"
    assert record["endpoint"] == "http://worker:8090/v1"


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


def test_to_data_sorts_mapping_keys_and_hides_private_attributes() -> None:
    class Result:
        def __init__(self) -> None:
            self.z = 1
            self._private = "hidden"
            self.a = State.READY

    assert to_data(Result()) == {"a": "ready", "z": 1}
    assert to_data(frozenset({"vision", "batching"})) == ["batching", "vision"]
