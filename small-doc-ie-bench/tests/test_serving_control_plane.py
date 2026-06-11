from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from docie_bench.serving.control_plane import ControlPlane, to_data


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


def test_to_data_sorts_mapping_keys_and_hides_private_attributes() -> None:
    class Result:
        def __init__(self) -> None:
            self.z = 1
            self._private = "hidden"
            self.a = State.READY

    assert to_data(Result()) == {"a": "ready", "z": 1}
    assert to_data(frozenset({"vision", "batching"})) == ["batching", "vision"]
