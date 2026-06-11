from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from docie_bench.serving.cli import create_app


class FakePlane:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def list_models(self) -> list[dict[str, object]]:
        return [
            {"state": "ready", "name": "zeta", "size_bytes": 20},
            {"state": "ready", "name": "alpha", "size_bytes": 10},
        ]

    async def show_model(self, name: str) -> dict[str, object]:
        self.calls.append(("show_model", name))
        return {"name": name, "revision": "abc123"}

    async def pull_model(self, name: str, **kwargs) -> dict[str, object]:
        self.calls.append(("pull_model", name, kwargs))
        return {"name": name, "state": "ready"}

    async def remove_model(self, name: str) -> dict[str, object]:
        self.calls.append(("remove_model", name))
        return {"removed": name}

    async def list_runtimes(self) -> list[dict[str, object]]:
        return [{"available": True, "name": "llamacpp"}]

    async def probe_runtime(self, name: str) -> dict[str, object]:
        self.calls.append(("probe_runtime", name))
        return {"available": True, "name": name}

    async def list_deployments(self) -> list[dict[str, object]]:
        return [{"name": "invoice", "state": "running"}]

    async def deployment_status(self, name: str) -> dict[str, object]:
        self.calls.append(("deployment_status", name))
        return {"name": name, "state": "running"}

    async def serve(self, model: str, **kwargs) -> dict[str, object]:
        self.calls.append(("serve", model, kwargs))
        return {"name": kwargs["name"], "state": "running"}

    async def start(self, name: str) -> dict[str, object]:
        self.calls.append(("start", name))
        return {"name": name, "state": "running"}

    async def stop(self, name: str) -> dict[str, object]:
        self.calls.append(("stop", name))
        return {"name": name, "state": "stopped"}

    async def plan(self, model: str, **kwargs) -> dict[str, object]:
        self.calls.append(("plan", model, kwargs))
        return {"compatible": True, "model": model, **kwargs}


runner = CliRunner()


def test_json_output_is_compact_deterministic_and_preserves_backend_order() -> None:
    result = runner.invoke(create_app(FakePlane()), ["--json", "model", "list"])

    assert result.exit_code == 0, result.output
    assert result.output == (
        '[{"name":"zeta","size_bytes":20,"state":"ready"},'
        '{"name":"alpha","size_bytes":10,"state":"ready"}]\n'
    )


def test_human_list_output_is_a_readable_table() -> None:
    result = runner.invoke(create_app(FakePlane()), ["model", "list"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split() == ["NAME", "SIZE_BYTES", "STATE"]
    assert "zeta" in lines[1]
    assert "alpha" in lines[2]


@pytest.mark.parametrize(
    ("arguments", "expected_call"),
    [
        (["model", "show", "tiny"], ("show_model", "tiny")),
        (
            [
                "model",
                "pull",
                "org/tiny",
                "--runtime",
                "vllm",
                "--revision",
                "abc",
                "--trust-remote-code",
            ],
            (
                "pull_model",
                "org/tiny",
                {"revision": "abc", "runtime": "vllm", "trust_remote_code": True},
            ),
        ),
        (["model", "remove", "tiny"], ("remove_model", "tiny")),
        (["runtime", "probe", "vllm"], ("probe_runtime", "vllm")),
        (["status", "invoice"], ("deployment_status", "invoice")),
        (
            ["serve", "org/tiny", "--name", "invoice", "--runtime", "llamacpp", "--replicas", "2"],
            ("serve", "org/tiny", {"name": "invoice", "replicas": 2, "runtime": "llamacpp"}),
        ),
        (["start", "invoice"], ("start", "invoice")),
        (["stop", "invoice"], ("stop", "invoice")),
        (
            ["plan", "org/tiny", "--runtime", "vllm", "--replicas", "3"],
            ("plan", "org/tiny", {"replicas": 3, "runtime": "vllm"}),
        ),
    ],
)
def test_commands_delegate_to_the_control_plane(arguments: list[str], expected_call: tuple) -> None:
    plane = FakePlane()

    result = runner.invoke(create_app(plane), arguments)

    assert result.exit_code == 0, result.output
    assert plane.calls == [expected_call]


@pytest.mark.parametrize("arguments", [["runtime", "list"], ["list"]])
def test_list_commands_are_available(arguments: list[str]) -> None:
    result = runner.invoke(create_app(FakePlane()), arguments)

    assert result.exit_code == 0, result.output
    assert "NAME" in result.output


def test_backend_is_constructed_lazily() -> None:
    calls = []

    def factory() -> FakePlane:
        calls.append("constructed")
        return FakePlane()

    app = create_app(plane_factory=factory)

    help_result = runner.invoke(app, ["--help"])
    list_result = runner.invoke(app, ["runtime", "list"])

    assert help_result.exit_code == 0
    assert list_result.exit_code == 0
    assert calls == ["constructed"]


def test_json_errors_are_machine_readable_and_exit_nonzero() -> None:
    class BrokenPlane(FakePlane):
        async def show_model(self, name: str) -> dict[str, object]:
            raise RuntimeError(f"unknown model: {name}")

    result = runner.invoke(create_app(BrokenPlane()), ["--json", "model", "show", "missing"])

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "error": {"message": "unknown model: missing", "type": "RuntimeError"}
    }
