from __future__ import annotations

import json
from typing import Any

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

    async def up(self, name: str, **kwargs) -> dict[str, object]:
        self.calls.append(("up", name, kwargs))
        return {"name": name, "state": "starting", "endpoint": "http://127.0.0.1:8088/v1"}

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
        (
            ["up", "nuextract3", "--port", "9000"],
            (
                "up",
                "nuextract3",
                {"context_length": 8192, "port": 9000, "deployment_name": None},
            ),
        ),
        (
            # A fleet names each node's record distinctly: the store name is
            # not unique across nodes (#530).
            ["up", "nuextract3", "--deployment-name", "nuextract3-ovh-02"],
            (
                "up",
                "nuextract3",
                {
                    "context_length": 8192,
                    "port": None,
                    "deployment_name": "nuextract3-ovh-02",
                },
            ),
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


def test_up_human_output_ends_with_the_benchmark_command() -> None:
    result = runner.invoke(create_app(FakePlane()), ["up", "nuextract3"])

    assert result.exit_code == 0, result.output
    non_empty = [line for line in result.output.splitlines() if line.strip()]
    assert non_empty[-1] == "docie-bench benchmark run --model-profile nuextract3"


def test_up_json_output_omits_the_human_hint() -> None:
    result = runner.invoke(create_app(FakePlane()), ["--json", "up", "nuextract3"])

    assert result.exit_code == 0, result.output
    assert "docie-bench benchmark run" not in result.output
    assert json.loads(result.output)["state"] == "starting"


def test_json_errors_are_machine_readable_and_exit_nonzero() -> None:
    class BrokenPlane(FakePlane):
        async def show_model(self, name: str) -> dict[str, object]:
            raise RuntimeError(f"unknown model: {name}")

    result = runner.invoke(create_app(BrokenPlane()), ["--json", "model", "show", "missing"])

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "error": {"message": "unknown model: missing", "type": "RuntimeError"}
    }


# --- fleet: the drain/resume path through the CENTRAL api (#634) -----------
#
# Exit codes are the contract here, not the output: a deployment hook gates on
# them, so every one of them is asserted rather than eyeballed.


class _Response:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=None, response=None  # type: ignore[arg-type]
            )


@pytest.fixture
def fleet_api(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A central api that answers from a script the test controls."""
    import httpx

    monkeypatch.setenv("DOCIE_API_URL", "https://studio.example.test")
    monkeypatch.setenv("DOCIE_API_KEY", "test-key")
    state: dict[str, Any] = {
        "posts": [],
        "post_response": _Response(200, {"node_id": "node-a", "models_offline": []}),
        "node_states": ["drained"],
        "gets": 0,
    }

    def fake_post(url: str, **kwargs: Any) -> _Response:
        state["posts"].append((url, kwargs.get("json"), kwargs.get("headers")))
        return state["post_response"]

    def fake_get(url: str, **kwargs: Any) -> _Response:
        state["gets"] += 1
        states = state["node_states"]
        drain_state = states[min(state["gets"] - 1, len(states) - 1)]
        return _Response(
            200,
            {
                "nodes": [
                    {
                        "node_id": "node-a",
                        "drain_state": drain_state,
                        "reporting": True,
                        "deployments": 1,
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "get", fake_get)
    return state


def test_fleet_drain_sends_the_acknowledgement_flag_and_the_key(
    fleet_api: dict[str, Any],
) -> None:
    result = runner.invoke(create_app(), ["fleet", "drain", "node-a"])

    assert result.exit_code == 0
    url, body, headers = fleet_api["posts"][0]
    assert url.endswith("/v1/serving/nodes/node-a/drain")
    assert body == {"acknowledge_outage": False}
    assert headers["Authorization"] == "Bearer test-key"


def test_fleet_drain_never_adds_the_acknowledgement_by_itself(
    fleet_api: dict[str, Any],
) -> None:
    """#634 is explicit: a refusal must not be worked around automatically."""
    fleet_api["post_response"] = _Response(
        409, {"detail": "would take 1 model(s) offline: lfm2.5-350m"}
    )

    result = runner.invoke(create_app(), ["fleet", "drain", "node-a", "--wait"])

    assert result.exit_code == 1
    # The api's own sentence, naming the model, not a generic failure.
    assert "lfm2.5-350m" in result.output
    assert len(fleet_api["posts"]) == 1
    assert fleet_api["posts"][0][1] == {"acknowledge_outage": False}


def test_fleet_drain_wait_exits_zero_once_the_runtimes_stop(
    fleet_api: dict[str, Any],
) -> None:
    fleet_api["node_states"] = ["draining", "drained"]

    result = runner.invoke(
        create_app(),
        ["--json", "fleet", "drain", "node-a", "--wait", "--timeout", "30"],
    )

    assert result.exit_code == 0
    assert '"safe_to_replace":true' in result.output.replace(" ", "")


def test_fleet_drain_wait_exits_nonzero_on_a_timeout(
    fleet_api: dict[str, Any],
) -> None:
    """The case a hook must not mistake for success: still holding runtimes
    when the deadline passes."""
    fleet_api["node_states"] = ["draining"]

    result = runner.invoke(
        create_app(),
        ["--json", "fleet", "drain", "node-a", "--wait", "--timeout", "0.1"],
    )

    assert result.exit_code == 1
    assert '"safe_to_replace":false' in result.output.replace(" ", "")


def test_fleet_status_names_the_readiness(fleet_api: dict[str, Any]) -> None:
    fleet_api["node_states"] = ["draining"]

    result = runner.invoke(create_app(), ["--json", "fleet", "status", "node-a"])

    assert result.exit_code == 0
    assert "routing_excluded" in result.output


def test_fleet_status_on_an_unknown_node_exits_nonzero(
    fleet_api: dict[str, Any],
) -> None:
    result = runner.invoke(create_app(), ["fleet", "status", "not-a-node"])

    assert result.exit_code == 1


def test_fleet_commands_refuse_without_a_central_api_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every other command in this CLI is node-local, so the missing variable
    has to say which commands need it."""
    monkeypatch.delenv("DOCIE_API_URL", raising=False)
    monkeypatch.setenv("DOCIE_API_KEY", "test-key")

    result = runner.invoke(create_app(), ["fleet", "status"])

    assert result.exit_code != 0
    assert "DOCIE_API_URL" in result.output


def test_fleet_commands_refuse_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCIE_API_URL", "https://studio.example.test")
    monkeypatch.delenv("DOCIE_API_KEY", raising=False)

    result = runner.invoke(create_app(), ["fleet", "status"])

    assert result.exit_code != 0
    assert "DOCIE_API_KEY" in result.output


def test_fleet_resume_posts_and_says_nothing_about_waiting(
    fleet_api: dict[str, Any],
) -> None:
    result = runner.invoke(create_app(), ["fleet", "resume", "node-a"])

    assert result.exit_code == 0
    assert fleet_api["posts"][0][0].endswith("/v1/serving/nodes/node-a/resume")
