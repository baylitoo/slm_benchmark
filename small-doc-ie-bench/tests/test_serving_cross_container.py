"""PR-1 cross-container serving reachability.

Proves the bind/advertise split: a deployed runtime BINDS an all-interfaces host
inside its container while the persisted ``DeploymentRecord`` ADVERTISES a name
every replica can resolve — so ``profile_resolver`` yields an endpoint the api
container (which never ran the deploy) can actually reach, instead of a
worker-local ``127.0.0.1`` loopback.

Pure unit tests: no network, no runtime process, no DB. The store deploy is
driven through a stub adapter whose ``start()`` HONORS ``spec.endpoint`` (unlike
``test_serving_supervisor.FakeAdapter``, which ignores it), so the recorded
endpoint is the real advertise URL and not the bind host.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from docie_bench.serving.control_plane import (
    ControlPlane,
    _DefaultSupervisor,
    reachable_launch,
)
from docie_bench.serving.profile_resolver import resolve_extraction_profile
from docie_bench.serving.runtime import (
    HealthResult,
    LlamaCppRuntime,
    RuntimeKind,
    RuntimeLaunchSpec,
    RuntimeProcess,
)
from docie_bench.serving.supervisor import (
    DeploymentRecord,
    DeploymentSpec,
    LifecycleState,
    PersistentSupervisor,
)

_BIND = "0.0.0.0"  # noqa: S104 - the in-container all-interfaces bind under test
_LOOPBACK = ("127.0.0.1", _BIND, "localhost")


class EndpointHonoringAdapter:
    """A stub runtime that records ``spec.endpoint`` (the advertise URL) verbatim.

    Mirrors the real adapters: ``endpoint()`` returns ``spec.endpoint`` when set,
    and ``start()`` stamps that onto the ``RuntimeProcess`` — which the supervisor
    persists as ``record.endpoint``. It never spawns a process or opens a socket.
    """

    def __init__(self) -> None:
        self.pid = 4242
        self.commands: list[tuple[str, ...]] = []

    def endpoint(self, spec: RuntimeLaunchSpec) -> str:
        return spec.endpoint.rstrip("/") if spec.endpoint else f"http://{spec.host}:{spec.port}/v1"

    def start(
        self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None
    ) -> RuntimeProcess:
        del log_path
        # Record what a real llama-server would bind, to prove --host is the bind host.
        self.commands.append(("--host", spec.host, "--port", str(spec.port)))
        return RuntimeProcess(spec.runtime, self.endpoint(spec), self.pid)

    def is_running(self, pid: int | None) -> bool:
        return pid == self.pid

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del pid, timeout

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        del spec, timeout
        return HealthResult(True, 200)


def _seed_store(root: Path) -> None:
    from docie_bench.serving.model_store import ModelStore

    model_gguf = root.parent / "model.gguf"
    mmproj_gguf = root.parent / "mmproj.gguf"
    model_gguf.write_bytes(b"GGUF-weights")
    mmproj_gguf.write_bytes(b"GGUF-mmproj")
    ModelStore(root).add_gguf(
        name="inv", family="nuextract3", model_gguf=model_gguf, mmproj=mmproj_gguf
    )


# ── the pure bind/advertise primitive ──────────────────────────────────────────


def test_reachable_launch_splits_bind_from_advertise() -> None:
    launch = reachable_launch(
        RuntimeLaunchSpec(
            runtime=RuntimeKind.LLAMACPP, model="/m/model.gguf", alias="inv", port=8088
        ),
        bind_host=_BIND,
        advertise_host="serving",
    )
    # BIND: the process binds all interfaces; ADVERTISE: the URL names the service.
    assert launch.host == _BIND
    assert launch.endpoint == "http://serving:8088/v1"

    adapter = LlamaCppRuntime(which=lambda _name: "llama-server")
    # endpoint() honors the override verbatim...
    assert adapter.endpoint(launch) == "http://serving:8088/v1"
    # ...while build_command still binds spec.host (0.0.0.0), never the advertise host.
    command = adapter.build_command(launch)
    assert command[command.index("--host") + 1] == _BIND
    assert "serving" not in command


def test_reachable_launch_uses_the_specs_own_port() -> None:
    # serve() leaves the RuntimeLaunchSpec default port (8000); the advertise URL
    # must derive from launch.port, not a hardcoded 8088.
    launch = reachable_launch(
        RuntimeLaunchSpec(runtime=RuntimeKind.VLLM, model="org/m", alias="a"),
        bind_host=_BIND,
        advertise_host="serving",
    )
    assert launch.port == 8000
    assert launch.endpoint == "http://serving:8000/v1"


def test_reachable_launch_leaves_remote_endpoint_untouched() -> None:
    # REMOTE's endpoint is the real upstream, not a local bind target.
    remote = RuntimeLaunchSpec(
        runtime=RuntimeKind.REMOTE, model="m", alias="r", endpoint="https://up.example/v1"
    )
    out = reachable_launch(remote, bind_host=_BIND, advertise_host="serving")
    assert out.endpoint == "https://up.example/v1"
    assert out is remote


# ── serve_store_model wires the split from settings/injected hosts ──────────────


def test_serve_store_model_binds_all_and_advertises_service_name(tmp_path: Path) -> None:
    root = tmp_path / "models"
    _seed_store(root)
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.LLAMACPP: EndpointHonoringAdapter()}
    )
    wrapper = _DefaultSupervisor(
        supervisor, planner=None, model_store_root=root, advertise_host="serving", bind_host=_BIND
    )
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    asyncio.run(plane.up("inv", port=8088))

    launch = supervisor.get("inv").spec.launch
    assert launch.host == _BIND  # bind: all interfaces inside the container
    assert launch.endpoint == "http://serving:8088/v1"  # advertise: the service name


def test_serve_generic_path_also_splits_bind_and_advertise(tmp_path: Path) -> None:
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.VLLM: EndpointHonoringAdapter()}
    )
    wrapper = _DefaultSupervisor(
        supervisor, planner=None, advertise_host="serving", bind_host=_BIND
    )
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]

    asyncio.run(plane.serve("org/model", name="dep", runtime="vllm", replicas=1))

    launch = supervisor.get("dep").spec.launch
    assert launch.host == _BIND
    assert launch.endpoint == "http://serving:8000/v1"


# ── the persisted record: cross-replica read yields the advertise endpoint ──────


def test_two_supervisors_over_one_state_read_the_advertise_endpoint(tmp_path: Path) -> None:
    # Simulates worker (deployer) + api (reader) sharing serving-state/deployments.json.
    root = tmp_path / "models"
    _seed_store(root)
    state = tmp_path / "deployments.json"

    deployer = PersistentSupervisor(
        state, adapters={RuntimeKind.LLAMACPP: EndpointHonoringAdapter()}
    )
    wrapper = _DefaultSupervisor(
        deployer, planner=None, model_store_root=root, advertise_host="serving", bind_host=_BIND
    )
    asyncio.run(ControlPlane(None, None, wrapper, None).up("inv", port=8088))  # type: ignore[arg-type]

    # A SECOND supervisor (the api container) loads the same file fresh.
    reader = PersistentSupervisor(
        state, adapters={RuntimeKind.LLAMACPP: EndpointHonoringAdapter()}
    )
    record = reader.get("inv")
    assert record.endpoint == "http://serving:8088/v1"
    # The contract: no persisted endpoint is a loopback/bind host.
    host = record.endpoint.split("//", 1)[1].split(":", 1)[0]
    assert host not in _LOOPBACK


# ── the fixed point: profile_resolver yields the advertise base_url ─────────────


def test_resolver_yields_cross_container_base_url(tmp_path: Path) -> None:
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        "profiles:\n  studio_default:\n    model: m\n    base_url: http://x/v1\n    api_key: k\n",
        encoding="utf-8",
    )
    record = DeploymentRecord(
        spec=DeploymentSpec(
            name="inv",
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.LLAMACPP,
                model="/m/model.gguf",
                alias="inv",
                host=_BIND,
                port=8088,
                endpoint="http://serving:8088/v1",
            ),
        ),
        state=LifecycleState.READY,
        endpoint="http://serving:8088/v1",
    )
    profile = resolve_extraction_profile(
        deployment="inv", models_config_path=models_yaml, deployments=[record]
    )
    assert profile.base_url == "http://serving:8088/v1"
    assert profile.model == "inv"  # served alias, not the GGUF path
    host = profile.base_url.split("//", 1)[1].split(":", 1)[0]
    assert host not in _LOOPBACK


# ── settings alias wiring (DOCIE_-prefixed env, mirrors DOCIE_SERVING_HOME) ──────


def test_settings_read_docie_prefixed_advertise_and_bind_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docie_bench.settings import Settings, get_settings

    monkeypatch.setenv("DOCIE_SERVING_ADVERTISE_HOST", "serving")
    monkeypatch.setenv("DOCIE_SERVING_BIND_HOST", _BIND)
    get_settings.cache_clear()
    try:
        settings = Settings()
        assert settings.serving_advertise_host == "serving"
        assert settings.serving_bind_host == _BIND
    finally:
        get_settings.cache_clear()


def test_settings_advertise_defaults_to_loopback_for_local_cli() -> None:
    # Local `docie up` runs same-host, so the default must stay 127.0.0.1 (no
    # compose-only service name leaks into the laptop path).
    from docie_bench.settings import Settings

    assert Settings().serving_advertise_host == "127.0.0.1"
