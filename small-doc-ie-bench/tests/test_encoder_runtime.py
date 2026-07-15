"""EncoderRuntime: the encoder shim as a managed control-plane deployment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from docie_bench.serving.runtime import (
    EncoderRuntime,
    HealthResult,
    LifecycleState,
    RuntimeConfigurationError,
    RuntimeKind,
    RuntimeLaunchSpec,
    default_runtime_adapters,
)
from docie_bench.serving.supervisor import DeploymentSpec, PersistentSupervisor

_REAL_FIND_SPEC = importlib.util.find_spec


def _spec(**overrides: object) -> RuntimeLaunchSpec:
    base: dict[str, object] = {
        "runtime": RuntimeKind.ENCODER,
        "model": "urchade/gliner_multi_pii-v1",
        "alias": "gliner-pii",
        "port": 8090,
    }
    base.update(overrides)
    return RuntimeLaunchSpec(**base)  # type: ignore[arg-type]


def _with_gliner(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    def fake_find_spec(name: str, *args: object, **kwargs: object):
        if name == "gliner":
            return object() if present else None
        return _REAL_FIND_SPEC(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


class FakeProcess:
    pid = 4242

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        pass


def test_default_adapters_include_encoder() -> None:
    adapters = default_runtime_adapters()
    assert isinstance(adapters[RuntimeKind.ENCODER], EncoderRuntime)


def test_build_command_uses_console_script_when_found() -> None:
    adapter = EncoderRuntime(which=lambda name: "/usr/bin/docie" if name == "docie" else None)
    command = adapter.build_command(_spec())
    assert command == (
        "/usr/bin/docie",
        "encoder",
        "--model",
        "urchade/gliner_multi_pii-v1",
        "--host",
        "127.0.0.1",
        "--port",
        "8090",
    )


def test_build_command_falls_back_to_cli_module() -> None:
    adapter = EncoderRuntime(which=lambda name: None)
    command = adapter.build_command(_spec(extra_args=("--threshold", "0.7")))
    assert command[:3] == (sys.executable, "-m", "docie_bench.serving.cli")
    assert command[3] == "encoder"
    assert command[-2:] == ("--threshold", "0.7")


def test_probe_incompatible_without_encoders_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_gliner(monkeypatch, present=False)
    adapter = EncoderRuntime(which=lambda name: "/usr/bin/docie" if name == "docie" else None)
    capabilities = adapter.probe(_spec())
    assert capabilities.installed is True  # CLI is launchable...
    assert capabilities.compatible is False  # ...but the analyzer library is missing
    assert any("encoders" in reason for reason in capabilities.reasons)


def test_probe_compatible_with_encoders_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_gliner(monkeypatch, present=True)
    adapter = EncoderRuntime(which=lambda name: "/usr/bin/docie" if name == "docie" else None)
    assert adapter.probe(_spec()).compatible is True


def test_health_probes_healthz_off_the_v1_endpoint() -> None:
    seen: list[str] = []

    def fake_get(url: str, timeout: float, headers: object) -> HealthResult:
        seen.append(url)
        return HealthResult(healthy=True, status_code=200)

    adapter = EncoderRuntime(health_get=fake_get, which=lambda name: "/usr/bin/docie")
    assert adapter.health(_spec()).healthy is True
    assert seen == ["http://127.0.0.1:8090/healthz"]


def test_start_fails_fast_without_encoders_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_gliner(monkeypatch, present=False)
    adapter = EncoderRuntime(which=lambda name: "/usr/bin/docie")
    with pytest.raises(RuntimeConfigurationError, match="encoders"):
        adapter.start(_spec())


def test_supervisor_round_trip_persists_encoder_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deploy → READY → reload from deployments.json with the encoder kind intact."""
    _with_gliner(monkeypatch, present=True)
    adapter = EncoderRuntime(
        popen_factory=lambda *args, **kwargs: FakeProcess(),
        health_get=lambda url, timeout, headers: HealthResult(healthy=True, status_code=200),
        which=lambda name: "/usr/bin/docie" if name == "docie" else None,
    )
    state_path = tmp_path / "deployments.json"
    supervisor = PersistentSupervisor(
        state_path, adapters={RuntimeKind.ENCODER: adapter}
    )
    supervisor.deploy(DeploymentSpec(name="gliner-pii", launch=_spec()))
    record = supervisor.await_ready("gliner-pii", timeout_s=5, sleep=lambda _s: None)
    assert record.state == LifecycleState.READY
    assert record.endpoint == "http://127.0.0.1:8090/v1"
    assert record.pid == 4242

    reloaded = PersistentSupervisor(state_path, adapters={RuntimeKind.ENCODER: adapter})
    persisted = reloaded.get("gliner-pii")
    assert persisted.spec.launch.runtime == RuntimeKind.ENCODER
    assert persisted.spec.launch.model == "urchade/gliner_multi_pii-v1"
