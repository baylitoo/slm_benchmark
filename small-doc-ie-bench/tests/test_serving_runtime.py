from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from docie_bench.serving.runtime import (
    HealthResult,
    LlamaCppRuntime,
    OllamaRuntime,
    RemoteRuntime,
    RuntimeConfigurationError,
    RuntimeKind,
    RuntimeLaunchSpec,
    RuntimeUnavailableError,
    VLLMRuntime,
)


class FakeProcess:
    def __init__(self, pid: int = 41) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        assert self.returncode is not None
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _spec(runtime: RuntimeKind, **overrides: Any) -> RuntimeLaunchSpec:
    values: dict[str, Any] = {
        "runtime": runtime,
        "model": "org/model",
        "alias": "invoice",
    }
    values.update(overrides)
    return RuntimeLaunchSpec(**values)


def test_vllm_builds_argv_without_shell_interpretation() -> None:
    model = "org/model; touch should-not-run"
    adapter = VLLMRuntime(which=lambda name: "/opt/bin/vllm")

    command = adapter.build_command(_spec(RuntimeKind.VLLM, model=model))

    assert command[0:3] == ("/opt/bin/vllm", "serve", model)
    assert model in command


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tensor_parallel_size": 2}, "tensor_parallel_size=1"),
        ({"gpu_memory_utilization": 0.5}, "invalid for vLLM CPU"),
        ({"dtype": "float16"}, "dtype must be"),
        ({"quantization": "awq"}, "benchmark validation"),
    ],
)
def test_vllm_cpu_configuration_is_conservatively_validated(
    overrides: dict[str, Any],
    message: str,
) -> None:
    adapter = VLLMRuntime(which=lambda name: "/opt/bin/vllm")
    spec = _spec(RuntimeKind.VLLM, device="cpu", **overrides)

    capabilities = adapter.probe(spec)

    assert capabilities.installed is True
    assert capabilities.compatible is False
    assert message in capabilities.reasons[0]
    with pytest.raises(RuntimeConfigurationError, match=message):
        adapter.start(spec)


def test_missing_external_runtime_is_optional_and_reported() -> None:
    adapter = VLLMRuntime(which=lambda name: None)
    spec = _spec(RuntimeKind.VLLM)

    capabilities = adapter.probe(spec)

    assert capabilities.installed is False
    assert capabilities.compatible is False
    with pytest.raises(RuntimeUnavailableError, match="not found"):
        adapter.start(spec)


def test_llamacpp_requires_gguf_and_builds_cpu_flags() -> None:
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(
        RuntimeKind.LLAMACPP,
        model="C:/models/invoice.gguf",
        context_length=4096,
        cpu_threads=12,
    )

    assert adapter.build_command(spec) == (
        "llama-server",
        "--model",
        "C:/models/invoice.gguf",
        "--alias",
        "invoice",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--ctx-size",
        "4096",
        "--threads",
        "12",
    )
    assert adapter.probe(_spec(RuntimeKind.LLAMACPP)).compatible is False


def test_start_uses_shell_false_and_tracks_lifecycle(tmp_path: Path) -> None:
    process = FakeProcess()
    call: dict[str, Any] = {}

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        call["command"] = command
        call.update(kwargs)
        return process

    adapter = OllamaRuntime(
        which=lambda name: "ollama",
        popen_factory=popen,
        health_get=lambda url, timeout, headers: HealthResult(True, 200),
    )
    spec = _spec(RuntimeKind.OLLAMA, port=11434)

    launched = adapter.start(spec, log_path=tmp_path / "ollama.log")

    assert launched.pid == process.pid
    assert call["command"] == ["ollama", "serve"]
    assert call["shell"] is False
    assert call["env"]["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert adapter.is_running(process.pid)
    assert adapter.health(spec).healthy
    adapter.shutdown(process.pid)
    assert process.terminated
    assert not adapter.is_running(process.pid)


def test_remote_runtime_is_processless_and_uses_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def health_get(url: str, timeout: float, headers: dict[str, str]) -> HealthResult:
        observed.update(url=url, timeout=timeout, headers=headers)
        return HealthResult(True, 200)

    monkeypatch.setenv("REMOTE_KEY", "secret")
    adapter = RemoteRuntime(health_get=health_get)
    spec = _spec(
        RuntimeKind.REMOTE,
        endpoint="https://models.example/v1",
        api_key_env="REMOTE_KEY",
    )

    process = adapter.start(spec)

    assert process.pid is None
    assert process.command == ()
    assert adapter.is_running(None)
    assert adapter.health(spec, timeout=4).healthy
    assert observed == {
        "url": "https://models.example/v1/models",
        "timeout": 4,
        "headers": {"Authorization": "Bearer secret"},
    }


def test_remote_runtime_rejects_embedded_credentials() -> None:
    adapter = RemoteRuntime()
    spec = _spec(
        RuntimeKind.REMOTE,
        endpoint="https://user:secret@models.example/v1",
    )

    assert adapter.probe(spec).compatible is False
    with pytest.raises(RuntimeConfigurationError, match="HTTP"):
        adapter.start(spec)


def test_version_detection_uses_argv_and_no_shell() -> None:
    observed: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="vllm 1.2.3\n", stderr="")

    adapter = VLLMRuntime(which=lambda name: "vllm", run_command=run)

    assert adapter.probe(_spec(RuntimeKind.VLLM)).version == "vllm 1.2.3"
    assert observed["command"] == ["vllm", "--version"]
    assert observed["kwargs"]["shell"] is False
