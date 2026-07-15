from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import psutil


class RuntimeKind(StrEnum):
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"
    REMOTE = "remote"
    ENCODER = "encoder"


class LifecycleState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


class RuntimeFeature(StrEnum):
    BATCHING = "batching"
    EMBEDDINGS = "embeddings"
    LORA = "lora"
    QUANTIZATION = "quantization"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLS = "tool_calls"
    VISION = "vision"


class RuntimeConfigurationError(ValueError):
    pass


class RuntimeUnavailableError(RuntimeError):
    pass


class RuntimeLaunchError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeLaunchSpec:
    runtime: RuntimeKind
    model: str
    alias: str
    host: str = "127.0.0.1"
    port: int = 8000
    endpoint: str | None = None
    executable: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    quantization: str | None = None
    context_length: int | None = None
    cpu_threads: int | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float | None = None
    api_key_env: str | None = None
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise RuntimeConfigurationError("model must not be empty")
        if not self.alias.strip():
            raise RuntimeConfigurationError("alias must not be empty")
        if any(character.isspace() or character in "/@" for character in self.host):
            raise RuntimeConfigurationError("host must be a hostname or IP address")
        if not 1 <= self.port <= 65535:
            raise RuntimeConfigurationError("port must be between 1 and 65535")
        if self.context_length is not None and self.context_length < 1:
            raise RuntimeConfigurationError("context_length must be positive")
        if self.cpu_threads is not None and self.cpu_threads < 1:
            raise RuntimeConfigurationError("cpu_threads must be positive")
        if self.tensor_parallel_size < 1:
            raise RuntimeConfigurationError("tensor_parallel_size must be positive")
        if self.gpu_memory_utilization is not None and not (0 < self.gpu_memory_utilization <= 1):
            raise RuntimeConfigurationError("gpu_memory_utilization must be in (0, 1]")
        if any("\x00" in value for value in self.extra_args):
            raise RuntimeConfigurationError("extra_args must not contain NUL bytes")
        if any("\x00" in key or "\x00" in value for key, value in self.env.items()):
            raise RuntimeConfigurationError("environment entries must not contain NUL bytes")


@dataclass(frozen=True)
class RuntimeCapabilities:
    runtime: RuntimeKind
    installed: bool
    compatible: bool
    version: str | None
    features: frozenset[RuntimeFeature]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeProcess:
    runtime: RuntimeKind
    endpoint: str
    pid: int | None
    command: tuple[str, ...] = ()
    started_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class HealthResult:
    healthy: bool
    status_code: int | None = None
    detail: str | None = None
    latency_seconds: float | None = None


class Process(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


PopenFactory = Callable[..., Process]
RunCommand = Callable[..., subprocess.CompletedProcess[str]]
HealthGet = Callable[[str, float, Mapping[str, str]], HealthResult]


def _default_health_get(url: str, timeout: float, headers: Mapping[str, str]) -> HealthResult:
    started = time.monotonic()
    request = urllib.request.Request(url, headers=dict(headers))  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = response.status
    except urllib.error.HTTPError as exc:
        return HealthResult(
            healthy=False,
            status_code=exc.code,
            detail=str(exc),
            latency_seconds=time.monotonic() - started,
        )
    except (OSError, urllib.error.URLError) as exc:
        return HealthResult(
            healthy=False,
            detail=str(exc),
            latency_seconds=time.monotonic() - started,
        )
    return HealthResult(
        healthy=200 <= status < 400,
        status_code=status,
        latency_seconds=time.monotonic() - started,
    )


class RuntimeAdapter:
    kind: RuntimeKind
    executable_names: tuple[str, ...] = ()
    features: frozenset[RuntimeFeature] = frozenset()
    health_path = "/health"

    def __init__(
        self,
        *,
        popen_factory: PopenFactory = subprocess.Popen,
        run_command: RunCommand = subprocess.run,
        health_get: HealthGet = _default_health_get,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self._popen_factory = popen_factory
        self._run_command = run_command
        self._health_get = health_get
        self._which = which
        self._processes: dict[int, Process] = {}

    def probe(self, spec: RuntimeLaunchSpec) -> RuntimeCapabilities:
        executable = self.resolve_executable(spec)
        installed = executable is not None
        reasons: tuple[str, ...] = ()
        try:
            self.validate(spec)
        except RuntimeConfigurationError as exc:
            reasons = (str(exc),)
        if not installed:
            reasons = (*reasons, f"{self.kind} executable was not found")
        return RuntimeCapabilities(
            runtime=self.kind,
            installed=installed,
            compatible=installed and not reasons,
            version=self.detect_version(executable) if executable else None,
            features=self.features,
            reasons=reasons,
        )

    def validate(self, spec: RuntimeLaunchSpec) -> None:
        if spec.runtime != self.kind:
            raise RuntimeConfigurationError(
                f"{self.kind} adapter cannot launch {spec.runtime} specifications"
            )

    def resolve_executable(self, spec: RuntimeLaunchSpec) -> str | None:
        if spec.executable:
            return self._which(spec.executable)
        for name in self.executable_names:
            if executable := self._which(name):
                return executable
        return None

    def detect_version(self, executable: str) -> str | None:
        try:
            result = self._run_command(
                [executable, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = (result.stdout or result.stderr).strip()
        return output.splitlines()[0] if output else None

    def endpoint(self, spec: RuntimeLaunchSpec) -> str:
        return spec.endpoint.rstrip("/") if spec.endpoint else f"http://{spec.host}:{spec.port}/v1"

    def build_command(self, spec: RuntimeLaunchSpec) -> tuple[str, ...]:
        raise NotImplementedError

    def build_environment(self, spec: RuntimeLaunchSpec) -> dict[str, str]:
        return {**os.environ, **dict(spec.env)}

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> RuntimeProcess:
        capabilities = self.probe(spec)
        if not capabilities.installed:
            raise RuntimeUnavailableError("; ".join(capabilities.reasons))
        if not capabilities.compatible:
            raise RuntimeConfigurationError("; ".join(capabilities.reasons))
        command = self.build_command(spec)
        log_handle: Any = subprocess.DEVNULL
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab")
        try:
            process = self._popen_factory(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=self.build_environment(spec),
                shell=False,
            )
        except OSError as exc:
            raise RuntimeLaunchError(f"Failed to launch {self.kind}: {exc}") from exc
        finally:
            if log_path is not None:
                log_handle.close()
        self._processes[process.pid] = process
        return RuntimeProcess(
            runtime=self.kind,
            endpoint=self.endpoint(spec),
            pid=process.pid,
            command=command,
        )

    def is_running(self, pid: int | None) -> bool:
        if pid is None:
            return False
        process = self._processes.get(pid)
        if process is not None:
            return process.poll() is None
        return bool(psutil.pid_exists(pid))

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        """Terminate the runtime process and WAIT until it is actually gone.

        Both branches block until the process exits (or the escalation
        timeout elapses): the owned-Popen branch always did, and the
        recovered-pid branch (a serving-container restart emptied
        ``_processes``) now does too instead of a fire-and-forget SIGTERM.
        The wait is load-bearing for the fit-before-evict gate: eviction
        frees a multi-GB victim precisely so a following ``assess_fit`` can
        observe the freed RAM — returning while the victim is still dying
        would let the gate approve an overcommit that OOMs.
        """
        if pid is None:
            return
        process = self._processes.pop(pid, None)
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
            return
        if not self.is_running(pid):
            return
        # Recovered pid (no Popen handle): terminate via psutil so we can WAIT
        # on a non-child process, escalating to kill() exactly like the owned
        # branch. NoSuchProcess at any point means "already gone" — done.
        try:
            external = psutil.Process(pid)
            external.terminate()
            try:
                external.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                external.kill()
                # A second timeout means unkillable (e.g. uninterruptible
                # I/O); nothing more can be done.
                with contextlib.suppress(psutil.TimeoutExpired):
                    external.wait(timeout=timeout)
        except psutil.NoSuchProcess:
            return

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        headers: dict[str, str] = {}
        if spec.api_key_env and (api_key := os.environ.get(spec.api_key_env)):
            headers["Authorization"] = f"Bearer {api_key}"
        return self._health_get(
            f"{self.endpoint(spec).removesuffix('/v1')}{self.health_path}",
            timeout,
            headers,
        )


class VLLMRuntime(RuntimeAdapter):
    kind = RuntimeKind.VLLM
    executable_names = ("vllm",)
    features = frozenset(
        {
            RuntimeFeature.BATCHING,
            RuntimeFeature.EMBEDDINGS,
            RuntimeFeature.LORA,
            RuntimeFeature.QUANTIZATION,
            RuntimeFeature.STRUCTURED_OUTPUT,
            RuntimeFeature.TOOL_CALLS,
            RuntimeFeature.VISION,
        }
    )

    def validate(self, spec: RuntimeLaunchSpec) -> None:
        super().validate(spec)
        if spec.device not in {"auto", "cpu", "cuda"}:
            raise RuntimeConfigurationError("vLLM device must be auto, cpu, or cuda")
        if spec.device == "cpu":
            if spec.tensor_parallel_size != 1:
                raise RuntimeConfigurationError("vLLM CPU requires tensor_parallel_size=1")
            if spec.gpu_memory_utilization is not None:
                raise RuntimeConfigurationError("gpu_memory_utilization is invalid for vLLM CPU")
            if spec.dtype not in {"auto", "bfloat16", "float32"}:
                raise RuntimeConfigurationError("vLLM CPU dtype must be auto, bfloat16, or float32")
            if spec.quantization is not None:
                raise RuntimeConfigurationError(
                    "vLLM CPU quantization requires runtime benchmark validation"
                )

    def build_command(self, spec: RuntimeLaunchSpec) -> tuple[str, ...]:
        self.validate(spec)
        executable = self.resolve_executable(spec)
        if executable is None:
            raise RuntimeUnavailableError("vllm executable was not found")
        command = [
            executable,
            "serve",
            spec.model,
            "--host",
            spec.host,
            "--port",
            str(spec.port),
            "--served-model-name",
            spec.alias,
            "--device",
            spec.device,
            "--dtype",
            spec.dtype,
            "--tensor-parallel-size",
            str(spec.tensor_parallel_size),
        ]
        if spec.context_length is not None:
            command.extend(["--max-model-len", str(spec.context_length)])
        if spec.quantization:
            command.extend(["--quantization", spec.quantization])
        if spec.gpu_memory_utilization is not None:
            command.extend(["--gpu-memory-utilization", str(spec.gpu_memory_utilization)])
        command.extend(spec.extra_args)
        return tuple(command)


class LlamaCppRuntime(RuntimeAdapter):
    kind = RuntimeKind.LLAMACPP
    executable_names = ("llama-server",)
    features = frozenset(
        {
            RuntimeFeature.BATCHING,
            RuntimeFeature.EMBEDDINGS,
            RuntimeFeature.LORA,
            RuntimeFeature.QUANTIZATION,
            RuntimeFeature.STRUCTURED_OUTPUT,
            RuntimeFeature.TOOL_CALLS,
            RuntimeFeature.VISION,
        }
    )

    def validate(self, spec: RuntimeLaunchSpec) -> None:
        super().validate(spec)
        if Path(spec.model).suffix.lower() != ".gguf":
            raise RuntimeConfigurationError("llama.cpp requires a GGUF model")

    def build_command(self, spec: RuntimeLaunchSpec) -> tuple[str, ...]:
        self.validate(spec)
        executable = self.resolve_executable(spec)
        if executable is None:
            raise RuntimeUnavailableError("llama-server executable was not found")
        command = [
            executable,
            "--model",
            spec.model,
            "--alias",
            spec.alias,
            "--host",
            spec.host,
            "--port",
            str(spec.port),
        ]
        if spec.context_length is not None:
            command.extend(["--ctx-size", str(spec.context_length)])
        if spec.cpu_threads is not None:
            command.extend(["--threads", str(spec.cpu_threads)])
        command.extend(spec.extra_args)
        return tuple(command)


class OllamaRuntime(RuntimeAdapter):
    kind = RuntimeKind.OLLAMA
    executable_names = ("ollama",)
    health_path = "/api/tags"
    features = frozenset(
        {
            RuntimeFeature.EMBEDDINGS,
            RuntimeFeature.QUANTIZATION,
            RuntimeFeature.STRUCTURED_OUTPUT,
            RuntimeFeature.TOOL_CALLS,
            RuntimeFeature.VISION,
        }
    )

    def endpoint(self, spec: RuntimeLaunchSpec) -> str:
        return spec.endpoint.rstrip("/") if spec.endpoint else f"http://{spec.host}:{spec.port}/v1"

    def build_environment(self, spec: RuntimeLaunchSpec) -> dict[str, str]:
        return {**super().build_environment(spec), "OLLAMA_HOST": f"{spec.host}:{spec.port}"}

    def build_command(self, spec: RuntimeLaunchSpec) -> tuple[str, ...]:
        self.validate(spec)
        executable = self.resolve_executable(spec)
        if executable is None:
            raise RuntimeUnavailableError("ollama executable was not found")
        return (executable, "serve", *spec.extra_args)


class RemoteRuntime(RuntimeAdapter):
    kind = RuntimeKind.REMOTE
    features = frozenset(RuntimeFeature)
    health_path = "/models"

    def validate(self, spec: RuntimeLaunchSpec) -> None:
        super().validate(spec)
        parsed = urllib.parse.urlsplit(spec.endpoint or "")
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise RuntimeConfigurationError("remote runtime requires an HTTP(S) endpoint")

    def probe(self, spec: RuntimeLaunchSpec) -> RuntimeCapabilities:
        reasons: tuple[str, ...] = ()
        try:
            self.validate(spec)
        except RuntimeConfigurationError as exc:
            reasons = (str(exc),)
        return RuntimeCapabilities(
            runtime=self.kind,
            installed=True,
            compatible=not reasons,
            version=None,
            features=self.features,
            reasons=reasons,
        )

    def build_command(self, spec: RuntimeLaunchSpec) -> tuple[str, ...]:
        self.validate(spec)
        return ()

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> RuntimeProcess:
        del log_path
        self.validate(spec)
        return RuntimeProcess(runtime=self.kind, endpoint=self.endpoint(spec), pid=None)

    def is_running(self, pid: int | None) -> bool:
        return True

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del pid, timeout

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        headers: dict[str, str] = {}
        if spec.api_key_env and (api_key := os.environ.get(spec.api_key_env)):
            headers["Authorization"] = f"Bearer {api_key}"
        return self._health_get(f"{self.endpoint(spec)}/models", timeout, headers)


class EncoderRuntime(RuntimeAdapter):
    """Launch ``docie encoder`` — the OpenAI-compatible shim over a
    token-classification model (see ``docie_bench.encoders.server``).

    ``spec.model`` is the encoder model id (e.g. a GLiNER HF id), not a GGUF
    path, so this deploys through the explicit-runtime ``serve`` path
    (``runtime="encoder"``), never the store path. Everything else — port
    allocation, deployment record, health probing (``/healthz``), reconciler
    overlay, load/unload/delete lifecycle — is inherited unchanged.
    """

    kind = RuntimeKind.ENCODER
    executable_names = ("docie", "docie-serving")
    health_path = "/healthz"
    features = frozenset()

    def resolve_executable(self, spec: RuntimeLaunchSpec) -> str | None:
        # The console script may be off PATH inside a container; the current
        # interpreter can always launch the CLI module instead (see
        # build_command), so the encoder never reads as "not installed".
        return super().resolve_executable(spec) or sys.executable

    def detect_version(self, executable: str) -> str | None:
        # The meaningful version is the analyzer library's, not the CLI's.
        try:
            return f"gliner {importlib.metadata.version('gliner')}"
        except importlib.metadata.PackageNotFoundError:
            return None

    def probe(self, spec: RuntimeLaunchSpec) -> RuntimeCapabilities:
        capabilities = super().probe(spec)
        if importlib.util.find_spec("gliner") is None:
            # Fail the deploy at probe time with the actionable reason, not
            # after a spawn whose child process dies on the same ImportError.
            return replace(
                capabilities,
                compatible=False,
                reasons=(
                    *capabilities.reasons,
                    "the 'encoders' extra is not installed on the serving node "
                    "(pip install 'small-doc-ie-bench[encoders]')",
                ),
            )
        return capabilities

    def build_command(self, spec: RuntimeLaunchSpec) -> tuple[str, ...]:
        self.validate(spec)
        # Base lookup (no interpreter fallback): a found console script runs
        # directly; otherwise launch the CLI module with this interpreter.
        found = RuntimeAdapter.resolve_executable(self, spec)
        base = (found,) if found else (sys.executable, "-m", "docie_bench.serving.cli")
        return (
            *base,
            "encoder",
            "--model",
            spec.model,
            "--host",
            spec.host,
            "--port",
            str(spec.port),
            *spec.extra_args,
        )


def default_runtime_adapters() -> dict[RuntimeKind, RuntimeAdapter]:
    return {
        RuntimeKind.VLLM: VLLMRuntime(),
        RuntimeKind.LLAMACPP: LlamaCppRuntime(),
        RuntimeKind.OLLAMA: OllamaRuntime(),
        RuntimeKind.REMOTE: RemoteRuntime(),
        RuntimeKind.ENCODER: EncoderRuntime(),
    }


def command_display(command: Sequence[str]) -> str:
    """Return a display-only command string. It must never be used for execution."""
    return (
        subprocess.list2cmdline(list(command))
        if sys.platform == "win32"
        else " ".join(repr(value) for value in command)
    )
