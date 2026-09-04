from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import json
import logging
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

logger = logging.getLogger(__name__)


class RuntimeKind(StrEnum):
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"
    REMOTE = "remote"
    ENCODER = "encoder"
    TRANSFORMERS = "transformers"
    MULTI_VECTOR = "multi_vector"


class LifecycleState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


class RuntimeFeature(StrEnum):
    BATCHING = "batching"
    EMBEDDINGS = "embeddings"
    LOGPROBS = "logprobs"
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


# llama-server's own --cache-type-k/-v allowed values (#382), verified against
# the live upstream tools/server/README.md -- shared by the draft-model
# variants too, but those aren't wired here.
_LLAMACPP_CACHE_TYPES = frozenset(
    {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
)

# llama-server's --spec-type (#400) supports several draft-model families
# too (draft-simple, draft-eagle3, ...), each needing a second checkpoint
# loaded alongside the primary one -- a separate, bigger feature (model
# pairing, VRAM budget for two models, tokenizer-family matching) than
# wiring the flag itself. Scoped to the n-gram types for this first pass:
# they speculate from n-grams already seen in the prompt/generation itself,
# needing no second model and no extra VRAM -- a strictly additive latency
# win on the RAM-constrained boxes this framework targets. Verified against
# the live upstream tools/server/README.md.
_LLAMACPP_NGRAM_SPEC_TYPES = frozenset({"ngram-simple", "ngram-cache"})


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
    # Default completion budget exposed by the deployment. This is routing /
    # generation metadata, not a llama-server process flag; callers may still
    # override it per request.
    max_tokens: int | None = None
    cpu_threads: int | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float | None = None
    api_key_env: str | None = None
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    # llama.cpp-specific multi-slot launch flags (#248/#321): other runtimes'
    # build_command never reads these.
    n_parallel: int = 1
    cache_reuse: int | None = None
    # llama.cpp only (#387): overrides the GGUF's own embedded chat_template
    # with an operator-supplied Jinja file -- e.g. patching tool-call
    # rendering onto a checkpoint whose baked-in template lacks it. Existence
    # is the caller's responsibility (same as `model` itself); the #290
    # health check re-verifies whatever template ends up live via GET /props
    # automatically, since that check queries the running server, not the
    # GGUF's static bytes -- an override gets exactly the same scrutiny a
    # baked-in template does, with no extra code.
    chat_template_file: str | None = None
    # llama.cpp only (#382): quantizing the KV cache is a direct, mechanical
    # RAM win on the RAM/VRAM-constrained boxes this framework targets --
    # q8_0 roughly halves KV cache size vs. f16 with near-zero accuracy
    # loss, q4_0 roughly quarters it with a small, model-dependent cost.
    # None (the default) omits both flags entirely, matching llama-server's
    # own f16 default byte-for-byte. --flash-attn is a *prerequisite* for
    # KV quant below f16 to actually take effect (otherwise it's silently a
    # no-op) -- build_command forces it on automatically whenever either
    # cache type is quantized, rather than requiring the operator to
    # separately remember to enable it.
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    # llama.cpp only (#402): caps how many tokens a thinking/reasoning model
    # can spend before it's forced to stop and answer -- -1 (the default)
    # is unrestricted, 0 ends thinking immediately, N>0 is a hard token cap.
    # Directly relevant to latency/cost: an unbounded reasoning model can
    # spend an arbitrary number of tokens "thinking" before ever producing a
    # usable answer, and interacts with the tool-budget-forced-answer
    # guarantee (#391) -- a forced round still has to wait through however
    # long the model wants to think before it answers. None omits the flag
    # entirely, matching llama-server's own -1/unrestricted default.
    reasoning_budget: int | None = None
    # Text injected before the end-of-thinking tag when reasoning_budget runs
    # out, so a budget-exhausted response reads as an intentional wrap-up
    # rather than an abrupt cut-off thought. Only meaningful alongside
    # reasoning_budget; validated together below.
    reasoning_budget_message: str | None = None
    # llama.cpp only (#400): n-gram speculative decoding -- a latency win
    # with no second model and no extra VRAM. Restricted to the n-gram
    # types only; the draft-model types (a second checkpoint, its own VRAM
    # budget and tokenizer-family matching) are a separate, bigger feature
    # deliberately not wired here.
    spec_type: str | None = None

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
        if self.max_tokens is not None and self.max_tokens < 1:
            raise RuntimeConfigurationError("max_tokens must be positive")
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
        if self.n_parallel < 1:
            raise RuntimeConfigurationError("n_parallel must be positive")
        if self.cache_reuse is not None and self.cache_reuse < 1:
            raise RuntimeConfigurationError("cache_reuse must be positive")
        if self.chat_template_file is not None and not self.chat_template_file.strip():
            raise RuntimeConfigurationError("chat_template_file must not be empty")
        if self.chat_template_file is not None and "\x00" in self.chat_template_file:
            raise RuntimeConfigurationError("chat_template_file must not contain NUL bytes")
        if self.cache_type_k is not None and self.cache_type_k not in _LLAMACPP_CACHE_TYPES:
            raise RuntimeConfigurationError(
                f"cache_type_k must be one of {sorted(_LLAMACPP_CACHE_TYPES)}"
            )
        if self.cache_type_v is not None and self.cache_type_v not in _LLAMACPP_CACHE_TYPES:
            raise RuntimeConfigurationError(
                f"cache_type_v must be one of {sorted(_LLAMACPP_CACHE_TYPES)}"
            )
        if self.reasoning_budget is not None and self.reasoning_budget < -1:
            raise RuntimeConfigurationError("reasoning_budget must be -1, 0, or positive")
        if self.reasoning_budget_message is not None:
            if self.reasoning_budget is None:
                raise RuntimeConfigurationError(
                    "reasoning_budget_message requires reasoning_budget to be set"
                )
            if "\x00" in self.reasoning_budget_message:
                raise RuntimeConfigurationError(
                    "reasoning_budget_message must not contain NUL bytes"
                )
        if self.spec_type is not None and self.spec_type not in _LLAMACPP_NGRAM_SPEC_TYPES:
            raise RuntimeConfigurationError(
                f"spec_type must be one of {sorted(_LLAMACPP_NGRAM_SPEC_TYPES)} "
                "(draft-model speculative types need a second checkpoint and "
                "aren't wired yet)"
            )


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
    # llama.cpp only (#353): the model's ACTUAL chat_template_caps.supports_tool_calls
    # from GET /props (see llamacpp_tool_calls_mismatch), captured whenever
    # chat_template_caps was present in the response. True/False is the real
    # signal; None means undetermined -- an older llama-server build, an
    # unreachable /props, or a runtime (vLLM, Ollama, ...) that never probes
    # this at all. Every non-llamacpp adapter leaves this at the default None.
    tool_calls_supported: bool | None = None


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


def _default_json_get(
    url: str, timeout: float, headers: Mapping[str, str]
) -> dict[str, Any] | None:
    """Best-effort GET+JSON-decode -- ``None`` on any failure (unreachable,
    non-200, not JSON). Used for capability-drift checks that must never
    turn a genuinely healthy deployment unhealthy just because an optional
    diagnostic endpoint is unavailable."""
    request = urllib.request.Request(url, headers=dict(headers))  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            if not (200 <= response.status < 300):
                return None
            body = response.read()
    except (OSError, urllib.error.URLError):
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


JsonGet = Callable[[str, float, Mapping[str, str]], "dict[str, Any] | None"]

SlotsGet = Callable[[str, float, Mapping[str, str]], "list[dict[str, Any]] | None"]


def _default_slots_get(
    url: str, timeout: float, headers: Mapping[str, str]
) -> list[dict[str, Any]] | None:
    """Best-effort GET+JSON-decode for ``GET /slots`` -- unlike ``/props``
    (a single JSON object), llama-server's own ``/slots`` returns a JSON
    ARRAY, one entry per processing slot. ``None`` on any failure
    (unreachable, non-200, not a JSON array): the same never-fail contract as
    ``_default_json_get`` -- a diagnostic endpoint that is disabled
    (``--no-slots``) or absent on an older build must never read as an error.
    """
    request = urllib.request.Request(url, headers=dict(headers))  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            if not (200 <= response.status < 300):
                return None
            body = response.read()
    except (OSError, urllib.error.URLError):
        return None
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    return [item for item in parsed if isinstance(item, dict)]


# Optional numeric fields llama-server MAY report per slot, varying by build
# (#315): some builds surface only prompt/cache state, others add per-slot
# prefill/decode timing directly on the slot entry (rather than only in a
# completion response's separate "timings" object). Every one of these is
# genuinely optional -- absence is normal, never an error.
_SLOT_NUMERIC_FIELDS = (
    "id_task",
    "n_past",
    "n_remain",
    "n_decoded",
    "cache_n",
    "prompt_n",
    "prompt_ms",
    "predicted_n",
    "predicted_ms",
    "tokens_per_second",
)

_NEXT_TOKEN_FIELDS = (
    "has_next_token",
    "has_new_line",
    "n_remain",
    "n_decoded",
    "stopped_eos",
    "stopped_limit",
    "stopped_word",
    "stopping_word",
)


def normalize_llamacpp_slot(slot: Mapping[str, Any]) -> dict[str, Any]:
    """Defensively project one raw ``GET /slots`` entry down to what the
    Observability slots card understands (#315).

    llama-server's ``/slots`` schema is NOT fixed across builds -- some
    report only prompt/cache state, others add per-slot prefill/decode
    timing, and field names have shifted release to release. Every field read
    here is optional and type-checked before use: a missing or unexpectedly-
    typed field is silently skipped, never raised on, so an older/newer/
    unknown llama-server build degrades to a smaller card instead of breaking
    the query.
    """
    def _is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    normalized: dict[str, Any] = {}
    if _is_int(slot.get("id")):
        normalized["id"] = slot["id"]
    if isinstance(slot.get("is_processing"), bool):
        normalized["is_processing"] = slot["is_processing"]
    if _is_int(slot.get("n_ctx")):
        normalized["n_ctx"] = slot["n_ctx"]
    prompt = slot.get("prompt")
    if isinstance(prompt, str):
        normalized["prompt"] = prompt[:200]
    for key in _SLOT_NUMERIC_FIELDS:
        value = slot.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized[key] = value
    next_token = slot.get("next_token")
    if isinstance(next_token, Mapping):
        normalized["next_token"] = {
            key: next_token[key]
            for key in _NEXT_TOKEN_FIELDS
            if key in next_token and isinstance(next_token[key], (bool, int, float, str))
        }
    return normalized


def fetch_llamacpp_slots(
    endpoint: str,
    *,
    timeout: float = 2,
    headers: Mapping[str, str] | None = None,
    slots_get: SlotsGet = _default_slots_get,
) -> tuple[dict[str, Any], ...]:
    """Query and normalize llama-server's ``GET /slots`` from a bare endpoint
    string -- no ``RuntimeLaunchSpec`` needed. This is the seam the
    Observability API route calls directly against a deployment's already-
    resolved ``endpoint`` (#315), and what ``LlamaCppRuntime.slots`` delegates
    to for callers that do have a spec. Never raises -- see
    ``_default_slots_get`` / ``normalize_llamacpp_slot``.
    """
    base = endpoint.rstrip("/").removesuffix("/v1")
    raw = slots_get(f"{base}/slots", timeout, dict(headers or {}))
    if raw is None:
        return ()
    return tuple(normalize_llamacpp_slot(slot) for slot in raw)


def llamacpp_tool_calls_mismatch(props: Mapping[str, Any]) -> str | None:
    """Compare llama-server's own ``GET /props`` report against
    ``RuntimeFeature.TOOL_CALLS`` (#290): ``LlamaCppRuntime.features``
    declares tool-calling support unconditionally, but the model's ACTUAL
    chat template is what decides whether that's true, per llama.cpp's own
    ``chat_template_caps.supports_tool_calls`` (see
    ``common/jinja/caps.h``/``caps.cpp`` -- the exact keys this checks
    against). ``None`` means no mismatch (or the field wasn't reported,
    an older llama-server build); a string is the loud warning to log.
    """
    caps = props.get("chat_template_caps")
    if not isinstance(caps, dict) or "supports_tool_calls" not in caps:
        return None
    if caps.get("supports_tool_calls") is False:
        return (
            "llama-server reports chat_template_caps.supports_tool_calls=false for "
            "this model's ACTUAL chat template, but LlamaCppRuntime advertises "
            "RuntimeFeature.TOOL_CALLS -- a request with 'tools' set will not get "
            "structured tool_calls back from this deployment"
        )
    return None


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
        json_get: JsonGet = _default_json_get,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self._popen_factory = popen_factory
        self._run_command = run_command
        self._health_get = health_get
        self._json_get = json_get
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

    def find_processes(self, spec: RuntimeLaunchSpec) -> tuple[int, ...]:
        """PIDs of live OS processes serving this spec (orphan reaping).

        The supervisor loses ``record.pid`` when a health-failure clears it
        while the process is still alive; a later stop/remove that kills only
        that (now ``None``) pid leaks the process and holds its multi-GB RAM
        forever, which then blocks every future deployment via the fit-check.
        This finds the process by its command line instead: matching the
        deployment's reserved ``--port`` AND ``--model`` tokens is unique to the
        deployment and still valid after the pid was lost — and safer than the
        recorded pid, which a PID reuse could point at an unrelated process.
        Runtimes with no owned local process (remote, shared ollama) return
        nothing.
        """
        needles = [str(spec.port)] if spec.port is not None else []
        if spec.model:
            needles.append(str(spec.model))
        if not needles:
            return ()
        found: list[int] = []
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                tokens = set(proc.info.get("cmdline") or ())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if all(needle in tokens for needle in needles):
                found.append(int(proc.info["pid"]))
        return tuple(found)

    def exit_status(self, pid: int | None) -> int | None:
        """Exit status of an OWNED process that has exited, else None.

        ``Popen.poll()`` yields the status once the child is reaped: a positive
        exit code, or a NEGATIVE signal number (``-9`` for SIGKILL). ``None``
        means the process is still running, or was not launched by this adapter
        (a recovered pid after a serving restart holds no Popen handle). This is
        what lets an OOM SIGKILL be told apart from an ordinary non-zero crash —
        an OOM-killed process writes nothing to its own stderr.
        """
        if pid is None:
            return None
        process = self._processes.get(pid)
        return process.poll() if process is not None else None

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
    # LOGPROBS (#335): llama-server's /v1/chat/completions supports OpenAI-shaped
    # logprobs/top_logprobs. Advertised here for llama.cpp only this round --
    # vLLM/Ollama parity is unverified, so they deliberately don't get it (see
    # extract.logprob_confidence, gated on ModelProfile.runtime == "llamacpp").
    features = frozenset(
        {
            RuntimeFeature.BATCHING,
            RuntimeFeature.EMBEDDINGS,
            RuntimeFeature.LOGPROBS,
            RuntimeFeature.LORA,
            RuntimeFeature.QUANTIZATION,
            RuntimeFeature.STRUCTURED_OUTPUT,
            RuntimeFeature.TOOL_CALLS,
            RuntimeFeature.VISION,
        }
    )

    def __init__(self, *, slots_get: SlotsGet = _default_slots_get, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._slots_get = slots_get

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
            "--jinja",
        ]
        if spec.chat_template_file is not None:
            command.extend(["--chat-template-file", spec.chat_template_file])
        if spec.cache_type_k is not None:
            command.extend(["--cache-type-k", spec.cache_type_k])
        if spec.cache_type_v is not None:
            command.extend(["--cache-type-v", spec.cache_type_v])
        # --flash-attn is a prerequisite for KV quant below f16 to actually
        # apply -- otherwise the type flags above are silently a no-op.
        # Forced on automatically rather than left to the operator to
        # separately remember (or to llama-server's own "auto" default,
        # which isn't guaranteed to enable it on every backend).
        if spec.cache_type_k not in (None, "f16") or spec.cache_type_v not in (None, "f16"):
            command.extend(["--flash-attn", "on"])
        if spec.reasoning_budget is not None:
            command.extend(["--reasoning-budget", str(spec.reasoning_budget)])
        if spec.reasoning_budget_message is not None:
            command.extend(["--reasoning-budget-message", spec.reasoning_budget_message])
        if spec.spec_type is not None:
            command.extend(["--spec-type", spec.spec_type])
        if spec.context_length is not None:
            ctx_size = spec.context_length
            if spec.n_parallel > 1:
                # llama-server splits --ctx-size across slots (effective
                # per-slot context = ctx_size / n_parallel), so the total
                # budget must be scaled up to give each slot the configured
                # context_length.
                ctx_size = spec.context_length * spec.n_parallel
            command.extend(["--ctx-size", str(ctx_size)])
        if spec.n_parallel > 1:
            command.extend(["--parallel", str(spec.n_parallel)])
        if spec.cpu_threads is not None:
            command.extend(["--threads", str(spec.cpu_threads)])
        if spec.cache_reuse is not None:
            command.extend(["--cache-reuse", str(spec.cache_reuse)])
        command.extend(spec.extra_args)
        return tuple(command)

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        """The base ``/health`` check, plus a capability-drift check (#290):
        only once the process is actually healthy, query ``GET /props`` and
        compare its ``chat_template_caps`` against what ``features``
        advertises. A mismatch is logged loudly -- never turns a healthy
        deployment unhealthy, since the deployment genuinely IS up; it's the
        TOOL_CALLS advertisement that would be a lie for THIS model's actual
        chat template.

        The raw ``chat_template_caps.supports_tool_calls`` verdict is also
        captured onto the returned ``HealthResult.tool_calls_supported`` --
        not just logged (#353) -- so a caller (the reconciler, then
        ``chat_api``) can act on it instead of it only ever reaching a log
        line nobody watches. Set whenever ``chat_template_caps`` was reported
        at all, True/False either way -- a healthy, tool-call-CAPABLE model
        gets ``True`` recorded explicitly, not left ``None``.
        """
        result = super().health(spec, timeout=timeout)
        if not result.healthy:
            return result
        base = self.endpoint(spec).removesuffix("/v1")
        headers: dict[str, str] = {}
        if spec.api_key_env and (api_key := os.environ.get(spec.api_key_env)):
            headers["Authorization"] = f"Bearer {api_key}"
        props = self._json_get(f"{base}/props", timeout, headers)
        tool_calls_supported: bool | None = None
        if props is not None:
            caps = props.get("chat_template_caps")
            if isinstance(caps, dict) and isinstance(caps.get("supports_tool_calls"), bool):
                tool_calls_supported = caps["supports_tool_calls"]
            mismatch = llamacpp_tool_calls_mismatch(props)
            if mismatch is not None:
                logger.warning("%s (alias=%r, endpoint=%r)", mismatch, spec.alias, base)
        return replace(result, tool_calls_supported=tool_calls_supported)

    def slots(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> tuple[dict[str, Any], ...]:
        """llama-server's own ``GET /slots`` introspection (#315): per-slot
        prompt state, KV cache reuse, and (on builds that report it)
        prefill/decode timing -- mirrors the ``GET /props`` call already in
        ``health()``. Never raises and never affects health: an unreachable
        or malformed ``/slots`` response yields an empty tuple, the same
        never-fail contract as the ``/props`` capability check above.
        """
        headers: dict[str, str] = {}
        if spec.api_key_env and (api_key := os.environ.get(spec.api_key_env)):
            headers["Authorization"] = f"Bearer {api_key}"
        return fetch_llamacpp_slots(
            self.endpoint(spec), timeout=timeout, headers=headers, slots_get=self._slots_get
        )


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

    def find_processes(self, spec: RuntimeLaunchSpec) -> tuple[int, ...]:
        # One shared `ollama serve` hosts every model; killing it by port would
        # take down unrelated deployments. Never orphan-reap here.
        return ()


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

    def find_processes(self, spec: RuntimeLaunchSpec) -> tuple[int, ...]:
        # A remote runtime owns no local process to reap.
        del spec
        return ()

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
        # Either analyzer library will do (the server auto-picks gliner2 for
        # GLiNER2 model ids). Fail the deploy at probe time with the
        # actionable reason, not after a spawn whose child dies on ImportError.
        if (
            importlib.util.find_spec("gliner") is None
            and importlib.util.find_spec("gliner2") is None
        ):
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


class TransformersRuntime(RuntimeAdapter):
    """Launch ``docie transformers`` — the OpenAI-compatible shim over an
    ``AutoModel`` checkpoint (see ``docie_bench.transformers_server.server``).

    The LAST-RESORT runtime: it serves a model with no GGUF (or an arch
    llama.cpp cannot load) directly from unquantized ``transformers`` weights,
    at ~2-3x a GGUF Q4's RAM and slower CPU inference — prefer a GGUF repo
    whenever one exists. ``spec.model`` is a local snapshot directory (seeded
    like an encoder) or an HF id; the family carries ``--trust-remote-code``
    into ``extra_args`` only when a custom-code checkpoint opts in. Everything
    else — port allocation, deployment record, health probing (``/healthz``),
    reconciler overlay, load/unload/delete lifecycle — is inherited unchanged.
    """

    kind = RuntimeKind.TRANSFORMERS
    executable_names = ("docie", "docie-serving")
    health_path = "/healthz"
    features = frozenset({RuntimeFeature.VISION})

    def resolve_executable(self, spec: RuntimeLaunchSpec) -> str | None:
        # The console script may be off PATH inside a container; the current
        # interpreter can always launch the CLI module instead (build_command),
        # so this runtime never reads as "not installed".
        return super().resolve_executable(spec) or sys.executable

    def detect_version(self, executable: str) -> str | None:
        # The meaningful version is transformers', not the CLI's.
        try:
            return f"transformers {importlib.metadata.version('transformers')}"
        except importlib.metadata.PackageNotFoundError:
            return None

    def probe(self, spec: RuntimeLaunchSpec) -> RuntimeCapabilities:
        capabilities = super().probe(spec)
        # Fail the deploy at probe time with the actionable reason, not after a
        # spawn whose child dies on ImportError.
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("transformers") is None
        ):
            return replace(
                capabilities,
                compatible=False,
                reasons=(
                    *capabilities.reasons,
                    "torch + transformers are not installed on the serving node "
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
            "transformers",
            "--model",
            spec.model,
            "--host",
            spec.host,
            "--port",
            str(spec.port),
            *spec.extra_args,
        )


class MultiVectorRuntime(RuntimeAdapter):
    """Launch ``docie multi-vector`` — the /v1/rerank shim over a
    sentence-transformers ``MultiVectorEncoder`` (ColBERT / PyLate late-
    interaction retriever; see ``docie_bench.multi_vector_server``).

    ``spec.model`` is a local safetensors snapshot directory (seeded like an
    encoder). MaxSim scoring over per-token embeddings runs in-process via
    ``encode_query`` / ``encode_document`` / ``similarity``. Everything else —
    port allocation, deployment record, health probing (``/healthz``),
    reconciler overlay, load/unload/delete lifecycle — is inherited unchanged.
    """

    kind = RuntimeKind.MULTI_VECTOR
    executable_names = ("docie", "docie-serving")
    health_path = "/healthz"
    features = frozenset()

    def resolve_executable(self, spec: RuntimeLaunchSpec) -> str | None:
        # The console script may be off PATH inside a container; the current
        # interpreter can always launch the CLI module instead (build_command),
        # so this runtime never reads as "not installed".
        return super().resolve_executable(spec) or sys.executable

    def detect_version(self, executable: str) -> str | None:
        # The meaningful version is sentence-transformers', not the CLI's.
        try:
            return f"sentence-transformers {importlib.metadata.version('sentence-transformers')}"
        except importlib.metadata.PackageNotFoundError:
            return None

    def probe(self, spec: RuntimeLaunchSpec) -> RuntimeCapabilities:
        capabilities = super().probe(spec)
        # Fail the deploy at probe time with the actionable reason, not after a
        # spawn whose child dies on ImportError. MultiVectorEncoder landed in
        # sentence-transformers 6.0 — an older install has the package but not
        # the class, so check for the class itself, not just the module.
        reason: str | None = None
        if importlib.util.find_spec("sentence_transformers") is None:
            reason = (
                "sentence-transformers is not installed on the serving node "
                "(pip install 'small-doc-ie-bench[encoders]')"
            )
        else:
            try:
                installed = importlib.metadata.version("sentence-transformers")
                if int(installed.split(".", 1)[0]) < 6:
                    reason = (
                        f"sentence-transformers {installed} is too old — MultiVectorEncoder "
                        "needs >= 6.0 (pip install 'small-doc-ie-bench[encoders]')"
                    )
            except (importlib.metadata.PackageNotFoundError, ValueError):
                reason = "could not read the installed sentence-transformers version"
        if reason is not None:
            return replace(
                capabilities, compatible=False, reasons=(*capabilities.reasons, reason)
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
            "multi-vector",
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
        RuntimeKind.TRANSFORMERS: TransformersRuntime(),
        RuntimeKind.MULTI_VECTOR: MultiVectorRuntime(),
    }


def command_display(command: Sequence[str]) -> str:
    """Return a display-only command string. It must never be used for execution."""
    return (
        subprocess.list2cmdline(list(command))
        if sys.platform == "win32"
        else " ".join(repr(value) for value in command)
    )
