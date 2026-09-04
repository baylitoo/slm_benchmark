from __future__ import annotations

import os
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
    fetch_llamacpp_slots,
    llamacpp_tool_calls_mismatch,
    normalize_llamacpp_slot,
)


class FakeProcess:
    # An intentionally implausible pid: after shutdown() the adapter falls back
    # to psutil.pid_exists, and a small pid like 41 EXISTS on Linux CI runners
    # (kernel threads), turning "is it still running?" into a platform lottery.
    def __init__(self, pid: int = 2_147_400_041) -> None:
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
        "--jinja",
        "--ctx-size",
        "4096",
        "--threads",
        "12",
    )
    assert adapter.probe(_spec(RuntimeKind.LLAMACPP)).compatible is False


def test_llamacpp_n_parallel_default_is_byte_identical_to_before() -> None:
    """n_parallel=1 (the default) must never emit --parallel and must never
    scale --ctx-size -- this is the existing, already-tested default path."""
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
        "--jinja",
        "--ctx-size",
        "4096",
        "--threads",
        "12",
    )


def test_llamacpp_n_parallel_scales_ctx_size_and_appends_parallel_flag() -> None:
    """--ctx-size is a SHARED total budget divided across slots by llama-server
    (confirmed against the ggml-org/llama.cpp server README): giving each slot
    the configured context_length requires ctx_size = context_length *
    n_parallel."""
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(
        RuntimeKind.LLAMACPP,
        model="C:/models/invoice.gguf",
        context_length=8192,
        n_parallel=4,
    )

    command = adapter.build_command(spec)

    assert "--ctx-size" in command
    assert command[command.index("--ctx-size") + 1] == "32768"
    assert "--parallel" in command
    assert command[command.index("--parallel") + 1] == "4"


def test_llamacpp_cache_reuse_emits_flag() -> None:
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(
        RuntimeKind.LLAMACPP,
        model="C:/models/invoice.gguf",
        cache_reuse=256,
    )

    command = adapter.build_command(spec)

    assert "--cache-reuse" in command
    assert command[command.index("--cache-reuse") + 1] == "256"


def test_launch_spec_rejects_invalid_n_parallel_and_cache_reuse() -> None:
    with pytest.raises(RuntimeConfigurationError, match="n_parallel"):
        _spec(RuntimeKind.LLAMACPP, n_parallel=0)
    with pytest.raises(RuntimeConfigurationError, match="cache_reuse"):
        _spec(RuntimeKind.LLAMACPP, cache_reuse=0)


def test_llamacpp_chat_template_file_emits_flag() -> None:
    # #387: overrides the GGUF's own embedded chat_template -- e.g. patching
    # tool-call rendering onto a checkpoint whose baked-in template lacks it
    # (verified live against LFM2.5-350M's own /props this milestone).
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(
        RuntimeKind.LLAMACPP,
        model="C:/models/invoice.gguf",
        chat_template_file="/templates/lfm2-with-tool-calls.jinja",
    )

    command = adapter.build_command(spec)

    assert "--chat-template-file" in command
    assert (
        command[command.index("--chat-template-file") + 1]
        == "/templates/lfm2-with-tool-calls.jinja"
    )


def test_llamacpp_omits_chat_template_file_flag_when_unset() -> None:
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(RuntimeKind.LLAMACPP, model="C:/models/invoice.gguf")

    command = adapter.build_command(spec)

    assert "--chat-template-file" not in command


def test_launch_spec_rejects_empty_or_nul_chat_template_file() -> None:
    with pytest.raises(RuntimeConfigurationError, match="chat_template_file"):
        _spec(RuntimeKind.LLAMACPP, chat_template_file="   ")
    with pytest.raises(RuntimeConfigurationError, match="chat_template_file"):
        _spec(RuntimeKind.LLAMACPP, chat_template_file="/bad\x00path.jinja")


def test_llamacpp_kv_cache_quant_emits_both_type_flags_and_forces_flash_attn() -> None:
    # #382: q8_0/q4_0 KV cache quant is a direct RAM win on the constrained
    # boxes this framework targets. --flash-attn is a prerequisite for the
    # quant to actually apply, forced on automatically rather than left to
    # the operator to separately remember.
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(
        RuntimeKind.LLAMACPP,
        model="C:/models/invoice.gguf",
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )

    command = adapter.build_command(spec)

    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--cache-type-v") + 1] == "q8_0"
    assert command[command.index("--flash-attn") + 1] == "on"


def test_llamacpp_kv_cache_quant_on_only_one_side_still_forces_flash_attn() -> None:
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(RuntimeKind.LLAMACPP, model="C:/models/invoice.gguf", cache_type_k="q4_0")

    command = adapter.build_command(spec)

    assert "--cache-type-v" not in command
    assert command[command.index("--flash-attn") + 1] == "on"


def test_llamacpp_omits_kv_cache_flags_when_unset() -> None:
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(RuntimeKind.LLAMACPP, model="C:/models/invoice.gguf")

    command = adapter.build_command(spec)

    assert "--cache-type-k" not in command
    assert "--cache-type-v" not in command
    assert "--flash-attn" not in command


def test_llamacpp_explicit_f16_cache_type_does_not_force_flash_attn() -> None:
    # f16 is llama-server's own default -- an operator explicitly spelling it
    # out is a no-op request, not a quantization request.
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(RuntimeKind.LLAMACPP, model="C:/models/invoice.gguf", cache_type_k="f16")

    command = adapter.build_command(spec)

    assert command[command.index("--cache-type-k") + 1] == "f16"
    assert "--flash-attn" not in command


def test_launch_spec_rejects_unknown_kv_cache_type() -> None:
    with pytest.raises(RuntimeConfigurationError, match="cache_type_k"):
        _spec(RuntimeKind.LLAMACPP, cache_type_k="q2_k")
    with pytest.raises(RuntimeConfigurationError, match="cache_type_v"):
        _spec(RuntimeKind.LLAMACPP, cache_type_v="not-a-real-type")


def test_llamacpp_reasoning_budget_emits_flag() -> None:
    # #402: caps how many tokens a thinking model spends before it's forced
    # to answer -- timely given lfm2.5-1.2b-thinking is actively deployed
    # this milestone.
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(RuntimeKind.LLAMACPP, model="C:/models/invoice.gguf", reasoning_budget=512)

    command = adapter.build_command(spec)

    assert command[command.index("--reasoning-budget") + 1] == "512"


def test_llamacpp_reasoning_budget_zero_and_negative_one_are_valid() -> None:
    # 0 = end thinking immediately, -1 = unrestricted -- both real,
    # meaningful values per llama-server's own docs, not "unset".
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    for value in (0, -1):
        spec = _spec(RuntimeKind.LLAMACPP, model="C:/models/invoice.gguf", reasoning_budget=value)
        command = adapter.build_command(spec)
        assert command[command.index("--reasoning-budget") + 1] == str(value)


def test_llamacpp_reasoning_budget_message_emits_alongside_budget() -> None:
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(
        RuntimeKind.LLAMACPP,
        model="C:/models/invoice.gguf",
        reasoning_budget=256,
        reasoning_budget_message="Wrapping up my analysis now.",
    )

    command = adapter.build_command(spec)

    assert (
        command[command.index("--reasoning-budget-message") + 1]
        == "Wrapping up my analysis now."
    )


def test_llamacpp_omits_reasoning_budget_flags_when_unset() -> None:
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(RuntimeKind.LLAMACPP, model="C:/models/invoice.gguf")

    command = adapter.build_command(spec)

    assert "--reasoning-budget" not in command
    assert "--reasoning-budget-message" not in command


def test_launch_spec_rejects_reasoning_budget_below_negative_one() -> None:
    with pytest.raises(RuntimeConfigurationError, match="reasoning_budget"):
        _spec(RuntimeKind.LLAMACPP, reasoning_budget=-2)


def test_launch_spec_rejects_reasoning_budget_message_without_a_budget() -> None:
    with pytest.raises(RuntimeConfigurationError, match="reasoning_budget_message"):
        _spec(RuntimeKind.LLAMACPP, reasoning_budget_message="Wrapping up.")


def test_llamacpp_ngram_speculative_decoding_emits_flag() -> None:
    # #400: ngram-simple/ngram-cache need no second model and no extra VRAM
    # -- a strictly additive latency win on the RAM-constrained boxes this
    # framework targets.
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    for spec_type in ("ngram-simple", "ngram-cache"):
        spec = _spec(RuntimeKind.LLAMACPP, model="C:/models/invoice.gguf", spec_type=spec_type)
        command = adapter.build_command(spec)
        assert command[command.index("--spec-type") + 1] == spec_type


def test_llamacpp_omits_spec_type_flag_when_unset() -> None:
    adapter = LlamaCppRuntime(which=lambda name: "llama-server")
    spec = _spec(RuntimeKind.LLAMACPP, model="C:/models/invoice.gguf")

    command = adapter.build_command(spec)

    assert "--spec-type" not in command


def test_launch_spec_rejects_draft_model_speculative_types() -> None:
    # The draft-model families (draft-simple, draft-eagle3, ...) need a
    # second checkpoint loaded alongside the primary one -- deliberately not
    # wired yet, so must be rejected rather than silently forwarded to a
    # llama-server that then errors (or worse, no-ops) at process start.
    with pytest.raises(RuntimeConfigurationError, match="spec_type"):
        _spec(RuntimeKind.LLAMACPP, spec_type="draft-simple")
    with pytest.raises(RuntimeConfigurationError, match="spec_type"):
        _spec(RuntimeKind.LLAMACPP, spec_type="not-a-real-type")


def test_llamacpp_tool_calls_mismatch_reads_the_real_chat_template_caps_schema() -> None:
    # Field name/shape confirmed against llama.cpp's own source (#290):
    # common/jinja/caps.h's caps struct, serialized via caps::to_map() in
    # caps.cpp, surfaced verbatim as GET /props's "chat_template_caps".
    supported = {"chat_template_caps": {"supports_tool_calls": True}}
    unsupported = {"chat_template_caps": {"supports_tool_calls": False}}
    assert llamacpp_tool_calls_mismatch(supported) is None
    assert "supports_tool_calls=false" in (llamacpp_tool_calls_mismatch(unsupported) or "")
    # An older llama-server build (or a malformed /props) reports nothing --
    # never treat "field absent" as a mismatch.
    assert llamacpp_tool_calls_mismatch({}) is None


def test_llamacpp_health_warns_on_capability_mismatch_but_stays_healthy(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = LlamaCppRuntime(
        which=lambda name: "llama-server",
        health_get=lambda url, timeout, headers: HealthResult(True, 200),
        json_get=lambda url, timeout, headers: {
            "chat_template_caps": {"supports_tool_calls": False}
        },
    )
    spec = _spec(RuntimeKind.LLAMACPP, alias="lfm2.5-vl-3b")

    with caplog.at_level("WARNING"):
        result = adapter.health(spec)

    assert result.healthy is True  # the deployment IS up -- never fail health over this
    assert any("supports_tool_calls=false" in r.message for r in caplog.records)
    assert any("lfm2.5-vl-3b" in r.message for r in caplog.records)


def test_llamacpp_health_records_tool_calls_supported_false() -> None:
    # #353: the raw chat_template_caps.supports_tool_calls verdict must land
    # on HealthResult.tool_calls_supported, not just a log line -- this is
    # what lets the reconciler/chat_api act on it.
    adapter = LlamaCppRuntime(
        which=lambda name: "llama-server",
        health_get=lambda url, timeout, headers: HealthResult(True, 200),
        json_get=lambda url, timeout, headers: {
            "chat_template_caps": {"supports_tool_calls": False}
        },
    )
    spec = _spec(RuntimeKind.LLAMACPP, alias="qwen3.8-2b-distill")

    result = adapter.health(spec)

    assert result.healthy is True
    assert result.tool_calls_supported is False


def test_llamacpp_health_records_tool_calls_supported_true() -> None:
    adapter = LlamaCppRuntime(
        which=lambda name: "llama-server",
        health_get=lambda url, timeout, headers: HealthResult(True, 200),
        json_get=lambda url, timeout, headers: {
            "chat_template_caps": {"supports_tool_calls": True}
        },
    )
    spec = _spec(RuntimeKind.LLAMACPP)

    result = adapter.health(spec)

    assert result.healthy is True
    assert result.tool_calls_supported is True


@pytest.mark.parametrize(
    "props",
    [
        {},  # older llama-server build: no chat_template_caps at all
        {"chat_template_caps": {}},  # caps present, field itself absent
        None,  # /props unreachable
    ],
)
def test_llamacpp_health_leaves_tool_calls_supported_none_when_undetermined(
    props: dict[str, Any] | None,
) -> None:
    adapter = LlamaCppRuntime(
        which=lambda name: "llama-server",
        health_get=lambda url, timeout, headers: HealthResult(True, 200),
        json_get=lambda url, timeout, headers: props,
    )
    spec = _spec(RuntimeKind.LLAMACPP)

    result = adapter.health(spec)

    assert result.healthy is True
    assert result.tool_calls_supported is None


def test_llamacpp_slots_tolerates_a_missing_field_schema() -> None:
    # Schema varies by llama-server build (#315): one slot reports the full
    # shape (including build-dependent timing/cache fields), the other only
    # the bare minimum an older build might report. Neither should raise, and
    # only genuinely-present, correctly-typed fields make it through.
    raw = [
        {
            "id": 0,
            "is_processing": True,
            "n_ctx": 4096,
            "prompt": "x" * 500,
            "prompt_ms": 12.5,
            "predicted_ms": 340.1,
            "cache_n": 128,
            "next_token": {"has_next_token": True, "n_remain": -1, "unexpected_field": object()},
            "unexpected_top_level_field": {"nested": "junk"},
        },
        {"id": 1, "is_processing": False},
    ]
    adapter = LlamaCppRuntime(
        which=lambda name: "llama-server",
        slots_get=lambda url, timeout, headers: raw,
    )
    spec = _spec(RuntimeKind.LLAMACPP, port=8090)

    slots = adapter.slots(spec)

    assert len(slots) == 2
    assert slots[0]["id"] == 0
    assert slots[0]["n_ctx"] == 4096
    assert len(slots[0]["prompt"]) <= 200  # truncated, not raw
    assert slots[0]["prompt_ms"] == 12.5
    assert slots[0]["cache_n"] == 128
    assert slots[0]["next_token"] == {"has_next_token": True, "n_remain": -1}
    assert "unexpected_top_level_field" not in slots[0]
    # The minimal slot carries only what it actually reported.
    assert slots[1] == {"id": 1, "is_processing": False}


def test_llamacpp_slots_never_raises_on_query_failure_and_never_touches_health() -> None:
    def boom(url: str, timeout: float, headers: dict[str, str]) -> list[dict] | None:
        return None  # mirrors _default_slots_get's contract on an unreachable/old build

    adapter = LlamaCppRuntime(
        which=lambda name: "llama-server",
        health_get=lambda url, timeout, headers: HealthResult(True, 200),
        slots_get=boom,
    )
    spec = _spec(RuntimeKind.LLAMACPP)

    assert adapter.slots(spec) == ()
    # health() doesn't even query /slots -- a slots-query failure must never
    # be able to flip an otherwise-healthy deployment to unhealthy.
    assert adapter.health(spec).healthy is True


def test_fetch_llamacpp_slots_works_from_a_bare_endpoint_string() -> None:
    # The Observability API route has a deployment's already-resolved
    # endpoint, not a RuntimeLaunchSpec -- this is the seam it calls directly.
    calls: list[str] = []

    def slots_get(url: str, timeout: float, headers: dict[str, str]) -> list[dict]:
        calls.append(url)
        return [{"id": 0, "is_processing": False}]

    result = fetch_llamacpp_slots("http://127.0.0.1:8090/v1", slots_get=slots_get)

    assert calls == ["http://127.0.0.1:8090/slots"]
    assert result == ({"id": 0, "is_processing": False},)


def test_normalize_llamacpp_slot_drops_wrong_typed_fields() -> None:
    # A boolean must never be read as the numeric id/n_ctx/timing fields --
    # bool is an int subclass in Python, an easy defensive-parsing mistake.
    assert normalize_llamacpp_slot({"id": True, "n_ctx": False, "prompt_ms": True}) == {}
    assert normalize_llamacpp_slot({}) == {}


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


def test_is_running_uses_psutil_for_pids_not_in_process_table() -> None:
    adapter = VLLMRuntime(which=lambda name: None)
    # Fallback branch: pid not tracked in _processes, must probe via psutil
    assert adapter.is_running(os.getpid())  # the test process itself is alive
    assert not adapter.is_running(2**31 - 1)  # beyond any OS PID space


def test_version_detection_uses_argv_and_no_shell() -> None:
    observed: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="vllm 1.2.3\n", stderr="")

    adapter = VLLMRuntime(which=lambda name: "vllm", run_command=run)

    assert adapter.probe(_spec(RuntimeKind.VLLM)).version == "vllm 1.2.3"
    assert observed["command"] == ["vllm", "--version"]
    assert observed["kwargs"]["shell"] is False


class _FakeProcInfo:
    def __init__(self, pid: int, cmdline: list[str]) -> None:
        self.info = {"pid": pid, "cmdline": cmdline}


def test_find_processes_matches_port_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # Orphan reaping keys on the deployment's unique port AND model in the
    # cmdline: both instances of a.gguf:8090 match; the other model, the other
    # port, and a stray process that merely mentions 8090 do not.
    from docie_bench.serving import runtime as runtime_mod

    procs = [
        _FakeProcInfo(10, ["llama-server", "--model", "/models/a.gguf", "--port", "8090"]),
        _FakeProcInfo(11, ["llama-server", "--model", "/models/b.gguf", "--port", "8091"]),
        _FakeProcInfo(12, ["python", "job.py", "8090"]),
        _FakeProcInfo(13, ["llama-server", "--model", "/models/a.gguf", "--port", "8090"]),
    ]
    monkeypatch.setattr(runtime_mod.psutil, "process_iter", lambda attrs=None: iter(procs))

    spec = _spec(RuntimeKind.LLAMACPP, model="/models/a.gguf", port=8090)
    assert set(LlamaCppRuntime().find_processes(spec)) == {10, 13}


def test_find_processes_exempts_shared_and_remote_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A shared ollama server and a remote endpoint own no reap-able local
    # process, so they must not even scan (killing ollama by port would take
    # down every model it hosts).
    from docie_bench.serving import runtime as runtime_mod

    def _boom(attrs: Any = None) -> Any:
        raise AssertionError("shared/remote runtimes must not scan processes")

    monkeypatch.setattr(runtime_mod.psutil, "process_iter", _boom)

    assert OllamaRuntime().find_processes(_spec(RuntimeKind.OLLAMA, port=11434)) == ()
    assert (
        RemoteRuntime().find_processes(_spec(RuntimeKind.REMOTE, endpoint="http://x/v1")) == ()
    )
