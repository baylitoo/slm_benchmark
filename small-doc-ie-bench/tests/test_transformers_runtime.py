"""Transformers last-resort runtime: arch gate, adapter command, family routing."""

from __future__ import annotations

from docie_bench.serving.arch_registry import (
    TRANSFORMERS_FAMILY,
    TRANSFORMERS_MEMORY_NOTE,
    resolve_family,
)
from docie_bench.serving.catalog import available_backends
from docie_bench.serving.model_store import get_family
from docie_bench.serving.runtime import (
    RuntimeKind,
    RuntimeLaunchSpec,
    TransformersRuntime,
    default_runtime_adapters,
)

# ── the "no servable GGUF" gate ─────────────────────────────────────────────


def test_safetensors_only_known_arch_falls_to_transformers() -> None:
    # A known llama.cpp arch but NO gguf in this repo: it cannot be served via
    # llama-server here, so the last resort applies — with the memory note.
    verdict = resolve_family(
        "llama", has_gguf=False, has_safetensors=True, has_mmproj=False
    )
    assert verdict.verdict == "supported"
    assert verdict.family == TRANSFORMERS_FAMILY
    assert verdict.runtime_note == TRANSFORMERS_MEMORY_NOTE


def test_safetensors_only_unknown_arch_falls_to_transformers() -> None:
    verdict = resolve_family(
        "some-new-arch", has_gguf=False, has_safetensors=True, has_mmproj=False
    )
    assert verdict.verdict == "supported"
    assert verdict.family == TRANSFORMERS_FAMILY
    assert verdict.runtime_note == TRANSFORMERS_MEMORY_NOTE


def test_gguf_present_never_suggests_transformers() -> None:
    # The hard gate: a servable GGUF is always preferred over transformers.
    verdict = resolve_family(
        "llama", has_gguf=True, has_safetensors=True, has_mmproj=False
    )
    assert verdict.family == "openai_chat"
    assert verdict.runtime_note != TRANSFORMERS_MEMORY_NOTE


def test_no_weights_at_all_is_unsupported() -> None:
    verdict = resolve_family(
        "llama", has_gguf=False, has_safetensors=False, has_mmproj=False
    )
    assert verdict.verdict == "unsupported"
    assert verdict.family is None


def test_gliner_still_wins_over_transformers_gate() -> None:
    # GLiNER checkpoints are safetensors-only too; the analyzer verdict must
    # still take precedence over the transformers last resort.
    verdict = resolve_family(
        "gliner", has_gguf=False, has_safetensors=True, has_mmproj=False
    )
    assert verdict.family == "encoder_gliner"


def test_no_gguf_no_safetensors_no_arch_is_unsupported() -> None:
    verdict = resolve_family(
        None, has_gguf=False, has_safetensors=False, has_mmproj=False
    )
    assert verdict.verdict == "unsupported"


# ── family contract ─────────────────────────────────────────────────────────


def test_transformers_families_registered() -> None:
    plain = get_family("transformers")
    assert plain.transformers_runtime is True
    assert plain.trust_remote_code is False
    trusted = get_family("transformers_trust_remote_code")
    assert trusted.transformers_runtime is True
    assert trusted.trust_remote_code is True


def test_available_backends_transformers_only() -> None:
    assert available_backends("transformers") == ["transformers"]


# ── runtime adapter ─────────────────────────────────────────────────────────


def test_transformers_runtime_registered() -> None:
    adapters = default_runtime_adapters()
    assert isinstance(adapters[RuntimeKind.TRANSFORMERS], TransformersRuntime)


def test_build_command_serves_model_dir() -> None:
    adapter = TransformersRuntime()
    spec = RuntimeLaunchSpec(
        runtime=RuntimeKind.TRANSFORMERS,
        model="/store/mymodel",
        alias="mymodel",
        host="127.0.0.1",
        port=8091,
    )
    command = adapter.build_command(spec)
    assert "transformers" in command
    assert "--model" in command
    assert "/store/mymodel" in command
    assert command[command.index("--port") + 1] == "8091"
    # No trust flag unless the family opts in via extra_args.
    assert "--trust-remote-code" not in command


def test_build_command_carries_trust_flag() -> None:
    adapter = TransformersRuntime()
    spec = RuntimeLaunchSpec(
        runtime=RuntimeKind.TRANSFORMERS,
        model="/store/custom",
        alias="custom",
        port=8091,
        extra_args=("--trust-remote-code",),
    )
    assert "--trust-remote-code" in adapter.build_command(spec)
