from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from docie_bench.llm.model_catalog import (
    ModelCapabilities,
    append_profile,
    build_profile_config,
    default_profile_name,
    detect_capabilities,
)
from docie_bench.llm.model_profiles import load_model_profiles

_MODELS_YAML = """\
# A comment that must survive appends.
profiles:
  existing_text:
    model: qwen2.5:1.5b
    base_url: http://localhost:11434/v1
    response_format_style: openai_json_schema
"""


def _models_config(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(_MODELS_YAML, encoding="utf-8")
    return path


def _install_show(monkeypatch: pytest.MonkeyPatch, payload: dict | None) -> None:
    """Stub Ollama /api/show; payload=None simulates an unreachable server."""

    class _Response(io.BytesIO):
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def fake_urlopen(request: object, *, timeout: float | None = None) -> _Response:
        if payload is None:
            raise OSError("connection refused")
        return _Response(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(
        "docie_bench.llm.model_catalog.urllib.request.urlopen", fake_urlopen
    )


def test_detect_capabilities_reads_vision_from_capabilities_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_show(
        monkeypatch, {"capabilities": ["completion", "vision"], "details": {"family": "gemma4"}}
    )
    caps = detect_capabilities("gemma4:12b")
    assert caps == ModelCapabilities(vision=True, family="gemma4", detected=True)


def test_detect_capabilities_infers_vision_from_family_when_no_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_show(monkeypatch, {"details": {"families": ["clip", "llama"]}})
    caps = detect_capabilities("llava:7b")
    assert caps.vision is True
    assert caps.detected is True


def test_detect_capabilities_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_show(monkeypatch, {"capabilities": ["completion"], "details": {"family": "phi3"}})
    caps = detect_capabilities("nuextract:3.8b")
    assert caps.vision is False
    assert caps.family == "phi3"


def test_detect_capabilities_unreachable_returns_not_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_show(monkeypatch, None)
    caps = detect_capabilities("whatever:latest")
    assert caps == ModelCapabilities(vision=False, family=None, detected=False)


def test_build_profile_config_defaults_to_json_object() -> None:
    cfg = build_profile_config("gemma4:e2b")
    assert cfg["response_format_style"] == "json_object"
    assert cfg["prompt_profile"] == "strict_extraction_v1"
    assert cfg["temperature"] == 0.0
    assert "stop_sequences" not in cfg
    assert "vision" not in cfg


def test_build_profile_config_special_cases_nuextract_by_name() -> None:
    cfg = build_profile_config("hf.co/numind/NuExtract:Q4_K_M")
    assert cfg["response_format_style"] == "none"
    assert cfg["prompt_profile"] == "nuextract_v1"
    assert cfg["stop_sequences"] == ["<|end-output|>"]
    assert cfg["max_tokens"] == 2000


def test_build_profile_config_vision_and_overrides() -> None:
    cfg = build_profile_config(
        "gemma4:12b", vision=True, response_format="json_object", prompt_profile="custom_v2"
    )
    assert cfg["vision"] is True
    assert cfg["vision_max_pages"] == 8
    assert cfg["prompt_profile"] == "custom_v2"


def test_default_profile_name_handles_hf_tags() -> None:
    name = default_profile_name("hf.co/yuxinlu1/gemma-4-12B-coder-v1-GGUF:Q4_K_M")
    assert name == "ollama_gemma_4_12b_coder_v1_gguf_q4_k_m"
    assert default_profile_name("gemma4:e2b") == "ollama_gemma4_e2b"


def test_append_profile_preserves_comments_and_loads(tmp_path: Path) -> None:
    path = _models_config(tmp_path)
    cfg = build_profile_config("gemma4:e2b", vision=True)
    append_profile(path, "ollama_gemma4_e2b_vision", cfg)

    text = path.read_text(encoding="utf-8")
    assert "# A comment that must survive appends." in text  # comments preserved
    profiles = load_model_profiles(path)
    assert "existing_text" in profiles  # prior profile intact
    added = profiles["ollama_gemma4_e2b_vision"]
    assert added.model == "gemma4:e2b"
    assert added.vision is True
    assert added.response_format_style == "json_object"


def test_append_profile_handles_special_chars_in_model_and_stop(tmp_path: Path) -> None:
    path = _models_config(tmp_path)
    model = "hf.co/numind/NuExtract:Q4_K_M"
    append_profile(path, "ollama_nuextract", build_profile_config(model))
    profiles = load_model_profiles(path)
    assert profiles["ollama_nuextract"].model == model
    assert profiles["ollama_nuextract"].stop_sequences == ("<|end-output|>",)


def test_append_profile_rejects_duplicate_name(tmp_path: Path) -> None:
    path = _models_config(tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        append_profile(path, "existing_text", build_profile_config("gemma4:e2b"))
