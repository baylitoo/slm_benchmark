"""Generate benchmark model profiles from an Ollama model with sane defaults.

`docie-bench models add <model>` uses this to turn a pulled Ollama model into a
ready `models.yaml` profile: it auto-detects vision capability via the Ollama
`/api/show` endpoint and picks a working `response_format_style`/`prompt_profile`
so the new profile extracts correctly without hand-tuning.

Format heuristic (Ollama):
    * a model whose name contains "nuextract" -> response_format "none" with the
      NuExtract prompt profile and its end-of-output stop token;
    * everything else -> "json_object". This is the safe universal Ollama default;
      "openai_json_schema" returns empty content on several Ollama models, so it is
      never auto-selected (pass --response-format to override).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from docie_bench.llm.model_profiles import load_model_profiles

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
# Ollama detail families that imply a multimodal projector when the newer
# `capabilities` array is absent (older Ollama versions).
_VISION_FAMILIES = {"clip", "mllama", "gemma3", "gemma4", "qwen2vl", "qwen2.5vl", "llava"}


@dataclass(frozen=True)
class ModelCapabilities:
    """What we could learn about a model from Ollama's /api/show."""

    vision: bool
    family: str | None
    detected: bool


def _ollama_api_host(base_url: str) -> str:
    """Map an OpenAI-compatible base_url (.../v1) to the Ollama native API root."""
    return base_url.rstrip("/").removesuffix("/v1").rstrip("/")


def detect_capabilities(
    model: str, base_url: str = DEFAULT_OLLAMA_BASE_URL, *, timeout: float = 10.0
) -> ModelCapabilities:
    """Query Ollama /api/show for a model's capabilities.

    Returns detected=False (rather than raising) when Ollama is unreachable or the
    model is not pulled, so the caller can fall back to text-only + an explicit flag.
    """
    host = _ollama_api_host(base_url)
    request = urllib.request.Request(  # noqa: S310
        f"{host}/api/show",
        data=json.dumps({"name": model}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError):
        return ModelCapabilities(vision=False, family=None, detected=False)

    capabilities = payload.get("capabilities") or []
    details = payload.get("details") or {}
    families = details.get("families") or (
        [details["family"]] if details.get("family") else []
    )
    vision = "vision" in capabilities or any(fam in _VISION_FAMILIES for fam in families)
    family = families[0] if families else None
    return ModelCapabilities(vision=vision, family=family, detected=True)


def build_profile_config(
    model: str,
    *,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    vision: bool = False,
    response_format: str | None = None,
    prompt_profile: str | None = None,
) -> dict[str, Any]:
    """Build a models.yaml profile mapping for an Ollama model."""
    is_nuextract = "nuextract" in model.lower()
    if response_format is None:
        response_format = "none" if is_nuextract else "json_object"
    if prompt_profile is None:
        prompt_profile = "nuextract_v1" if is_nuextract else "strict_extraction_v1"

    cfg: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": "local-not-used",
        "response_format_style": response_format,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2000 if is_nuextract else 1200,
        "timeout_seconds": 600,
        "prompt_profile": prompt_profile,
    }
    if is_nuextract:
        cfg["stop_sequences"] = ["<|end-output|>"]
    if vision:
        cfg["vision"] = True
        cfg["vision_max_pages"] = 8
        cfg["vision_pdf_dpi"] = 150
    return cfg


def default_profile_name(model: str) -> str:
    """Derive a profile name from a model tag.

    The tag/variant after ':' is kept so gemma4:e2b and gemma4:12b don't collide
    (gemma4:e2b -> ollama_gemma4_e2b; hf.co/user/repo:Q4 -> ollama_repo_q4).
    """
    tail = model.split("/")[-1]
    sanitized = re.sub(r"[^a-z0-9]+", "_", tail.lower()).strip("_")
    return f"ollama_{sanitized}" if sanitized else "ollama_model"


def _render_profile_block(name: str, cfg: dict[str, Any]) -> str:
    """Render a single profile as a 2-space-indented YAML block under `profiles:`."""
    dumped = yaml.safe_dump(
        {name: cfg}, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    return "\n".join(("  " + line) if line.strip() else line for line in dumped.splitlines())


def append_profile(models_config_path: Path, name: str, cfg: dict[str, Any]) -> None:
    """Append a profile to models.yaml, preserving existing content and comments.

    The new block is text-appended (so comments survive) and then verified by
    reloading: the new profile must be present and every prior profile intact, or
    the file is restored and a ValueError is raised.
    """
    existing = load_model_profiles(models_config_path)
    if name in existing:
        raise ValueError(f"Profile {name!r} already exists in {models_config_path}")

    original = models_config_path.read_text(encoding="utf-8")
    block = _render_profile_block(name, cfg)
    models_config_path.write_text(
        original.rstrip("\n") + "\n\n" + block + "\n", encoding="utf-8", newline="\n"
    )

    try:
        reloaded = load_model_profiles(models_config_path)
    except Exception:
        models_config_path.write_text(original, encoding="utf-8", newline="\n")
        raise
    if name not in reloaded or not set(existing).issubset(reloaded):
        models_config_path.write_text(original, encoding="utf-8", newline="\n")
        raise ValueError("Appending the profile corrupted the models config; reverted the file.")
