from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    base_url: str
    api_key: str
    response_format_style: str = "openai_json_schema"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 900
    timeout_seconds: float = 180.0
    prompt_profile: str = "strict_extraction_v1"
    stop_sequences: tuple[str, ...] = ()
    vision: bool = False
    vision_max_pages: int = 8
    vision_pdf_dpi: int = 150


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_model_profiles(path: str | Path) -> dict[str, ModelProfile]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    profiles: dict[str, ModelProfile] = {}
    for name, cfg in data.get("profiles", {}).items():
        cfg = {key: _expand_env(value) for key, value in cfg.items()}
        api_key_env = cfg.get("api_key_env")
        api_key = os.environ.get(api_key_env, "") if api_key_env else cfg.get("api_key", "")
        profiles[name] = ModelProfile(
            name=name,
            model=cfg["model"],
            base_url=cfg["base_url"].rstrip("/"),
            api_key=api_key or "local-not-used",
            response_format_style=cfg.get("response_format_style", "openai_json_schema"),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            max_tokens=int(cfg.get("max_tokens", 900)),
            timeout_seconds=float(cfg.get("timeout_seconds", 180)),
            prompt_profile=cfg.get("prompt_profile", "strict_extraction_v1"),
            stop_sequences=tuple(cfg.get("stop_sequences") or ()),
            vision=bool(cfg.get("vision", False)),
            vision_max_pages=int(cfg.get("vision_max_pages", 8)),
            vision_pdf_dpi=int(cfg.get("vision_pdf_dpi", 150)),
        )
    return profiles


def load_judge_profile(path: str | Path, profile_name: str | None = None) -> ModelProfile:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    selected_name = profile_name or data.get("judge", {}).get("profile")
    if not selected_name:
        raise ValueError(
            "LLM judge evaluation requires --judge-profile or judge.profile in models.yaml"
        )
    profiles = load_model_profiles(config_path)
    try:
        return profiles[selected_name]
    except KeyError as exc:
        raise ValueError(f"Unknown judge profile {selected_name!r}") from exc
