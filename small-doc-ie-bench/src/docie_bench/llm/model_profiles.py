from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Profile kinds the gateway can dispatch. "passthrough" proxies to an
# OpenAI-compatible upstream (the default — every existing profile). Solution
# kinds are handled by a local adapter (see docie_bench.serving.solutions).
VALID_PROFILE_KINDS = frozenset({"passthrough", "ocr", "pipeline"})


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
    capability_discovery: str = "disabled"
    retry_max_attempts: int = 2
    retry_backoff_base_seconds: float = 1.0
    retry_backoff_max_seconds: float = 8.0
    retry_jitter_seconds: float = 0.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_seconds: float = 30.0
    max_concurrency: int = 4
    queue_limit: int = 32
    queue_timeout_seconds: float = 30.0
    # Solution routing. "passthrough" (default) proxies to base_url; other kinds
    # are served by a local adapter. `options` carries adapter-specific config.
    kind: str = "passthrough"
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in VALID_PROFILE_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(VALID_PROFILE_KINDS)}, got {self.kind!r}"
            )
        if self.capability_discovery not in {"disabled", "optional", "required"}:
            raise ValueError("capability_discovery must be disabled, optional, or required")
        for name in (
            "retry_max_attempts",
            "circuit_breaker_failure_threshold",
            "max_concurrency",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.queue_limit < 0:
            raise ValueError("queue_limit must be at least 0")
        for name in (
            "retry_backoff_base_seconds",
            "retry_backoff_max_seconds",
            "retry_jitter_seconds",
            "circuit_breaker_reset_seconds",
            "queue_timeout_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


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
        kind = cfg.get("kind", "passthrough")
        model = cfg.get("model", "")
        base_url = str(cfg.get("base_url", "")).rstrip("/")
        # Passthrough profiles still require an upstream; solution kinds (ocr/pipeline)
        # are served by a local adapter and may omit model/base_url.
        if kind == "passthrough" and (not model or not base_url):
            raise ValueError(
                f"profile {name!r}: 'model' and 'base_url' are required for passthrough profiles"
            )
        profiles[name] = ModelProfile(
            name=name,
            model=model,
            base_url=base_url,
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
            capability_discovery=cfg.get("capability_discovery", "disabled"),
            retry_max_attempts=int(cfg.get("retry_max_attempts", 2)),
            retry_backoff_base_seconds=float(cfg.get("retry_backoff_base_seconds", 1)),
            retry_backoff_max_seconds=float(cfg.get("retry_backoff_max_seconds", 8)),
            retry_jitter_seconds=float(cfg.get("retry_jitter_seconds", 0)),
            circuit_breaker_failure_threshold=int(
                cfg.get("circuit_breaker_failure_threshold", 5)
            ),
            circuit_breaker_reset_seconds=float(cfg.get("circuit_breaker_reset_seconds", 30)),
            max_concurrency=int(cfg.get("max_concurrency", 4)),
            queue_limit=int(cfg.get("queue_limit", 32)),
            queue_timeout_seconds=float(cfg.get("queue_timeout_seconds", 30)),
            kind=kind,
            options=dict(cfg.get("options") or {}),
        )
    return profiles


class ProfileWriteError(ValueError):
    """A ``kind: pipeline`` profile could not be validated or written."""


class ProfileConflictError(ProfileWriteError):
    """The profile name already exists in the target models.yaml."""


def add_pipeline_profile(
    path: str | Path,
    *,
    name: str,
    extractor: str,
    ocr_backend: str | None = None,
    ocr_model: str | None = None,
    language: str | None = None,
) -> ModelProfile:
    """Append a ``kind: pipeline`` (OCR->LLM) profile to the models.yaml at `path`.

    Text-spliced rather than a full ``yaml.safe_load``/``yaml.safe_dump`` round-trip
    of the whole file. ``save_registry`` (benchmark/registry.py) can safely do a full
    round-trip because its target (``data/datasets.yaml``) is machine-only data with
    no comments; ``configs/models.yaml`` is a hand-documented file (see its own header
    comment and the per-profile notes throughout) and a full re-dump would silently
    discard every one of them. This locates the top-level ``profiles:`` key by text
    search, inserts the new entry right before the next top-level (column-0,
    non-comment) line -- or at EOF if none follows, which is this file's current shape
    -- and writes atomically via temp file + ``os.replace``, the one part of
    ``save_registry``'s pattern that does carry over unchanged here.

    Create-only: raises ``ProfileConflictError`` if `name` already exists. Updating an
    existing entry in place is deliberately out of scope -- text-splicing a REPLACE is
    materially harder than an INSERT (matching the existing block's exact line range
    without a real YAML round-trip) and isn't needed for the "author a new pipeline
    profile" slice this exists for.

    Validates the same rules ``serving.solutions.PipelineSolution`` enforces at request
    time (kept in sync manually -- there is no shared source of truth for these), so a
    misconfigured profile fails at authoring time rather than mid-benchmark: exactly one
    of `ocr_backend`/`ocr_model` (never both, never neither -- the solution silently
    prefers `ocr_model` if both are set, which would make a profile's config lie about
    what actually runs); `extractor` must name an existing ``kind: passthrough`` profile;
    `ocr_model` (if given) must name an existing ``kind: passthrough`` profile with
    ``vision: true``; `ocr_backend` (if given) must be a real OCR backend name
    (validated via ``get_ocr_backend``, the exact fail-fast check PipelineSolution
    itself does at construction).
    """
    config_path = Path(path)
    existing = load_model_profiles(config_path) if config_path.exists() else {}

    name = name.strip()
    if not name:
        raise ProfileWriteError("name is required")
    if name in existing:
        raise ProfileConflictError(f"profile {name!r} already exists")

    extractor = extractor.strip()
    if not extractor:
        raise ProfileWriteError("extractor is required")
    if bool(ocr_backend) == bool(ocr_model):
        raise ProfileWriteError("provide exactly one of ocr_backend or ocr_model")

    extractor_profile = existing.get(extractor)
    if extractor_profile is None:
        raise ProfileWriteError(f"extractor profile {extractor!r} is not configured")
    if extractor_profile.kind != "passthrough":
        raise ProfileWriteError(f"extractor {extractor!r} must be a passthrough LLM profile")

    options: dict[str, Any] = {"extractor": extractor}
    if ocr_model:
        vision_profile = existing.get(ocr_model)
        if vision_profile is None:
            raise ProfileWriteError(f"ocr_model profile {ocr_model!r} is not configured")
        if vision_profile.kind != "passthrough" or not vision_profile.vision:
            raise ProfileWriteError(
                f"ocr_model {ocr_model!r} must be a passthrough profile with vision=true"
            )
        options["ocr_model"] = ocr_model
    else:
        # Local import: keep the OCR backend stack (tesseract/paddleocr) an optional
        # dependency for callers that only need to read/parse profiles.
        from docie_bench.ocr.factory import get_ocr_backend

        try:
            get_ocr_backend(str(ocr_backend), language=language)  # fail fast on a bad name
        except ValueError as exc:
            raise ProfileWriteError(str(exc)) from exc
        options["ocr_backend"] = ocr_backend
    if language:
        options["language"] = language

    _splice_profile(config_path, name, {"kind": "pipeline", "options": options})
    return load_model_profiles(config_path)[name]


def _splice_profile(path: Path, name: str, entry: dict[str, Any]) -> None:
    """Insert one ``name: entry`` mapping under `path`'s top-level ``profiles:`` key.

    See ``add_pipeline_profile`` for why this splices text instead of a full
    safe_load/safe_dump round-trip.
    """
    text = path.read_text(encoding="utf-8") if path.exists() else "profiles:\n"
    if not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)

    profiles_idx = next(
        (i for i, line in enumerate(lines) if line.rstrip("\n") == "profiles:"), None
    )
    if profiles_idx is None:
        raise ProfileWriteError(f"{path} has no top-level 'profiles:' key")

    insert_at = len(lines)
    for i in range(profiles_idx + 1, len(lines)):
        stripped = lines[i].rstrip("\n")
        # Next top-level (column-0) key ends the profiles block; comments and blank
        # lines don't count (they appear at column 0 throughout the file's prose).
        if stripped and not stripped.startswith((" ", "\t", "#")):
            insert_at = i
            break

    block_yaml = yaml.safe_dump({name: entry}, sort_keys=False, allow_unicode=True)
    indented = "".join(
        f"  {line}" if line.strip() else line for line in block_yaml.splitlines(keepends=True)
    )
    new_block = "\n" + indented

    new_text = "".join(lines[:insert_at]) + new_block + "".join(lines[insert_at:])

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(new_text, encoding="utf-8")
    tmp_path.replace(path)


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
