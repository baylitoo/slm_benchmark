from __future__ import annotations

import os
import re
import uuid
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
    # `or {}`, not `.get("profiles", {})`: a `profiles:` key present with nothing
    # under it (e.g. every profile was just deleted from the file) parses to
    # {"profiles": None}, and `.get` returns that None verbatim since the key
    # exists -- only `or {}` degrades it to the same empty-registry behavior as a
    # missing `profiles:` key entirely, instead of an AttributeError on `.items()`.
    for name, cfg in (data.get("profiles") or {}).items():
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


class ProfileNotFoundError(ProfileWriteError):
    """No profile named `name` exists in the target models.yaml."""


_VALID_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# YAML 1.1 (PyYAML's implicit resolver) coerces these bare scalars to bool/null, not
# str -- load_model_profiles would key the parsed profile by True/False/None instead
# of the string a caller typed, silently defeating the "name already exists" check
# below (a second, distinct `name:` entry would splice in right alongside the first).
# Case-insensitive: yes/Yes/YES all resolve the same way.
_YAML_RESERVED_WORDS = frozenset(
    {"y", "n", "yes", "no", "true", "false", "on", "off", "null", "~"}
)


def _validate_new_profile_name(name: str, existing: Mapping[str, ModelProfile]) -> str:
    """Shared create-only name validation for ``add_pipeline_profile``/``add_ocr_profile``.

    Strips, checks non-empty, charset, YAML-reserved-word (see the module comment
    above ``_YAML_RESERVED_WORDS`` for why), and collision against `existing`. Returns
    the stripped name on success. Both writers need exactly this and nothing more —
    the divergence between a pipeline profile (extractor + exactly-one-of
    ocr_backend/ocr_model) and an ocr profile (backend only, no LLM stage) starts
    after the name is settled, so only this prefix is shared.
    """
    name = name.strip()
    if not name:
        raise ProfileWriteError("name is required")
    if not _VALID_PROFILE_NAME.fullmatch(name):
        raise ProfileWriteError(
            "name must start with a letter or digit and contain only letters, "
            "digits, '_', '.', or '-'"
        )
    if name.lower() in _YAML_RESERVED_WORDS:
        raise ProfileWriteError(
            f"name {name!r} is a YAML boolean/null literal and would not round-trip "
            "as a string key"
        )
    if name in existing:
        raise ProfileConflictError(f"profile {name!r} already exists")
    return name


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
    materially harder than either an INSERT or a DELETE (see ``delete_pipeline_profile``,
    which *is* provided) because a REPLACE has to regenerate valid YAML for a block whose
    existing shape (nested ``options:``, key order, any hand-added comments) isn't known
    ahead of time, where INSERT only ever has to place well-formed YAML this function
    itself generated and DELETE only ever has to find a block's start/end and remove it.

    Validates rules that are AT LEAST as strict as what ``serving.solutions.
    PipelineSolution`` enforces at request time (kept in sync manually -- there is no
    shared source of truth for these), so a misconfigured profile fails at authoring
    time rather than mid-benchmark: exactly one of `ocr_backend`/`ocr_model` (never
    both, never neither -- the solution silently prefers `ocr_model` if both are set,
    which would make a profile's config lie about what actually runs); `extractor` must
    name an existing ``kind: passthrough`` profile; `ocr_model` (if given) must name an
    existing ``kind: passthrough`` profile with ``vision: true``; `ocr_backend` (if
    given) must be a real OCR backend name (validated via ``get_ocr_backend``, the
    exact fail-fast check PipelineSolution itself does at construction). NOTE: the
    ``vision: true`` requirement on `ocr_model` is intentionally STRICTER than
    PipelineSolution's own runtime check, which does not currently verify `.vision` --
    this rejects some `ocr_model` choices that would in fact run today via a
    hand-edited entry. Erring stricter at write-time, not a parity bug.
    """
    config_path = Path(path)
    existing = load_model_profiles(config_path) if config_path.exists() else {}
    name = _validate_new_profile_name(name, existing)

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
    try:
        return load_model_profiles(config_path)[name]
    except KeyError as exc:
        # Two concurrent writers can race the read-modify-write below (no file lock);
        # the loser's own insert can be clobbered by the winner's replace() landing
        # after it. Surface as a clear, retryable error instead of a raw KeyError/500.
        raise ProfileWriteError(
            f"profile {name!r} was written but is missing from {config_path} on "
            "re-read -- likely a concurrent write to the same file; retry"
        ) from exc


def add_ocr_profile(
    path: str | Path,
    *,
    name: str,
    backend: str,
    language: str | None = None,
) -> ModelProfile:
    """Append a ``kind: ocr`` (OCR-only, no LLM stage) profile to the models.yaml at `path`.

    The narrower sibling of ``add_pipeline_profile`` -- #188 authored `kind: pipeline`
    profiles from the Studio UI and explicitly deferred `kind: ocr` as "a related but
    separate gap". Unlike a pipeline profile, an ocr profile has no extractor and no
    choice between a built-in backend and a vision-model transcriber: ``serving.
    solutions.OcrSolution`` reads only ``options.backend`` (default "tesseract") and
    optional ``options.language``, and returns the backend's raw transcribed text as
    the completion -- there is no JSON extraction stage. That means a ``kind: ocr``
    profile is not a schema-scored benchmark model in its own right (the benchmark
    runner expects a completion it can parse against an extraction schema, which
    OCR-only output isn't); it's reachable directly through the gateway by name
    (``/v1/chat/completions`` with ``model=<name>``), or reusable as a pipeline
    profile's `ocr_backend` choice made durable under its own name. NOTE: no Studio
    UI surface (Playground, Benchmark) invokes a `kind: ocr` profile yet -- creating
    one here does not by itself make it reachable from anywhere in the Studio UI.

    IMPORTANT: the options key is ``backend``, not ``ocr_backend`` -- that's
    `PipelineSolution`'s key for the same concept. ``OcrSolution.__init__`` reads only
    ``options.get("backend", ...)`` with no fallback, so writing ``ocr_backend`` here
    would silently run the tesseract default instead of what the caller asked for.

    Shares ``_validate_new_profile_name`` and ``_splice_profile`` with
    ``add_pipeline_profile`` (see their docstrings for why validation is create-only
    and the file is text-spliced rather than round-tripped); the backend-name
    validation itself mirrors ``OcrSolution.__init__``'s own fail-fast ``get_ocr_backend``
    call so a bad name is rejected at authoring time, not mid-request.
    """
    config_path = Path(path)
    existing = load_model_profiles(config_path) if config_path.exists() else {}
    name = _validate_new_profile_name(name, existing)

    backend = backend.strip() if backend else ""
    if not backend:
        raise ProfileWriteError("backend is required")

    # Local import: keep the OCR backend stack (tesseract/paddleocr) an optional
    # dependency for callers that only need to read/parse profiles.
    from docie_bench.ocr.factory import get_ocr_backend

    try:
        get_ocr_backend(backend, language=language)  # fail fast on a bad name
    except ValueError as exc:
        raise ProfileWriteError(str(exc)) from exc

    options: dict[str, Any] = {"backend": backend}
    if language:
        options["language"] = language

    _splice_profile(config_path, name, {"kind": "ocr", "options": options})
    try:
        return load_model_profiles(config_path)[name]
    except KeyError as exc:
        # Same concurrent-write race as add_pipeline_profile's re-read -- see its
        # comment above for the mechanics.
        raise ProfileWriteError(
            f"profile {name!r} was written but is missing from {config_path} on "
            "re-read -- likely a concurrent write to the same file; retry"
        ) from exc


def _splice_profile(path: Path, name: str, entry: dict[str, Any]) -> None:
    """Insert one ``name: entry`` mapping under `path`'s top-level ``profiles:`` key.

    See ``add_pipeline_profile`` for why this splices text instead of a full
    safe_load/safe_dump round-trip.

    KNOWN LIMITATION, not fixed here: comments don't count as block-enders (so the
    search continues past them looking for the next real top-level key), which means
    a comment written to document the section AFTER `profiles:` (e.g. "# end of
    profiles, judge config below") can end up sitting ABOVE the newly-inserted entry
    instead of below it -- still valid, correctly-nested YAML, just a misplaced
    comment. Narrowing this (stop at the first top-level comment unless it's
    immediately followed by more indented profile content) is a real improvement but
    is deferred -- low frequency, cosmetic only, and getting the heuristic exactly
    right is more risk than this fix is worth right now.
    """
    text = path.read_text(encoding="utf-8") if path.exists() else "profiles:\n"
    if not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)

    profiles_idx = next(
        (i for i, line in enumerate(lines) if line.rstrip() == "profiles:"), None
    )
    if profiles_idx is None:
        raise ProfileWriteError(f"{path} has no top-level 'profiles:' key")

    insert_at = len(lines)
    for i in range(profiles_idx + 1, len(lines)):
        stripped = lines[i].rstrip()
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
    # Per-call-unique suffix: a fixed name (e.g. "models.yaml.tmp") would let two
    # concurrent callers stage into the SAME temp file before either calls replace(),
    # which can silently lose one writer's insert even though each individual
    # write_text()/replace() pair is atomic on its own.
    tmp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(new_text, encoding="utf-8")
    tmp_path.replace(path)


def delete_pipeline_profile(path: str | Path, *, name: str) -> None:
    """Remove an existing ``kind: pipeline`` or ``kind: ocr`` profile from the
    models.yaml at `path`.

    The DELETE counterpart to ``add_pipeline_profile``'s create-only INSERT -- not an
    UPDATE (see that function's own docstring for why an in-place REPLACE is a harder
    text-splice problem than either of these). Text-spliced for the same reason
    ``add_pipeline_profile`` is: ``configs/models.yaml`` is hand-documented and a full
    ``yaml.safe_load``/``yaml.safe_dump`` round-trip would silently discard every
    comment in it.

    ``passthrough`` profiles can't be deleted through this path: a live deployment, or
    another pipeline profile's ``extractor``/``ocr_model``, can reference one by name,
    and this module has no way to check for -- let alone warn about -- that at delete
    time. Restricting to ``pipeline``/``ocr`` keeps the blast radius to profiles this
    API itself created and fully owns the lifecycle of.

    Raises ``ProfileNotFoundError`` if `name` doesn't exist, or ``ProfileWriteError``
    if it exists but isn't a ``pipeline``/``ocr`` profile.
    """
    config_path = Path(path)
    name = name.strip()
    existing = load_model_profiles(config_path) if config_path.exists() else {}

    profile = existing.get(name)
    if profile is None:
        raise ProfileNotFoundError(f"profile {name!r} does not exist")
    if profile.kind not in {"pipeline", "ocr"}:
        raise ProfileWriteError(
            f"profile {name!r} has kind={profile.kind!r}; only pipeline/ocr profiles "
            "can be deleted through this API"
        )

    _remove_profile_block(config_path, name)

    reloaded = load_model_profiles(config_path)
    if name in reloaded:
        # Same lost-update race add_pipeline_profile guards against (no file lock):
        # a concurrent writer's os.replace() landing after ours can re-introduce the
        # block we just removed if their read-modify-write started from a copy of
        # the file that still had it. Surface as a clear, retryable error rather
        # than silently reporting success on a delete that didn't actually stick.
        raise ProfileWriteError(
            f"profile {name!r} was deleted but is still present in {config_path} on "
            "re-read -- likely a concurrent write to the same file; retry"
        )


def _remove_profile_block(path: Path, name: str) -> None:
    """Remove one ``name: ...`` entry from `path`'s top-level ``profiles:`` key.

    See ``delete_pipeline_profile`` for why this splices text instead of a full
    safe_load/safe_dump round-trip.

    Deliberately uses a STRICTER block-end rule than ``_splice_profile``'s own insert-
    side scan. That scan treats comments as non-terminating (see its KNOWN LIMITATION
    docstring) because misplacing a *new* block relative to an existing comment is only
    ever cosmetic. Deleting is a different risk: this file's real per-profile
    documentation comments sit at the SAME 2-space indent as the profile keys
    themselves (see configs/models.yaml -- e.g. the six-line comment directly above
    ``nuextract3:``), not at column 0, so reusing the insert-side "comments don't
    count" rule here would silently swallow a NEIGHBORING profile's documentation into
    the deleted range whenever a comment happens to fall in the gap right after the
    target block. The rule below instead stops at the first non-blank line that isn't
    more indented than the profile key itself (indentation <= 2) -- whether that line
    is the next profile's key, a comment (documenting that next profile, or the next
    top-level section), or the next top-level section itself -- so a delete can never
    remove a comment it doesn't unambiguously own. Blank lines are still swallowed
    (matching the insert side) so a mid-file delete doesn't leave a stray double-blank
    gap; only at EOF (no next sibling to swallow the leading blank forward into) does
    this also walk backward over blank lines immediately before the block, so a
    create-then-delete of the last profile in the file doesn't accumulate one extra
    blank line per cycle.

    Refuses (via ``ProfileWriteError``, before writing anything) rather than guesses
    if `name` doesn't appear under ``profiles:`` at the expected 2-space indent, or
    appears more than once -- e.g. a hand-edited file with a duplicate top-level key,
    where ``load_model_profiles`` (plain ``yaml.safe_load``) silently resolves to
    whichever occurrence PyYAML parses last, but this text scan has no principled way
    to know which physical block that was. Without this guard, a duplicate key could
    let a `kind: passthrough` block get deleted out from under a caller who validated
    a *different*, later `kind: pipeline` occurrence of the same name.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    profiles_idx = next(
        (i for i, line in enumerate(lines) if line.rstrip() == "profiles:"), None
    )
    if profiles_idx is None:
        raise ProfileWriteError(f"{path} has no top-level 'profiles:' key")

    key_pattern = re.compile(r"^  " + re.escape(name) + r":(\s|$)")
    matches: list[int] = []
    section_end = len(lines)
    for i in range(profiles_idx + 1, len(lines)):
        stripped = lines[i].rstrip()
        # Same section-end rule as _splice_profile: next top-level (column-0) key
        # ends the profiles block; comments and blank lines don't count there.
        if stripped and not stripped.startswith((" ", "\t", "#")):
            section_end = i
            break
        if key_pattern.match(lines[i]):
            matches.append(i)

    if not matches:
        raise ProfileWriteError(f"profile {name!r} not found under 'profiles:' in {path}")
    if len(matches) > 1:
        raise ProfileWriteError(
            f"profile {name!r} appears {len(matches)} times under 'profiles:' in "
            f"{path}; refusing to guess which block to remove"
        )
    block_start = matches[0]

    block_end = section_end
    for i in range(block_start + 1, section_end):
        line = lines[i]
        if not line.strip():
            continue  # blank line inside the block -- swallow, doesn't end it
        indent = len(line) - len(line.lstrip(" "))
        if indent <= 2:
            block_end = i
            break

    if block_end == len(lines):
        # EOF only: nothing after the block to swallow the leading blank forward
        # into (the mid-file case above already does that via block_end), so walk
        # the start back over it here instead.
        while block_start > profiles_idx + 1 and not lines[block_start - 1].strip():
            block_start -= 1

    new_text = "".join(lines[:block_start]) + "".join(lines[block_end:])

    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-call-unique suffix -- same race this guards against in _splice_profile.
    tmp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
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
