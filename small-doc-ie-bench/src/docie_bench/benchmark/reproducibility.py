from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from docie_bench.llm.model_profiles import ModelProfile

MANIFEST_VERSION = 1
TERMINAL_TASK_STATES = frozenset({"completed", "failed"})


class ResumeDriftError(ValueError):
    """Raised when an existing run cannot safely resume with the requested inputs."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def profile_snapshot(profile: ModelProfile) -> dict[str, Any]:
    if is_dataclass(profile) and not isinstance(profile, type):
        snapshot = asdict(profile)
    else:
        snapshot = vars(profile).copy()
    snapshot.pop("api_key", None)
    snapshot["stop_sequences"] = list(getattr(profile, "stop_sequences", ()))
    base_url = getattr(profile, "base_url", None)
    if base_url:
        parsed_url = urlsplit(base_url)
        if parsed_url.username or parsed_url.password:
            hostname = parsed_url.hostname or ""
            netloc = f"{hostname}:{parsed_url.port}" if parsed_url.port else hostname
            snapshot["base_url"] = urlunsplit(
                (parsed_url.scheme, netloc, parsed_url.path, parsed_url.query, parsed_url.fragment)
            )
    return snapshot


def git_snapshot(cwd: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.strip()  # noqa: S607
        except (OSError, subprocess.CalledProcessError):
            return ""

    sha = run("rev-parse", "HEAD") or None
    status = run("status", "--porcelain=v1", "--untracked-files=no")
    untracked = [
        name
        for name in run("ls-files", "--others", "--exclude-standard").splitlines()
        if name.startswith(("src/", "tests/", "configs/", "scripts/"))
    ]
    diff = run("diff", "--binary", "HEAD")
    untracked_hashes = {name: hash_file(cwd / name) for name in untracked if (cwd / name).is_file()}
    dirty_files = [*status.splitlines(), *(f"?? {name}" for name in untracked)]
    return {
        "sha": sha,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "working_tree_hash": stable_hash({"diff": diff, "untracked": untracked_hashes})
        if dirty_files
        else None,
    }


def dependency_snapshot() -> dict[str, str]:
    distributions = sorted(
        (
            (dist.metadata["Name"] or "unknown", dist.version)
            for dist in importlib.metadata.distributions()
        ),
        key=lambda item: item[0].lower(),
    )
    return dict(distributions)


def system_snapshot() -> dict[str, Any]:
    memory_bytes = None
    try:
        import psutil

        memory_bytes = psutil.virtual_memory().total
    except ImportError:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any, *, indent: int | None = None) -> None:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent, default=str)
    atomic_write_text(path, content + "\n")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError as exc:
            raise FileExistsError(f"Run manifest is immutable and already exists: {path}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(row) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl_recover(path: Path) -> list[dict[str, Any]]:
    """Load JSONL, repairing only a truncated final record.

    Invalid records before the final non-empty line indicate corruption and are never ignored.
    """
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        if not line.strip():
            valid_bytes += len(line)
            continue
        try:
            rows.append(json.loads(line))
            valid_bytes += len(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_final = index == len(lines) - 1
            if not is_final:
                raise ValueError(f"Corrupt JSONL record {index + 1} in {path}") from exc
            with path.open("r+b") as handle:
                handle.truncate(valid_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            break
    return rows


def validate_resume_manifest(
    existing: dict[str, Any],
    requested: dict[str, Any],
    requested_task_ids: Iterable[str],
) -> list[str]:
    drift: list[str] = []
    if existing.get("manifest_version") != MANIFEST_VERSION:
        drift.append(
            f"manifest_version: {existing.get('manifest_version')!r} -> {MANIFEST_VERSION!r}"
        )
    if existing.get("input_fingerprint") != requested.get("input_fingerprint"):
        old_inputs = existing.get("inputs", {})
        new_inputs = requested.get("inputs", {})
        for key in sorted(set(old_inputs) | set(new_inputs)):
            if stable_hash(old_inputs.get(key)) != stable_hash(new_inputs.get(key)):
                drift.append(f"inputs.{key} changed")
    existing_ids = set(existing.get("task_ids", []))
    new_ids = set(requested_task_ids)
    if existing_ids != new_ids:
        drift.append(
            f"task set changed (added={len(new_ids - existing_ids)}, "
            f"removed={len(existing_ids - new_ids)})"
        )
    if drift:
        raise ResumeDriftError("Cannot resume because run inputs drifted:\n- " + "\n- ".join(drift))
    warnings: list[str] = []
    if stable_hash(existing.get("environment")) != stable_hash(requested.get("environment")):
        warnings.append("Execution environment changed since the run was created")
    old_concurrency = existing.get("invocation", {}).get("concurrency")
    new_concurrency = requested.get("invocation", {}).get("concurrency")
    if old_concurrency != new_concurrency:
        warnings.append(f"Concurrency changed from {old_concurrency} to {new_concurrency}")
    return warnings
