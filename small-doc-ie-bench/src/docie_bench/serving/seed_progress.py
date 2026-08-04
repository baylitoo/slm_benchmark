"""Pollable seed-download progress — a realtime-independent fallback.

The seed job streams download progress to the run's realtime ``progress`` topic
(a live percentage bar in the Studio). When realtime is unavailable (no gateway
token, a dropped subscription) the UI degrades to polling the run STATUS, which
is coarse — "running", no percentage. This sidecar closes that gap: the seed job
also writes the latest progress to a tiny JSON file on the shared serving volume,
keyed by the run's channel, and the api serves it so the polling fallback can
render the SAME percentage bar. Best-effort throughout — a disk hiccup must never
fail a download.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _home() -> Path:
    return Path(
        os.environ.get(
            "DOCIE_SERVING_HOME",
            Path.home() / ".local" / "share" / "docie-bench" / "serving",
        )
    )


def _progress_dir() -> Path:
    return _home() / "seed-progress"


def _progress_path(channel: str) -> Path | None:
    """The sidecar path for ``channel``, or None when the (slugified) channel
    would escape the progress directory."""
    slug = _SAFE.sub("_", channel).strip("_")
    if not slug:
        return None
    base = _progress_dir().resolve()
    path = (base / f"{slug}.json").resolve()
    if base not in path.parents:
        return None
    return path


def write_progress(channel: str, payload: dict[str, Any]) -> None:
    """Persist the latest progress for ``channel`` (atomic; best-effort)."""
    path = _progress_path(channel)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def read_progress(channel: str) -> dict[str, Any] | None:
    """The last-written progress for ``channel`` (None when absent/unreadable)."""
    path = _progress_path(channel)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def clear_progress(channel: str) -> None:
    """Remove the sidecar once a seed settles (best-effort)."""
    path = _progress_path(channel)
    if path is None:
        return
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
