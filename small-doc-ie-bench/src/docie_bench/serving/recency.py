"""Per-deployment recency sidecars: ``<serving-home>/recency/<name>``.

``last_served`` must be written by the code path that actually serves traffic
— the scaled ``worker`` extract path — but ``deployments.json`` has exactly ONE
writer (the single-replica ``serving`` service, P1). Resolution (design doc §1,
fix #5): each extract writes a tiny single-key sidecar file per deployment
(atomic ``os.replace``), and the serving-side reconciler folds the sidecars
into ``DeploymentRecord.last_served`` each cycle. ``last_served`` is a
monotonic max-timestamp, so last-write-wins between concurrent workers is the
CORRECT semantics and a lost update is harmless (an idle unload slips one
cycle at worst).

Everything here is best-effort: recency is an optimization input (LRU/idle in
PR-4), so a disk hiccup must never fail an extraction or a reconcile cycle.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _serving_home() -> Path:
    # Must match control_plane.from_defaults / profile_resolver so every
    # container reads and writes the SAME shared serving-state volume.
    return Path(
        os.environ.get(
            "DOCIE_SERVING_HOME",
            Path.home() / ".local" / "share" / "docie-bench" / "serving",
        )
    )


def recency_dir(home: Path | None = None) -> Path:
    return (home if home is not None else _serving_home()) / "recency"


def _filename(name: str) -> str:
    """A filesystem-safe sidecar filename for a deployment name.

    Deployment names are aliases and normally already safe; anything else
    (path separators, ``:`` on Windows, ...) is mapped to a stable digest so a
    hostile or odd name can never traverse out of the recency dir. Both the
    stamper and the folder use this same mapping, so they always agree.
    """
    if _SAFE_NAME.match(name):
        return name
    return "sha256-" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def stamp(name: str, *, timestamp: float | None = None, home: Path | None = None) -> None:
    """Record "deployment ``name`` served a request now" (best-effort, atomic).

    Single-key write via temp-file + ``os.replace``: concurrent stampers
    last-write-wins, which is correct for a monotonic max-timestamp. Never
    raises — recency must not fail the extraction that produced it.
    """
    try:
        directory = recency_dir(home)
        directory.mkdir(parents=True, exist_ok=True)
        value = timestamp if timestamp is not None else time.time()
        target = directory / _filename(name)
        temporary = directory / f".{target.name}.tmp"
        temporary.write_text(f"{value:.6f}", encoding="utf-8")
        os.replace(temporary, target)
    except OSError:
        logger.warning("could not stamp recency for deployment %r", name, exc_info=True)


def read_for(names: list[str], *, home: Path | None = None) -> dict[str, float]:
    """Read the sidecar timestamps for ``names`` (missing/corrupt => absent).

    Keyed by deployment name (the reconciler folds by record name); reads only
    the sidecars for known records, so stray files are simply ignored.
    """
    directory = recency_dir(home)
    result: dict[str, float] = {}
    for name in names:
        path = directory / _filename(name)
        try:
            result[name] = float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return result


__all__ = ["stamp", "read_for", "recency_dir"]
