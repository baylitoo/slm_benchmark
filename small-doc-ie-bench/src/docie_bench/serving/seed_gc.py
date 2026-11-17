"""Reclaim seed-download leftovers on the node that downloaded them.

A successful seed cleans up after itself; a SIGKILL mid-download leaves the
multi-GB ``models/.hf-downloads/<name>`` staging dir and the run's
``seed-progress`` sidecar behind. Both live on the serving node's own volume,
so the sweep runs in that node's reconciler, not on the central worker.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from docie_bench.serving.paths import model_store_root
from docie_bench.serving.seed_progress import prune_stale

# A staging dir is only resumed when the SAME store name is re-seeded, and a
# week-old .part costs more on disk than restarting the download does.
SEED_LEFTOVER_MAX_AGE_S = 7 * 86400.0


def gc_seed_leftovers(
    *,
    max_age_s: float = SEED_LEFTOVER_MAX_AGE_S,
    staging_root: Path | None = None,
    now: float | None = None,
) -> dict[str, int]:
    """Prune stale seed staging dirs and progress sidecars (best-effort)."""
    root = staging_root if staging_root is not None else model_store_root() / ".hf-downloads"
    current = time.time() if now is None else now
    removed_staging = 0
    try:
        children = list(root.iterdir())
    except OSError:
        children = []
    for child in children:
        try:
            # Newest mtime under the dir: the dir's own mtime misses writes
            # into an existing .part file.
            stamps = [child.stat().st_mtime]
            stamps += [p.stat().st_mtime for p in child.rglob("*")]
        except OSError:
            continue
        if current - max(stamps) > max_age_s:
            shutil.rmtree(child, ignore_errors=True)
            removed_staging += 1
    removed_progress = prune_stale(max_age_s=max_age_s)
    return {"deleted_seed_staging": removed_staging, "deleted_seed_progress": removed_progress}


__all__ = ["SEED_LEFTOVER_MAX_AGE_S", "gc_seed_leftovers"]
