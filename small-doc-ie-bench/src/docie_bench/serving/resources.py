"""Resource tracker for the single-replica ``serving`` service (PR-2).

Measures what the sizing math needs and nothing else: node total/free RAM,
per-runtime-process RSS, and a calibrated per-model memory footprint. All
measurement happens **inside the serving container** (design doc §2) — a
number measured in the api container would describe the wrong cgroup — and is
published to Postgres by the reconciler so the api/Studio read a snapshot, not
a lie.

Node memory — and how Docker Desktop lies (design doc §2). Inside a container
``psutil.virtual_memory()`` reads the WSL2 VM's ``/proc/meminfo``: an elastic
total that says nothing about what THIS container may use. cgroup v2 numbers
(``/sys/fs/cgroup/memory.max`` / ``memory.current``) are authoritative — but
only when compose actually sets a limit; an unlimited cgroup reads the ``max``
sentinel and the only honest fallback is the VM view. So:

* cgroup-v2-first: a real ``memory.max`` limit wins, ``free = max - current``.
* ``memory.max == "max"`` (or no cgroup at all) => psutil VM fallback.
* every reading carries ``source: "cgroup" | "vm"`` so the UI can badge a VM
  number as soft instead of presenting it as truth.

Per-model footprint. The prediction reuses the planner's FORMULA but never its
input path (design doc fix #6): weights come from ``ModelStoreEntry.size_bytes``
or an on-disk ``stat`` of the GGUF — **never** the registry, whose ``plan()``
raises on store models::

    predicted = weights x quant_factor + kv_per_token x ctx x n_parallel + overhead

Calibration against reality: llama.cpp mmaps the GGUF, so RSS is low right
after load and climbs as pages fault in — calibrating from fresh RSS
under-counts and later over-commits the node. The tracker therefore records a
model's footprint only from STEADY-STATE samples (a run of consecutive
``hot`` observations whose RSS has stopped moving) and persists them per model
in sidecar files on the serving-state volume, so sizing improves over time and
survives restarts. The working number is always
``footprint = max(observed_steady_rss, predicted)`` — trust the measurement
once there is one, stay conservative about the ramp.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# Default cgroup-v2 controller mount inside a container.
CGROUP_V2_ROOT = Path("/sys/fs/cgroup")

# The cgroup-v2 sentinel for "no limit configured": the number is meaningless
# for sizing, so the reader falls back to the VM view (flagged as soft).
_CGROUP_UNLIMITED_SENTINEL = "max"

# KV-cache bytes per context token per parallel slot. This is exactly the
# planner's formula constant (planner.py::_estimate_memory_gb prices 0.25 GiB
# per 4096-token context per concurrent slot => 65536 bytes/token) — reuse the
# FORMULA, not the plumbing (design doc §2/fix #6).
KV_CACHE_BYTES_PER_TOKEN = 65_536

# Fixed llama-server runtime slab on top of weights + KV (arena, buffers;
# design doc §2 brackets it 0.3-0.5 GB — take the conservative top end).
RUNTIME_OVERHEAD_BYTES = 512 * 1024 * 1024

# llama-server's default --ctx-size when a deployment specifies none.
DEFAULT_CONTEXT_LENGTH = 4096

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _serving_home() -> Path:
    # Must match recency.py / control_plane.from_defaults so every process on
    # the shared serving-state volume reads and writes the SAME sidecars.
    return Path(
        os.environ.get(
            "DOCIE_SERVING_HOME",
            Path.home() / ".local" / "share" / "docie-bench" / "serving",
        )
    )


# --------------------------------------------------------------- node memory


@dataclass(frozen=True)
class NodeMemory:
    """One node RAM reading, honest about where the numbers came from.

    ``reclaimable_bytes`` flags the page-cache adjustment applied to
    ``free_bytes`` (cgroup readings only; see :func:`read_cgroup_memory`): a
    non-zero value says "free was computed against the WORKING SET, not raw
    ``memory.current``" — 0 means no adjustment was applicable/measurable.
    """

    total_bytes: int
    free_bytes: int
    source: str  # "cgroup" (authoritative limit) | "vm" (soft: VM/host view)
    reclaimable_bytes: int = 0  # inactive_file page cache excluded from "used"


def _read_inactive_file_bytes(root: Path) -> int:
    """``inactive_file`` from ``memory.stat`` (0 when missing/corrupt).

    Best-effort by design: an authoritative ``memory.max`` limit must never be
    discarded because the reclaim adjustment could not be measured.
    """
    try:
        text = (root / "memory.stat").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "inactive_file":
            try:
                return max(int(parts[1]), 0)
            except ValueError:
                return 0
    return 0


def read_cgroup_memory(root: Path = CGROUP_V2_ROOT) -> NodeMemory | None:
    """cgroup-v2 memory reading, or ``None`` when it cannot be authoritative.

    ``None`` means: no cgroup-v2 files (not in a container / cgroup v1), an
    unreadable controller, or ``memory.max`` holding the ``max`` sentinel (no
    limit configured => the cgroup ceiling is not a sizing denominator).
    ``memory.current`` is best-effort: missing/corrupt reads as 0 used rather
    than discarding an authoritative limit.

    Reclaimable page cache (PR-2 review fix): ``memory.current`` INCLUDES page
    cache, and llama.cpp mmaps multi-GB GGUFs — those file pages linger as
    cache long after an unload, so ``limit - current`` would keep reporting
    the node nearly full and wreck sizing. Standard cgroup-v2 working-set
    accounting subtracts the reclaimable file cache::

        working_set = memory.current - memory.stat:inactive_file
        free        = memory.max - working_set

    We deliberately subtract only ``inactive_file`` (not all of ``file``):
    ``active_file`` pages are recently referenced — e.g. the resident weights
    of a HOT model — and pricing them as free would over-commit the node;
    ``inactive_file`` is what the kernel reclaims first under pressure. The
    adjustment is flagged in the reading (``reclaimable_bytes``) so the
    published snapshot can say the number is reclaim-adjusted.
    """
    try:
        raw_limit = (root / "memory.max").read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if raw_limit == _CGROUP_UNLIMITED_SENTINEL:
        return None
    try:
        total = int(raw_limit)
    except ValueError:
        return None
    if total <= 0:
        return None
    try:
        current = int((root / "memory.current").read_text(encoding="ascii").strip())
    except (OSError, ValueError, UnicodeDecodeError):
        current = 0
    # Never subtract more than what is actually accounted as used.
    reclaimable = min(_read_inactive_file_bytes(root), max(current, 0))
    working_set = max(current - reclaimable, 0)
    return NodeMemory(
        total_bytes=total,
        free_bytes=max(total - working_set, 0),
        source="cgroup",
        reclaimable_bytes=reclaimable,
    )


def read_vm_memory() -> NodeMemory:
    """psutil view of the VM/host — the soft fallback, flagged ``source=vm``.

    ``psutil.virtual_memory().available`` already accounts for reclaimable
    cache, so no explicit adjustment applies (``reclaimable_bytes=0``).
    """
    import psutil

    virtual = psutil.virtual_memory()
    return NodeMemory(
        total_bytes=int(virtual.total),
        free_bytes=int(virtual.available),
        source="vm",
    )


def read_node_memory(
    *,
    cgroup_root: Path = CGROUP_V2_ROOT,
    vm_reader: Callable[[], NodeMemory] = read_vm_memory,
) -> NodeMemory:
    """cgroup-v2-first node memory; VM fallback is flagged, never silent."""
    reading = read_cgroup_memory(cgroup_root)
    if reading is not None:
        return reading
    return vm_reader()


# ---------------------------------------------------------- per-process RSS


def process_rss(pid: int) -> int:
    """RSS of one live process; a vanished/unreadable process reads as 0.

    The single RSS sampler for the serving service — the reconciler's
    per-deployment sampling is consolidated here (PR-2) so every consumer
    prices a process the same way.
    """
    try:
        import psutil

        return int(psutil.Process(pid).memory_info().rss)
    except Exception:  # noqa: BLE001 - NoSuchProcess/AccessDenied => unmeasurable
        return 0


# ------------------------------------------------------- predicted footprint


def predict_footprint_bytes(
    weights_bytes: int,
    *,
    context_length: int | None = None,
    n_parallel: int = 1,
    quant_factor: float = 1.0,
    mmproj_bytes: int = 0,
    overhead_bytes: int = RUNTIME_OVERHEAD_BYTES,
) -> int:
    """The planner FORMULA priced from store/on-disk weights (design doc §2).

    ``predicted = weights x quant_factor + kv_per_token x ctx x n_parallel +
    overhead (+ mmproj for vision families)``. ``quant_factor`` is ~1.0 for a
    GGUF already at its target quantization (the file IS the resident
    weights); it exists for callers pricing a hypothetical re-quantization.
    """
    if weights_bytes < 0:
        raise ValueError("weights_bytes must be non-negative")
    if quant_factor <= 0:
        raise ValueError("quant_factor must be positive")
    context = (
        context_length
        if context_length is not None and context_length > 0
        else DEFAULT_CONTEXT_LENGTH
    )
    slots = max(1, n_parallel)
    kv_cache = KV_CACHE_BYTES_PER_TOKEN * context * slots
    return int(weights_bytes * quant_factor) + kv_cache + overhead_bytes + max(0, mmproj_bytes)


def predicted_footprint_for_model(
    *,
    size_bytes: int | None,
    model_path: str | None = None,
    context_length: int | None = None,
    n_parallel: int = 1,
    quant_factor: float = 1.0,
    mmproj_bytes: int = 0,
) -> int | None:
    """Predicted footprint with the PR-2 weights contract: store size or stat.

    Weights come from ``ModelStoreEntry.size_bytes`` when the caller has a
    catalog row, else an on-disk ``stat`` of the GGUF — NEVER the registry
    manifest, which does not know store models (design doc fix #6). ``None``
    when the weights are unknowable (no size, no readable file): the caller
    must treat the model as unpriceable rather than pretend a number.
    """
    weights = size_bytes
    if weights is None and model_path:
        try:
            weights = Path(model_path).stat().st_size
        except OSError:
            return None
    if weights is None or weights <= 0:
        return None
    return predict_footprint_bytes(
        weights,
        context_length=context_length,
        n_parallel=n_parallel,
        quant_factor=quant_factor,
        mmproj_bytes=mmproj_bytes,
    )


def footprint_bytes(predicted: int, observed_steady_rss: int | None) -> int:
    """``max(observed_steady_rss, predicted)`` — the design's calibration rule.

    Trust the measurement once there is one; fall back to the formula for
    models never yet run. Never the minimum: a fresh mmap'd RSS below the
    prediction must not shrink the budget (design doc §2).
    """
    return max(predicted, observed_steady_rss or 0)


# Calibration-key scheme version. v1 keyed on the file BASENAME — but the
# canonical store names EVERY model's weights ``model.gguf``
# (<root>/<name>/model.gguf), so v1 collapsed all store models into ONE shared
# calibration entry and poisoned sizing across models. The prefix versions the
# scheme so pre-v2 sidecars can never be read again (FootprintStore purges
# them on sight).
_FOOTPRINT_KEY_PREFIX = "v2-"


def footprint_key(model: str) -> str:
    """Stable per-MODEL calibration key from a launch model reference.

    Keyed by the FULL reference (path), never the bare basename: uniqueness
    comes from a digest of the whole string, so ``<root>/a/model.gguf`` and
    ``<root>/b/model.gguf`` — distinct store models with identical basenames —
    calibrate independently. Two deployments launching the SAME path still
    share one calibration. The basename is kept in the key (when
    filesystem-safe) purely so sidecar files stay human-readable; a hostile
    name degrades to digest-only and can never traverse out of the footprints
    directory.
    """
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]
    name = Path(model).name if model else ""
    if name and _SAFE_NAME.match(name):
        return f"{_FOOTPRINT_KEY_PREFIX}{name}-{digest}"
    return f"{_FOOTPRINT_KEY_PREFIX}sha256-{digest}"


# ------------------------------------------------------ persisted footprints


class FootprintStore:
    """Per-model observed steady-state RSS, persisted on the serving volume.

    One tiny JSON sidecar per model under ``<serving-home>/footprints/``
    (atomic single-key ``os.replace`` writes — the recency.py pattern), so
    calibration is DB-optional, survives restarts, and has exactly one writer
    (the tracker inside the single-replica serving service). ``record_steady``
    keeps the MAX of steady-state samples: successive steady readings can
    still creep up as stragglers page in, and sizing must stay conservative.
    Everything is best-effort — a disk hiccup must never fail a reconcile.

    Methods take the raw MODEL REFERENCE (launch model path) and derive the
    sidecar name via :func:`footprint_key` exactly once. Pre-v2 sidecars
    (basename-keyed — one shared ``model.gguf`` entry for every store model,
    i.e. cross-model-poisoned data) are purged on construction so they can
    never leak into sizing again.
    """

    def __init__(self, home: Path | None = None) -> None:
        self._directory = (home if home is not None else _serving_home()) / "footprints"
        self._purge_legacy_keys()

    @property
    def directory(self) -> Path:
        return self._directory

    def get(self, model: str) -> int | None:
        """Last calibrated steady-state RSS for ``model`` (None = never observed)."""
        path = self._directory / footprint_key(model)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rss = int(payload["rss_bytes"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return rss if rss > 0 else None

    def record_steady(self, model: str, rss_bytes: int) -> None:
        """Persist one steady-state sample (monotonic max; best-effort)."""
        if rss_bytes <= 0:
            return
        current = self.get(model)
        if current is not None and current >= rss_bytes:
            return
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            target = self._directory / footprint_key(model)
            temporary = self._directory / f".{target.name}.tmp"
            temporary.write_text(
                json.dumps(
                    {"rss_bytes": int(rss_bytes), "updated": time.time()}, sort_keys=True
                ),
                encoding="utf-8",
            )
            os.replace(temporary, target)
        except OSError:  # pragma: no cover - disk hiccup must not fail a cycle
            logger.warning("could not persist footprint for %r", model, exc_info=True)

    def _purge_legacy_keys(self) -> None:
        """Delete every pre-v2 sidecar (and stray tmp) — best-effort.

        v1 keys were file basenames, which the canonical store collapses to a
        single shared ``model.gguf`` entry across ALL models: that data is
        cross-model poisoned by construction, so it is invalidated (deleted),
        never migrated — there is no way to know which model wrote it.
        """
        try:
            entries = list(self._directory.iterdir())
        except OSError:  # missing directory: nothing to purge
            return
        for entry in entries:
            if not entry.name.startswith(_FOOTPRINT_KEY_PREFIX):
                with contextlib.suppress(OSError):
                    entry.unlink()


# --------------------------------------------------------------- the tracker


class Observation(Protocol):
    """What the tracker needs from one reconciler observation (duck-typed to
    avoid a resources->reconciler import cycle)."""

    @property
    def name(self) -> str: ...

    @property
    def phase(self) -> str: ...

    @property
    def rss_bytes(self) -> int: ...

    @property
    def model(self) -> str: ...


@dataclass(frozen=True)
class NodeSnapshot:
    """The single-row ``serving_node`` payload published each reconcile cycle."""

    total_bytes: int
    free_bytes: int
    source: str  # "cgroup" | "vm" — the UI's soft-number badge input
    sum_rss_bytes: int  # sum of live (hot/loading) runtime RSS this cycle
    # Reclaimable page cache excluded from "used" when computing free_bytes
    # (cgroup working-set accounting; 0 = no adjustment applied). Published so
    # readers know the free number is reclaim-adjusted, not raw memory.current.
    reclaimable_bytes: int = 0


class ResourceTracker:
    """Per-cycle node snapshot + steady-state footprint calibration.

    Fed the reconciler's observations each cycle (outside the supervisor lock
    — everything here is measurement, never mutation). Calibration rules:

    * ``loading``-phase samples are NEVER recorded — that is the mmap ramp the
      design explicitly excludes (RSS is still climbing as pages fault in).
    * a ``hot`` sample counts only after ``steady_hot_cycles`` consecutive hot
      observations AND once RSS has stopped moving (relative delta vs the
      previous cycle within ``stability_fraction``) — early-hot cycles are
      still ramping even though health already passes.
    * any non-hot observation resets that deployment's streak.
    """

    def __init__(
        self,
        *,
        memory_reader: Callable[[], NodeMemory] | None = None,
        footprints: FootprintStore | None = None,
        steady_hot_cycles: int = 6,
        stability_fraction: float = 0.02,
    ) -> None:
        if steady_hot_cycles < 2:
            # A single hot sample has no previous reading to prove stability.
            raise ValueError("steady_hot_cycles must be at least 2")
        if not 0 < stability_fraction < 1:
            raise ValueError("stability_fraction must be in (0, 1)")
        self._memory_reader = memory_reader if memory_reader is not None else read_node_memory
        self.footprints = footprints if footprints is not None else FootprintStore()
        self.steady_hot_cycles = steady_hot_cycles
        self.stability_fraction = stability_fraction
        self._hot_streak: dict[str, int] = {}
        self._previous_rss: dict[str, int] = {}

    def observe_cycle(self, observations: Sequence[Observation]) -> NodeSnapshot:
        """Fold one cycle's observations: calibrate, then snapshot the node."""
        self._calibrate(observations)
        memory = self._memory_reader()
        sum_rss = sum(
            observed.rss_bytes
            for observed in observations
            if observed.phase in {"hot", "loading"} and observed.rss_bytes > 0
        )
        return NodeSnapshot(
            total_bytes=memory.total_bytes,
            free_bytes=memory.free_bytes,
            source=memory.source,
            sum_rss_bytes=sum_rss,
            reclaimable_bytes=memory.reclaimable_bytes,
        )

    def footprint_for(self, model: str, predicted: int) -> int:
        """Calibrated working footprint: ``max(observed_steady, predicted)``."""
        return footprint_bytes(predicted, self.footprints.get(model))

    def _calibrate(self, observations: Iterable[Observation]) -> None:
        seen: set[str] = set()
        for observed in observations:
            seen.add(observed.name)
            if observed.phase != "hot" or observed.rss_bytes <= 0:
                # Loading (mmap ramp), cold, evicted, failed: never a
                # calibration sample, and any streak restarts from zero.
                self._hot_streak.pop(observed.name, None)
                self._previous_rss.pop(observed.name, None)
                continue
            previous = self._previous_rss.get(observed.name)
            streak = self._hot_streak.get(observed.name, 0) + 1
            self._hot_streak[observed.name] = streak
            self._previous_rss[observed.name] = observed.rss_bytes
            if previous is None or streak < self.steady_hot_cycles:
                continue  # not enough consecutive hot samples to call it steady
            if abs(observed.rss_bytes - previous) > self.stability_fraction * previous:
                continue  # still climbing: pages are faulting in
            self.footprints.record_steady(observed.model, observed.rss_bytes)
        # Forget deployments that no longer exist.
        for stale in set(self._hot_streak) - seen:
            del self._hot_streak[stale]
        for stale in set(self._previous_rss) - seen:
            del self._previous_rss[stale]


# ----------------------------------------------------------------- publish


def publish_snapshot_via_catalog(snapshot: NodeSnapshot) -> None:
    """Default snapshot publisher: single-row upsert, best-effort, DB-optional.

    With no ``DATABASE_URL`` (or a DB hiccup) the publish is SKIPPED and the
    cycle goes on — the repair loop must never depend on Postgres (design doc
    fix #8); the api then reports the observed surface as unavailable instead
    of serving a stale number.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        ModelCatalog().publish_node_snapshot(
            total_bytes=snapshot.total_bytes,
            free_bytes=snapshot.free_bytes,
            source=snapshot.source,
            sum_rss_bytes=snapshot.sum_rss_bytes,
            reclaimable_bytes=snapshot.reclaimable_bytes,
        )
    except CatalogUnavailableError:
        logger.debug("no DATABASE_URL: node snapshot not published this cycle")
    except Exception:  # noqa: BLE001 - a DB hiccup must never break the cycle
        logger.warning("could not publish serving node snapshot", exc_info=True)


__all__ = [
    "CGROUP_V2_ROOT",
    "DEFAULT_CONTEXT_LENGTH",
    "KV_CACHE_BYTES_PER_TOKEN",
    "RUNTIME_OVERHEAD_BYTES",
    "FootprintStore",
    "NodeMemory",
    "NodeSnapshot",
    "Observation",
    "ResourceTracker",
    "footprint_bytes",
    "footprint_key",
    "predict_footprint_bytes",
    "predicted_footprint_for_model",
    "process_rss",
    "publish_snapshot_via_catalog",
    "read_cgroup_memory",
    "read_node_memory",
    "read_vm_memory",
]
