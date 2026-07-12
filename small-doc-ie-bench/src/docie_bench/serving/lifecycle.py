"""Dynamic load/unload lifecycle primitives (PR-4, design doc §4).

Everything here executes ONLY inside the single-replica ``serving`` service —
the one process holding the runtime Popen handles (P1). The pieces:

* :func:`assess_fit` — ONE fit policy for every load-shaped decision: the same
  ``max(calibrated steady RSS, predicted)`` footprint the sizing engine and
  the reconciler's gated-restart gate price, checked against live
  cgroup-aware node free RAM minus the SAME explicit safety margin the Sizing
  tab surfaces. Fail-open on anything unknowable (no model file, unmeasurable
  node): the gate exists to stop OOM storms, not to block legitimate loads on
  measurement hiccups.
* :func:`load_timeout_s` — the size-aware ``await_ready`` budget (design fix
  #7): a large CPU GGUF can legitimately outrun the flat 60s default, and a
  timeout would fail an extraction that would have been ready at 90s. The
  budget scales off the on-disk weights.
* :func:`select_victims` — LRU eviction victim selection with every storm
  guard from the design: pinned deployments are NEVER chosen, a just-loaded
  deployment is protected by ``min_hot_s``, at most ``max_evictions`` victims
  per attempt (rate limit), and **fit-before-evict**: when even the maximum
  allowed evictions cannot cover the deficit, NOTHING is evicted (never
  evict-to-not-fit).
* :class:`LoadCoordinator` — the global cold-start pileup lock. All loads
  funnel through a per-deployment ``threading.Lock`` in this one process, so
  N concurrent requests (from any number of scaled workers, the api, or a
  step retry) trigger exactly ONE spawn; an already-hot deployment is an
  idempotent no-op, which is what makes a re-fired load event harmless.
  Admission (fit gate + eviction) additionally runs under ONE
  coordinator-wide lock with in-flight reservations, so two concurrent loads
  of DIFFERENT deployments can never both price themselves against the same
  ``free_bytes`` and jointly overcommit the node.

Unload itself lives on ``PersistentSupervisor.unload`` (the record mutation)
and ``control_plane._DefaultSupervisor.unload`` (the placement-row UPDATE) —
this module only decides *who* to unload.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from docie_bench.serving.resources import (
    FootprintStore,
    NodeMemory,
    footprint_bytes,
    predicted_footprint_for_model,
    read_node_memory,
)
from docie_bench.serving.runtime import LifecycleState, RuntimeLaunchSpec
from docie_bench.serving.supervisor import (
    DeploymentRecord,
    DesiredState,
    PersistentSupervisor,
)

logger = logging.getLogger(__name__)

# Size-aware cold-load budget (design fix #7): a floor generous enough for any
# small model, plus time to fault a large GGUF in from disk at a conservative
# CPU-node read rate. A 4 GB model prices to ~4 minutes — deliberately roomy;
# the request "fails honestly only after the generous timeout".
LOAD_TIMEOUT_FLOOR_S = 120.0
LOAD_TIMEOUT_BYTES_PER_SECOND = 24 * 1024 * 1024
LOAD_TIMEOUT_CEILING_S = 1800.0


class LoadError(RuntimeError):
    """A load-on-demand could not complete (no fit, or readiness timeout)."""


@dataclass(frozen=True)
class FitDecision:
    """One fit-gate verdict, with the numbers eviction math needs.

    ``needed_bytes``/``free_bytes`` are ``None`` exactly when the decision was
    fail-open (unpriceable model or unmeasurable node) — ``fits`` is then True
    and there is nothing to evict for.
    """

    fits: bool
    needed_bytes: int | None
    free_bytes: int | None
    margin_bytes: int
    reason: str


def mmproj_bytes_from_launch(launch: RuntimeLaunchSpec) -> int:
    """Size of the vision projector a launch will load (0 when none/unreadable).

    ``needs_mmproj`` families launch with ``--mmproj <path>`` in
    ``extra_args`` (wired by ``ModelStore.family_launch_args``); llama-server
    loads that projector fully resident, so the fit gate must price it.
    Unreadable/missing degrades to 0 (fail-open, like every other unknowable
    in the gate).
    """
    arguments = list(launch.extra_args)
    for index, argument in enumerate(arguments):
        path: str | None = None
        if argument == "--mmproj" and index + 1 < len(arguments):
            path = arguments[index + 1]
        elif argument.startswith("--mmproj="):
            path = argument.partition("=")[2]
        if path:
            try:
                return Path(path).stat().st_size
            except OSError:
                return 0
    return 0


def assess_fit(
    record: DeploymentRecord,
    *,
    footprints: FootprintStore | None = None,
    margin_fraction: float | None = None,
    memory_reader: Callable[[], NodeMemory] | None = None,
) -> FitDecision:
    """The one fit policy (see module docstring); reconciler + loads share it.

    ``footprints`` supplies the steady-state calibration sidecars; ``None``
    consults the default store on the serving volume so calibration is never
    silently ignored. ``margin_fraction=None`` reads the settings knob.
    ``memory_reader`` is the live node-RAM source (injection seam: the
    reconciler passes its module-global so existing monkeypatches keep
    working; tests inject a fixed reading).
    """
    from docie_bench.serving.sizing import safety_margin_bytes
    from docie_bench.settings import get_settings

    launch = record.spec.launch
    predicted = predicted_footprint_for_model(
        size_bytes=None,
        model_path=launch.model,
        context_length=launch.context_length,
        mmproj_bytes=mmproj_bytes_from_launch(launch),
    )
    if predicted is None:
        return FitDecision(True, None, None, 0, "")
    store = footprints if footprints is not None else FootprintStore()
    needed = footprint_bytes(predicted, store.get(launch.model))
    reader = memory_reader if memory_reader is not None else read_node_memory
    try:
        memory = reader()
    except Exception:  # noqa: BLE001 - unmeasurable => fail-open
        return FitDecision(True, needed, None, 0, "")
    fraction = (
        margin_fraction
        if margin_fraction is not None
        else get_settings().serving_sizing_margin_fraction
    )
    margin = safety_margin_bytes(memory.total_bytes, fraction)
    if memory.free_bytes - margin < needed:
        return FitDecision(
            False,
            needed,
            memory.free_bytes,
            margin,
            (
                f"needs ~{needed} bytes (max of calibrated steady-state RSS and "
                f"predicted weights + kv-cache + overhead + mmproj) "
                f"but only {memory.free_bytes} free minus the {margin}-byte safety "
                f"margin leaves {memory.free_bytes - margin} available"
            ),
        )
    return FitDecision(True, needed, memory.free_bytes, margin, "")


def load_timeout_s(
    model_path: str | None,
    *,
    size_bytes: int | None = None,
    floor_s: float = LOAD_TIMEOUT_FLOOR_S,
    bytes_per_second: float = LOAD_TIMEOUT_BYTES_PER_SECOND,
    ceiling_s: float = LOAD_TIMEOUT_CEILING_S,
) -> float:
    """Generous, size-aware ``await_ready`` budget for a cold load (fix #7).

    Scaled off the on-disk weights (``size_bytes`` wins when the caller knows
    it; else a best-effort ``stat`` of ``model_path``), floored so a small or
    un-stat-able model still gets a real budget, and ceilinged so a corrupt
    size can never park a request for hours.
    """
    weights = size_bytes
    if weights is None and model_path:
        try:
            weights = Path(model_path).stat().st_size
        except OSError:
            weights = None
    if weights is None or weights <= 0:
        return floor_s
    return min(max(floor_s, weights / bytes_per_second + 60.0), ceiling_s)


def releasable_bytes(record: DeploymentRecord, footprints: FootprintStore) -> int:
    """RAM an eviction of ``record`` is expected to free (0 = unpriceable).

    Same pricing as the fit gate — ``max(calibrated steady RSS, predicted)``
    — so "what a victim frees" and "what a candidate needs" are one currency.
    An unpriceable victim reads 0 and is skipped by selection: evicting it
    would free an unknown amount, which must never be counted toward a fit.
    """
    launch = record.spec.launch
    predicted = predicted_footprint_for_model(
        size_bytes=None,
        model_path=launch.model,
        context_length=launch.context_length,
        mmproj_bytes=mmproj_bytes_from_launch(launch),
    )
    if predicted is None:
        return 0
    return footprint_bytes(predicted, footprints.get(launch.model))


def _is_hot(record: DeploymentRecord) -> bool:
    return (
        record.spec.desired_state == DesiredState.RUNNING
        and record.state == LifecycleState.READY
    )


def select_victims(
    records: Iterable[DeploymentRecord],
    *,
    deficit_bytes: int,
    now: float,
    min_hot_s: float,
    max_evictions: int,
    price: Callable[[DeploymentRecord], int],
) -> list[str] | None:
    """LRU eviction victims covering ``deficit_bytes``, or None to evict nothing.

    Candidates are HOT deployments only. All the storm guards live here:

    * ``pinned`` deployments are never candidates;
    * a deployment hot for less than ``min_hot_s`` (per its ``loaded_at``
      spawn stamp) is never a candidate — a just-loaded model must not be
      immediately re-evicted by the load that follows it;
    * victims are taken least-recently-served first (never-served sorts
      oldest — nothing has needed it yet);
    * at most ``max_evictions`` victims per attempt (the per-cycle rate
      limit);
    * **fit-before-evict**: if the allowed victims cannot cover the deficit,
      return ``None`` — the caller must evict NOTHING and fail the load
      honestly, never trade a working deployment for one that still won't
      fit.
    """
    if deficit_bytes <= 0:
        return []
    eligible = [
        record
        for record in records
        if _is_hot(record)
        and not record.pinned
        and (record.loaded_at is None or now - record.loaded_at >= min_hot_s)
    ]
    eligible.sort(key=lambda record: record.last_served or record.loaded_at or 0.0)
    victims: list[str] = []
    freed = 0
    for record in eligible:
        if len(victims) >= max_evictions:
            break
        gain = price(record)
        if gain <= 0:
            continue  # unpriceable: an unknown gain must not be counted
        victims.append(record.spec.name)
        freed += gain
        if freed >= deficit_bytes:
            return victims
    return None


class LoadCoordinator:
    """Serving-side load orchestrator: one spawn per deployment, ever.

    Per-deployment ``threading.Lock``s funnel every load — worker
    load-on-demand events, the api's Load button, Inngest step retries — into
    a single spawn (the design's "global by construction" pileup lock: only
    this replica binds ports, so serializing here serializes the world).
    Holding a load lock does NOT hold the supervisor lock across the whole
    ``await_ready`` poll, so the reconciler and other handlers keep running
    while a large GGUF loads.

    ``unload`` is the eviction executor (injected so the control plane can
    route it through the placement-row UPDATE); ``assess`` is the fit gate
    (injected in tests). ``min_hot_s``/``max_evictions`` default from
    settings at call time so env knobs act without a restart.

    Concurrent-load admission (review fix): the per-deployment locks
    serialize loads of the SAME name only, so two threads loading DIFFERENT
    deployments would each run ``assess_fit`` against the same measured
    ``free_bytes`` — free RAM does not drop until the freshly spawned model
    faults its pages in, so both gates could pass and jointly OOM the node.
    Admission therefore runs under one ``_admission_lock`` and every admitted
    load holds an in-flight reservation of its priced ``needed_bytes`` until
    it reaches READY (or fails); a later admission subtracts the outstanding
    reservations from the measured free RAM before deciding. Eviction is
    symmetric: ``RuntimeAdapter.shutdown`` WAITS for the victim process to
    exit (owned Popen and recovered-pid alike), so the freed RAM is
    observable before the admitted load spawns.
    """

    def __init__(
        self,
        supervisor: PersistentSupervisor,
        *,
        unload: Callable[[str], object] | None = None,
        assess: Callable[[DeploymentRecord], FitDecision] | None = None,
        footprints: FootprintStore | None = None,
        min_hot_s: float | None = None,
        max_evictions: int | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        timeout_for: Callable[[str], float] = load_timeout_s,
    ) -> None:
        self.supervisor = supervisor
        self._unload: Callable[[str], object] = (
            unload if unload is not None else supervisor.unload
        )
        self._footprints = footprints if footprints is not None else FootprintStore()
        self._assess: Callable[[DeploymentRecord], FitDecision] = (
            assess
            if assess is not None
            else (lambda record: assess_fit(record, footprints=self._footprints))
        )
        self._min_hot_s = min_hot_s
        self._max_evictions = max_evictions
        self._clock = clock
        self._sleep = sleep
        self._timeout_for = timeout_for
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()
        # One admission at a time + outstanding reservations (name -> priced
        # needed_bytes) for loads admitted but not yet READY/failed. See the
        # class docstring: this is what stops two concurrent loads of
        # different deployments from double-spending the same free_bytes.
        self._admission_lock = threading.Lock()
        self._inflight: dict[str, int] = {}

    # ------------------------------------------------------------------ knobs
    def _eviction_knobs(self) -> tuple[float, int]:
        from docie_bench.settings import get_settings

        if self._min_hot_s is not None and self._max_evictions is not None:
            return self._min_hot_s, self._max_evictions
        settings = get_settings()
        min_hot = (
            self._min_hot_s
            if self._min_hot_s is not None
            else float(settings.serving_min_hot_seconds)
        )
        max_evictions = (
            self._max_evictions
            if self._max_evictions is not None
            else int(settings.serving_max_evictions_per_cycle)
        )
        return min_hot, max_evictions

    def _lock_for(self, name: str) -> threading.Lock:
        with self._registry_lock:
            return self._locks.setdefault(name, threading.Lock())

    # ------------------------------------------------------------------- load
    def load(self, name: str) -> DeploymentRecord:
        """Bring ``name`` hot; idempotent; blocks until READY or raises.

        Steps under the per-deployment lock: already-hot no-op, fit gate
        (evicting LRU victims when — and only when — that makes it fit), flip
        ``desired -> RUNNING`` + spawn, then a size-aware ``await_ready``.
        Raises :class:`LoadError` with the honest reason on no-fit or
        readiness timeout; a KeyError for an unknown deployment propagates.
        """
        with self._lock_for(name):
            with self.supervisor.lock:
                record = self.supervisor.get(name)
                if _is_hot(record) and record.endpoint:
                    logger.debug("load %r: already hot — idempotent no-op", name)
                    return replace(record)
                snapshot = replace(record)
            # Admission is globally serialized and leaves a reservation (see
            # class docstring); the reservation is released only once the load
            # reaches READY (RSS then backs the memory) or fails.
            with self._admission_lock:
                decision = self._make_room(snapshot)
                self._inflight[name] = max(decision.needed_bytes or 0, 0)
            try:
                with self.supervisor.lock:
                    record = self.supervisor.get(name)
                    spec = replace(record.spec, desired_state=DesiredState.RUNNING)
                    record = self.supervisor.deploy(spec)
                if record.state == LifecycleState.READY:
                    return record
                timeout = self._timeout_for(spec.launch.model)
                record = self.supervisor.await_ready(
                    name, timeout_s=timeout, sleep=self._sleep
                )
                if record.state != LifecycleState.READY:
                    raise LoadError(
                        f"deployment {name!r} did not become ready within the "
                        f"{timeout:.0f}s size-aware load budget "
                        f"(state={record.state.value}, last_error={record.last_error!r})"
                    )
                return record
            finally:
                with self._admission_lock:
                    self._inflight.pop(name, None)

    # --------------------------------------------------------------- eviction
    def _make_room(self, record: DeploymentRecord) -> FitDecision:
        """Fit-gate ``record``; evict LRU victims iff that makes it fit (§4).

        Callers hold ``_admission_lock``. Outstanding in-flight reservations
        (other admitted-but-not-yet-READY loads) are subtracted from the
        measured free RAM before deciding, so concurrent loads of different
        deployments cannot both spend the same bytes. Returns the decision so
        the caller can reserve ``needed_bytes`` for this load.
        """
        decision = self._assess(record)
        name = record.spec.name
        reserved = sum(
            bytes_ for other, bytes_ in self._inflight.items() if other != name
        )
        if decision.needed_bytes is None or decision.free_bytes is None:
            if decision.fits:
                # Fail-open decision (unpriceable model or unmeasurable node):
                # nothing to reserve, nothing to evict for.
                return decision
            # A non-fitting decision always carries numbers (fail-open paths
            # return fits=True); guard against a custom gate that does not.
            raise LoadError(f"deployment {name!r} does not fit: {decision.reason}")
        effective_free = decision.free_bytes - reserved
        if effective_free - decision.margin_bytes >= decision.needed_bytes:
            return decision
        deficit = decision.needed_bytes + decision.margin_bytes - effective_free
        min_hot_s, max_evictions = self._eviction_knobs()
        with self.supervisor.lock:
            candidates = [
                replace(other)
                for other in self.supervisor.records().values()
                if other.spec.name != name
            ]
        victims = select_victims(
            candidates,
            deficit_bytes=deficit,
            now=self._clock(),
            min_hot_s=min_hot_s,
            max_evictions=max_evictions,
            price=lambda victim: releasable_bytes(victim, self._footprints),
        )
        if victims is None:
            detail = decision.reason or (
                f"needs ~{decision.needed_bytes} bytes but only {effective_free} "
                f"effectively free ({decision.free_bytes} measured minus "
                f"{reserved} reserved by in-flight loads) minus the "
                f"{decision.margin_bytes}-byte safety margin"
            )
            raise LoadError(
                f"deployment {name!r} does not fit and eviction cannot make it "
                f"fit (pinned/min-hot/rate-limit guards; never evict-to-not-fit): "
                f"{detail}"
            )
        for victim in victims:
            logger.info(
                "evicting %r (LRU victim) to make room for loading %r", victim, name
            )
            # The unload path waits for the victim process to exit (adapter
            # shutdown blocks), so the freed RAM is real by the time we return.
            self._unload(victim)
        return decision


__all__ = [
    "FitDecision",
    "LoadCoordinator",
    "LoadError",
    "assess_fit",
    "load_timeout_s",
    "mmproj_bytes_from_launch",
    "releasable_bytes",
    "select_victims",
]
