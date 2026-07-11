"""Background reconciler for the single-replica ``serving`` service (PR-1).

Periodically observes every deployment record, repairs what should be running
(gated — never a respawn storm), and publishes the observed state to the
Postgres ``model_placement`` surface the Studio reads. Kills the "a dead
llama-server still shows ready" staleness at its root: nothing else in the
codebase re-observes a deployment after it reaches READY.

Placement in the topology (design doc §0.1/P1): this runs ONLY inside the
dedicated single-replica ``serving`` compose service — the one process that
holds the ``subprocess.Popen`` handles — against the SAME shared
``PersistentSupervisor`` the lifecycle handlers use. It must never run on a
scaled ``worker`` replica: N reconcilers would clobber ``deployments.json``
and mis-probe the round-robin advertise host. ``worker.py`` enforces this by
starting it only for ``DOCIE_WORKER_ROLE=serving``.

Concurrency (fix #4): a whole observation pass runs under the supervisor's
thread lock (the same lock every handler mutation takes), off the event loop
via ``asyncio.to_thread`` so health probes and ``fsync`` never stall the
Inngest Connect heartbeats. Publishing to Postgres happens OUTSIDE the lock —
observations are immutable snapshots, and a slow DB must not block handlers.

What a cycle does per record (design doc §1):

* Liveness by HEALTH, not pid: pid presence is a hint; ``phase=hot`` /
  ``health_ok=true`` are published only on a real ``/health`` pass. A
  PID-reuse guard (``create_time`` mismatch, ``NoSuchProcess`` => dead) keeps
  a recycled pid from impersonating a runtime.
* Fast death for READY vs slow-load tolerance for STARTING: a crashed or hung
  previously-READY runtime is declared failed within ``ready_miss_threshold``
  misses (2-3), while a cold-loading GGUF keeps the deploy path's generous
  ``health_failure_threshold=60`` and is never killed here.
* Gated restart — the sharp edge of naive ``reconcile_all()``: a dead
  ``desired=RUNNING`` deployment is respawned only when (a) restart budget
  remains AND (b) a live fit-check passes, so a model that OOMs on load can
  not crash->respawn->OOM in a tight loop.
* Folds the per-deployment recency sidecars (written by the scaled workers)
  into ``DeploymentRecord.last_served`` — the PR-4 LRU/idle input.
* Publishes the observed row (phase/pid/create_time/rss/health/last_error;
  ``endpoint=""`` whenever not live) best-effort: with no DATABASE_URL the
  cycle still repairs and ``_save()``s ``deployments.json`` (which the api
  reads fresh per request), it just skips the Postgres publish.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from docie_bench.serving import recency
from docie_bench.serving.runtime import LifecycleState, RuntimeKind
from docie_bench.serving.supervisor import (
    Activation,
    DeploymentRecord,
    DesiredState,
    PersistentSupervisor,
    RestartPolicy,
    _default_create_time,
)

logger = logging.getLogger(__name__)

# Fixed runtime slab assumed on top of the model weights when fit-checking a
# restart (llama-server arena, KV cache floor). PR-2 replaces this heuristic
# with the calibrated resource tracker; PR-1 only needs a storm brake.
_RESTART_OVERHEAD_BYTES = 512 * 1024 * 1024

# create_time is stable for the lifetime of a process; allow a small slack for
# float rounding across psutil reads.
_CREATE_TIME_TOLERANCE_S = 1.0

# RuntimeKind -> the "engine" label the placement rows/resolver key on.
_ENGINE_BY_RUNTIME: dict[RuntimeKind, str] = {
    RuntimeKind.LLAMACPP: "llama-server",
    RuntimeKind.OLLAMA: "ollama",
    RuntimeKind.VLLM: "vllm",
    RuntimeKind.REMOTE: "remote",
}


@dataclass(frozen=True)
class ObservedDeployment:
    """One immutable per-cycle observation, ready to publish."""

    name: str
    engine: str
    state: str  # LifecycleState value
    phase: str  # hot | loading | cold | evicted | failed
    endpoint: str  # "" whenever not live (never None: the column is NOT NULL)
    pid: int | None
    pid_create_time: float | None
    rss_bytes: int
    health_ok: bool
    last_error: str | None


def default_fit_check(record: DeploymentRecord) -> tuple[bool, str]:
    """Cheap PR-1 fit gate: on-disk weights + fixed overhead vs available RAM.

    Fail-open on anything unknowable (no model file, psutil failure): the gate
    exists to stop crash->OOM->respawn storms, not to block legitimate repairs
    on measurement hiccups. PR-2's resource tracker replaces this.
    """
    try:
        weights = Path(record.spec.launch.model).stat().st_size
    except OSError:
        return True, ""
    if weights <= 0:
        return True, ""
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - unmeasurable => fail-open
        return True, ""
    needed = weights + _RESTART_OVERHEAD_BYTES
    if available < needed:
        return False, (
            f"needs ~{needed} bytes (weights {weights} + overhead) "
            f"but only {available} available"
        )
    return True, ""


def _default_rss(pid: int) -> int:
    try:
        import psutil

        return int(psutil.Process(pid).memory_info().rss)
    except Exception:  # noqa: BLE001 - a vanished process reads as 0
        return 0


def _publish_via_catalog(observations: list[ObservedDeployment]) -> None:
    """Default publisher: upsert the observed rows (best-effort, DB-optional)."""
    if not observations:
        return
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        catalog = ModelCatalog()
        for observed in observations:
            catalog.publish_observed(
                observed.name,
                engine=observed.engine,
                state=observed.state,
                endpoint=observed.endpoint,
                phase=observed.phase,
                pid=observed.pid,
                pid_create_time=observed.pid_create_time,
                rss_bytes=observed.rss_bytes,
                health_ok=observed.health_ok,
                last_error=observed.last_error,
            )
    except CatalogUnavailableError:
        logger.debug("no DATABASE_URL: observed state not published (repair-only cycle)")
    except Exception:  # noqa: BLE001 - a DB hiccup must never break the repair loop
        logger.warning("could not publish observed serving state", exc_info=True)


class ServingReconciler:
    """The in-``serving`` observation/repair loop (see module docstring)."""

    def __init__(
        self,
        supervisor: PersistentSupervisor,
        *,
        interval_s: float = 10.0,
        ready_miss_threshold: int = 3,
        fit_check: Callable[[DeploymentRecord], tuple[bool, str]] = default_fit_check,
        rss_reader: Callable[[int], int] = _default_rss,
        create_time: Callable[[int], float | None] = _default_create_time,
        publisher: Callable[[list[ObservedDeployment]], None] = _publish_via_catalog,
        recency_home: Path | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if ready_miss_threshold < 1:
            raise ValueError("ready_miss_threshold must be positive")
        self.supervisor = supervisor
        self.interval_s = interval_s
        self.ready_miss_threshold = ready_miss_threshold
        self._fit_check = fit_check
        self._rss_reader = rss_reader
        self._create_time = create_time
        self._publisher = publisher
        self._recency_home = recency_home
        # Reconciler-local memory: which deployments were observed READY (so a
        # later miss is a fast-death candidate, not a slow cold-load) and how
        # many consecutive misses each has accrued since.
        self._was_ready: set[str] = set()
        self._misses: dict[str, int] = {}

    # ------------------------------------------------------------------ loop
    async def run_forever(self, *, stop_event: asyncio.Event | None = None) -> None:
        """Cycle until cancelled (or ``stop_event`` is set). Never raises."""
        logger.info(
            "serving reconciler started (interval=%.1fs, ready_miss_threshold=%d)",
            self.interval_s,
            self.ready_miss_threshold,
        )
        while stop_event is None or not stop_event.is_set():
            try:
                # The cycle blocks (health probes + fsync) while holding the
                # supervisor lock -> a worker thread, never the event loop.
                await asyncio.to_thread(self.run_cycle)
            except Exception:  # noqa: BLE001 - one bad cycle must not kill the loop
                logger.exception("serving reconcile cycle failed")
            if stop_event is None:
                await asyncio.sleep(self.interval_s)
            else:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), self.interval_s)

    # ----------------------------------------------------------------- cycle
    def run_cycle(self) -> list[ObservedDeployment]:
        """One observation/repair pass; returns what was (or would be) published."""
        with self.supervisor.lock:
            observations = self._observe_locked()
        # Publish OUTSIDE the lock: observations are immutable snapshots and a
        # slow database must not block the lifecycle handlers.
        self._publisher(observations)
        return observations

    def _observe_locked(self) -> list[ObservedDeployment]:
        records = self.supervisor.records()
        names = sorted(records)
        # Fold worker-written recency sidecars first so this cycle's view (and
        # PR-4's idle/LRU decisions) see the freshest last_served.
        stamps = recency.read_for(names, home=self._recency_home)
        if stamps:
            self.supervisor.fold_recency(stamps)
        observations: list[ObservedDeployment] = []
        for name in names:
            record = records.get(name)
            if record is None:  # removed concurrently within this cycle
                continue
            observations.append(self._observe_one(name, record))
        # Forget reconciler-local memory of deployments that no longer exist.
        self._was_ready.intersection_update(records)
        for stale in set(self._misses) - set(records):
            del self._misses[stale]
        return observations

    def _observe_one(self, name: str, record: DeploymentRecord) -> ObservedDeployment:
        if record.spec.desired_state == DesiredState.STOPPED:
            return self._observe_stopped(name, record)
        return self._observe_running(name, record)

    def _observe_stopped(self, name: str, record: DeploymentRecord) -> ObservedDeployment:
        # Ensure no stray process survives a STOPPED desire (reconcile()'s
        # STOPPED branch is exactly that teardown and is idempotent).
        if record.pid is not None:
            record = self.supervisor.reconcile(name)
        self._forget(name)
        phase = "evicted" if record.activation == Activation.MANAGED else "cold"
        return self._observation(name, record, phase=phase, health_ok=False)

    def _observe_running(self, name: str, record: DeploymentRecord) -> ObservedDeployment:
        alive, reuse_error = self._process_alive(record)
        if not alive:
            return self._observe_dead(name, record, reuse_error)

        adapter = self.supervisor.adapters[record.spec.launch.runtime]
        health = adapter.health(
            record.spec.launch, timeout=record.spec.health_timeout_seconds
        )
        if health.healthy:
            record = self.supervisor.observe_health(name, healthy=True)
            self._was_ready.add(name)
            self._misses[name] = 0
            return self._observation(name, record, phase="hot", health_ok=True)

        detail = health.detail or f"health check returned status {health.status_code}"
        if name in self._was_ready:
            # Previously READY and now unreachable: the FAST path. Do not wait
            # out the deploy-time health_failure_threshold=60 (~10 min).
            misses = self._misses.get(name, 0) + 1
            self._misses[name] = misses
            if misses >= self.ready_miss_threshold:
                logger.warning(
                    "deployment %s unresponsive after READY (%d misses): declaring failed",
                    name,
                    misses,
                )
                self._forget(name)
                allowed, reason = self._restart_allowed(record)
                record = self.supervisor.mark_failed(
                    name, f"unresponsive after READY: {detail}", shutdown=True
                )
                return self._maybe_restart(name, record, allowed, reason)
            record = self.supervisor.observe_health(name, healthy=False, detail=detail)
            return self._observation(name, record, phase="loading", health_ok=False)

        # Never READY yet: a cold-loading GGUF. High tolerance — observe, never
        # kill; the record's own health_failure_threshold governs the deploy
        # path's degrade logic, not this loop.
        record = self.supervisor.observe_health(name, healthy=False, detail=detail)
        return self._observation(name, record, phase="loading", health_ok=False)

    def _observe_dead(
        self, name: str, record: DeploymentRecord, reuse_error: str | None
    ) -> ObservedDeployment:
        self._forget(name)
        already_declared = record.pid is None and record.state == LifecycleState.FAILED
        allowed, reason = self._restart_allowed(record)
        if not already_declared:
            logger.warning(
                "deployment %s runtime process is gone (%s)", name, reuse_error or "exited"
            )
            # shutdown=False always: the process is gone, and for a PID-reuse
            # mismatch a shutdown would SIGTERM an unrelated process.
            record = self.supervisor.mark_failed(name, reuse_error, shutdown=False)
        return self._maybe_restart(name, record, allowed, reason)

    def _maybe_restart(
        self, name: str, record: DeploymentRecord, allowed: bool, reason: str
    ) -> ObservedDeployment:
        if allowed:
            logger.info("deployment %s: gated restart approved; respawning", name)
            record = self.supervisor.reconcile(name)
            phase = {
                LifecycleState.READY: "hot",
                LifecycleState.STARTING: "loading",
                LifecycleState.DEGRADED: "loading",
            }.get(record.state, "failed")
            return self._observation(
                name, record, phase=phase, health_ok=record.state == LifecycleState.READY
            )
        if reason:
            logger.warning("deployment %s stays failed: restart withheld (%s)", name, reason)
        last_error = record.last_error
        if reason:
            last_error = f"{last_error or 'runtime process exited'} | restart withheld: {reason}"
        return self._observation(
            name, record, phase="failed", health_ok=False, last_error=last_error
        )

    # -------------------------------------------------------------- plumbing
    def _process_alive(self, record: DeploymentRecord) -> tuple[bool, str | None]:
        """(alive, reuse_error): health-independent process liveness + reuse guard."""
        adapter = self.supervisor.adapters[record.spec.launch.runtime]
        if not adapter.is_running(record.pid):
            return False, None
        if record.pid is not None and record.pid_create_time is not None:
            actual = self._create_time(record.pid)  # None on NoSuchProcess => dead
            if actual is None or abs(actual - record.pid_create_time) > _CREATE_TIME_TOLERANCE_S:
                return False, (
                    f"pid {record.pid} no longer names the spawned runtime "
                    f"(create_time mismatch); treating as dead"
                )
        return True, None

    def _restart_allowed(self, record: DeploymentRecord) -> tuple[bool, str]:
        if record.spec.restart_policy == RestartPolicy.NEVER:
            return False, "restart policy is 'never'"
        if record.restart_count >= record.spec.max_restarts:
            return False, (
                f"restart budget exhausted "
                f"({record.restart_count}/{record.spec.max_restarts})"
            )
        fits, reason = self._fit_check(record)
        if not fits:
            return False, f"fit-check failed: {reason}"
        return True, ""

    def _forget(self, name: str) -> None:
        self._was_ready.discard(name)
        self._misses.pop(name, None)

    def _observation(
        self,
        name: str,
        record: DeploymentRecord,
        *,
        phase: str,
        health_ok: bool,
        last_error: str | None = None,
    ) -> ObservedDeployment:
        live = phase == "hot"
        rss = 0
        if record.pid is not None and phase in {"hot", "loading"}:
            rss = self._rss_reader(record.pid)
        return ObservedDeployment(
            name=name,
            engine=_ENGINE_BY_RUNTIME.get(record.spec.launch.runtime, "llama-server"),
            state=record.state.value,
            phase=phase,
            # "" whenever not live: readers treat "" as "no live endpoint", so
            # stale routing into a dead endpoint stops within a cycle or two.
            endpoint=(record.endpoint or "") if live else "",
            pid=record.pid,
            pid_create_time=record.pid_create_time,
            rss_bytes=rss,
            health_ok=health_ok,
            last_error=last_error if last_error is not None else record.last_error,
        )


__all__ = ["ServingReconciler", "ObservedDeployment", "default_fit_check"]
