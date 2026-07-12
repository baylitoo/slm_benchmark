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
* Publishes the ``serving_node`` resource snapshot (PR-2): the
  ``ResourceTracker`` folds this cycle's observations (steady-state footprint
  calibration, live-RSS sum) and reads node total/free RAM cgroup-v2-first,
  then the snapshot is upserted as the single ``serving_node`` row — same
  best-effort no-DB degradation as the observed rows.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from docie_bench.serving import recency
from docie_bench.serving.resources import (
    FootprintStore,
    NodeSnapshot,
    ResourceTracker,
    footprint_bytes,
    predicted_footprint_for_model,
    process_rss,
    publish_snapshot_via_catalog,
    read_node_memory,
)
from docie_bench.serving.runtime import LifecycleState, RuntimeKind, RuntimeLaunchSpec
from docie_bench.serving.supervisor import (
    Activation,
    DeploymentRecord,
    DesiredState,
    PersistentSupervisor,
    RestartPolicy,
    _default_create_time,
)

logger = logging.getLogger(__name__)

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


class ReconcilerSingletonError(RuntimeError):
    """Another live serving replica already holds the reconciler lease."""


@dataclass
class ReconcilerLease:
    """Advisory single-reconciler lease on the shared serving-state volume.

    ``deploy.replicas: 1`` in compose is advisory — ``--scale serving=2`` (or a
    copy-pasted override file) silently bypasses it, and N reconcilers on the
    shared volume are exactly the multi-writer clobber P1 designed out. This
    lease turns that misconfiguration into a loud refusal: every reconciler
    heartbeats ``<serving-home>/reconciler-lease.json`` (atomic single-key
    write, same pattern as the recency sidecars) each cycle, and a starting
    reconciler REFUSES to run when a fresh lease from a DIFFERENT instance
    exists. Advisory, not a distributed lock: a perfectly simultaneous double
    start can still race the first claim — the guard exists to catch the
    realistic "second replica joins an already-running fleet" case, not to be
    Paxos. ``--scale serving=N`` for N>1 remains forbidden regardless.
    """

    path: Path
    instance_id: str
    stale_after_s: float = 60.0
    clock: Callable[[], float] = field(default=time.time)

    def claim(self) -> None:
        """Take the lease or raise ``ReconcilerSingletonError`` if held live.

        A stale lease (older than ``stale_after_s`` — e.g. this same container
        restarted, or a crashed replica) is overwritten. A fresh lease from
        this same ``instance_id`` is simply re-claimed (idempotent restart).
        """
        holder = self._read()
        if holder is not None:
            other_instance, stamped_at = holder
            age = self.clock() - stamped_at
            if other_instance != self.instance_id and 0 <= age < self.stale_after_s:
                raise ReconcilerSingletonError(
                    f"another serving replica ({other_instance!r}) heartbeated the "
                    f"reconciler lease {age:.1f}s ago (< {self.stale_after_s:.0f}s): "
                    f"refusing to start a second reconciler. The serving service is "
                    f"single-replica by design — never `--scale serving=2`; scale "
                    f"back to 1 (or remove the stale replica) and restart."
                )
        self.refresh()

    def refresh(self) -> None:
        """Heartbeat the lease (best-effort, atomic; called every cycle)."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            temporary.write_text(
                json.dumps(
                    {"instance": self.instance_id, "timestamp": self.clock()},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError:  # pragma: no cover - disk hiccup must not kill the loop
            logger.warning("could not heartbeat reconciler lease %s", self.path, exc_info=True)

    def release(self) -> None:
        """Drop the lease on clean shutdown iff it is still ours (best-effort)."""
        holder = self._read()
        if holder is not None and holder[0] == self.instance_id:
            with contextlib.suppress(OSError):
                self.path.unlink()

    def _read(self) -> tuple[str, float] | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return str(payload["instance"]), float(payload["timestamp"])
        except (OSError, ValueError, KeyError, TypeError):
            return None  # missing/corrupt lease reads as "unheld"


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
    # Launch model reference (GGUF path) — the resource tracker's calibration
    # key, so a per-model footprint survives redeploys under new names.
    model: str = ""


def _mmproj_bytes_from_launch(launch: RuntimeLaunchSpec) -> int:
    """Size of the vision projector a launch will load (0 when none/unreadable).

    ``needs_mmproj`` families launch with ``--mmproj <path>`` in
    ``extra_args`` (wired by ``ModelStore.family_launch_args``); llama-server
    loads that projector fully resident, so the fit gate must price it —
    ``predicted_footprint_for_model`` already takes ``mmproj_bytes``, this
    recovers the number from the record. Unreadable/missing degrades to 0
    (fail-open, like every other unknowable in the gate).
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


def default_fit_check(
    record: DeploymentRecord, *, tracker: ResourceTracker | None = None
) -> tuple[bool, str]:
    """Restart fit gate, priced by the PR-2 resource tracker's numbers.

    ``predicted_footprint_for_model`` (on-disk weights + KV(ctx) + runtime
    overhead + mmproj for vision launches), lifted to the CALIBRATED working
    footprint ``max(observed_steady_rss, predicted)`` — the whole point of
    steady-state calibration is that a model measured to need more than its
    formula must be priced at the measurement, or the gate re-approves the
    exact OOM the budget exists to stop. Checked against the cgroup-aware node
    free RAM — the same numbers the published snapshot reports, replacing
    PR-1's psutil-only heuristic (which read the elastic WSL2 VM total inside
    a container). Fail-open on anything unknowable (no model file,
    unmeasurable node): the gate exists to stop crash->OOM->respawn storms,
    not to block legitimate repairs on measurement hiccups.

    ``tracker`` supplies the calibration sidecars; the reconciler binds its
    own tracker here. Without one (direct/standalone calls) the default
    ``FootprintStore`` on the serving volume is consulted, so calibration is
    never silently ignored.
    """
    launch = record.spec.launch
    predicted = predicted_footprint_for_model(
        size_bytes=None,
        model_path=launch.model,
        context_length=launch.context_length,
        mmproj_bytes=_mmproj_bytes_from_launch(launch),
    )
    if predicted is None:
        return True, ""
    footprints = tracker.footprints if tracker is not None else FootprintStore()
    needed = footprint_bytes(predicted, footprints.get(launch.model))
    try:
        free = read_node_memory().free_bytes
    except Exception:  # noqa: BLE001 - unmeasurable => fail-open
        return True, ""
    if free < needed:
        return False, (
            f"needs ~{needed} bytes (max of calibrated steady-state RSS and "
            f"predicted weights + kv-cache + overhead + mmproj) "
            f"but only {free} available"
        )
    return True, ""


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
        rss_reader: Callable[[int], int] = process_rss,
        create_time: Callable[[int], float | None] = _default_create_time,
        publisher: Callable[[list[ObservedDeployment]], None] = _publish_via_catalog,
        recency_home: Path | None = None,
        healthy_reset_threshold: int = 30,
        lease: ReconcilerLease | None = None,
        tracker: ResourceTracker | None = None,
        snapshot_publisher: Callable[[NodeSnapshot], None] = publish_snapshot_via_catalog,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if ready_miss_threshold < 1:
            raise ValueError("ready_miss_threshold must be positive")
        if healthy_reset_threshold < 1:
            raise ValueError("healthy_reset_threshold must be positive")
        self.supervisor = supervisor
        self.interval_s = interval_s
        self.ready_miss_threshold = ready_miss_threshold
        self.healthy_reset_threshold = healthy_reset_threshold
        self._rss_reader = rss_reader
        self._create_time = create_time
        self._publisher = publisher
        self._recency_home = recency_home
        self._lease = lease
        # PR-2 resource tracker: node RAM snapshot + steady-state footprint
        # calibration each cycle. The default persists footprint sidecars
        # under the same serving home the recency fold reads, so tests that
        # redirect recency_home never write calibration into the real home.
        if tracker is None:
            tracker = ResourceTracker(footprints=FootprintStore(home=recency_home))
        self._tracker = tracker
        # The default fit gate must consume THIS reconciler's calibration
        # sidecars (max(calibrated, predicted)), not price restarts from the
        # raw prediction — bind the tracker in. An injected fit_check is used
        # verbatim (tests / custom gates).
        if fit_check is default_fit_check:
            self._fit_check: Callable[[DeploymentRecord], tuple[bool, str]] = (
                lambda record: default_fit_check(record, tracker=self._tracker)
            )
        else:
            self._fit_check = fit_check
        self._snapshot_publisher = snapshot_publisher
        # Reconciler-local memory: which deployments were observed READY (so a
        # later miss is a fast-death candidate, not a slow cold-load) and how
        # many consecutive misses each has accrued since. _was_ready is a CACHE,
        # not the authority: was-readiness is also derived from the persisted
        # record state (READY survives deployments.json reloads), so a serving
        # restart never resets death-detection for a hung runtime back to the
        # ~10-minute slow-load tolerance (see _was_previously_ready).
        self._was_ready: set[str] = set()
        self._misses: dict[str, int] = {}
        # Consecutive healthy cycles per deployment: after
        # ``healthy_reset_threshold`` of them the restart budget is forgiven
        # (reset to 0), so the budget bounds crash STORMS without becoming a
        # lifetime cap. Any miss or death zeroes the streak.
        self._healthy_streak: dict[str, int] = {}

    def claim_singleton(self) -> None:
        """Claim the singleton lease before starting the loop.

        Raises ``ReconcilerSingletonError`` when another live serving replica
        holds it (the caller must then NOT start this reconciler). No-op when
        no lease was configured (tests / embedded use).
        """
        if self._lease is not None:
            self._lease.claim()

    # ------------------------------------------------------------------ loop
    async def run_forever(self, *, stop_event: asyncio.Event | None = None) -> None:
        """Cycle until cancelled (or ``stop_event`` is set). Never raises."""
        logger.info(
            "serving reconciler started (interval=%.1fs, ready_miss_threshold=%d)",
            self.interval_s,
            self.ready_miss_threshold,
        )
        try:
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
        finally:
            # Clean shutdown drops the singleton lease so a replacement replica
            # does not have to wait out stale_after_s (a crash leaves the lease
            # to expire naturally — that is the point of the staleness window).
            if self._lease is not None:
                self._lease.release()

    # ----------------------------------------------------------------- cycle
    def run_cycle(self) -> list[ObservedDeployment]:
        """One observation/repair pass; returns what was (or would be) published."""
        with self.supervisor.lock:
            observations = self._observe_locked()
        # Publish OUTSIDE the lock: observations are immutable snapshots and a
        # slow database must not block the lifecycle handlers.
        self._publisher(observations)
        # Resource snapshot (PR-2), also outside the lock: pure measurement
        # (cgroup/psutil reads + footprint sidecar writes). Best-effort — a
        # measurement hiccup must never mark the repair cycle failed.
        try:
            snapshot = self._tracker.observe_cycle(observations)
        except Exception:  # noqa: BLE001 - unmeasurable node != broken cycle
            logger.warning("resource snapshot failed this cycle", exc_info=True)
        else:
            # The publisher gets the same containment as the measurement: an
            # injected publisher that raises (or a non-CatalogUnavailable DB
            # driver surprise) must not abort the cycle tail — the lease
            # heartbeat below is what keeps a second replica refusing.
            try:
                self._snapshot_publisher(snapshot)
            except Exception:  # noqa: BLE001 - publish hiccup != broken cycle
                logger.warning("node snapshot publish failed this cycle", exc_info=True)
        # Heartbeat the singleton lease each completed cycle so a second
        # serving replica joining later sees a live holder and refuses.
        if self._lease is not None:
            self._lease.refresh()
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
        for stale in set(self._healthy_streak) - set(records):
            del self._healthy_streak[stale]
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
            record = self._credit_healthy_cycle(name, record)
            return self._observation(name, record, phase="hot", health_ok=True)

        self._healthy_streak.pop(name, None)  # any miss restarts the streak
        detail = health.detail or f"health check returned status {health.status_code}"
        if self._was_previously_ready(name, record):
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
    def _was_previously_ready(self, name: str, record: DeploymentRecord) -> bool:
        """Was this deployment ever READY? Survives a serving-service restart.

        The in-memory ``_was_ready`` set is process-local; if it were the only
        signal, restarting the serving container would re-enter the generous
        slow-load tolerance for a HUNG (alive-but-unreachable) runtime and take
        ~10 minutes to re-declare it dead. The persisted record state closes
        that hole: ``observe_health(healthy=False)`` deliberately never flips
        READY off (state transitions on misses are this loop's policy), so a
        record that reached READY still says READY in ``deployments.json``
        after a reload — derived was-readiness, no extra persistence needed.
        """
        return name in self._was_ready or record.state == LifecycleState.READY

    def _credit_healthy_cycle(self, name: str, record: DeploymentRecord) -> DeploymentRecord:
        """Count a healthy cycle; forgive the restart budget after a streak.

        ``restart_count`` otherwise only ratchets up, so ``max_restarts=5``
        would be a LIFETIME cap: five crashes spread over weeks of healthy
        uptime would leave the deployment permanently unrepairable. After
        ``healthy_reset_threshold`` consecutive healthy cycles (default 30 —
        five minutes at the 10s interval) the budget resets to zero. A crash
        storm never earns the reset: every miss or death zeroes the streak.
        """
        streak = self._healthy_streak.get(name, 0) + 1
        self._healthy_streak[name] = streak
        if streak >= self.healthy_reset_threshold and record.restart_count > 0:
            logger.info(
                "deployment %s healthy for %d consecutive cycles: "
                "resetting restart budget (was %d/%d)",
                name,
                streak,
                record.restart_count,
                record.spec.max_restarts,
            )
            record = self.supervisor.reset_restart_budget(name)
        return record

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
        self._healthy_streak.pop(name, None)

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
            model=record.spec.launch.model,
        )


__all__ = [
    "ServingReconciler",
    "ObservedDeployment",
    "ReconcilerLease",
    "ReconcilerSingletonError",
    "default_fit_check",
]
