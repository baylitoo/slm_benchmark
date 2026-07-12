from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from docie_bench.serving.runtime import (
    LifecycleState,
    RuntimeAdapter,
    RuntimeKind,
    RuntimeLaunchSpec,
    default_runtime_adapters,
)


class DesiredState(StrEnum):
    RUNNING = "running"
    STOPPED = "stopped"


class Activation(StrEnum):
    """Who put a STOPPED deployment in that state (lifecycle-control metadata).

    ``MANUAL``  — a user pressed Stop: stays cold, never auto-reloaded.
    ``MANAGED`` — the autoloader evicted it for memory (PR-4): a request may
    auto-reload it. Lives in ``deployments.json`` (NOT Postgres) by design so
    the DB-optional routing contract survives (design doc §1, fix #5).
    """

    MANUAL = "manual"
    MANAGED = "managed"


def _default_create_time(pid: int) -> float | None:
    """psutil ``create_time`` for ``pid``, or None when unknowable.

    ``NoSuchProcess`` (pid already gone) and any access error map to None —
    "cannot prove it is our process" — rather than bubbling out of a reconcile
    (design doc fix #8).
    """
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 - NoSuchProcess/AccessDenied/etc => unknowable
        return None


class RestartPolicy(StrEnum):
    ALWAYS = "always"
    NEVER = "never"
    ON_FAILURE = "on_failure"


class SupervisorStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeploymentSpec:
    name: str
    launch: RuntimeLaunchSpec
    desired_state: DesiredState = DesiredState.RUNNING
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE
    max_restarts: int = 5
    health_failure_threshold: int = 3
    health_timeout_seconds: float = 2

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("deployment name must not be empty")
        if self.max_restarts < 0:
            raise ValueError("max_restarts must be non-negative")
        if self.health_failure_threshold < 1:
            raise ValueError("health_failure_threshold must be positive")
        if self.health_timeout_seconds <= 0:
            raise ValueError("health_timeout_seconds must be positive")


@dataclass
class DeploymentRecord:
    spec: DeploymentSpec
    state: LifecycleState = LifecycleState.STOPPED
    pid: int | None = None
    endpoint: str | None = None
    restart_count: int = 0
    consecutive_health_failures: int = 0
    last_error: str | None = None
    updated_at: float = field(default_factory=time.time)
    # psutil create_time() captured at spawn: the PID-reuse guard. A later
    # observer compares the live process's create_time against this — mismatch
    # (or NoSuchProcess) means "not our process => dead", so a recycled pid can
    # never keep a stale record alive (design doc §1 step 1). None => never
    # captured (pre-PR-1 records, or psutil could not see the fresh process).
    pid_create_time: float | None = None
    # Lifecycle-control metadata (PR-1 adds the fields; PR-4 drives them).
    # Persisted in deployments.json — deliberately NOT in Postgres (fix #5).
    activation: Activation = Activation.MANUAL
    pinned: bool = False
    # Unix timestamp of the last request served through this deployment, folded
    # in from the per-deployment recency sidecars by the reconciler (monotonic
    # max — last-write-wins is correct semantics). None => never served.
    last_served: float | None = None
    # True once a runtime process was observed to start and then exit on its own
    # (crash / bind collision) — as opposed to a launch that never spawned (a
    # missing binary raises before start). The reallocation caller uses this to
    # tell "port collided, try another" from "unfixable, don't churn ports".
    # Transient/advisory: not restored on reload (a fresh process reconciles anew).
    # metadata serialize=False keeps this internal signal out of every JSON-facing
    # projection (to_data, deployments.json) so it never leaks into the API/Studio.
    exited_after_start: bool = field(default=False, metadata={"serialize": False})
    # Byte offset into logs_dir/{name}.log captured immediately BEFORE the current
    # process was spawned. The exit branch reads only bytes past this offset so the
    # surfaced tail is *this* attempt's stderr, not a prior restart's (the log is
    # opened "ab" and accumulates across restarts). serialize=False for the same
    # reason: it churns every reconcile and is meaningless outside this process.
    log_offset: int = field(default=0, metadata={"serialize": False})

    def __post_init__(self) -> None:
        if self.log_offset < 0:
            raise ValueError("log_offset must be non-negative")


class PersistentSupervisor:
    """Small single-node desired-state reconciler with durable JSON state.

    Thread-safety (PR-1): every public mutator takes ``self.lock`` (a
    reentrant thread lock). In the single-replica ``serving`` service the
    background reconciler and the Inngest lifecycle handlers share ONE
    supervisor instance — one ``_records`` dict, one ``deployments.json`` —
    and interleaved read-modify-``_save()`` cycles would lose writes (design
    doc fix #4). The reconciler additionally holds the same lock across a
    whole observation cycle (it runs off the event loop via
    ``asyncio.to_thread``, hence a *thread* lock, not an asyncio one).
    ``await_ready`` deliberately locks per ``reconcile()`` call, not around
    its whole sleep loop, so a slow model load never starves the handlers.
    """

    def __init__(
        self,
        state_path: str | Path,
        *,
        adapters: Mapping[RuntimeKind, RuntimeAdapter] | None = None,
        clock: Callable[[], float] = time.time,
        logs_dir: str | Path | None = None,
        create_time: Callable[[int], float | None] = _default_create_time,
    ) -> None:
        self.state_path = Path(state_path)
        self.logs_dir = Path(logs_dir) if logs_dir is not None else self.state_path.parent / "logs"
        self.adapters = dict(adapters or default_runtime_adapters())
        self._clock = clock
        self._create_time = create_time
        self._lock = threading.RLock()
        self._records = self._load()

    @property
    def lock(self) -> threading.RLock:
        """The lock serializing ALL supervisor mutations (see class docstring)."""
        return self._lock

    def list(self) -> tuple[DeploymentRecord, ...]:
        with self._lock:
            return tuple(replace(self._records[name]) for name in sorted(self._records))

    def get(self, name: str) -> DeploymentRecord:
        try:
            return self._records[name]
        except KeyError as exc:
            raise KeyError(f"Unknown deployment {name!r}") from exc

    def records(self) -> dict[str, DeploymentRecord]:
        """The LIVE records dict — callers MUST hold ``self.lock``.

        Reconciler seam: the reconciler needs the real records (not the
        defensive copies ``list()`` returns) to read transient fields and
        decide per-record actions, all under one lock for the whole cycle.
        """
        return self._records

    def deploy(self, spec: DeploymentSpec) -> DeploymentRecord:
        with self._lock:
            current = self._records.get(spec.name)
            if current is not None and current.spec.launch != spec.launch:
                if current.pid is not None:
                    self.adapters[current.spec.launch.runtime].shutdown(current.pid)
                current = None
            self._records[spec.name] = DeploymentRecord(
                spec=spec,
                state=current.state if current else LifecycleState.STOPPED,
                pid=current.pid if current else None,
                endpoint=current.endpoint if current else None,
                restart_count=current.restart_count if current else 0,
                consecutive_health_failures=(
                    current.consecutive_health_failures if current else 0
                ),
                last_error=current.last_error if current else None,
                updated_at=self._clock(),
                pid_create_time=current.pid_create_time if current else None,
                activation=current.activation if current else Activation.MANUAL,
                pinned=current.pinned if current else False,
                last_served=current.last_served if current else None,
            )
            self._save()
            return self.reconcile(spec.name)

    def stop(self, name: str) -> DeploymentRecord:
        with self._lock:
            record = self.get(name)
            record.spec = replace(record.spec, desired_state=DesiredState.STOPPED)
            # A user Stop is always MANUAL: it stays cold and is never
            # auto-reloaded. The managed/evicted flavor is set by the PR-4
            # unload path, never here.
            record.activation = Activation.MANUAL
            return self.reconcile(name)

    def remove(self, name: str) -> None:
        with self._lock:
            record = self.get(name)
            if record.pid is not None:
                self.adapters[record.spec.launch.runtime].shutdown(record.pid)
            del self._records[name]
            self._save()

    def mark_failed(
        self, name: str, error: str | None, *, shutdown: bool = False
    ) -> DeploymentRecord:
        """Declare a deployment dead WITHOUT respawning it (reconciler seam).

        ``reconcile()`` conflates "observe dead" with "repair by respawn"; the
        background reconciler must be able to declare death when the restart
        gate (budget / fit-check) says no. ``shutdown=False`` is the default
        because the usual caller has already established the process is gone —
        and for a PID-REUSE mismatch a shutdown would SIGTERM an unrelated
        process. Pass ``shutdown=True`` only for a hung-but-alive process the
        caller wants torn down (fast-death of an unresponsive READY runtime).
        """
        with self._lock:
            record = self.get(name)
            if shutdown and record.pid is not None:
                self.adapters[record.spec.launch.runtime].shutdown(record.pid)
            if error is None and record.pid is not None:
                log_path = self.logs_dir / f"{record.spec.name}.log"
                error = self._log_tail(log_path, record.log_offset) or "runtime process exited"
            record.pid = None
            record.pid_create_time = None
            record.state = LifecycleState.FAILED
            record.last_error = error or record.last_error or "runtime process exited"
            record.updated_at = self._clock()
            self._save()
            return record

    def observe_health(
        self, name: str, *, healthy: bool, detail: str | None = None
    ) -> DeploymentRecord:
        """Record one health observation without spawn/kill side effects.

        The reconciler probes health itself (so it can apply its own fast/slow
        thresholds) and writes the outcome through this narrow seam instead of
        ``reconcile()`` (which would re-probe AND auto-respawn). A pass makes
        the record READY and clears the failure streak; a miss increments the
        streak and records the reason but does NOT change ``state`` — state
        transitions on misses are the caller's policy (fast-death via
        ``mark_failed``), so a transient blip never flips READY off and stops
        routing by itself.
        """
        with self._lock:
            record = self.get(name)
            if healthy:
                record.state = LifecycleState.READY
                record.consecutive_health_failures = 0
                record.last_error = None
            else:
                record.consecutive_health_failures += 1
                record.last_error = detail or "health check failed"
            record.updated_at = self._clock()
            self._save()
            return record

    def reset_restart_budget(self, name: str) -> DeploymentRecord:
        """Zero ``restart_count`` after a sustained healthy streak (reconciler seam).

        Without this the budget only ever ratchets up: a deployment that
        crashed ``max_restarts`` times over WEEKS of otherwise-healthy uptime
        would be permanently unrepairable, because nothing ever forgives old
        restarts. The reconciler calls this once a deployment has stayed
        healthy for ``healthy_reset_threshold`` consecutive cycles, so the
        budget bounds crash *storms* (rapid crash->respawn loops never get a
        healthy streak) without becoming a lifetime cap. No-op when the count
        is already zero (no churn, no _save).
        """
        with self._lock:
            record = self.get(name)
            if record.restart_count == 0:
                return record
            record.restart_count = 0
            record.updated_at = self._clock()
            self._save()
            return record

    def fold_recency(self, timestamps: Mapping[str, float]) -> bool:
        """Fold per-deployment recency sidecars into ``last_served`` (max-wins).

        ``last_served`` is a monotonic max-timestamp: the scaled workers stamp
        sidecars (they must never write deployments.json — single-writer, P1)
        and the reconciler folds them here each cycle. Returns True when
        anything changed (and was saved).
        """
        with self._lock:
            changed = False
            for name, timestamp in timestamps.items():
                record = self._records.get(name)
                if record is None:
                    continue
                if record.last_served is None or timestamp > record.last_served:
                    record.last_served = timestamp
                    changed = True
            if changed:
                self._save()
            return changed

    def reconcile_all(self) -> tuple[DeploymentRecord, ...]:
        with self._lock:
            return tuple(self.reconcile(name) for name in sorted(self._records))

    def reconcile(self, name: str) -> DeploymentRecord:
        with self._lock:
            return self._reconcile_locked(name)

    def _reconcile_locked(self, name: str) -> DeploymentRecord:
        record = self.get(name)
        adapter = self.adapters[record.spec.launch.runtime]
        if record.spec.desired_state == DesiredState.STOPPED:
            adapter.shutdown(record.pid)
            record.pid = None
            record.pid_create_time = None
            record.endpoint = None
            record.state = LifecycleState.STOPPED
            record.consecutive_health_failures = 0
            record.updated_at = self._clock()
            self._save()
            return record

        log_path = self.logs_dir / f"{record.spec.name}.log"
        running = record.endpoint is not None and adapter.is_running(record.pid)
        if not running:
            is_restart = record.endpoint is not None
            if record.pid is not None:
                # The process was up on the prior pass and has since exited on its
                # own — surface the real reason (its stderr tail from THIS attempt,
                # via the pre-spawn offset) instead of a bare, undiagnosable string,
                # and flag it so the deploy caller can reallocate off a bad port.
                record.pid = None
                record.pid_create_time = None
                record.state = LifecycleState.FAILED
                record.exited_after_start = True
                record.last_error = self._log_tail(log_path, record.log_offset) or (
                    "runtime process exited"
                )
            if not self._may_restart(record):
                record.updated_at = self._clock()
                self._save()
                return record
            # Capture the log size BEFORE spawning so the next exit branch tails
            # only this process's output (the log is opened "ab" and accumulates).
            record.log_offset = self._log_size(log_path)
            try:
                process = adapter.start(
                    record.spec.launch,
                    log_path=log_path,
                )
            except Exception as exc:
                # Never spawned (e.g. missing binary): NOT a started-then-exited
                # failure, so do not signal reallocation — it fails identically on
                # every port. The exception message is the honest reason.
                record.state = LifecycleState.FAILED
                record.exited_after_start = False
                record.last_error = str(exc)
                record.restart_count += 1
                record.updated_at = self._clock()
                self._save()
                return record
            record.pid = process.pid
            # Capture create_time AT SPAWN so later observers can prove the pid
            # still names this process (PID-reuse guard). None when the process
            # exited before psutil saw it, or has no pid (REMOTE).
            record.pid_create_time = (
                self._create_time(process.pid) if process.pid is not None else None
            )
            record.endpoint = process.endpoint
            record.state = LifecycleState.STARTING
            record.last_error = None
            record.exited_after_start = False
            if is_restart:
                record.restart_count += 1

        health = adapter.health(
            record.spec.launch,
            timeout=record.spec.health_timeout_seconds,
        )
        if health.healthy:
            record.state = LifecycleState.READY
            record.consecutive_health_failures = 0
            record.last_error = None
        else:
            record.consecutive_health_failures += 1
            record.last_error = health.detail or (
                f"health check returned status {health.status_code}"
            )
            if record.consecutive_health_failures >= record.spec.health_failure_threshold:
                record.state = LifecycleState.DEGRADED
                if record.pid is not None and self._may_restart(record):
                    adapter.shutdown(record.pid)
                    record.pid = None
                    record.pid_create_time = None
            else:
                record.state = LifecycleState.STARTING
        record.updated_at = self._clock()
        self._save()
        return record

    def await_ready(
        self,
        name: str,
        *,
        timeout_s: float = 60.0,
        interval_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> DeploymentRecord:
        """Re-run reconcile() until the deployment is READY or the timeout elapses.

        The deploy path spawns the runtime and probes health ONCE immediately —
        but the model takes seconds to load, so that first probe sees
        "Connection refused" and the record freezes at STARTING (the API
        cannot reach a container-local endpoint, and before PR-1's reconciler
        nothing re-probed periodically). reconcile() performs the health check
        here in the deploying service, where the ADVERTISE endpoint
        (``DOCIE_SERVING_ADVERTISE_HOST:port``, via ``reachable_launch``)
        resolves back to this same single-replica container, so this bounded
        loop re-probes until the process is actually serving. Returns whatever
        state is reached — an honest final record, never a busy-wait past
        ``timeout_s``.

        A *terminally* FAILED record (exited/never-spawned with no restarts left)
        short-circuits immediately: reconcile keeps respawning while restarts
        remain (returning STARTING), so once it returns FAILED with ``_may_restart``
        false nothing more will happen — polling on would just burn the whole
        ``timeout_s``. This is what keeps collision recovery (serve_store_model /
        serve) from blocking ~60s per exhausted attempt. A slow-loading model stays
        alive (STARTING/DEGRADED), never trips this, and keeps waiting.
        """
        deadline = self._clock() + timeout_s
        record = self.reconcile(name)
        while record.state != LifecycleState.READY and self._clock() < deadline:
            if record.state == LifecycleState.FAILED and not self._may_restart(record):
                break
            sleep(interval_s)
            record = self.reconcile(name)
        return record

    @staticmethod
    def _log_size(log_path: Path) -> int:
        try:
            return log_path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _log_tail(
        log_path: Path, offset: int, *, max_bytes: int = 2048, max_lines: int = 20
    ) -> str | None:
        """Return the runtime stderr written since ``offset`` (bounded), or None.

        Reads only the bytes past the pre-spawn ``offset`` so a prior restart's
        output (the log is opened "ab") never masks this attempt's real error, and
        caps the result so a chatty runtime cannot bloat ``deployments.json`` (the
        tail is persisted onto ``last_error``). None => nothing new/unreadable, so
        the caller keeps its generic fallback string.
        """
        try:
            with log_path.open("rb") as handle:
                handle.seek(max(0, offset))
                data = handle.read()
        except OSError:
            return None
        if not data:
            return None
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        lines = text.splitlines()[-max_lines:]
        tail = "\n".join(lines)
        if len(tail) > max_bytes:
            tail = tail[-max_bytes:]
        return tail

    def _may_restart(self, record: DeploymentRecord) -> bool:
        if record.endpoint is None and record.state == LifecycleState.STOPPED:
            return True
        if record.restart_count >= record.spec.max_restarts:
            return False
        if record.state == LifecycleState.FAILED:
            return record.spec.restart_policy in {
                RestartPolicy.ALWAYS,
                RestartPolicy.ON_FAILURE,
            }
        return record.spec.restart_policy != RestartPolicy.NEVER or record.restart_count == 0

    def _load(self) -> dict[str, DeploymentRecord]:
        if not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return {
                name: _record_from_dict(value)
                for name, value in payload.get("deployments", {}).items()
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SupervisorStateError(f"Invalid supervisor state: {self.state_path}") from exc

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "deployments": {
                name: _record_to_dict(record) for name, record in sorted(self._records.items())
            },
        }
        temporary = self.state_path.with_name(f".{self.state_path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.state_path)


def _record_to_dict(record: DeploymentRecord) -> dict[str, Any]:
    value = asdict(record)
    value["state"] = record.state.value
    value["activation"] = record.activation.value
    value["spec"]["desired_state"] = record.spec.desired_state.value
    value["spec"]["restart_policy"] = record.spec.restart_policy.value
    value["spec"]["launch"]["runtime"] = record.spec.launch.runtime.value
    value["spec"]["launch"]["extra_args"] = list(record.spec.launch.extra_args)
    value["spec"]["launch"]["env"] = dict(record.spec.launch.env)
    # In-memory-only reallocation signals: drop them from the persisted/served
    # payload so they never leak into deployments.json (log_offset churns every
    # reconcile), the /deployments API, or the Studio deployments table.
    # _record_from_dict already ignores them (they reset to defaults on reload).
    value.pop("exited_after_start", None)
    value.pop("log_offset", None)
    return value


def _record_from_dict(value: dict[str, Any]) -> DeploymentRecord:
    spec_value = dict(value["spec"])
    launch_value = dict(spec_value.pop("launch"))
    launch_value["runtime"] = RuntimeKind(launch_value["runtime"])
    launch_value["extra_args"] = tuple(launch_value.get("extra_args", ()))
    launch = RuntimeLaunchSpec(**launch_value)
    spec = DeploymentSpec(
        launch=launch,
        desired_state=DesiredState(spec_value.pop("desired_state")),
        restart_policy=RestartPolicy(spec_value.pop("restart_policy")),
        **spec_value,
    )
    raw_create_time = value.get("pid_create_time")
    raw_last_served = value.get("last_served")
    return DeploymentRecord(
        spec=spec,
        state=LifecycleState(value["state"]),
        pid=value.get("pid"),
        endpoint=value.get("endpoint"),
        restart_count=int(value.get("restart_count", 0)),
        consecutive_health_failures=int(value.get("consecutive_health_failures", 0)),
        last_error=value.get("last_error"),
        updated_at=float(value.get("updated_at", 0)),
        # New PR-1 fields default safely for records written by older code.
        pid_create_time=float(raw_create_time) if raw_create_time is not None else None,
        activation=Activation(value.get("activation", Activation.MANUAL.value)),
        pinned=bool(value.get("pinned", False)),
        last_served=float(raw_last_served) if raw_last_served is not None else None,
    )
