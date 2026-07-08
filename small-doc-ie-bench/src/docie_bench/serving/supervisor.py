from __future__ import annotations

import json
import os
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
    # True once a runtime process was observed to start and then exit on its own
    # (crash / bind collision) — as opposed to a launch that never spawned (a
    # missing binary raises before start). The reallocation caller uses this to
    # tell "port collided, try another" from "unfixable, don't churn ports".
    # Transient/advisory: not restored on reload (a fresh process reconciles anew).
    exited_after_start: bool = False
    # Byte offset into logs_dir/{name}.log captured immediately BEFORE the current
    # process was spawned. The exit branch reads only bytes past this offset so the
    # surfaced tail is *this* attempt's stderr, not a prior restart's (the log is
    # opened "ab" and accumulates across restarts).
    log_offset: int = 0

    def __post_init__(self) -> None:
        if self.log_offset < 0:
            raise ValueError("log_offset must be non-negative")


class PersistentSupervisor:
    """Small single-node desired-state reconciler with durable JSON state."""

    def __init__(
        self,
        state_path: str | Path,
        *,
        adapters: Mapping[RuntimeKind, RuntimeAdapter] | None = None,
        clock: Callable[[], float] = time.time,
        logs_dir: str | Path | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.logs_dir = Path(logs_dir) if logs_dir is not None else self.state_path.parent / "logs"
        self.adapters = dict(adapters or default_runtime_adapters())
        self._clock = clock
        self._records = self._load()

    def list(self) -> tuple[DeploymentRecord, ...]:
        return tuple(replace(self._records[name]) for name in sorted(self._records))

    def get(self, name: str) -> DeploymentRecord:
        try:
            return self._records[name]
        except KeyError as exc:
            raise KeyError(f"Unknown deployment {name!r}") from exc

    def deploy(self, spec: DeploymentSpec) -> DeploymentRecord:
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
            consecutive_health_failures=current.consecutive_health_failures if current else 0,
            last_error=current.last_error if current else None,
            updated_at=self._clock(),
        )
        self._save()
        return self.reconcile(spec.name)

    def stop(self, name: str) -> DeploymentRecord:
        record = self.get(name)
        record.spec = replace(record.spec, desired_state=DesiredState.STOPPED)
        return self.reconcile(name)

    def remove(self, name: str) -> None:
        record = self.get(name)
        if record.pid is not None:
            self.adapters[record.spec.launch.runtime].shutdown(record.pid)
        del self._records[name]
        self._save()

    def reconcile_all(self) -> tuple[DeploymentRecord, ...]:
        return tuple(self.reconcile(name) for name in sorted(self._records))

    def reconcile(self, name: str) -> DeploymentRecord:
        record = self.get(name)
        adapter = self.adapters[record.spec.launch.runtime]
        if record.spec.desired_state == DesiredState.STOPPED:
            adapter.shutdown(record.pid)
            record.pid = None
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
        "Connection refused" and the record freezes at STARTING forever (there
        is no background reconcile, and the API cannot reach the worker-local
        endpoint). reconcile() performs the health check here in the worker,
        where 127.0.0.1 IS reachable, so this bounded loop re-probes until the
        process is actually serving. Returns whatever state is reached — an
        honest final record, never a busy-wait past ``timeout_s``.
        """
        deadline = self._clock() + timeout_s
        record = self.reconcile(name)
        while record.state != LifecycleState.READY and self._clock() < deadline:
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
    value["spec"]["desired_state"] = record.spec.desired_state.value
    value["spec"]["restart_policy"] = record.spec.restart_policy.value
    value["spec"]["launch"]["runtime"] = record.spec.launch.runtime.value
    value["spec"]["launch"]["extra_args"] = list(record.spec.launch.extra_args)
    value["spec"]["launch"]["env"] = dict(record.spec.launch.env)
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
    return DeploymentRecord(
        spec=spec,
        state=LifecycleState(value["state"]),
        pid=value.get("pid"),
        endpoint=value.get("endpoint"),
        restart_count=int(value.get("restart_count", 0)),
        consecutive_health_failures=int(value.get("consecutive_health_failures", 0)),
        last_error=value.get("last_error"),
        updated_at=float(value.get("updated_at", 0)),
    )
