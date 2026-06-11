from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, cast

from sqlalchemy import func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, selectinload, sessionmaker

from docie_bench.orchestrator.models import BenchmarkRun, BenchmarkTask, RunArtifact, RunEvent
from docie_bench.orchestrator.schemas import ArtifactInput, RunCreate

TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled"}


class OrchestratorError(RuntimeError):
    pass


class NotFoundError(OrchestratorError):
    pass


class LeaseConflictError(OrchestratorError):
    pass


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class OrchestratorService:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    def create_run(self, request: RunCreate) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        now = utcnow()
        with self.sessions.begin() as session:
            run = BenchmarkRun(
                id=run_id,
                name=request.name,
                owner=request.owner,
                status="queued",
                notes=request.notes,
                tags=sorted(set(request.tags)),
                metadata_json=request.metadata,
                manifest_json=request.manifest,
                created_at=now,
                updated_at=now,
            )
            session.add(run)
            for spec in request.tasks:
                session.add(
                    BenchmarkTask(
                        id=str(uuid.uuid4()),
                        run_id=run_id,
                        task_key=spec.key,
                        status="queued",
                        payload_json=spec.payload,
                        max_attempts=spec.max_attempts,
                        created_at=now,
                        updated_at=now,
                    )
                )
            self._event(session, run_id, "run.created", data={"task_count": len(request.tasks)})
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.sessions() as session:
            run = session.scalar(
                select(BenchmarkRun)
                .where(BenchmarkRun.id == run_id)
                .options(
                    selectinload(BenchmarkRun.tasks),
                    selectinload(BenchmarkRun.events),
                    selectinload(BenchmarkRun.artifacts),
                )
            )
            if run is None:
                raise NotFoundError(f"Run {run_id} not found")
            return self._run_dict(run, include_tasks=True)

    def query_runs(
        self,
        *,
        owner: str | None = None,
        status: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        statement = select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc())
        if not tag:
            statement = statement.limit(limit)
        if owner:
            statement = statement.where(BenchmarkRun.owner == owner)
        if status:
            statement = statement.where(BenchmarkRun.status == status)
        if query:
            pattern = f"%{query.lower()}%"
            statement = statement.where(
                or_(
                    func.lower(BenchmarkRun.name).like(pattern),
                    func.lower(BenchmarkRun.notes).like(pattern),
                )
            )
        with self.sessions() as session:
            runs = list(session.scalars(statement))
            if tag:
                runs = [run for run in runs if tag in run.tags][:limit]
            return [self._run_dict(run, include_tasks=False) for run in runs]

    def claim_task(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        run_id: str | None = None,
    ) -> dict[str, Any] | None:
        for _ in range(5):
            now = utcnow()
            self.recover_expired(now=now)
            with self.sessions.begin() as session:
                statement = (
                    select(BenchmarkTask)
                    .join(BenchmarkRun)
                    .where(
                        BenchmarkTask.status == "queued",
                        BenchmarkRun.status.in_(["queued", "running"]),
                    )
                    .order_by(BenchmarkTask.created_at, BenchmarkTask.task_key)
                    .limit(1)
                )
                if run_id:
                    statement = statement.where(BenchmarkTask.run_id == run_id)
                task = session.scalar(statement)
                if task is None:
                    return None
                token = str(uuid.uuid4())
                expires = now + dt.timedelta(seconds=lease_seconds)
                claimed = cast(
                    CursorResult[Any],
                    session.execute(
                        update(BenchmarkTask)
                        .where(BenchmarkTask.id == task.id, BenchmarkTask.status == "queued")
                        .values(
                            status="leased",
                            worker_id=worker_id,
                            lease_token=token,
                            lease_expires_at=expires,
                            heartbeat_at=now,
                            attempt=BenchmarkTask.attempt + 1,
                            updated_at=now,
                            error_text=None,
                        )
                    ),
                )
                if claimed.rowcount != 1:
                    continue
                run = session.get(BenchmarkRun, task.run_id)
                assert run is not None
                if run.status == "queued":
                    run.status = "running"
                    run.started_at = now
                run.updated_at = now
                self._event(
                    session,
                    task.run_id,
                    "task.leased",
                    task_id=task.id,
                    data={"worker_id": worker_id, "lease_expires_at": expires.isoformat()},
                )
                return {
                    "id": task.id,
                    "run_id": task.run_id,
                    "task_key": task.task_key,
                    "payload": task.payload_json,
                    "attempt": task.attempt,
                    "max_attempts": task.max_attempts,
                    "worker_id": worker_id,
                    "lease_token": token,
                    "lease_expires_at": expires,
                }
        return None

    def heartbeat(
        self, *, task_id: str, worker_id: str, lease_token: str, lease_seconds: int
    ) -> dict[str, Any]:
        now = utcnow()
        expires = now + dt.timedelta(seconds=lease_seconds)
        with self.sessions.begin() as session:
            changed = cast(
                CursorResult[Any],
                session.execute(
                    update(BenchmarkTask)
                    .where(
                        BenchmarkTask.id == task_id,
                        BenchmarkTask.status == "leased",
                        BenchmarkTask.worker_id == worker_id,
                        BenchmarkTask.lease_token == lease_token,
                        BenchmarkTask.lease_expires_at > now,
                    )
                    .values(heartbeat_at=now, lease_expires_at=expires, updated_at=now)
                ),
            )
            if changed.rowcount != 1:
                raise LeaseConflictError(
                    "Task lease is missing, expired, or owned by another worker"
                )
        return {"task_id": task_id, "lease_expires_at": expires}

    def complete_task(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_token: str,
        result: dict[str, Any],
        artifacts: list[ArtifactInput] | None = None,
    ) -> dict[str, Any]:
        now = utcnow()
        with self.sessions.begin() as session:
            task = self._leased_task(session, task_id, worker_id, lease_token, now)
            task.status = "completed"
            task.result_json = result
            task.finished_at = now
            task.updated_at = now
            task.lease_token = None
            task.lease_expires_at = None
            for artifact in artifacts or []:
                session.add(
                    RunArtifact(
                        id=str(uuid.uuid4()),
                        run_id=task.run_id,
                        task_id=task.id,
                        **artifact.model_dump(),
                    )
                )
            self._event(session, task.run_id, "task.completed", task_id=task.id)
            self._refresh_run(session, task.run_id, now)
        return self.get_run_task(task_id)

    def fail_task(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_token: str,
        error: str,
        retryable: bool = True,
    ) -> dict[str, Any]:
        now = utcnow()
        with self.sessions.begin() as session:
            task = self._leased_task(session, task_id, worker_id, lease_token, now)
            will_retry = retryable and task.attempt < task.max_attempts
            task.status = "queued" if will_retry else "failed"
            task.error_text = error
            task.worker_id = None
            task.lease_token = None
            task.lease_expires_at = None
            task.heartbeat_at = None
            task.updated_at = now
            task.finished_at = None if will_retry else now
            self._event(
                session,
                task.run_id,
                "task.retrying" if will_retry else "task.failed",
                task_id=task.id,
                message=error,
                data={"attempt": task.attempt, "max_attempts": task.max_attempts},
            )
            self._refresh_run(session, task.run_id, now)
        return self.get_run_task(task_id)

    def recover_expired(self, *, now: dt.datetime | None = None) -> int:
        now = now or utcnow()
        recovered = 0
        with self.sessions.begin() as session:
            tasks = list(
                session.scalars(
                    select(BenchmarkTask).where(
                        BenchmarkTask.status == "leased", BenchmarkTask.lease_expires_at <= now
                    )
                )
            )
            for task in tasks:
                retry = task.attempt < task.max_attempts
                task.status = "queued" if retry else "failed"
                task.error_text = "Worker lease expired"
                task.worker_id = None
                task.lease_token = None
                task.lease_expires_at = None
                task.heartbeat_at = None
                task.updated_at = now
                task.finished_at = None if retry else now
                self._event(
                    session,
                    task.run_id,
                    "task.recovered" if retry else "task.failed",
                    task_id=task.id,
                    message="Worker lease expired",
                )
                self._refresh_run(session, task.run_id, now)
                recovered += 1
        return recovered

    def cancel_run(self, run_id: str, reason: str = "") -> dict[str, Any]:
        now = utcnow()
        with self.sessions.begin() as session:
            run = session.get(BenchmarkRun, run_id)
            if run is None:
                raise NotFoundError(f"Run {run_id} not found")
            session.execute(
                update(BenchmarkTask)
                .where(
                    BenchmarkTask.run_id == run_id,
                    BenchmarkTask.status.in_(["queued", "leased"]),
                )
                .values(
                    status="cancelled",
                    worker_id=None,
                    lease_token=None,
                    lease_expires_at=None,
                    heartbeat_at=None,
                    finished_at=now,
                    updated_at=now,
                )
            )
            run.status = "cancelled"
            run.finished_at = now
            run.updated_at = now
            self._event(session, run_id, "run.cancelled", message=reason)
        return self.get_run(run_id)

    def resume_run(self, run_id: str, reason: str = "") -> dict[str, Any]:
        now = utcnow()
        with self.sessions.begin() as session:
            run = session.get(BenchmarkRun, run_id)
            if run is None:
                raise NotFoundError(f"Run {run_id} not found")
            changed = cast(
                CursorResult[Any],
                session.execute(
                    update(BenchmarkTask)
                    .where(
                        BenchmarkTask.run_id == run_id,
                        BenchmarkTask.status.in_(["failed", "cancelled"]),
                        BenchmarkTask.attempt < BenchmarkTask.max_attempts,
                    )
                    .values(status="queued", finished_at=None, updated_at=now)
                ),
            )
            if changed.rowcount == 0:
                raise OrchestratorError("Run has no retryable failed or cancelled tasks")
            run.status = "running" if run.started_at else "queued"
            run.finished_at = None
            run.updated_at = now
            self._event(session, run_id, "run.resumed", message=reason)
        return self.get_run(run_id)

    def get_run_task(self, task_id: str) -> dict[str, Any]:
        with self.sessions() as session:
            task = session.get(BenchmarkTask, task_id)
            if task is None:
                raise NotFoundError(f"Task {task_id} not found")
            return self._task_dict(task)

    def _leased_task(
        self, session: Session, task_id: str, worker_id: str, lease_token: str, now: dt.datetime
    ) -> BenchmarkTask:
        task = session.scalar(
            select(BenchmarkTask).where(
                BenchmarkTask.id == task_id,
                BenchmarkTask.status == "leased",
                BenchmarkTask.worker_id == worker_id,
                BenchmarkTask.lease_token == lease_token,
                BenchmarkTask.lease_expires_at > now,
            )
        )
        if task is None:
            raise LeaseConflictError("Task lease is missing, expired, or owned by another worker")
        return task

    def _refresh_run(self, session: Session, run_id: str, now: dt.datetime) -> None:
        run = session.get(BenchmarkRun, run_id)
        if run is None:
            raise OrchestratorError(f"Run {run_id} not found during refresh")
        statuses = list(
            session.scalars(select(BenchmarkTask.status).where(BenchmarkTask.run_id == run_id))
        )
        if all(status == "completed" for status in statuses):
            run.status = "completed"
            run.finished_at = now
            self._event(session, run_id, "run.completed")
        elif all(status in TERMINAL_TASK_STATUSES for status in statuses):
            run.status = "failed" if "failed" in statuses else "cancelled"
            run.finished_at = now
            self._event(session, run_id, f"run.{run.status}")
        elif run.status != "cancelled":
            run.status = "running"
        run.updated_at = now

    @staticmethod
    def _event(
        session: Session,
        run_id: str,
        event_type: str,
        *,
        task_id: str | None = None,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            RunEvent(
                run_id=run_id,
                task_id=task_id,
                event_type=event_type,
                message=message,
                data_json=data or {},
            )
        )

    @classmethod
    def _run_dict(cls, run: BenchmarkRun, *, include_tasks: bool) -> dict[str, Any]:
        counts: dict[str, int] = {}
        if include_tasks:
            for task in run.tasks:
                counts[task.status] = counts.get(task.status, 0) + 1
        result = {
            "id": run.id,
            "name": run.name,
            "owner": run.owner,
            "status": run.status,
            "notes": run.notes,
            "tags": run.tags,
            "metadata": run.metadata_json,
            "manifest": run.manifest_json,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
        }
        if include_tasks:
            result.update(
                {
                    "progress": {"total": len(run.tasks), "by_status": counts},
                    "tasks": [
                        cls._task_dict(task) for task in sorted(run.tasks, key=lambda x: x.task_key)
                    ],
                    "events": [
                        {
                            "id": event.id,
                            "task_id": event.task_id,
                            "type": event.event_type,
                            "message": event.message,
                            "data": event.data_json,
                            "created_at": event.created_at,
                        }
                        for event in run.events
                    ],
                    "artifacts": [
                        {
                            "id": artifact.id,
                            "task_id": artifact.task_id,
                            "name": artifact.name,
                            "uri": artifact.uri,
                            "sha256": artifact.sha256,
                            "size_bytes": artifact.size_bytes,
                            "media_type": artifact.media_type,
                        }
                        for artifact in run.artifacts
                    ],
                }
            )
        return result

    @staticmethod
    def _task_dict(task: BenchmarkTask) -> dict[str, Any]:
        return {
            "id": task.id,
            "run_id": task.run_id,
            "task_key": task.task_key,
            "status": task.status,
            "payload": task.payload_json,
            "result": task.result_json,
            "error": task.error_text,
            "attempt": task.attempt,
            "max_attempts": task.max_attempts,
            "worker_id": task.worker_id,
            "lease_expires_at": task.lease_expires_at,
            "heartbeat_at": task.heartbeat_at,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "finished_at": task.finished_at,
        }
