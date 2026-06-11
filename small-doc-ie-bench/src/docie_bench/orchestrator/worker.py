from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from docie_bench.orchestrator.artifacts import ArtifactStore
from docie_bench.orchestrator.schemas import ArtifactInput
from docie_bench.orchestrator.service import LeaseConflictError, OrchestratorService


@dataclass(frozen=True)
class ArtifactOutput:
    name: str
    content: bytes
    media_type: str = "application/octet-stream"


@dataclass(frozen=True)
class TaskOutput:
    result: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactOutput] = field(default_factory=list)


Executor = Callable[[dict[str, Any]], TaskOutput | Awaitable[TaskOutput]]


class BenchmarkWorker:
    """Claims and executes one task at a time with lease heartbeats."""

    def __init__(
        self,
        *,
        worker_id: str,
        service: OrchestratorService,
        executor: Executor,
        artifact_store: ArtifactStore,
        lease_seconds: int = 60,
        heartbeat_seconds: float | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.service = service
        self.executor = executor
        self.artifact_store = artifact_store
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds or max(0.25, lease_seconds / 3)

    async def run_once(self, *, run_id: str | None = None) -> bool:
        task = self.service.claim_task(
            worker_id=self.worker_id, lease_seconds=self.lease_seconds, run_id=run_id
        )
        if task is None:
            return False
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(task, stop))
        try:
            if inspect.iscoroutinefunction(self.executor):
                output = await self.executor(task["payload"])
            else:
                output = await asyncio.to_thread(self.executor, task["payload"])
            if inspect.isawaitable(output):
                output = await output
            artifacts = []
            for artifact in output.artifacts:
                stored = self.artifact_store.put(
                    run_id=task["run_id"],
                    task_id=task["id"],
                    name=artifact.name,
                    content=artifact.content,
                    media_type=artifact.media_type,
                )
                artifacts.append(ArtifactInput(**stored.__dict__))
            self.service.complete_task(
                task_id=task["id"],
                worker_id=self.worker_id,
                lease_token=task["lease_token"],
                result=output.result,
                artifacts=artifacts,
            )
        except LeaseConflictError:
            raise
        except Exception as exc:
            self.service.fail_task(
                task_id=task["id"],
                worker_id=self.worker_id,
                lease_token=task["lease_token"],
                error=repr(exc),
                retryable=True,
            )
        finally:
            stop.set()
            await heartbeat
        return True

    async def _heartbeat(self, task: dict[str, Any], stop: asyncio.Event) -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.heartbeat_seconds)
                return
            except TimeoutError:
                self.service.heartbeat(
                    task_id=task["id"],
                    worker_id=self.worker_id,
                    lease_token=task["lease_token"],
                    lease_seconds=self.lease_seconds,
                )


def json_artifact(name: str, value: Any) -> ArtifactOutput:
    return ArtifactOutput(
        name=name,
        content=json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"),
        media_type="application/json",
    )
