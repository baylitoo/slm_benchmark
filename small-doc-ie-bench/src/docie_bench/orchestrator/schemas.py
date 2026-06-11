from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TaskSpec(BaseModel):
    key: str = Field(min_length=1, max_length=128)
    payload: dict[str, Any]
    max_attempts: int = Field(default=3, ge=1, le=100)


class RunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    owner: str = Field(min_length=1, max_length=200)
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    manifest: dict[str, Any] = Field(default_factory=dict)
    tasks: list[TaskSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_task_keys(self) -> RunCreate:
        keys = [task.key for task in self.tasks]
        if len(keys) != len(set(keys)):
            raise ValueError("Task keys must be unique within a run")
        return self


class ArtifactInput(BaseModel):
    name: str
    uri: str
    sha256: str
    size_bytes: int = Field(ge=0)
    media_type: str = "application/octet-stream"


class ClaimRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=200)
    lease_seconds: int = Field(default=60, ge=1, le=3600)
    run_id: str | None = None


class LeaseRequest(BaseModel):
    worker_id: str
    lease_token: str
    lease_seconds: int = Field(default=60, ge=1, le=3600)


class CompleteRequest(BaseModel):
    worker_id: str
    lease_token: str
    result: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactInput] = Field(default_factory=list)


class FailRequest(BaseModel):
    worker_id: str
    lease_token: str
    error: str
    retryable: bool = True


class RunAction(BaseModel):
    reason: str = ""


RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
TaskStatus = Literal["queued", "leased", "completed", "failed", "cancelled"]
