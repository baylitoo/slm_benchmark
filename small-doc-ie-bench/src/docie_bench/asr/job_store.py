"""Persistence and artifact access for durable ASR transcription jobs."""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, Table, delete, select, text
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload, sessionmaker
from sqlalchemy.schema import CreateIndex, CreateTable

from docie_bench.asr.job_models import ASRJob, ASRJobArtifact, ASRJobItem, utcnow
from docie_bench.storage.db import get_session_factory
from docie_bench.studio.store import ArtifactBlobStore, StoredBlob, default_blob_store

ARTIFACT_URI_PREFIX = "/v1/audio/transcription-jobs"
_ASR_TABLES_LOCK_KEY = 0x0D0C1E12
TERMINAL_STATUSES = frozenset({"completed", "completed_with_errors", "failed", "cancelled"})


class ASRJobStoreUnavailableError(RuntimeError):
    pass


class ASRJobStore:
    def __init__(
        self,
        factory: sessionmaker[Session] | None,
        blobs: ArtifactBlobStore,
    ) -> None:
        self.factory = factory
        self.blobs = blobs

    @property
    def enabled(self) -> bool:
        return self.factory is not None

    @contextmanager
    def _session(self) -> Iterator[Session]:
        if self.factory is None:
            raise ASRJobStoreUnavailableError(
                "Durable ASR jobs require DATABASE_URL and the shared artifact store"
            )
        session = self.factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def claim(
        self,
        *,
        event_id: str,
        tenant_id: str,
        idempotency_key: str,
        channel: str,
        deployment: str,
        model: str,
        options: dict[str, Any],
        raw_retention: str,
        raw_expires_at: dt.datetime | None,
        items: Sequence[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Create a queued job, or return the tenant's existing deduplicated job."""
        try:
            with self._session() as session:
                existing = session.scalar(
                    _job_query().where(
                        ASRJob.tenant_id == tenant_id,
                        ASRJob.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    return "exists", _job_to_dict(existing)
                job = ASRJob(
                    event_id=event_id,
                    tenant_id=tenant_id,
                    idempotency_key=idempotency_key,
                    channel=channel,
                    deployment=deployment,
                    model=model,
                    status="queued",
                    total_items=len(items),
                    options_json=dict(options),
                    raw_retention=raw_retention,
                    raw_expires_at=raw_expires_at,
                )
                session.add(job)
                session.add_all(
                    ASRJobItem(
                        job_event_id=event_id,
                        position=position,
                        filename=str(item["filename"])[:300],
                        input_relkey=str(item["relkey"]),
                        input_sha256=str(item["sha256"]),
                        input_size_bytes=int(item["size_bytes"]),
                        mime_type=str(item["mime_type"]),
                        reference_text=(
                            str(item["reference"]) if item.get("reference") is not None else None
                        ),
                    )
                    for position, item in enumerate(items)
                )
                session.flush()
                session.refresh(job)
                return "claimed", _job_to_dict(job)
        except IntegrityError:
            # Two API replicas can race the same tenant/idempotency key. The
            # unique constraint chooses the winner; fetch it in a fresh tx.
            with self._session() as session:
                existing = session.scalar(
                    _job_query().where(
                        ASRJob.tenant_id == tenant_id,
                        ASRJob.idempotency_key == idempotency_key,
                    )
                )
                if existing is None:
                    raise
                return "exists", _job_to_dict(existing)

    def get(self, event_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            job = session.scalar(
                _job_query().where(
                    ASRJob.event_id == event_id,
                    ASRJob.tenant_id == tenant_id,
                )
            )
            return _job_to_dict(job) if job is not None else None

    def get_internal(self, event_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            job = session.scalar(_job_query().where(ASRJob.event_id == event_id))
            return _job_to_dict(job) if job is not None else None

    def list(self, *, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._session() as session:
            jobs = session.scalars(
                _job_query()
                .where(ASRJob.tenant_id == tenant_id)
                .order_by(ASRJob.created_at.desc())
                .limit(max(1, min(limit, 500)))
            ).all()
            return [_job_to_dict(job) for job in jobs]

    def start(self, event_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            job = session.scalar(_job_query().where(ASRJob.event_id == event_id))
            if job is None:
                return None
            if job.status in TERMINAL_STATUSES:
                return _job_to_dict(job)
            job.status = "cancelling" if job.cancel_requested_at else "running"
            job.started_at = job.started_at or utcnow()
            job.updated_at = utcnow()
            session.flush()
            return _job_to_dict(job)

    def discard_queued(self, event_id: str) -> None:
        """Remove a job that could not be enqueued so its key can be retried."""
        with self._session() as session:
            job = session.get(ASRJob, event_id)
            if job is None or job.status != "queued":
                return
            session.execute(delete(ASRJobItem).where(ASRJobItem.job_event_id == event_id))
            session.delete(job)

    def begin_item(self, event_id: str, position: int, *, attempt: int) -> bool:
        with self._session() as session:
            job = session.get(ASRJob, event_id)
            if job is None or job.cancel_requested_at is not None:
                return False
            item = session.scalar(
                select(ASRJobItem).where(
                    ASRJobItem.job_event_id == event_id,
                    ASRJobItem.position == position,
                )
            )
            if item is None or item.status in {"completed", "failed", "cancelled"}:
                return False
            item.status = "running"
            item.attempts = max(item.attempts, attempt)
            item.started_at = item.started_at or utcnow()
            item.updated_at = utcnow()
            return True

    def complete_item(
        self,
        event_id: str,
        position: int,
        *,
        result: dict[str, Any],
        metrics: dict[str, Any] | None,
        artifacts: Sequence[tuple[str, str, StoredBlob]],
        attempts: int,
    ) -> None:
        with self._session() as session:
            item = session.scalar(
                select(ASRJobItem).where(
                    ASRJobItem.job_event_id == event_id,
                    ASRJobItem.position == position,
                )
            )
            if item is None:
                raise KeyError(f"ASR job item {event_id}:{position} not found")
            item.status = "completed"
            item.detected_language = _optional_text(result.get("language"), 32)
            item.duration_seconds = _optional_float(result.get("duration"))
            item.processing_seconds = _optional_float(result.get("processing_seconds"))
            item.result_json = dict(result)
            item.metrics_json = dict(metrics) if metrics is not None else None
            item.error_text = None
            item.attempts = max(item.attempts, attempts)
            item.completed_at = utcnow()
            item.updated_at = utcnow()
            for name, kind, blob in artifacts:
                _upsert_artifact(
                    session,
                    event_id=event_id,
                    item_position=position,
                    name=name,
                    kind=kind,
                    blob=blob,
                )

    def fail_item(self, event_id: str, position: int, *, error: str, attempts: int) -> None:
        with self._session() as session:
            item = session.scalar(
                select(ASRJobItem).where(
                    ASRJobItem.job_event_id == event_id,
                    ASRJobItem.position == position,
                )
            )
            if item is None:
                raise KeyError(f"ASR job item {event_id}:{position} not found")
            item.status = "failed"
            item.error_text = error[:20_000]
            item.attempts = max(item.attempts, attempts)
            item.completed_at = utcnow()
            item.updated_at = utcnow()

    def add_job_artifact(self, event_id: str, *, name: str, kind: str, blob: StoredBlob) -> None:
        with self._session() as session:
            _upsert_artifact(
                session,
                event_id=event_id,
                item_position=None,
                name=name,
                kind=kind,
                blob=blob,
            )

    def request_cancel(self, event_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            job = session.scalar(
                _job_query().where(
                    ASRJob.event_id == event_id,
                    ASRJob.tenant_id == tenant_id,
                )
            )
            if job is None:
                return None
            if job.status not in TERMINAL_STATUSES:
                job.cancel_requested_at = job.cancel_requested_at or utcnow()
                if job.status == "queued":
                    job.status = "cancelled"
                    job.completed_at = utcnow()
                    for item in job.items:
                        item.status = "cancelled"
                        item.completed_at = job.completed_at
                    _schedule_raw_retention(job)
                else:
                    job.status = "cancelling"
                job.updated_at = utcnow()
                session.flush()
            return _job_to_dict(job)

    def is_cancel_requested(self, event_id: str) -> bool:
        with self._session() as session:
            value = session.scalar(
                select(ASRJob.cancel_requested_at).where(ASRJob.event_id == event_id)
            )
            return value is not None

    def settle(self, event_id: str, *, metrics: dict[str, Any]) -> dict[str, Any]:
        with self._session() as session:
            job = session.scalar(_job_query().where(ASRJob.event_id == event_id))
            if job is None:
                raise KeyError(f"ASR job {event_id!r} not found")
            completed = sum(item.status == "completed" for item in job.items)
            failed = sum(item.status == "failed" for item in job.items)
            if job.cancel_requested_at is not None:
                status = "cancelled"
                for item in job.items:
                    if item.status in {"pending", "running"}:
                        item.status = "cancelled"
                        item.completed_at = utcnow()
            elif failed:
                status = "completed_with_errors"
            else:
                status = "completed"
            job.status = status
            job.completed_items = completed
            job.failed_items = failed
            job.metrics_json = dict(metrics)
            job.completed_at = utcnow()
            job.updated_at = utcnow()
            _schedule_raw_retention(job)
            session.flush()
            return _job_to_dict(job)

    def fail_job(self, event_id: str, *, error: str) -> dict[str, Any] | None:
        with self._session() as session:
            job = session.scalar(_job_query().where(ASRJob.event_id == event_id))
            if job is None:
                return None
            if job.status not in TERMINAL_STATUSES:
                job.status = "failed"
                job.error_text = error[:20_000]
                job.completed_at = utcnow()
                job.updated_at = utcnow()
                _schedule_raw_retention(job)
                session.flush()
            return _job_to_dict(job)

    def open_artifact(
        self, artifact_id: str, *, tenant_id: str
    ) -> tuple[dict[str, Any], bytes] | None:
        with self._session() as session:
            row = session.scalar(
                select(ASRJobArtifact)
                .join(ASRJob, ASRJob.event_id == ASRJobArtifact.job_event_id)
                .where(
                    ASRJobArtifact.id == artifact_id,
                    ASRJob.tenant_id == tenant_id,
                )
            )
            if row is None:
                return None
            meta = _artifact_to_dict(row)
            relkey = row.relkey
        try:
            return meta, self.blobs.read(relkey)
        except (FileNotFoundError, ValueError):
            return None

    def gc(
        self,
        *,
        max_age_days: int,
        max_jobs: int,
        now: dt.datetime | None = None,
    ) -> dict[str, int]:
        """Expire retained raw inputs and bound completed job history.

        Blob deletion is intentionally left to the shared mark-and-sweep in
        ``RunStore.gc``; it sees ASR references too, so cross-domain content
        dedup can never cause this store to delete another feature's blob.
        """
        current = now or utcnow()
        cutoff = current - dt.timedelta(days=max_age_days)
        with self._session() as session:
            expired_raw = session.scalars(
                select(ASRJobItem)
                .join(ASRJob, ASRJob.event_id == ASRJobItem.job_event_id)
                .where(
                    ASRJob.raw_expires_at.is_not(None),
                    ASRJob.raw_expires_at <= current,
                    ASRJobItem.input_relkey.is_not(None),
                )
            ).all()
            for item in expired_raw:
                item.input_relkey = None

            jobs = session.scalars(
                select(ASRJob).order_by(ASRJob.created_at.desc())
            ).all()
            doomed: set[str] = set()
            retained = 0
            for job in jobs:
                created = _aware(job.created_at)
                if job.status in TERMINAL_STATUSES and (
                    created < cutoff or retained >= max_jobs
                ):
                    doomed.add(job.event_id)
                else:
                    retained += 1
            if doomed:
                session.execute(
                    delete(ASRJobArtifact).where(ASRJobArtifact.job_event_id.in_(doomed))
                )
                session.execute(delete(ASRJobItem).where(ASRJobItem.job_event_id.in_(doomed)))
                session.execute(delete(ASRJob).where(ASRJob.event_id.in_(doomed)))
            return {
                "deleted_asr_jobs": len(doomed),
                "expired_asr_inputs": len(expired_raw),
                "retained_asr_jobs": retained,
            }


def ensure_asr_job_tables(engine: Engine) -> bool:
    tables = (
        ASRJob.__table__,
        ASRJobItem.__table__,
        ASRJobArtifact.__table__,
    )
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                text("SELECT pg_advisory_xact_lock(:key)"), {"key": _ASR_TABLES_LOCK_KEY}
            )
        existed = sa_inspect(connection).has_table("asr_jobs")
        for table in tables:
            assert isinstance(table, Table)
            connection.execute(CreateTable(table, if_not_exists=True))
            for index in table.indexes:
                connection.execute(CreateIndex(index, if_not_exists=True))
    return not existed


def default_asr_job_store() -> ASRJobStore:
    return ASRJobStore(get_session_factory(), default_blob_store())


def _job_query() -> Any:
    return select(ASRJob).options(
        selectinload(ASRJob.items),
        selectinload(ASRJob.artifacts),
    )


def _upsert_artifact(
    session: Session,
    *,
    event_id: str,
    item_position: int | None,
    name: str,
    kind: str,
    blob: StoredBlob,
) -> None:
    row = session.scalar(
        select(ASRJobArtifact).where(
            ASRJobArtifact.job_event_id == event_id,
            ASRJobArtifact.name == name,
        )
    )
    if row is None:
        row = ASRJobArtifact(
            id=str(uuid.uuid4()),
            job_event_id=event_id,
            item_position=item_position,
            name=name[:300],
            kind=kind[:32],
            relkey=blob.relkey,
            sha256=blob.sha256,
            size_bytes=blob.size_bytes,
            media_type=blob.media_type,
        )
        session.add(row)
    else:
        row.item_position = item_position
        row.kind = kind[:32]
        row.relkey = blob.relkey
        row.sha256 = blob.sha256
        row.size_bytes = blob.size_bytes
        row.media_type = blob.media_type


def _job_to_dict(job: ASRJob) -> dict[str, Any]:
    artifacts_by_item: dict[int | None, list[dict[str, Any]]] = {}
    for artifact in job.artifacts:
        artifacts_by_item.setdefault(artifact.item_position, []).append(
            _artifact_to_dict(artifact)
        )
    return {
        "event_id": job.event_id,
        "tenant_id": job.tenant_id,
        "idempotency_key": job.idempotency_key,
        "channel": job.channel,
        "deployment": job.deployment,
        "model": job.model,
        "status": job.status,
        "total_items": job.total_items,
        "completed_items": job.completed_items,
        "failed_items": job.failed_items,
        "options": job.options_json or {},
        "metrics": job.metrics_json,
        "error": job.error_text,
        "raw_retention": job.raw_retention,
        "raw_expires_at": _isoformat(job.raw_expires_at),
        "cancel_requested_at": _isoformat(job.cancel_requested_at),
        "created_at": _isoformat(job.created_at),
        "started_at": _isoformat(job.started_at),
        "completed_at": _isoformat(job.completed_at),
        "updated_at": _isoformat(job.updated_at),
        "artifacts": artifacts_by_item.get(None, []),
        "items": [
            {
                "position": item.position,
                "filename": item.filename,
                "input_sha256": item.input_sha256,
                "input_size_bytes": item.input_size_bytes,
                "raw_available": bool(item.input_relkey),
                "mime_type": item.mime_type,
                "reference": item.reference_text,
                "status": item.status,
                "detected_language": item.detected_language,
                "duration_seconds": item.duration_seconds,
                "processing_seconds": item.processing_seconds,
                "result": item.result_json,
                "metrics": item.metrics_json,
                "error": item.error_text,
                "attempts": item.attempts,
                "started_at": _isoformat(item.started_at),
                "completed_at": _isoformat(item.completed_at),
                "artifacts": artifacts_by_item.get(item.position, []),
            }
            for item in job.items
        ],
    }


def _artifact_to_dict(row: ASRJobArtifact) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "kind": row.kind,
        "sha256": row.sha256,
        "size_bytes": row.size_bytes,
        "media_type": row.media_type,
        "uri": f"{ARTIFACT_URI_PREFIX}/{row.job_event_id}/artifacts/{row.id}",
    }


def _isoformat(value: dt.datetime | None) -> str | None:
    return _aware(value).isoformat() if value is not None else None


def _aware(value: dt.datetime) -> dt.datetime:
    return value.replace(tzinfo=dt.UTC) if value.tzinfo is None else value


def _optional_text(value: Any, limit: int) -> str | None:
    return str(value)[:limit] if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _schedule_raw_retention(job: ASRJob) -> None:
    assert job.completed_at is not None
    if job.raw_retention == "delete_after_completion":
        for item in job.items:
            item.input_relkey = None
        job.raw_expires_at = job.completed_at
    elif job.raw_retention == "retain_7d":
        job.raw_expires_at = job.completed_at + dt.timedelta(days=7)
    elif job.raw_retention == "retain_30d":
        job.raw_expires_at = job.completed_at + dt.timedelta(days=30)


__all__ = [
    "ASRJobStore",
    "ASRJobStoreUnavailableError",
    "TERMINAL_STATUSES",
    "default_asr_job_store",
    "ensure_asr_job_tables",
]
