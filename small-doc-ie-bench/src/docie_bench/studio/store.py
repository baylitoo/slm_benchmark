"""Blob store + run index service for durable Studio benchmark results.

``ArtifactBlobStore`` is a content-addressed, atomic-write store rooted at a
shared directory (a Docker volume or an S3/MinIO mount). Reads resolve a
*store-relative* key against the store root, so a run written by the worker is
readable from the ``api`` replica with no shared knowledge of the worker's local
paths — the property PR-2 exists to guarantee.

``RunStore`` wraps the Postgres index: it claims runs idempotently, records
metrics + artifact references on completion, resolves artifacts for
authenticated download (tenant-filtered), and garbage-collects old runs together
with any blobs no surviving run still references.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from docie_bench.settings import get_settings
from docie_bench.storage.db import get_session_factory
from docie_bench.studio.models import StudioRun, StudioRunArtifact

logger = logging.getLogger("docie_bench.studio.store")

ARTIFACT_URI_PREFIX = "/v1/studio/artifacts"


@dataclass(frozen=True)
class StoredBlob:
    """A blob committed to the store, addressed by its store-relative key."""

    relkey: str
    sha256: str
    size_bytes: int
    media_type: str


class ArtifactBlobStore:
    """Content-addressed, atomic blob store rooted at a shared directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _relkey(self, digest: str, name: str) -> str:
        # Fan out on the digest so a directory never holds unbounded entries, and
        # keep the human-readable name as the leaf. POSIX separators keep the key
        # stable across OSes (it is only ever joined back onto ``root``).
        return f"{digest[:2]}/{digest}/{name}"

    def put(
        self, *, name: str, content: bytes, media_type: str = "application/octet-stream"
    ) -> StoredBlob:
        safe_name = Path(name).name
        if not safe_name or safe_name != name:
            raise ValueError("Artifact name must be a plain file name")
        digest = hashlib.sha256(content).hexdigest()
        relkey = self._relkey(digest, safe_name)
        destination = self.root / relkey
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            fd, temporary = tempfile.mkstemp(prefix=f".{safe_name}.", dir=destination.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return StoredBlob(
            relkey=relkey, sha256=digest, size_bytes=len(content), media_type=media_type
        )

    def path_for(self, relkey: str) -> Path:
        """Resolve a store-relative key to an absolute path *inside* the root.

        Guards against traversal (``..``) and absolute keys so a poisoned DB row
        can never point the download endpoint outside the store.
        """
        root = self.root.resolve()
        candidate = (root / relkey).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Artifact key escapes the store root")
        return candidate

    def read(self, relkey: str) -> bytes:
        return self.path_for(relkey).read_bytes()

    def exists(self, relkey: str) -> bool:
        try:
            return self.path_for(relkey).is_file()
        except ValueError:
            return False

    def delete(self, relkey: str) -> bool:
        """Delete a blob and prune now-empty digest directories. Idempotent."""
        try:
            path = self.path_for(relkey)
        except ValueError:
            return False
        removed = False
        if path.is_file():
            path.unlink(missing_ok=True)
            removed = True
        root = self.root.resolve()
        parent = path.parent
        while parent != root and parent.is_dir():
            try:
                next(parent.iterdir())
                break  # not empty
            except StopIteration:
                parent.rmdir()
                parent = parent.parent
        return removed


def default_blob_store() -> ArtifactBlobStore:
    return ArtifactBlobStore(get_settings().artifact_store_dir)


class RunStoreUnavailableError(RuntimeError):
    """Raised when the run index is used without a configured database."""


class RunStore:
    """Durable index for Studio benchmark runs (Postgres + blob store)."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] | None,
        blob_store: ArtifactBlobStore,
    ) -> None:
        self._sessions = session_factory
        self.blobs = blob_store

    @property
    def enabled(self) -> bool:
        return self._sessions is not None

    @contextmanager
    def _session(self) -> Iterator[Session]:
        if self._sessions is None:
            raise RunStoreUnavailableError("Studio run index requires a configured DATABASE_URL")
        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -- write path -------------------------------------------------------

    def claim(
        self,
        *,
        event_id: str,
        idempotency_key: str,
        tenant_id: str,
        dataset: str | None = None,
        model_profile: str | None = None,
        schema_name: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Reserve a run row *before* doing any work.

        Returns ``("claimed", record)`` when this caller owns the run and must
        execute it, or ``("exists", record)`` when it must NOT run again:

          - same ``event_id``, already ``completed`` -> ``exists`` (a redelivery
            of a finished run);
          - same ``event_id``, still ``running``/``failed`` -> ``claimed`` (this
            is our own run being retried/resumed — Inngest is at-least-once, so a
            failed attempt must be allowed to run again);
          - a *different* ``event_id`` with the same ``idempotency_key`` (a
            duplicate trigger, e.g. a double-click) -> ``exists``.
        """
        with self._session() as session:
            existing = session.get(StudioRun, event_id)
            if existing is not None:
                if existing.status == "completed":
                    return "exists", _run_to_dict(existing)
                # Our own run retrying: reset to running and re-execute.
                existing.status = "running"
                existing.error_text = None
                session.flush()
                return "claimed", _run_to_dict(existing)
            run = StudioRun(
                event_id=event_id,
                idempotency_key=idempotency_key,
                tenant_id=tenant_id or "anonymous",
                status="running",
                dataset=dataset,
                model_profile=model_profile,
                schema_name=schema_name,
            )
            session.add(run)
            try:
                session.flush()
            except IntegrityError:
                # A duplicate trigger already owns the logical run under a
                # different event id (unique idempotency_key). Do not double-run.
                session.rollback()
                found = session.scalars(
                    select(StudioRun).where(StudioRun.idempotency_key == idempotency_key)
                ).first()
                if found is None:  # pragma: no cover - defensive; row must exist
                    raise
                return "exists", _run_to_dict(found)
            return "claimed", _run_to_dict(run)

    def complete(
        self,
        *,
        event_id: str,
        metrics: dict[str, Any] | None,
        artifacts: Sequence[tuple[str, StoredBlob]],
    ) -> dict[str, Any]:
        with self._session() as session:
            run = session.get(StudioRun, event_id)
            if run is None:  # pragma: no cover - complete always follows claim
                raise RunStoreUnavailableError(f"No claimed run for event {event_id!r}")
            run.status = "completed"
            run.metrics_json = metrics
            run.error_text = None
            # Replace any partial artifacts from a prior attempt.
            run.artifacts.clear()
            session.flush()
            for name, blob in artifacts:
                run.artifacts.append(
                    StudioRunArtifact(
                        id=uuid.uuid4().hex,
                        run_event_id=event_id,
                        name=name,
                        relkey=blob.relkey,
                        sha256=blob.sha256,
                        size_bytes=blob.size_bytes,
                        media_type=blob.media_type,
                    )
                )
            session.flush()
            return _run_to_dict(run)

    def fail(self, *, event_id: str, error: str) -> dict[str, Any] | None:
        with self._session() as session:
            run = session.get(StudioRun, event_id)
            if run is None:
                return None
            run.status = "failed"
            run.error_text = error[:4000]
            session.flush()
            return _run_to_dict(run)

    # -- read path (tenant-scoped) ----------------------------------------

    def get_run(self, event_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        with self._session() as session:
            run = session.get(StudioRun, event_id)
            if run is None or run.tenant_id != tenant_id:
                return None  # 404, not 403 — never confirm another tenant's run
            return _run_to_dict(run)

    def run_owner(self, event_id: str) -> str | None:
        """Owning tenant of a run, or ``None`` if no durable row exists.

        Lets the run-status route distinguish "not a benchmark run" (fall through
        to the Inngest proxy) from "someone else's run" (404) — so a cross-tenant
        id can never be answered by the unauthenticated proxy path.
        """
        with self._session() as session:
            run = session.get(StudioRun, event_id)
            return run.tenant_id if run is not None else None

    def list_runs(self, *, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.scalars(
                select(StudioRun)
                .where(StudioRun.tenant_id == tenant_id)
                .order_by(StudioRun.created_at.desc())
                .limit(limit)
            ).all()
            return [_run_to_dict(run) for run in rows]

    def open_artifact(
        self, artifact_id: str, *, tenant_id: str
    ) -> tuple[dict[str, Any], bytes] | None:
        """Resolve an artifact by id → bytes, filtered by owning tenant.

        Bytes are read purely from ``artifact_id → DB row → store root``; no path
        travels in the job payload, which is what makes the download reachable
        from a non-worker replica.
        """
        with self._session() as session:
            artifact = session.get(StudioRunArtifact, artifact_id)
            if artifact is None:
                return None
            run = session.get(StudioRun, artifact.run_event_id)
            if run is None or run.tenant_id != tenant_id:
                return None  # cross-tenant → 404
            meta = _artifact_to_dict(artifact)
            relkey = artifact.relkey
        content = self.blobs.read(relkey)
        return meta, content

    # -- retention / GC ---------------------------------------------------

    def gc(
        self,
        *,
        max_age_days: int,
        max_runs: int,
        now: dt.datetime | None = None,
    ) -> dict[str, int]:
        """Delete runs older than ``max_age_days`` or beyond the newest
        ``max_runs``, then delete any blob no surviving artifact references.

        Age is applied first; the count cap then trims the survivors. Blobs are
        content-addressed, so a blob shared by a retained run is kept.
        """
        current = now or dt.datetime.now(dt.UTC)
        cutoff = current - dt.timedelta(days=max_age_days)
        with self._session() as session:
            all_runs = session.scalars(
                select(StudioRun).order_by(StudioRun.created_at.desc())
            ).all()
            doomed_ids: set[str] = set()
            survivors = 0
            for run in all_runs:
                created = run.created_at
                if created is not None and created.tzinfo is None:
                    created = created.replace(tzinfo=dt.UTC)
                too_old = created is not None and created < cutoff
                if too_old or survivors >= max_runs:
                    doomed_ids.add(run.event_id)
                else:
                    survivors += 1
            if not doomed_ids:
                return {"deleted_runs": 0, "deleted_blobs": 0, "retained_runs": survivors}

            doomed_relkeys = set(
                session.scalars(
                    select(StudioRunArtifact.relkey).where(
                        StudioRunArtifact.run_event_id.in_(doomed_ids)
                    )
                ).all()
            )
            # Delete child artifact rows explicitly: a core DELETE does not fire
            # ORM cascade, and sqlite does not enforce ON DELETE CASCADE by default.
            session.execute(
                delete(StudioRunArtifact).where(StudioRunArtifact.run_event_id.in_(doomed_ids))
            )
            session.execute(delete(StudioRun).where(StudioRun.event_id.in_(doomed_ids)))
            session.flush()
            # A relkey is safe to delete only if no *surviving* artifact uses it.
            still_referenced = set(
                session.scalars(
                    select(StudioRunArtifact.relkey).where(
                        StudioRunArtifact.relkey.in_(doomed_relkeys)
                    )
                ).all()
            )
            orphans = doomed_relkeys - still_referenced

        deleted_blobs = sum(1 for relkey in orphans if self.blobs.delete(relkey))
        return {
            "deleted_runs": len(doomed_ids),
            "deleted_blobs": deleted_blobs,
            "retained_runs": survivors,
        }


def _isoformat(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.isoformat()


def _artifact_to_dict(artifact: StudioRunArtifact) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "name": artifact.name,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "media_type": artifact.media_type,
        # Addressable, path-independent URI; never the worker-local file path.
        "uri": f"{ARTIFACT_URI_PREFIX}/{artifact.id}",
    }


def _run_to_dict(run: StudioRun) -> dict[str, Any]:
    return {
        "event_id": run.event_id,
        "idempotency_key": run.idempotency_key,
        "tenant_id": run.tenant_id,
        "status": run.status,
        "dataset": run.dataset,
        "model_profile": run.model_profile,
        "schema_name": run.schema_name,
        "metrics": run.metrics_json,
        "error": run.error_text,
        "created_at": _isoformat(run.created_at),
        "updated_at": _isoformat(run.updated_at),
        "artifacts": [_artifact_to_dict(a) for a in run.artifacts],
    }


def default_run_store() -> RunStore:
    """Build a RunStore from process defaults (shared blob dir + app DB)."""
    return RunStore(get_session_factory(), default_blob_store())


__all__ = [
    "ARTIFACT_URI_PREFIX",
    "ArtifactBlobStore",
    "RunStore",
    "RunStoreUnavailableError",
    "StoredBlob",
    "default_blob_store",
    "default_run_store",
]
