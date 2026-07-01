"""Durable run/artifact store for DocIE Studio benchmark jobs.

The Inngest ``benchmark/run.requested`` function runs on a background *worker*
whose local filesystem is unreachable from the ``api``/``web`` replicas. This
package makes a run's results *addressable*:

  - a content-addressed blob store on a shared volume (``ArtifactBlobStore``),
  - a Postgres index keyed by ``event_id`` (``StudioRun``/``StudioRunArtifact``),
  - a service (``RunStore``) that claims runs idempotently, records metrics +
    artifact references, resolves artifacts by id for authenticated download,
    and garbage-collects old runs (rows *and* orphaned blobs).

Metrics summaries (small) live in Postgres; ``report.html`` and the potentially
large ``predictions.jsonl`` live only in the blob store.
"""

from __future__ import annotations

from docie_bench.studio.models import StudioRun, StudioRunArtifact
from docie_bench.studio.store import (
    ArtifactBlobStore,
    RunStore,
    StoredBlob,
    default_blob_store,
    default_run_store,
)

__all__ = [
    "ArtifactBlobStore",
    "RunStore",
    "StoredBlob",
    "StudioRun",
    "StudioRunArtifact",
    "default_blob_store",
    "default_run_store",
]
