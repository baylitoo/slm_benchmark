"""POST /v1/studio/comparisons: the CLI's compare_runs logic, reachable live.

Exercises the refactored ``build_comparison_payload`` core (decoded metrics
dicts in, no filesystem I/O) plus the tenant-scoped API route wired on top of
the existing durable ``StudioRun`` index -- entirely against a sqlite store,
no live model stack.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import docie_bench.api as api
import docie_bench.inngest.studio_api as studio_api
from docie_bench.benchmark.comparison import build_comparison_payload
from docie_bench.security import TenantQuotaManager
from docie_bench.storage.db import Base
from docie_bench.studio.store import ArtifactBlobStore, RunStore


def _make_store(db_path: Path, blob_root: Path) -> tuple[RunStore, sessionmaker[Session]]:
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    return RunStore(factory, ArtifactBlobStore(blob_root)), factory


def _row(doc_id: str, *, correct: bool, latency: float = 10) -> dict:
    return {
        "doc_id": doc_id,
        "model_profile": "model",
        "schema_name": "invoice",
        "language": "en",
        "ok": True,
        "latency_ms": latency,
        "validation": {"valid": True},
        "score": {
            "field_accuracy": float(correct),
            "avg_similarity": float(correct),
            "fields": [
                {"field": "invoice_number", "correct": correct, "similarity": float(correct)}
            ],
        },
    }


# ---------------------------------------------------------------------------
# build_comparison_payload: pure core, no I/O
# ---------------------------------------------------------------------------


def test_build_comparison_payload_matches_compare_runs_shape() -> None:
    baseline_metrics = {"rows": [_row("one", correct=False)]}
    candidate_metrics = {"rows": [_row("one", correct=True)]}

    payload = build_comparison_payload(
        baseline_metrics,
        candidate_metrics,
        baseline_meta={"event_id": "ev-base"},
        candidate_meta={"event_id": "ev-cand"},
    )

    assert payload["verdict"] == "pass"
    assert payload["baseline"] == {"event_id": "ev-base"}
    assert payload["candidate"] == {"event_id": "ev-cand"}
    aggregate = next(
        item
        for item in payload["comparisons"]
        if item["dimension"] == "aggregate" and item["metric"] == "field_accuracy"
    )
    assert aggregate["delta"] == 1.0
    assert "root_causes" in payload
    assert "judge_calibration" in payload


def test_build_comparison_payload_applies_budgets() -> None:
    baseline_metrics = {"rows": [_row("one", correct=True)]}
    candidate_metrics = {"rows": [_row("one", correct=False)]}

    payload = build_comparison_payload(
        baseline_metrics,
        candidate_metrics,
        baseline_meta={},
        candidate_meta={},
        budgets=[{"name": "accuracy", "metric": "field_accuracy", "max_regression": 0.1}],
    )

    assert payload["verdict"] == "fail"
    assert payload["budget_checks"][0]["reason"] == "budget_exceeded"
    assert payload["root_causes"]["documents"]


# ---------------------------------------------------------------------------
# POST /v1/studio/comparisons: tenant-scoped, backed by StudioRun rows
# ---------------------------------------------------------------------------


def _client_with_store(store: RunStore, monkeypatch) -> TestClient:
    monkeypatch.setattr(studio_api, "default_run_store", lambda: store)
    from docie_bench import security

    manager = TenantQuotaManager(
        api_keys={"secret-a": "tenant-a", "secret-b": "tenant-b"},
        auth_required=True,
        requests_per_window=100,
        window_seconds=60,
        max_concurrent=10,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)
    return TestClient(api.app)


def _complete_run(store: RunStore, *, event_id: str, tenant: str, rows: list[dict]) -> None:
    store.claim(event_id=event_id, idempotency_key=f"k-{event_id}", tenant_id=tenant, dataset="ds")
    store.complete(event_id=event_id, metrics={"rows": rows}, artifacts=[])


def test_comparison_endpoint_returns_verdict_for_owned_runs(tmp_path: Path, monkeypatch) -> None:
    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    _complete_run(store, event_id="base", tenant="tenant-a", rows=[_row("one", correct=False)])
    _complete_run(store, event_id="cand", tenant="tenant-a", rows=[_row("one", correct=True)])
    client = _client_with_store(store, monkeypatch)

    resp = client.post(
        "/v1/studio/comparisons",
        json={"baseline_event_id": "base", "candidate_event_id": "cand"},
        headers={"X-API-Key": "secret-a"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "pass"
    assert body["baseline"]["event_id"] == "base"
    assert body["candidate"]["event_id"] == "cand"


def test_comparison_endpoint_requires_auth(tmp_path: Path, monkeypatch) -> None:
    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    client = _client_with_store(store, monkeypatch)

    resp = client.post(
        "/v1/studio/comparisons",
        json={"baseline_event_id": "base", "candidate_event_id": "cand"},
    )

    assert resp.status_code == 401


def test_comparison_endpoint_rejects_cross_tenant_runs(tmp_path: Path, monkeypatch) -> None:
    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    _complete_run(store, event_id="base", tenant="tenant-a", rows=[_row("one", correct=True)])
    _complete_run(store, event_id="cand", tenant="tenant-b", rows=[_row("one", correct=True)])
    client = _client_with_store(store, monkeypatch)

    resp = client.post(
        "/v1/studio/comparisons",
        json={"baseline_event_id": "base", "candidate_event_id": "cand"},
        headers={"X-API-Key": "secret-a"},
    )

    # candidate belongs to tenant-b -> 404, never a leaked cross-tenant comparison
    assert resp.status_code == 404


def test_comparison_endpoint_rejects_unknown_run(tmp_path: Path, monkeypatch) -> None:
    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    _complete_run(store, event_id="base", tenant="tenant-a", rows=[_row("one", correct=True)])
    client = _client_with_store(store, monkeypatch)

    resp = client.post(
        "/v1/studio/comparisons",
        json={"baseline_event_id": "base", "candidate_event_id": "does-not-exist"},
        headers={"X-API-Key": "secret-a"},
    )

    assert resp.status_code == 404


def test_comparison_endpoint_rejects_run_without_metrics_yet(tmp_path: Path, monkeypatch) -> None:
    store, _ = _make_store(tmp_path / "s.db", tmp_path / "b")
    _complete_run(store, event_id="base", tenant="tenant-a", rows=[_row("one", correct=True)])
    # Still running: claimed but never completed, so metrics_json is None.
    store.claim(event_id="cand", idempotency_key="k-cand", tenant_id="tenant-a", dataset="ds")
    client = _client_with_store(store, monkeypatch)

    resp = client.post(
        "/v1/studio/comparisons",
        json={"baseline_event_id": "base", "candidate_event_id": "cand"},
        headers={"X-API-Key": "secret-a"},
    )

    assert resp.status_code == 409
