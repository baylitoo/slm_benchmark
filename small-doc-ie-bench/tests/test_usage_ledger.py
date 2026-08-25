"""Per-deployment usage ledger: the store's aggregation math (percentile /
window / tenant scoping), the degrade-on-database-trouble write contract, the
``GET /v1/studio/usage`` route, and the extract/embed/rerank recording seams
(the chat seams are exercised in tests/test_chat_api.py against the same
httpx.MockTransport harness the surface itself is tested with)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import docie_bench.api as api
from docie_bench.storage.db import dispose_engine, init_engine, session_scope
from docie_bench.studio.models import UsageRecord, utcnow
from docie_bench.studio.usage_store import (
    aggregate_usage,
    percentile,
    record_usage,
    usage_summary,
)

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


@pytest.fixture
def usage_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    yield
    dispose_engine()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from docie_bench import security
    from docie_bench.security import TenantQuotaManager

    monkeypatch.setattr(
        security,
        "get_quota_manager",
        lambda: TenantQuotaManager(
            api_keys={"key-a": TENANT_A, "key-b": TENANT_B},
            auth_required=True,
            requests_per_window=100,
            window_seconds=60,
            max_concurrent=10,
        ),
    )
    return TestClient(api.app)


def _insert_row(
    *,
    deployment: str,
    tenant_id: str = TENANT_A,
    surface: str = "chat",
    status: str = "ok",
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    latency_ms: int = 100,
    age: dt.timedelta = dt.timedelta(0),
) -> None:
    with session_scope() as session:
        assert session is not None
        session.add(
            UsageRecord(
                deployment=deployment,
                surface=surface,
                tenant_id=tenant_id,
                status=status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                created_at=utcnow() - age,
            )
        )


# ── pure aggregation math ────────────────────────────────────────────────────


def test_percentile_empty_is_none() -> None:
    assert percentile([], 0.95) is None


def test_percentile_single_value() -> None:
    assert percentile([42], 0.95) == 42.0


def test_percentile_nearest_rank() -> None:
    # 20 values 1..20: nearest-rank p95 = ceil(0.95 * 20) = 19th value.
    assert percentile(list(range(1, 21)), 0.95) == 19.0
    # 100 values 1..100: 95th value.
    assert percentile(list(range(1, 101)), 0.95) == 95.0
    # p50 of 4 values = 2nd value (nearest rank, not interpolated).
    assert percentile([10, 20, 30, 40], 0.5) == 20.0


def test_aggregate_usage_folds_per_deployment() -> None:
    now = dt.datetime(2026, 8, 25, 12, 0, tzinfo=dt.UTC)
    rows: list[tuple[str, str, int | None, int | None, int, dt.datetime | None]] = [
        ("lfm2.5-350m", "ok", 10, 5, 100, now - dt.timedelta(minutes=5)),
        ("lfm2.5-350m", "ok", 20, 10, 200, now),
        ("lfm2.5-350m", "error", None, None, 300, now - dt.timedelta(minutes=1)),
        ("nuextract3", "ok", None, None, 50, now - dt.timedelta(hours=1)),
    ]
    busiest, other = aggregate_usage(rows)
    assert busiest["deployment"] == "lfm2.5-350m"  # sorted by requests desc
    assert busiest["requests"] == 3
    assert busiest["errors"] == 1
    assert busiest["prompt_tokens"] == 30  # None tokens contribute 0
    assert busiest["completion_tokens"] == 15
    assert busiest["avg_latency_ms"] == 200.0
    assert busiest["p95_latency_ms"] == 300.0
    assert busiest["last_used_at"] == now.isoformat()
    assert other["deployment"] == "nuextract3"
    assert other["requests"] == 1
    assert other["errors"] == 0


def test_aggregate_usage_handles_naive_timestamps() -> None:
    # sqlite hands back naive datetimes; they must normalize, not TypeError.
    naive = dt.datetime(2026, 8, 25, 12, 0)
    (entry,) = aggregate_usage([("lfm2.5-350m", "ok", 1, 1, 10, naive)])
    assert entry["last_used_at"] == "2026-08-25T12:00:00+00:00"


# ── store: record + summary against a real (sqlite) database ────────────────


def test_record_and_summarize_roundtrip(usage_database) -> None:
    assert record_usage(
        deployment="lfm2.5-350m",
        surface="chat",
        tenant_id=TENANT_A,
        latency_ms=120,
        prompt_tokens=7,
        completion_tokens=3,
    )
    assert record_usage(
        deployment="lfm2.5-350m",
        surface="extract",
        tenant_id=TENANT_A,
        latency_ms=480,
        status="error",
    )
    (entry,) = usage_summary(tenant_id=TENANT_A, window_hours=24)
    assert entry["deployment"] == "lfm2.5-350m"
    assert entry["requests"] == 2
    assert entry["errors"] == 1
    assert entry["prompt_tokens"] == 7
    assert entry["completion_tokens"] == 3
    assert entry["avg_latency_ms"] == 300.0
    assert entry["p95_latency_ms"] == 480.0
    assert entry["last_used_at"] is not None


def test_summary_respects_the_window(usage_database) -> None:
    _insert_row(deployment="lfm2.5-350m", age=dt.timedelta(hours=1))
    _insert_row(deployment="lfm2.5-350m", age=dt.timedelta(days=3))
    _insert_row(deployment="nuextract3", age=dt.timedelta(days=12))
    def summarize(hours: int) -> dict[str, dict[str, Any]]:
        return {
            e["deployment"]: e for e in usage_summary(tenant_id=TENANT_A, window_hours=hours)
        }

    day = summarize(24)
    assert day.keys() == {"lfm2.5-350m"}
    assert day["lfm2.5-350m"]["requests"] == 1
    assert summarize(7 * 24)["lfm2.5-350m"]["requests"] == 2
    assert summarize(30 * 24).keys() == {"lfm2.5-350m", "nuextract3"}


def test_summary_is_tenant_scoped(usage_database) -> None:
    _insert_row(deployment="lfm2.5-350m", tenant_id=TENANT_A)
    _insert_row(deployment="nuextract3", tenant_id=TENANT_B)
    (entry,) = usage_summary(tenant_id=TENANT_A, window_hours=24)
    assert entry["deployment"] == "lfm2.5-350m"


def test_record_without_database_degrades_to_false() -> None:
    dispose_engine()
    assert (
        record_usage(deployment="lfm2.5-350m", surface="chat", tenant_id=TENANT_A, latency_ms=1)
        is False
    )
    assert usage_summary(tenant_id=TENANT_A, window_hours=24) == []


def test_record_never_raises_when_database_errors(monkeypatch) -> None:
    from sqlalchemy.exc import OperationalError

    from docie_bench.studio import usage_store

    def broken_session_scope():
        raise OperationalError("insert", None, Exception("database is down"))

    monkeypatch.setattr(usage_store, "session_scope", broken_session_scope)
    assert (
        usage_store.record_usage(
            deployment="lfm2.5-350m", surface="chat", tenant_id=TENANT_A, latency_ms=1
        )
        is False
    )


# ── GET /v1/studio/usage ─────────────────────────────────────────────────────


def test_usage_route_aggregates_for_the_caller(usage_database, client) -> None:
    _insert_row(
        deployment="lfm2.5-350m",
        tenant_id=TENANT_A,
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=100,
    )
    _insert_row(deployment="nuextract3", tenant_id=TENANT_B)
    response = client.get("/v1/studio/usage", headers={"X-API-Key": "key-a"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["window"] == "24h"
    (entry,) = body["deployments"]
    assert entry["deployment"] == "lfm2.5-350m"
    assert entry["requests"] == 1
    assert entry["prompt_tokens"] == 10
    assert entry["completion_tokens"] == 5


def test_usage_route_window_switch(usage_database, client) -> None:
    _insert_row(deployment="lfm2.5-350m", age=dt.timedelta(days=3))
    empty = client.get("/v1/studio/usage?window=24h", headers={"X-API-Key": "key-a"})
    assert empty.json()["deployments"] == []
    week = client.get("/v1/studio/usage?window=7d", headers={"X-API-Key": "key-a"})
    assert len(week.json()["deployments"]) == 1


def test_usage_route_rejects_unknown_window(client) -> None:
    response = client.get("/v1/studio/usage?window=90d", headers={"X-API-Key": "key-a"})
    assert response.status_code == 400
    assert "window" in response.json()["detail"]


def test_usage_route_requires_auth(client) -> None:
    assert client.get("/v1/studio/usage").status_code == 401


# ── recording seams: extract (api.py) + embed/rerank (chat_api.py) ──────────


def test_finalize_response_records_extract_usage(usage_database) -> None:
    from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation, Usage

    response = ExtractionResponse(
        request_id="req-1",
        schema_name="invoice",
        model_profile="nuextract3",
        document_hash=None,
        result={},
        validation=ExtractionValidation(valid=True),
        usage=Usage(prompt_tokens=100, completion_tokens=40, total_tokens=140),
        latency_ms=1234,
    )
    api.finalize_response(response, tenant_id=TENANT_A)
    (entry,) = usage_summary(tenant_id=TENANT_A, window_hours=24)
    assert entry["deployment"] == "nuextract3"
    assert entry["requests"] == 1
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 40
    assert entry["p95_latency_ms"] == 1234.0


@pytest.fixture
def chat_surface(monkeypatch) -> TestClient:
    from docie_bench.agents.api import configure_http_transport
    from docie_bench.chat_api import router as chat_router
    from docie_bench.llm.model_profiles import ModelProfile
    from docie_bench.serving.profile_resolver import ProfileResolutionError

    upstream = ModelProfile(
        name="lfm-embed", model="lfm-embed-served", base_url="http://upstream/v1", api_key="k"
    )

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "lfm-embed":
            return upstream
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": body["model"],
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                },
            )
        assert request.url.path.endswith("/rerank")
        return httpx.Response(
            200, json={"model": body["model"], "results": [{"index": 0, "relevance_score": 0.9}]}
        )

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(chat_router)
    yield TestClient(app)
    configure_http_transport(None)


def test_embeddings_and_rerank_record_their_surfaces(chat_surface, monkeypatch) -> None:
    from docie_bench.studio import usage_store

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        usage_store, "record_usage", lambda **kwargs: calls.append(kwargs) or True
    )
    assert (
        chat_surface.post(
            "/v1/embeddings", json={"model": "lfm-embed", "input": "invoice total"}
        ).status_code
        == 200
    )
    assert (
        chat_surface.post(
            "/v1/rerank",
            json={"model": "lfm-embed", "query": "total", "documents": ["a", "b"]},
        ).status_code
        == 200
    )
    embed_call, rerank_call = calls
    assert embed_call["surface"] == "embed"
    assert embed_call["deployment"] == "lfm-embed"
    assert embed_call["status"] == "ok"
    # llama-server's embeddings usage block has prompt_tokens only.
    assert embed_call["prompt_tokens"] == 4
    assert embed_call["completion_tokens"] is None
    assert rerank_call["surface"] == "rerank"
    assert rerank_call["status"] == "ok"
