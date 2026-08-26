from __future__ import annotations

import base64
import inspect
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import docie_bench.asr.jobs_api as jobs_api
import docie_bench.asr.jobs_worker as jobs_worker
import docie_bench.security as security
from docie_bench.api import app
from docie_bench.asr.job_store import ASRJobStore, default_asr_job_store
from docie_bench.asr.jobs_worker import ASRJobItemError, process_transcription_job
from docie_bench.asr.routing import ASRRoute
from docie_bench.security import TenantQuotaManager
from docie_bench.settings import get_settings
from docie_bench.storage.db import dispose_engine, init_engine
from docie_bench.studio.store import default_run_store


def _wav(payload: bytes = b"audio") -> bytes:
    size = 36 + len(payload)
    return (
        b"RIFF"
        + size.to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + b"\x01\x00\x01\x00"
        + (16_000).to_bytes(4, "little")
        + (32_000).to_bytes(4, "little")
        + b"\x02\x00\x10\x00data"
        + len(payload).to_bytes(4, "little")
        + payload
    )


@pytest.fixture
def asr_job_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ASRJobStore:
    dispose_engine()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'jobs.db'}")
    monkeypatch.setenv("ARTIFACT_STORE_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()
    init_engine()
    manager = TenantQuotaManager(
        api_keys={"secret-a": "tenant-a", "secret-b": "tenant-b"},
        auth_required=True,
        requests_per_window=10_000,
        read_requests_per_window=10_000,
        window_seconds=60,
        max_concurrent=100,
    )
    monkeypatch.setattr(security, "get_quota_manager", lambda: manager)
    yield default_asr_job_store()
    dispose_engine()
    get_settings.cache_clear()


def _prepared_item(
    store: ASRJobStore,
    *,
    filename: str,
    payload: bytes,
    reference: str | None = None,
) -> dict[str, Any]:
    blob = store.blobs.put(name=filename, content=payload, media_type="audio/wav")
    return {
        "filename": filename,
        "relkey": blob.relkey,
        "sha256": blob.sha256,
        "size_bytes": blob.size_bytes,
        "mime_type": "audio/wav",
        "reference": reference,
    }


def _claim(
    store: ASRJobStore,
    *,
    event_id: str,
    tenant_id: str = "tenant-a",
    items: list[dict[str, Any]],
    key: str | None = None,
    retention: str = "delete_after_completion",
) -> None:
    outcome, _ = store.claim(
        event_id=event_id,
        tenant_id=tenant_id,
        idempotency_key=key or f"key-{event_id}",
        channel=f"asr:{event_id}",
        deployment="speech-one",
        model="whisper-small",
        options={"temperature": 0.0},
        raw_retention=retention,
        raw_expires_at=None,
        items=items,
    )
    assert outcome == "claimed"


class ImmediateStep:
    async def run(self, _name: str, fn: Any) -> Any:
        value = fn()
        return await value if inspect.isawaitable(value) else value


def _verbose(text: str = "hello world") -> dict[str, Any]:
    return {
        "task": "transcribe",
        "language": "en",
        "duration": 2.0,
        "text": text,
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 2.0,
                "text": text,
                "tokens": [],
                "temperature": 0.0,
            }
        ],
        "processing_seconds": 1.0,
        "real_time_factor": 0.5,
        "model": "whisper-small",
        "backend": "fake-asr",
    }


def test_trigger_is_idempotent_and_tenant_scoped(
    asr_job_env: ASRJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = ASRRoute("speech-one", "http://asr-runtime:8093/v1", "whisper-small")
    monkeypatch.setattr(jobs_api, "resolve_asr_route", lambda _model: route)
    sent: list[str] = []

    async def fake_send(_client: Any, event: Any) -> list[str]:
        sent.append(event.id)
        return [event.id]

    monkeypatch.setattr(jobs_api, "send_or_503", fake_send)
    body = {
        "model": "speech-one",
        "recordings": [
            {
                "filename": "clip.wav",
                "content_b64": base64.b64encode(_wav()).decode("ascii"),
                "reference": "hello world",
            }
        ],
        "idempotency_key": "customer-request-42",
    }
    client = TestClient(app)
    first = client.post(
        "/v1/audio/transcription-jobs", json=body, headers={"X-API-Key": "secret-a"}
    )
    second = client.post(
        "/v1/audio/transcription-jobs", json=body, headers={"X-API-Key": "secret-a"}
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json()["deduplicated"] is True
    assert len(sent) == 1

    job_id = first.json()["job_id"]
    assert client.get(
        f"/v1/audio/transcription-jobs/{job_id}", headers={"X-API-Key": "secret-a"}
    ).status_code == 200
    assert client.get(
        f"/v1/audio/transcription-jobs/{job_id}", headers={"X-API-Key": "secret-b"}
    ).status_code == 404


@pytest.mark.asyncio
async def test_batch_partial_failure_produces_all_formats_and_weighted_metrics(
    asr_job_env: ASRJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [
        _prepared_item(
            asr_job_env, filename="good.wav", payload=_wav(b"one"), reference="hello world"
        ),
        _prepared_item(asr_job_env, filename="bad.wav", payload=_wav(b"two")),
    ]
    _claim(asr_job_env, event_id="asr-batch", items=items)
    route = ASRRoute("speech-one", "http://asr-runtime:8093/v1", "whisper-small")
    monkeypatch.setattr(jobs_worker, "resolve_asr_route", lambda _deployment: route)

    async def fake_request(
        _route: ASRRoute, *, item: dict[str, Any], options: dict[str, Any], store: ASRJobStore
    ) -> dict[str, Any]:
        del options, store
        if item["filename"] == "bad.wav":
            error = ASRJobItemError("corrupt audio")
            error.status_code = 422
            raise error
        return _verbose()

    monkeypatch.setattr(jobs_worker, "_request_verbose_transcription", fake_request)
    monkeypatch.setattr(jobs_worker, "record_usage", lambda **_kwargs: True)
    monkeypatch.setattr(jobs_worker, "stamp", lambda _deployment: None)
    result = await process_transcription_job(
        {
            "channel": "asr:asr-batch",
            "tenant_id": "tenant-a",
            "deployment": "speech-one",
            "model": "whisper-small",
            "options": {"temperature": 0.0},
            "items": items,
        },
        event_id="asr-batch",
        step=ImmediateStep(),
        store=asr_job_env,
    )
    assert result["status"] == "completed_with_errors"
    assert result["completed_items"] == 1
    assert result["failed_items"] == 1
    assert result["metrics"]["wer"] == 0.0
    assert result["metrics"]["cer"] == 0.0
    assert result["metrics"]["real_time_factor"] == 0.5
    assert result["items"][1]["error"] == "corrupt audio"
    assert {artifact["kind"] for artifact in result["items"][0]["artifacts"]} == {
        "text",
        "verbose_json",
        "srt",
        "vtt",
    }
    assert result["artifacts"][0]["kind"] == "manifest"
    # Safe default: raw references are cleared on settlement; output artifacts remain.
    assert all(item["raw_available"] is False for item in result["items"])


@pytest.mark.asyncio
async def test_transient_item_retries_without_duplicate_artifacts(
    asr_job_env: ASRJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [_prepared_item(asr_job_env, filename="retry.wav", payload=_wav())]
    _claim(asr_job_env, event_id="asr-retry", items=items)
    route = ASRRoute("speech-one", "http://asr-runtime:8093/v1", "whisper-small")
    monkeypatch.setattr(jobs_worker, "resolve_asr_route", lambda _deployment: route)
    monkeypatch.setattr(jobs_worker, "_RETRY_DELAYS", (0.0, 0.0, 0.0))
    calls = 0

    async def flaky(
        _route: ASRRoute, *, item: dict[str, Any], options: dict[str, Any], store: ASRJobStore
    ) -> dict[str, Any]:
        nonlocal calls
        del item, options, store
        calls += 1
        if calls < 3:
            error = ASRJobItemError("runtime warming")
            error.status_code = 503
            raise error
        return _verbose("recovered")

    monkeypatch.setattr(jobs_worker, "_request_verbose_transcription", flaky)
    monkeypatch.setattr(jobs_worker, "record_usage", lambda **_kwargs: True)
    monkeypatch.setattr(jobs_worker, "stamp", lambda _deployment: None)
    result = await process_transcription_job(
        {
            "tenant_id": "tenant-a",
            "deployment": "speech-one",
            "model": "whisper-small",
            "options": {},
            "items": items,
        },
        event_id="asr-retry",
        step=ImmediateStep(),
        store=asr_job_env,
    )
    assert calls == 3
    assert result["items"][0]["attempts"] == 3
    assert len(result["items"][0]["artifacts"]) == 4
    assert len({artifact["name"] for artifact in result["items"][0]["artifacts"]}) == 4

    async def must_not_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("a DB-completed item must not call the ASR runtime on replay")

    monkeypatch.setattr(jobs_worker, "_request_verbose_transcription", must_not_run)
    replay = await jobs_worker._process_item(
        asr_job_env,
        event_id="asr-retry",
        position=0,
        item=items[0],
        route=route,
        tenant_id="tenant-a",
        options={},
    )
    assert replay["status"] == "completed"
    assert replay["attempts"] == 3


@pytest.mark.asyncio
async def test_cancellation_stops_before_next_item_and_is_durable(
    asr_job_env: ASRJobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [_prepared_item(asr_job_env, filename="cancel.wav", payload=_wav())]
    _claim(asr_job_env, event_id="asr-cancel", items=items, retention="retain_7d")
    cancelled = asr_job_env.request_cancel("asr-cancel", tenant_id="tenant-a")
    assert cancelled is not None
    assert cancelled["status"] == "cancelled"
    monkeypatch.setattr(
        jobs_worker,
        "resolve_asr_route",
        lambda _deployment: ASRRoute(
            "speech-one", "http://asr-runtime:8093/v1", "whisper-small"
        ),
    )
    result = await process_transcription_job(
        {
            "tenant_id": "tenant-a",
            "deployment": "speech-one",
            "model": "whisper-small",
            "options": {},
            "items": items,
        },
        event_id="asr-cancel",
        step=ImmediateStep(),
        store=asr_job_env,
    )
    assert result["status"] == "cancelled"
    assert result["items"][0]["status"] == "cancelled"
    assert result["items"][0]["raw_available"] is True
    assert result["raw_expires_at"] is not None


def test_artifact_download_fails_closed_across_tenants(
    asr_job_env: ASRJobStore,
) -> None:
    items = [_prepared_item(asr_job_env, filename="owned.wav", payload=_wav())]
    _claim(asr_job_env, event_id="asr-owned", items=items)
    blob = asr_job_env.blobs.put(
        name="owned.txt", content=b"tenant secret", media_type="text/plain"
    )
    asr_job_env.add_job_artifact(
        "asr-owned", name="owned.txt", kind="text", blob=blob
    )
    record = asr_job_env.get("asr-owned", tenant_id="tenant-a")
    assert record is not None
    artifact_id = record["artifacts"][0]["id"]
    assert asr_job_env.open_artifact(artifact_id, tenant_id="tenant-a") is not None
    assert asr_job_env.open_artifact(artifact_id, tenant_id="tenant-b") is None
    client = TestClient(app)
    uri = record["artifacts"][0]["uri"]
    owned = client.get(uri, headers={"X-API-Key": "secret-a"})
    assert owned.status_code == 200
    assert owned.content == b"tenant secret"
    assert client.get(uri, headers={"X-API-Key": "secret-b"}).status_code == 404
    assert client.get(
        uri.replace("asr-owned", "a-different-job"),
        headers={"X-API-Key": "secret-a"},
    ).status_code == 404


def test_same_idempotency_key_isolated_per_tenant(asr_job_env: ASRJobStore) -> None:
    item = _prepared_item(asr_job_env, filename="same.wav", payload=_wav())
    _claim(
        asr_job_env,
        event_id="tenant-a-job",
        tenant_id="tenant-a",
        items=[item],
        key="shared-client-key",
    )
    _claim(
        asr_job_env,
        event_id="tenant-b-job",
        tenant_id="tenant-b",
        items=[item],
        key="shared-client-key",
    )
    assert asr_job_env.get("tenant-a-job", tenant_id="tenant-b") is None
    assert asr_job_env.get("tenant-b-job", tenant_id="tenant-a") is None


def test_studio_retry_cannot_delete_blob_still_referenced_by_asr(
    asr_job_env: ASRJobStore,
) -> None:
    shared = asr_job_env.blobs.put(
        name="shared.txt", content=b"same bytes", media_type="text/plain"
    )
    item = _prepared_item(asr_job_env, filename="input.wav", payload=_wav())
    _claim(asr_job_env, event_id="asr-shared", items=[item])
    asr_job_env.add_job_artifact(
        "asr-shared", name="shared.txt", kind="text", blob=shared
    )

    studio = default_run_store()
    studio.claim(
        event_id="studio-run",
        idempotency_key="studio-key",
        tenant_id="tenant-a",
    )
    studio.complete(event_id="studio-run", metrics={}, artifacts=[("shared.txt", shared)])
    replacement = asr_job_env.blobs.put(
        name="shared.txt", content=b"new bytes", media_type="text/plain"
    )
    studio.complete(
        event_id="studio-run", metrics={}, artifacts=[("shared.txt", replacement)]
    )
    assert asr_job_env.blobs.exists(shared.relkey)
