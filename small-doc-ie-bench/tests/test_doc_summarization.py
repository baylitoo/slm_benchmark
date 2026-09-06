"""Upload-time rolling document summarization (#430)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from docie_bench.doc_summarization import summarize_document
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.mcp_servers import docs_search
from docie_bench.serving.placement_resolver import (
    PlacementNotFoundError,
    PlacementNotReadyError,
)
from docie_bench.settings import get_settings

PROFILE = ModelProfile(
    name="lfm2.5-350m", model="lfm2.5-350m-served", base_url="http://upstream/v1", api_key="k"
)


@pytest.fixture(autouse=True)
def _clear_extraction_cache() -> None:
    docs_search._EXTRACTION_CACHE.clear()


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("DOC_SUMMARY_MODEL", "store:lfm2.5-350m")
    yield
    get_settings.cache_clear()


def _completion(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


async def test_summarize_document_is_a_noop_when_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # doc_summary_model defaults to "store:lfm2.5-350m" -- explicitly unset
    # it here rather than relying on the default to represent "disabled".
    monkeypatch.setenv("DOC_SUMMARY_MODEL", "")
    get_settings.cache_clear()
    doc = tmp_path / "a.txt"
    doc.write_text("hello")
    await summarize_document(doc)
    assert docs_search.read_summary(doc) is None
    get_settings.cache_clear()


async def test_summarize_document_marks_unavailable_when_model_is_not_a_catalog_entry(
    enabled: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        raise PlacementNotFoundError(f"no placement for {model_profile!r}")

    async def fake_trigger(_name: str) -> None:
        return None  # not a real catalog entry -- nothing to load

    monkeypatch.setattr("docie_bench.doc_summarization.resolve_extraction_profile", fake_resolver)
    monkeypatch.setattr("docie_bench.doc_summarization.trigger_deployment_load", fake_trigger)
    doc = tmp_path / "a.txt"
    doc.write_text("hello")
    await summarize_document(doc)
    assert docs_search.read_summary(doc) == {"state": "unavailable", "summary": None}


async def test_summarize_document_fires_load_on_demand_and_waits_for_it_to_come_up(
    enabled: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The default profile is a store: model that's very likely not live yet
    # the first time an upload fires this -- mirrors chat_api's own
    # load-on-demand seam instead of just giving up.
    monkeypatch.setattr("docie_bench.doc_summarization._LOAD_POLL_INTERVAL_SECONDS", 0.01)
    attempts = {"n": 0}

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PlacementNotReadyError("still loading")
        return PROFILE

    async def fake_trigger(name: str) -> tuple[str, float]:
        assert name == "lfm2.5-350m"  # "store:" prefix stripped
        return (name, 1.0)

    monkeypatch.setattr("docie_bench.doc_summarization.resolve_extraction_profile", fake_resolver)
    monkeypatch.setattr("docie_bench.doc_summarization.trigger_deployment_load", fake_trigger)
    monkeypatch.setattr(
        "docie_bench.doc_summarization.extract_page_texts", lambda _path: {1: "page one"}
    )

    doc = tmp_path / "a.txt"
    doc.write_text("hello")

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: _completion("a summary")))
    await summarize_document(doc, http_client=client)
    await client.aclose()

    assert attempts["n"] >= 3
    assert docs_search.read_summary(doc) == {"state": "ready", "summary": "a summary"}


async def test_summarize_document_marks_unavailable_when_trigger_send_fails(
    enabled: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fastapi import HTTPException

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        raise PlacementNotFoundError(f"no placement for {model_profile!r}")

    async def fake_trigger(_name: str) -> tuple[str, float]:
        # trigger_deployment_load's own "never deployed yet" branch fires
        # send_or_503 unguarded -- a transport failure there surfaces as
        # this exception rather than returning None.
        raise HTTPException(status_code=503, detail="inngest send failed")

    monkeypatch.setattr("docie_bench.doc_summarization.resolve_extraction_profile", fake_resolver)
    monkeypatch.setattr("docie_bench.doc_summarization.trigger_deployment_load", fake_trigger)
    doc = tmp_path / "a.txt"
    doc.write_text("hello")
    await summarize_document(doc)
    assert docs_search.read_summary(doc) == {"state": "unavailable", "summary": None}


async def test_summarize_document_never_raises_and_marks_failed_on_a_surprise_error(
    enabled: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def exploding_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        raise RuntimeError("something nobody anticipated")

    monkeypatch.setattr(
        "docie_bench.doc_summarization.resolve_extraction_profile", exploding_resolver
    )
    doc = tmp_path / "a.txt"
    doc.write_text("hello")
    await summarize_document(doc)  # must not raise
    assert docs_search.read_summary(doc)["state"] == "failed"


async def test_summarize_document_marks_unavailable_when_load_never_completes(
    enabled: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("docie_bench.doc_summarization._LOAD_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr("docie_bench.doc_summarization._MAX_LOAD_WAIT_SECONDS", 0.05)

    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        raise PlacementNotReadyError("still loading")

    async def fake_trigger(name: str) -> tuple[str, float]:
        return (name, 0.02)

    monkeypatch.setattr("docie_bench.doc_summarization.resolve_extraction_profile", fake_resolver)
    monkeypatch.setattr("docie_bench.doc_summarization.trigger_deployment_load", fake_trigger)

    doc = tmp_path / "a.txt"
    doc.write_text("hello")
    await summarize_document(doc)
    assert docs_search.read_summary(doc) == {"state": "unavailable", "summary": None}


async def test_summarize_document_marks_failed_when_extraction_fails(
    enabled: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "docie_bench.doc_summarization.resolve_extraction_profile", lambda **_: PROFILE
    )
    doc = tmp_path / "missing.txt"  # never written -- extraction must fail cleanly
    await summarize_document(doc)
    assert docs_search.read_summary(doc)["state"] == "failed"


async def test_summarize_document_rolls_pages_into_a_capped_running_summary(
    enabled: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "docie_bench.doc_summarization.resolve_extraction_profile", lambda **_: PROFILE
    )
    # extract_page_texts's real backends only ever emit a single page="1" for
    # a plain .txt (see ocr.base.text_to_blocks) -- a genuine multi-page
    # fixture needs a real PDF, so the rolling loop itself is exercised
    # against a synthetic 3-page extraction instead.
    monkeypatch.setattr(
        "docie_bench.doc_summarization.extract_page_texts",
        lambda _path: {1: "page one", 2: "page two", 3: "page three"},
    )
    monkeypatch.setenv("DOC_SUMMARY_CHUNK_PAGES", "1")
    monkeypatch.setenv("DOC_SUMMARY_MAX_CHARS", "100")
    get_settings.cache_clear()

    doc = tmp_path / "invoice.txt"
    doc.write_text("irrelevant -- extract_page_texts is mocked above")

    requests: list[dict[str, object]] = []

    long_completion = "x" * 150  # deliberately over DOC_SUMMARY_MAX_CHARS=100

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return _completion(long_completion)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await summarize_document(doc, http_client=client)
    await client.aclose()

    assert len(requests) == 3  # one page per chunk (DOC_SUMMARY_CHUNK_PAGES=1)
    assert "Summary so far" not in requests[0]["messages"][0]["content"]
    assert "Summary so far" in requests[1]["messages"][0]["content"]

    sidecar = docs_search.read_summary(doc)
    assert sidecar["state"] == "ready"
    assert sidecar["summary"] == "x" * 100  # truncated to doc_summary_max_chars


async def test_summarize_document_marks_failed_on_upstream_error_but_keeps_partial_summary(
    enabled: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "docie_bench.doc_summarization.resolve_extraction_profile", lambda **_: PROFILE
    )
    monkeypatch.setattr(
        "docie_bench.doc_summarization.extract_page_texts",
        lambda _path: {1: "page one", 2: "page two"},
    )
    monkeypatch.setenv("DOC_SUMMARY_CHUNK_PAGES", "1")
    get_settings.cache_clear()

    doc = tmp_path / "invoice.txt"
    doc.write_text("irrelevant -- extract_page_texts is mocked above")

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _completion("first chunk summary")
        return httpx.Response(500, text="upstream exploded")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await summarize_document(doc, http_client=client)
    await client.aclose()

    sidecar = docs_search.read_summary(doc)
    assert sidecar["state"] == "failed"
    assert sidecar["summary"] == "first chunk summary"
