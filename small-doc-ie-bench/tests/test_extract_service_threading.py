"""Regression: ExtractionService.extract_from_file ran its OCR/PDF-rasterize
step as a plain synchronous call inside an `async def` -- no `asyncio.to_thread`,
no executor. With the API served by a single uvicorn worker (one event loop),
that call blocked the ENTIRE process for its duration: not just other requests
to the same model, every concurrent request on the server, including unrelated
tenants and health checks. `serving/solutions.py` already offloads its own OCR
calls via `asyncio.to_thread` -- this file's calls were the one path that
didn't. Fixed by wrapping each call site (`processor_from_settings(...).process`,
`load_document_images`, and the pipeline's own copies of both) in
`asyncio.to_thread`. These tests assert the blocking work actually runs off the
event loop's thread, not just that the result comes back correct (which passed
before the fix too).
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation, OCRBlock
from docie_bench.vision import DocumentImage


class _ThreadRecordingBackend:
    name = "fake"

    def __init__(self) -> None:
        self.call_thread_idents: list[int] = []

    def version(self) -> str:
        return "1"

    def configuration(self) -> dict:
        return {}

    def extract(self, path: Path) -> list[OCRBlock]:
        self.call_thread_idents.append(threading.get_ident())
        return [OCRBlock(id="b1", text="hello", source="manual")]


async def _fake_extract_blocks(self, **kwargs):
    return "response"


@pytest.mark.asyncio
async def test_plain_profile_ocr_runs_off_the_event_loop_thread(
    monkeypatch, tmp_path: Path
) -> None:
    document = tmp_path / "document.txt"
    document.write_text("hello", encoding="utf-8")
    backend = _ThreadRecordingBackend()

    monkeypatch.setattr(
        "docie_bench.ocr.service.get_ocr_backend", lambda *a, **k: backend
    )
    monkeypatch.setattr(
        "docie_bench.extract.service.get_settings",
        lambda: SimpleNamespace(log_document_content=False),
    )
    monkeypatch.setattr(ExtractionService, "_extract_blocks", _fake_extract_blocks)
    profile = ModelProfile(
        name="test", model="test", base_url="http://example.test/v1", api_key="test"
    )
    service = ExtractionService(profile)

    main_thread_ident = threading.get_ident()
    result = await service.extract_from_file(
        path=document, ocr_backend_name="fake", schema_name="invoice"
    )

    assert result == "response"
    assert backend.call_thread_idents[0] != main_thread_ident


@pytest.mark.asyncio
async def test_vision_profile_rasterization_runs_off_the_event_loop_thread(
    monkeypatch, tmp_path: Path
) -> None:
    document = tmp_path / "document.pdf"
    document.write_bytes(b"%PDF-1.4 fake")
    recorded_idents: list[int] = []

    def fake_load_document_images(path, *, max_pages, pdf_dpi):
        recorded_idents.append(threading.get_ident())
        return [DocumentImage(page=1, media_type="image/png", data=b"fake")]

    monkeypatch.setattr(
        "docie_bench.extract.service.load_document_images", fake_load_document_images
    )
    monkeypatch.setattr(ExtractionService, "_extract_blocks", _fake_extract_blocks)
    profile = ModelProfile(
        name="test-vision",
        model="test-vision",
        base_url="http://example.test/v1",
        api_key="test",
        vision=True,
    )
    service = ExtractionService(profile)

    main_thread_ident = threading.get_ident()
    result = await service.extract_from_file(
        path=document, ocr_backend_name="tesseract", schema_name="invoice"
    )

    assert result == "response"
    assert recorded_idents[0] != main_thread_ident


@pytest.mark.asyncio
async def test_pipeline_backend_ocr_runs_off_the_event_loop_thread(
    monkeypatch, tmp_path: Path
) -> None:
    document = tmp_path / "invoice.txt"
    document.write_text("irrelevant", encoding="utf-8")
    backend = _ThreadRecordingBackend()

    async def fake_extract_from_text(self, **kwargs):
        return ExtractionResponse(
            request_id="r1",
            schema_name=kwargs["schema_name"],
            model_profile=self.profile.name,
            document_hash=kwargs["document_hash"],
            result={},
            validation=ExtractionValidation(valid=True),
            latency_ms=1,
        )

    monkeypatch.setattr(
        "docie_bench.ocr.service.get_ocr_backend", lambda *a, **k: backend
    )
    monkeypatch.setattr(
        "docie_bench.extract.service.get_settings",
        lambda: SimpleNamespace(log_document_content=False),
    )
    monkeypatch.setattr(ExtractionService, "extract_from_text", fake_extract_from_text)
    profile = ModelProfile(
        name="pipeline-profile",
        model="pipeline-profile",
        base_url="",
        api_key="unused",
        kind="pipeline",
        options={"extractor": "extractor-profile", "ocr_backend": "fake"},
    )
    profiles = {
        "extractor-profile": ModelProfile(
            name="extractor-profile",
            model="extractor-model",
            base_url="http://example.test/v1",
            api_key="test",
        )
    }
    service = ExtractionService(profile, profiles=profiles)

    main_thread_ident = threading.get_ident()
    await service.extract_from_file(
        path=document, ocr_backend_name="tesseract", schema_name="invoice"
    )

    assert backend.call_thread_idents[0] != main_thread_ident
