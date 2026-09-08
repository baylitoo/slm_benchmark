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
async def test_vision_profile_hash_file_runs_off_the_event_loop_thread(
    monkeypatch, tmp_path: Path
) -> None:
    document = tmp_path / "document.pdf"
    document.write_bytes(b"%PDF-1.4 fake")
    recorded_idents: list[int] = []

    def fake_hash_file(path):
        recorded_idents.append(threading.get_ident())
        return "sha256:fake"

    monkeypatch.setattr(
        "docie_bench.extract.service.load_document_images",
        lambda path, *, max_pages, pdf_dpi: [
            DocumentImage(page=1, media_type="image/png", data=b"fake")
        ],
    )
    monkeypatch.setattr("docie_bench.extract.service.hash_file", fake_hash_file)
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


class _ThreadRecordingClient:
    """Stands in for OpenAICompatibleClient -- lets _extract_blocks run for
    real (so its own _data_urls call executes), without a real HTTP call."""

    last_response_format_style = None

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    async def chat_json(self, **kwargs):
        return {"document_type": "resume", "name": {"value": "Jane Doe"}}, None, {}

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_extract_blocks_image_url_encoding_runs_off_the_event_loop_thread(
    monkeypatch, tmp_path: Path
) -> None:
    """The main vision extraction path's image_urls (built from _data_urls,
    fed to chat_json) went straight to DocumentImage.data_url() in a list
    comprehension on the event loop before this fix."""
    document = tmp_path / "document.pdf"
    document.write_bytes(b"%PDF-1.4 fake")
    recorded_idents: list[int] = []

    def fake_data_urls(images):
        recorded_idents.append(threading.get_ident())
        return [f"data:image/png;base64,fake{i}" for i in range(len(images))]

    monkeypatch.setattr(
        "docie_bench.extract.service.load_document_images",
        lambda path, *, max_pages, pdf_dpi: [
            DocumentImage(page=1, media_type="image/png", data=b"fake")
        ],
    )
    monkeypatch.setattr("docie_bench.extract.service._data_urls", fake_data_urls)
    monkeypatch.setattr(
        "docie_bench.extract.service.OpenAICompatibleClient", _ThreadRecordingClient
    )
    profile = ModelProfile(
        name="test-vision",
        model="test-vision",
        base_url="http://example.test/v1",
        api_key="test",
        vision=True,
    )
    service = ExtractionService(profile)

    main_thread_ident = threading.get_ident()
    response = await service.extract_from_file(
        path=document,
        ocr_backend_name="tesseract",
        schema_name="resume",
        schema_mode="dynamic",
        dynamic_schema={"document_type": "resume", "fields": [{"name": "name", "type": "string"}]},
    )

    assert response.result["name"]["value"] == "Jane Doe"
    assert recorded_idents[0] != main_thread_ident


@pytest.mark.asyncio
async def test_vlm_ocr_text_data_url_encoding_runs_off_the_event_loop_thread(
    monkeypatch, tmp_path: Path
) -> None:
    """The pipeline's VLM-as-OCR transcription request (_vlm_ocr_text) went
    straight to DocumentImage.data_url() in a list comprehension on the event
    loop before this fix."""
    document = tmp_path / "document.pdf"
    document.write_bytes(b"%PDF-1.4 fake")
    recorded_idents: list[int] = []

    def fake_data_urls(images):
        recorded_idents.append(threading.get_ident())
        return [f"data:image/png;base64,fake{i}" for i in range(len(images))]

    class FakeHttpResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "transcribed text"}}],
            }

    class FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            return FakeHttpResponse()

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
        "docie_bench.extract.service.load_document_images",
        lambda path, *, max_pages, pdf_dpi: [
            DocumentImage(page=1, media_type="image/png", data=b"fake")
        ],
    )
    monkeypatch.setattr("docie_bench.extract.service._data_urls", fake_data_urls)
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: FakeHttpClient())
    monkeypatch.setattr(ExtractionService, "extract_from_text", fake_extract_from_text)
    profile = ModelProfile(
        name="pipeline-profile",
        model="pipeline-profile",
        base_url="",
        api_key="unused",
        kind="pipeline",
        options={"extractor": "extractor-profile", "ocr_model": "vision-profile"},
    )
    profiles = {
        "extractor-profile": ModelProfile(
            name="extractor-profile",
            model="extractor-model",
            base_url="http://example.test/v1",
            api_key="test",
        ),
        "vision-profile": ModelProfile(
            name="vision-profile",
            model="vision-model",
            base_url="http://vision.test/v1",
            api_key="test",
            vision=True,
        ),
    }
    service = ExtractionService(profile, profiles=profiles)

    main_thread_ident = threading.get_ident()
    await service.extract_from_file(
        path=document, ocr_backend_name="tesseract", schema_name="invoice"
    )

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
