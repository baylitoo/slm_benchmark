"""Benchmark-side wiring for kind="pipeline" profiles.

ModelProfile.kind/.options (PR #50) and serving.solutions.PipelineSolution
already let the Studio gateway serve an OCR->LLM pipeline as a chat
completion. The benchmark runner never read profile.kind at all -- a
kind="pipeline" profile fell through ExtractionService.extract_from_file's
plain-OCR branch, which OCRs with the GLOBAL default backend (ignoring the
profile's own options) and then tries to call the pipeline profile itself
as if it were a real LLM (it isn't -- the real work is delegated to
options.extractor). These tests cover the fix: ExtractionService now
dispatches kind="pipeline" to its own OCR step -- EITHER a built-in backend
(options.ocr_backend) or a deployed vision model doing VLM-as-OCR
(options.ocr_model) -- followed by a delegated extract_from_text call on the
resolved extractor profile, reusing 100% of the normal text-extraction
machinery (prompts, schema validation, grounding, nuextract normalization)
unchanged.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation, OCRBlock
from docie_bench.vision import DocumentImage


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def version(self) -> str:
        return "1"

    def configuration(self) -> dict:
        return {}

    def extract(self, path: Path) -> list[OCRBlock]:
        self.calls.append(path)
        return [OCRBlock(id="b1", text="hello from ocr", source="manual")]


def _pipeline_profile(**options: object) -> ModelProfile:
    return ModelProfile(
        name="pipeline-profile",
        model="pipeline-profile",
        base_url="",
        api_key="unused",
        kind="pipeline",
        options={"extractor": "extractor-profile", "ocr_backend": "fake", **options},
    )


def _extractor_profile(kind: str = "passthrough") -> ModelProfile:
    return ModelProfile(
        name="extractor-profile",
        model="extractor-model",
        base_url="http://example.test/v1",
        api_key="test",
        kind=kind,
    )


def _vision_profile(*, kind: str = "passthrough", vision: bool = True) -> ModelProfile:
    return ModelProfile(
        name="vision-profile",
        model="vision-model",
        base_url="http://vision.test/v1",
        api_key="test",
        kind=kind,
        vision=vision,
    )


class FakeHttpResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> object:
        return self._payload


class FakeHttpClient:
    """Stands in for `async with httpx.AsyncClient() as client: ...` --
    _vlm_ocr_text constructs its own client per call (no DI, mirroring the
    gateway's PipelineSolution having http_client injected instead), so the
    test monkeypatches httpx.AsyncClient itself rather than passing a
    transport."""

    def __init__(self, response: FakeHttpResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def __aenter__(self) -> FakeHttpClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, url: str, **kwargs: object) -> FakeHttpResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


@pytest.mark.asyncio
async def test_pipeline_ocrs_with_its_own_backend_then_delegates_to_extractor(
    monkeypatch, tmp_path: Path
) -> None:
    document = tmp_path / "invoice.txt"
    document.write_text("irrelevant", encoding="utf-8")
    backend = FakeBackend()
    captured: list[dict] = []

    async def fake_extract_blocks(self, **kwargs):
        captured.append({"profile": self.profile.name, **kwargs})
        return ExtractionResponse(
            request_id="r1",
            schema_name=kwargs["schema_name"],
            model_profile=self.profile.name,  # the INNER extractor's name
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
    monkeypatch.setattr(ExtractionService, "_extract_blocks", fake_extract_blocks)

    profiles = {"extractor-profile": _extractor_profile()}
    service = ExtractionService(_pipeline_profile(), profiles=profiles)

    response = await service.extract_from_file(
        path=document, ocr_backend_name="tesseract", schema_name="invoice"
    )

    # OCR ran with the PROFILE's own backend ("fake"), not a global default.
    assert backend.calls == [document]
    # The extraction that actually ran was on the resolved extractor profile,
    # over the OCR'd text -- not the pipeline profile itself.
    assert captured[0]["profile"] == "extractor-profile"
    assert captured[0]["blocks"][0].text == "hello from ocr"
    assert captured[0]["schema_name"] == "invoice"
    # document_hash must come from the OCR result, not be dropped -- it feeds
    # reproducibility/caching elsewhere in the benchmark, so losing it in a
    # future refactor would be a real (not cosmetic) regression.
    assert captured[0]["document_hash"] is not None
    assert captured[0]["schema_mode"] == "static"
    assert captured[0]["language"] is None
    assert captured[0]["metadata"] == {}
    # The response is reported under the PIPELINE profile's name (what the
    # caller asked to benchmark), even though _extract_blocks ran as the
    # inner extractor.
    assert response.model_profile == "pipeline-profile"


@pytest.mark.asyncio
async def test_pipeline_requires_extractor_option() -> None:
    profile = ModelProfile(
        name="p", model="p", base_url="", api_key="x", kind="pipeline", options={}
    )
    service = ExtractionService(profile, profiles={})

    with pytest.raises(ValueError, match="options.extractor"):
        await service.extract_from_file(
            path=Path("doc.txt"), ocr_backend_name="tesseract", schema_name="invoice"
        )


@pytest.mark.asyncio
async def test_pipeline_requires_known_extractor_profile() -> None:
    service = ExtractionService(_pipeline_profile(), profiles={})

    with pytest.raises(ValueError, match="not configured"):
        await service.extract_from_file(
            path=Path("doc.txt"), ocr_backend_name="tesseract", schema_name="invoice"
        )


@pytest.mark.asyncio
async def test_pipeline_requires_passthrough_extractor() -> None:
    profiles = {"extractor-profile": _extractor_profile(kind="ocr")}
    service = ExtractionService(_pipeline_profile(), profiles=profiles)

    with pytest.raises(ValueError, match="passthrough"):
        await service.extract_from_file(
            path=Path("doc.txt"), ocr_backend_name="tesseract", schema_name="invoice"
        )


@pytest.mark.asyncio
async def test_pipeline_vlm_ocr_transcribes_then_delegates_to_extractor(
    monkeypatch, tmp_path: Path
) -> None:
    document = tmp_path / "invoice.png"
    document.write_bytes(b"not a real png, never read")
    captured: list[dict] = []
    http_client = FakeHttpClient(
        FakeHttpResponse({"choices": [{"message": {"content": "hello from the vlm"}}]})
    )

    async def fake_extract_blocks(self, **kwargs):
        captured.append({"profile": self.profile.name, **kwargs})
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
        lambda path, **kw: [DocumentImage(page=1, media_type="image/png", data=b"png")],
    )
    monkeypatch.setattr(
        "docie_bench.extract.service.httpx.AsyncClient", lambda: http_client
    )
    monkeypatch.setattr(ExtractionService, "_extract_blocks", fake_extract_blocks)

    profiles = {
        "extractor-profile": _extractor_profile(),
        "vision-profile": _vision_profile(),
    }
    service = ExtractionService(
        _pipeline_profile(ocr_model="vision-profile"), profiles=profiles
    )

    response = await service.extract_from_file(
        path=document, ocr_backend_name="tesseract", schema_name="invoice"
    )

    # The vision model was called with the OCR transcription prompt + the
    # image, not the extraction schema/prompt -- that's the extractor's job.
    assert len(http_client.calls) == 1
    sent = http_client.calls[0]["json"]
    assert sent["model"] == "vision-model"
    content = sent["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "Transcribe" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    # The vision model's transcription became the extractor's input text --
    # not the pipeline profile's own (unused) endpoint.
    assert captured[0]["profile"] == "extractor-profile"
    assert captured[0]["blocks"][0].text == "hello from the vlm"
    assert captured[0]["document_hash"] is not None
    assert response.model_profile == "pipeline-profile"


@pytest.mark.asyncio
async def test_pipeline_vlm_ocr_requires_known_profile() -> None:
    profiles = {"extractor-profile": _extractor_profile()}
    service = ExtractionService(
        _pipeline_profile(ocr_model="missing-vision-profile"), profiles=profiles
    )

    with pytest.raises(ValueError, match="not configured"):
        await service.extract_from_file(
            path=Path("doc.png"), ocr_backend_name="tesseract", schema_name="invoice"
        )


@pytest.mark.asyncio
async def test_pipeline_vlm_ocr_requires_passthrough_vision_profile() -> None:
    profiles = {
        "extractor-profile": _extractor_profile(),
        "vision-profile": _vision_profile(vision=False),  # not actually a vision model
    }
    service = ExtractionService(
        _pipeline_profile(ocr_model="vision-profile"), profiles=profiles
    )

    with pytest.raises(ValueError, match="vision deployment"):
        await service.extract_from_file(
            path=Path("doc.png"), ocr_backend_name="tesseract", schema_name="invoice"
        )
