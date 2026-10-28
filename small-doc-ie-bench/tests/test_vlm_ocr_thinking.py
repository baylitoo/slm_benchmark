"""VLM OCR transcription never asks the model to think.

``_extract_group`` sends ``enable_thinking: False`` unconditionally, because
extraction never wants thinking. Transcription has the same property for a
stronger reason -- it asks for a verbatim transcript -- but the suppression was
conditional there, and the condition read the wrong profile: the gate consulted
the PIPELINE profile's options while the request goes to the ``ocr_model``
vision deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation
from docie_bench.vision import DocumentImage


class _Response:
    status_code = 200

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "page one text"}}]}

    def raise_for_status(self) -> None:
        return None

    @property
    def text(self) -> str:
        return "page one text"


@pytest.fixture
def posted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """Capture every OCR transcription request body."""
    captured: list[dict[str, Any]] = []

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> _Response:
            captured.append({"url": url, **kwargs})
            return _Response()

    monkeypatch.setattr(
        "docie_bench.extract.service.load_document_images",
        lambda path, *, max_pages, pdf_dpi: [
            DocumentImage(page=1, media_type="image/png", data=b"fake")
        ],
    )
    monkeypatch.setattr(
        "docie_bench.extract.service._data_urls",
        lambda images: ["data:image/png;base64,ZmFrZQ=="],
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: _Client())

    async def fake_extract_from_text(self: ExtractionService, **kwargs: Any):
        return ExtractionResponse(
            request_id="r1",
            schema_name=kwargs["schema_name"],
            model_profile=self.profile.name,
            document_hash=kwargs["document_hash"],
            result={},
            validation=ExtractionValidation(valid=True),
            latency_ms=1,
        )

    monkeypatch.setattr(ExtractionService, "extract_from_text", fake_extract_from_text)
    return captured


def _service(**vision_options: Any) -> ExtractionService:
    pipeline = ModelProfile(
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
            model="lfm3-vl-2b",
            base_url="http://vision.test/v1",
            api_key="test",
            vision=True,
            options=vision_options,
        ),
    }
    return ExtractionService(pipeline, profiles=profiles)


async def _transcribe(service: ExtractionService, tmp_path: Path) -> None:
    document = tmp_path / "invoice.png"
    document.write_bytes(b"not-a-real-image")
    await service.extract_from_file(
        path=document, ocr_backend_name="vlm", schema_name="invoice"
    )


@pytest.mark.asyncio
async def test_transcription_suppresses_thinking_with_no_option_set_anywhere(
    posted: list[dict[str, Any]], tmp_path: Path
) -> None:
    """The bug. Neither profile carries ``no_think`` and the service was not
    constructed with ``disable_thinking``, so a thinking VLM -- LFM2.5-VL,
    LFM3-VL -- spent its budget reasoning about a verbatim transcript, and
    anything it emitted before the transcript landed in the OCR text that
    extraction then ran on.
    """
    await _transcribe(_service(), tmp_path)

    assert posted, "no transcription request was made"
    body = posted[0]["json"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_the_request_still_goes_to_the_vision_deployment(
    posted: list[dict[str, Any]], tmp_path: Path
) -> None:
    """The gate used to read the pipeline profile's options while the request
    goes here -- so a ``no_think`` set on the vision deployment was ignored."""
    await _transcribe(_service(), tmp_path)

    assert posted[0]["url"].startswith("http://vision.test/v1")
    assert posted[0]["json"]["model"] == "lfm3-vl-2b"


@pytest.mark.asyncio
async def test_an_option_on_the_vision_profile_is_no_longer_needed(
    posted: list[dict[str, Any]], tmp_path: Path
) -> None:
    """Suppression is unconditional, so setting it changes nothing -- the point
    being that NOT setting it no longer changes anything either."""
    await _transcribe(_service(no_think=True), tmp_path)

    assert posted[0]["json"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_transcription_asks_for_plain_text_not_json(
    posted: list[dict[str, Any]], tmp_path: Path
) -> None:
    """The property the suppression protects: this is an unconstrained
    plain-text completion, so there is no JSON schema to make a thinking
    preamble harmless."""
    await _transcribe(_service(), tmp_path)

    body = posted[0]["json"]
    assert "response_format" not in body
    assert body["temperature"] == 0
