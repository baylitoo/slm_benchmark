from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from docie_bench.benchmark.judge import EvaluationMode
from docie_bench.benchmark.runner import summarize
from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.openai_client import OpenAICompatibleClient
from docie_bench.vision import DocumentImage, load_document_images


def _profile(*, vision: bool = False) -> ModelProfile:
    return ModelProfile(
        name="test",
        model="test-model",
        base_url="http://example.test/v1",
        api_key="test",
        response_format_style="json_object",
        vision=vision,
    )


def test_load_document_images_normalizes_image_to_png(tmp_path: Path) -> None:
    path = tmp_path / "scan.jpg"
    Image.new("RGB", (8, 6), "white").save(path, format="JPEG")

    images = load_document_images(path)

    assert len(images) == 1
    assert images[0].page == 1
    assert images[0].media_type == "image/png"
    assert images[0].data.startswith(b"\x89PNG")
    assert base64.b64decode(images[0].data_url().split(",", 1)[1]) == images[0].data


def test_load_document_images_rasterizes_pdf_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePixmap:
        def tobytes(self, output: str) -> bytes:
            assert output == "png"
            return b"png-page"

    class FakePage:
        def get_pixmap(self, *, matrix, alpha: bool):
            assert matrix == ("matrix", 150 / 72, 150 / 72)
            assert alpha is False
            return FakePixmap()

    class FakeDocument:
        page_count = 2

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def load_page(self, index: int):
            assert index in {0, 1}
            return FakePage()

    fake_fitz = SimpleNamespace(
        open=lambda path: FakeDocument(),
        Matrix=lambda x, y: ("matrix", x, y),
    )
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    images = load_document_images(Path("scan.pdf"), max_pages=2)

    assert [image.page for image in images] == [1, 2]
    assert [image.data for image in images] == [b"png-page", b"png-page"]


@pytest.mark.asyncio
async def test_openai_client_sends_multimodal_user_content() -> None:
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    class FakeHttpClient:
        async def post(self, path: str, *, json: dict):
            captured["path"] = path
            captured["payload"] = json
            return FakeResponse()

        async def aclose(self):
            return None

    client = OpenAICompatibleClient(_profile())
    await client._client.aclose()
    client._client = FakeHttpClient()

    await client.chat_json(
        system_prompt="system",
        user_prompt="extract",
        schema_name="invoice",
        schema={"type": "object"},
        image_urls=["data:image/png;base64,cG5n"],
    )

    content = captured["payload"]["messages"][1]["content"]
    assert content == [
        {"type": "text", "text": "extract"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}},
    ]


@pytest.mark.asyncio
async def test_vision_profile_bypasses_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "invoice.png"
    path.write_bytes(b"document")
    captured = {}

    monkeypatch.setattr(
        "docie_bench.extract.service.load_document_images",
        lambda *args, **kwargs: [DocumentImage(page=1, media_type="image/png", data=b"png")],
    )

    async def fake_extract_blocks(self, **kwargs):
        captured.update(kwargs)
        return "response"

    monkeypatch.setattr(ExtractionService, "_extract_blocks", fake_extract_blocks)

    response = await ExtractionService(_profile(vision=True)).extract_from_file(
        path=path,
        ocr_backend_name="pdf_text",
        schema_name="invoice",
    )

    assert response == "response"
    assert captured["blocks"] == []
    assert captured["images"][0].data == b"png"


@pytest.mark.asyncio
async def test_vision_backend_requires_vision_profile(tmp_path: Path) -> None:
    path = tmp_path / "invoice.png"
    path.write_bytes(b"document")

    with pytest.raises(ValueError, match="requires a model profile"):
        await ExtractionService(_profile()).extract_from_file(
            path=path,
            ocr_backend_name="vision",
            schema_name="invoice",
        )


def test_benchmark_summary_labels_vision_path() -> None:
    metrics = summarize(
        [
            {
                "model_profile": "vision-model",
                "ingestion_path": "vision",
                "ok": True,
                "latency_ms": 10,
                "validation": {"valid": True},
                "score": {},
            }
        ],
        eval_mode=EvaluationMode.GROUND_TRUTH,
    )

    assert metrics["summary"][0]["ingestion_path"] == "vision"
