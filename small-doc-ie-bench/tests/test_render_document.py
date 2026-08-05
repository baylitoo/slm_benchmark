"""POST /v1/studio/render-document: rasterize an upload to page-image data URLs."""

from __future__ import annotations

import asyncio
import base64
import io

import pytest
from fastapi import HTTPException
from PIL import Image

from docie_bench.inngest.studio_api import RenderDocumentRequest, render_document


def _png_b64(size: tuple[int, int] = (4, 4)) -> str:
    img = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render(b64: str, filename: str) -> dict:
    return asyncio.run(
        render_document(
            RenderDocumentRequest(content_b64=b64, filename=filename), tenant=None  # type: ignore[arg-type]
        )
    )


def test_png_returns_one_image_data_url() -> None:
    result = _render(_png_b64(), "photo.png")
    assert result["pages"] == 1
    assert len(result["images"]) == 1
    # Normalized to a PNG data URL a vision model accepts.
    assert result["images"][0].startswith("data:image/png;base64,")


def test_bad_base64_is_400() -> None:
    with pytest.raises(HTTPException) as exc:
        _render("!!!not-base64!!!", "x.png")
    assert exc.value.status_code == 400


def test_unsupported_suffix_is_400() -> None:
    # A .docx (not PDF/image) is rejected by the rasterizer as a client error.
    with pytest.raises(HTTPException) as exc:
        _render(_png_b64(), "resume.docx")
    assert exc.value.status_code == 400
