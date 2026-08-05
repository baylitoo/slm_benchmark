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


def _minimal_pdf_b64() -> str:
    """A 1-page vector-text PDF (Helvetica) with correct xref offsets — its text
    rasterizes sharper at higher DPI, so page-image size scales with DPI."""
    stream = b"BT /F1 11 Tf 50 780 Td (Resume: Senior Engineer) Tj ET"
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        f"<</Length {len(stream)}>>stream\n".encode() + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<</Size {len(objs) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n".encode()
        + b"%%EOF"
    )
    return base64.b64encode(out.getvalue()).decode("ascii")


def test_pdf_dpi_scales_page_resolution() -> None:
    from PIL import Image

    pdf = _minimal_pdf_b64()
    lo = _render_dpi(pdf, 150)
    hi = _render_dpi(pdf, 300)
    assert lo["pages"] == hi["pages"] == 1

    def _dims(data_url: str) -> tuple[int, int]:
        b64 = data_url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).size

    lo_w, _ = _dims(lo["images"][0])
    hi_w, _ = _dims(hi["images"][0])
    # 300 DPI is 2x the linear resolution of 150 DPI — roughly twice as wide.
    assert hi_w > lo_w * 1.8


def _render_dpi(b64: str, dpi: int) -> dict:
    return asyncio.run(
        render_document(
            RenderDocumentRequest(content_b64=b64, filename="doc.pdf", dpi=dpi),
            tenant=None,  # type: ignore[arg-type]
        )
    )
