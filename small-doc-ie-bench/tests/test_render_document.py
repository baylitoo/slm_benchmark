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
    assert result["total_pages"] == 1
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


def _minimal_pdf_b64(*, page_count: int = 1) -> str:
    """A vector-text PDF (Helvetica) with correct xref offsets, ``page_count``
    pages — its text rasterizes sharper at higher DPI, so page-image size
    scales with DPI."""
    page_ids = list(range(3, 3 + page_count))
    content_ids = list(range(page_ids[-1] + 1, page_ids[-1] + 1 + page_count))
    font_id = content_ids[-1] + 1

    objs: dict[int, bytes] = {
        1: b"<</Type/Catalog/Pages 2 0 R>>",
        2: f"<</Type/Pages/Kids[{' '.join(f'{pid} 0 R' for pid in page_ids)}]"
        f"/Count {page_count}>>".encode(),
        font_id: b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    }
    for i, (page_id, content_id) in enumerate(zip(page_ids, content_ids, strict=True), 1):
        stream = f"BT /F1 11 Tf 50 780 Td (Page {i}) Tj ET".encode()
        objs[page_id] = (
            f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]/Contents {content_id} 0 R"
            f"/Resources<</Font<</F1 {font_id} 0 R>>>>>>".encode()
        )
        objs[content_id] = f"<</Length {len(stream)}>>stream\n".encode() + stream + b"\nendstream"

    max_id = max(objs)
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for i in range(1, max_id + 1):
        offsets[i] = out.tell()
        out.write(f"{i} 0 obj\n".encode() + objs[i] + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {max_id + 1}\n".encode() + b"0000000000 65535 f \n")
    for i in range(1, max_id + 1):
        out.write(f"{offsets[i]:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<</Size {max_id + 1}/Root 1 0 R>>\nstartxref\n{xref}\n".encode() + b"%%EOF"
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


def test_multipage_pdf_thumbnail_with_explicit_page_succeeds() -> None:
    # This is the reported bug: a page-1-only thumbnail preview (max_pages=1)
    # against a multi-page PDF used to ALWAYS 400 -- load_document_images
    # rasterized every page, then rejected because 5 > 1. Passing pages=[1]
    # explicitly must render just that page, no reject, regardless of the
    # document's true length.
    pdf = _minimal_pdf_b64(page_count=5)
    result = asyncio.run(
        render_document(
            RenderDocumentRequest(
                content_b64=pdf, filename="doc.pdf", max_pages=1, pages=[1]
            ),
            tenant=None,  # type: ignore[arg-type]
        )
    )
    assert result["pages"] == 1
    assert len(result["images"]) == 1
    # The response reports the PDF's TRUE page count even though only page 1
    # was rendered -- len(images) alone would misleadingly say "1 page total".
    assert result["total_pages"] == 5


def test_multipage_pdf_without_explicit_pages_still_rejects_over_max_pages() -> None:
    # Regression guard: the vision-send path's intentional policy (reject a
    # document with more pages than the model will see, rather than silently
    # truncating it) must still apply when `pages` is NOT given.
    pdf = _minimal_pdf_b64(page_count=5)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            render_document(
                RenderDocumentRequest(content_b64=pdf, filename="doc.pdf", max_pages=2),
                tenant=None,  # type: ignore[arg-type]
            )
        )
    assert exc.value.status_code == 400
    assert "vision_max_pages is 2" in exc.value.detail


def test_total_pages_reflects_true_count_when_all_pages_rendered() -> None:
    pdf = _minimal_pdf_b64(page_count=3)
    result = asyncio.run(
        render_document(
            RenderDocumentRequest(content_b64=pdf, filename="doc.pdf", max_pages=8),
            tenant=None,  # type: ignore[arg-type]
        )
    )
    assert result["pages"] == 3
    assert result["total_pages"] == 3
