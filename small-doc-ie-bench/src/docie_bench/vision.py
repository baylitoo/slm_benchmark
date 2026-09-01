from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from liteparse import LiteParse
from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class DocumentImage:
    page: int
    media_type: str
    data: bytes

    def data_url(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"


def load_document_images(
    path: Path,
    *,
    max_pages: int = 8,
    pdf_dpi: int = 150,
    pages: list[int] | None = None,
) -> list[DocumentImage]:
    """Load an image document or rasterize PDF pages for a vision-capable model.

    ``pages``, when given, is an explicit list of 1-indexed page numbers to
    rasterize -- ONLY those pages are rendered, and the ``max_pages`` reject
    check does not apply (the caller told us exactly what it wants, e.g. a
    cheap page-1 thumbnail preview). ``None`` (the default) keeps the existing
    contract: rasterize every page, then reject if there are more than
    ``max_pages`` -- the vision-send path relies on this to guarantee the
    model actually sees the whole document it's told about.
    """
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _rasterize_pdf(path, max_pages=max_pages, pdf_dpi=pdf_dpi, pages=pages)
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return [_normalize_image(path, page=1)]
    raise ValueError(
        f"Vision ingestion supports PDF and image files only, got {path.suffix or '<no suffix>'}"
    )


def _normalize_image(path: Path, *, page: int) -> DocumentImage:
    with Image.open(path) as image:
        image.load()
        return _image_to_png(image, page=page)


def _image_to_png(image: Image.Image, *, page: int) -> DocumentImage:
    # PNG is broadly supported by OpenAI-compatible multimodal gateways and avoids
    # passing through TIFF/multipage container details that gateways often reject.
    normalized = image.convert("RGB")
    output = io.BytesIO()
    normalized.save(output, format="PNG", optimize=True)
    return DocumentImage(page=page, media_type="image/png", data=output.getvalue())


def _rasterize_pdf(
    path: Path, *, max_pages: int, pdf_dpi: int, pages: list[int] | None = None
) -> list[DocumentImage]:
    # liteparse renders pages via PDFium; screenshot() returns PNG bytes per page.
    parser = LiteParse(dpi=float(pdf_dpi), quiet=True)
    screenshots = parser.screenshot(path, page_numbers=pages)
    if not screenshots:
        raise ValueError("PDF contains no pages")
    # An explicit page list is exactly what the caller asked for -- no
    # max_pages reject here, and it's cheap regardless of the PDF's total
    # length since liteparse only rasterizes the requested pages.
    if pages is None and len(screenshots) > max_pages:
        raise ValueError(f"PDF has {len(screenshots)} pages; vision_max_pages is {max_pages}")
    return [
        DocumentImage(page=shot.page_num, media_type="image/png", data=shot.image_bytes)
        for shot in sorted(screenshots, key=lambda shot: shot.page_num)
    ]
