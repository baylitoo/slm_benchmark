from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

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
    path: Path, *, max_pages: int = 8, pdf_dpi: int = 150
) -> list[DocumentImage]:
    """Load an image document or rasterize PDF pages for a vision-capable model."""
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _rasterize_pdf(path, max_pages=max_pages, pdf_dpi=pdf_dpi)
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


def _rasterize_pdf(path: Path, *, max_pages: int, pdf_dpi: int) -> list[DocumentImage]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF vision ingestion requires PyMuPDF. Install project dependencies or "
            "`pip install pymupdf`."
        ) from exc

    images: list[DocumentImage] = []
    with fitz.open(path) as document:
        if document.page_count > max_pages:
            raise ValueError(
                f"PDF has {document.page_count} pages; vision_max_pages is {max_pages}"
            )
        scale = pdf_dpi / 72
        matrix = fitz.Matrix(scale, scale)
        for page_index in range(document.page_count):
            pixmap = document.load_page(page_index).get_pixmap(matrix=matrix, alpha=False)
            images.append(
                DocumentImage(
                    page=page_index + 1,
                    media_type="image/png",
                    data=pixmap.tobytes("png"),
                )
            )
    if not images:
        raise ValueError("PDF contains no pages")
    return images
