from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from liteparse import LiteParse

from docie_bench.ocr.base import OCRBackend, stable_block_id, text_to_blocks
from docie_bench.schemas.common import BoundingBox, OCRBlock
from docie_bench.settings import get_settings


class PdfTextBackend(OCRBackend):
    """PDF text backend built on liteparse (PDFium spatial text + pluggable OCR).

    PDFium extracts a PDF's native text layer fast and with spatial geometry;
    pages without a usable text layer (scanned) are OCR'd by liteparse — the
    built-in Tesseract by default, or a VLM-backed server when ``ocr_server_url``
    is configured. The VLM route matters only for text-only extraction models
    that cannot read the page image themselves; vision-capable profiles bypass
    OCR entirely and receive page images directly (see ``docie_bench.vision``).
    """

    name = "pdf_text"

    def __init__(
        self,
        *,
        language: str | None = None,
        ocr_server_url: str | None = None,
        dpi: int | None = None,
    ) -> None:
        settings = get_settings()
        self._language = language or settings.ocr_language
        self._ocr_server_url = (
            ocr_server_url if ocr_server_url is not None else settings.ocr_server_url
        )
        self._dpi = int(dpi if dpi is not None else settings.ocr_dpi)

    def version(self) -> str:
        try:
            return f"1:liteparse-{version('liteparse')}"
        except PackageNotFoundError:
            return "1:liteparse-unknown"

    def configuration(self) -> dict[str, Any]:
        # Part of the OCR cache key: changing the OCR route/dpi/language must
        # invalidate cached artifacts.
        return {
            "engine": "liteparse",
            "dpi": self._dpi,
            "ocr_server_url": self._ocr_server_url or "",
            "language": self._language or "",
        }

    def extract(self, path: Path) -> list[OCRBlock]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return text_to_blocks(
                path.read_text(encoding="utf-8", errors="replace"), source="manual"
            )
        if suffix != ".pdf":
            raise ValueError(f"pdf_text backend supports .pdf and .txt only, got {path.suffix}")

        parser = LiteParse(
            ocr_enabled=True,
            ocr_server_url=self._ocr_server_url,
            ocr_language=self._language,
            dpi=float(self._dpi),
            quiet=True,
        )
        result = parser.parse(path)

        blocks: list[OCRBlock] = []
        for page in result.pages:
            for idx, item in enumerate(page.text_items):
                text = (item.text or "").strip()
                if not text:
                    continue
                bbox = BoundingBox(
                    x0=float(item.x),
                    y0=float(item.y),
                    x1=float(item.x) + float(item.width),
                    y1=float(item.y) + float(item.height),
                )
                confidence = getattr(item, "confidence", None)
                blocks.append(
                    OCRBlock(
                        id=stable_block_id(page.page_num, idx, text),
                        text=text,
                        page=page.page_num,
                        bbox=bbox,
                        source="pdf_text",
                        confidence=float(confidence) if confidence is not None else None,
                    )
                )
        return blocks
