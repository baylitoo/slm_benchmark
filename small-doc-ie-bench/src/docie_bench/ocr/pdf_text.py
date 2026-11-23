from __future__ import annotations

import inspect
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from liteparse import LiteParse

from docie_bench.ocr.base import OCRBackend, stable_block_id, text_to_blocks
from docie_bench.ocr.layout import merge_lines, reading_order
from docie_bench.schemas.common import BoundingBox, OCRBlock
from docie_bench.settings import get_settings

#: OCR engines LiteParse runs itself when no ``ocr_server_url`` is set.
OCR_ENGINES = ("tesseract", "ocria-nano")
_PULL_LOCK = threading.Lock()


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
        ocr_engine: str = "tesseract",
        ocr_server_url: str | None = None,
        dpi: int | None = None,
    ) -> None:
        settings = get_settings()
        self._language = language or settings.ocr_language
        self._ocr_server_url = (
            ocr_server_url if ocr_server_url is not None else settings.ocr_server_url
        )
        self._dpi = int(dpi if dpi is not None else settings.ocr_dpi)
        if ocr_engine not in OCR_ENGINES:
            raise ValueError(f"unknown liteparse OCR engine {ocr_engine!r}; expected {OCR_ENGINES}")
        # Always named: once OCRia-nano is pulled, LiteParse's own default
        # switches to it, which would silently change every extraction.
        self._ocr_engine = ocr_engine
        # The artifact and OCR benchmark rows group by this name, so each engine
        # needs its own; Tesseract keeps the historical "pdf_text".
        self.name = "pdf_text" if ocr_engine == "tesseract" else f"liteparse+{ocr_engine}"

    def version(self) -> str:
        return f"1:liteparse-{_liteparse_version()}"

    def configuration(self) -> dict[str, Any]:
        # Part of the OCR cache key: changing the OCR route/dpi/language must
        # invalidate cached artifacts.
        return {
            "engine": "liteparse",
            "layout": "xy-cut-v1",
            "dpi": self._dpi,
            "ocr_server_url": self._ocr_server_url or "",
            "language": self._language or "",
            "ocr_engine": self._ocr_engine,
        }

    def extract(self, path: Path) -> list[OCRBlock]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return text_to_blocks(
                path.read_text(encoding="utf-8", errors="replace"), source="manual"
            )
        if suffix != ".pdf":
            raise ValueError(f"pdf_text backend supports .pdf and .txt only, got {path.suffix}")

        engine: dict[str, Any] = {}
        if not self._ocr_server_url and _accepts_ocr_backend():
            if self._ocr_engine == "ocria-nano":
                _ensure_ocria_nano()
            engine["ocr_backend"] = self._ocr_engine
        elif self._ocr_engine != "tesseract" and not self._ocr_server_url:
            raise RuntimeError(
                f"the {self._ocr_engine!r} OCR engine needs liteparse>=2.21.0 "
                f"(installed: {_liteparse_version()})"
            )
        parser = LiteParse(
            ocr_enabled=True,
            ocr_server_url=self._ocr_server_url,
            ocr_language=self._language,
            dpi=float(self._dpi),
            quiet=True,
            **engine,
        )
        result = parser.parse(path)

        raw: list[OCRBlock] = []
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
                raw.append(
                    OCRBlock(
                        id=f"raw-{page.page_num}-{idx}",
                        text=text,
                        page=page.page_num,
                        bbox=bbox,
                        source="pdf_text",
                        confidence=float(confidence) if confidence is not None else None,
                    )
                )
        ordered = merge_lines(reading_order(raw))
        # Ids follow reading order so evidence ids stay stable for a given
        # document + layout algorithm.
        return [
            block.model_copy(update={"id": stable_block_id(block.page, idx, block.text)})
            for idx, block in enumerate(ordered)
        ]


def _liteparse_version() -> str:
    try:
        return version("liteparse")
    except PackageNotFoundError:
        return "unknown"


def _accepts_ocr_backend() -> bool:
    """Whether the installed LiteParse lets the caller pick its engine (2.21+)."""
    return "ocr_backend" in inspect.signature(LiteParse.__init__).parameters


def ocria_nano_supported() -> bool:
    """Whether this LiteParse can run OCRia-nano (its ``models`` API exists)."""
    import importlib.util

    return _accepts_ocr_backend() and importlib.util.find_spec("liteparse.models") is not None


def _ensure_ocria_nano() -> None:
    """Download OCRia-nano's model files once (LiteParse verifies each sha256)."""
    from liteparse import models  # type: ignore[attr-defined]  # liteparse>=2.21

    with _PULL_LOCK:
        installed = [str(getattr(m, "name", m)) for m in models.list()]
        if not any(name.startswith("ocria-nano") for name in installed):
            models.pull("ocria-nano", quiet=True)
