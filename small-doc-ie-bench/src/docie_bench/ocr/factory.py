from __future__ import annotations

from docie_bench.ocr.base import OCRBackend
from docie_bench.ocr.paddle_backend import PaddleOCRBackend
from docie_bench.ocr.pdf_text import PdfTextBackend
from docie_bench.ocr.tesseract_backend import TesseractBackend


def get_ocr_backend(name: str, *, language: str | None = None) -> OCRBackend:
    normalized = name.lower().strip()
    # "liteparse" is the canonical name; "pdf_text" is the legacy alias for the
    # same backend (PdfTextBackend IS liteparse — PDFium spatial text + an OCR
    # fallback). Lightweight and the sensible default for PDFs.
    if normalized in ("liteparse", "pdf_text"):
        return PdfTextBackend(language=language)
    if normalized == "tesseract":
        return TesseractBackend()
    if normalized == "paddleocr":
        return PaddleOCRBackend(lang=language or "en")
    raise ValueError(
        f"Unknown OCR backend {name!r}. Expected liteparse, tesseract, or paddleocr."
    )
