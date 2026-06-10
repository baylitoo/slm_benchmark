from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PIL import Image

from docie_bench.ocr.base import OCRBackend, text_to_blocks
from docie_bench.schemas.common import OCRBlock


class TesseractBackend(OCRBackend):
    name = "tesseract"

    def version(self) -> str:
        try:
            return f"1:pytesseract-{version('pytesseract')}"
        except PackageNotFoundError:
            return "1:pytesseract-unknown"

    def extract(self, path: Path) -> list[OCRBlock]:
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "Install optional dependency: pip install small-doc-ie-bench[ocr]"
            ) from exc

        if path.suffix.lower() == ".txt":
            return text_to_blocks(
                path.read_text(encoding="utf-8", errors="replace"), source="manual"
            )
        image = Image.open(path)
        text = pytesseract.image_to_string(image)
        return text_to_blocks(text, source="tesseract")
