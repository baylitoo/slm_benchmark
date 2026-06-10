from __future__ import annotations

from pathlib import Path

from PIL import Image

from docie_bench.ocr.base import OCRBackend, text_to_blocks


class TesseractBackend(OCRBackend):
    name = "tesseract"

    def extract(self, path: Path):
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install small-doc-ie-bench[ocr]") from exc

        if path.suffix.lower() == ".txt":
            return text_to_blocks(path.read_text(encoding="utf-8", errors="replace"), source="manual")
        image = Image.open(path)
        text = pytesseract.image_to_string(image)
        return text_to_blocks(text, source="tesseract")
