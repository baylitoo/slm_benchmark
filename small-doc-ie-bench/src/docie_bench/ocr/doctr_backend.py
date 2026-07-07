from __future__ import annotations

from pathlib import Path

from docie_bench.ocr.base import OCRBackend, text_to_blocks
from docie_bench.schemas.common import OCRBlock


class DocTRBackend(OCRBackend):
    """docTR OCR competitor behind the existing ``ocr`` kind (a text producer).

    docTR is a lighter, optional add: it registers as an OCR backend via the
    factory and reuses the whole OCR/pipeline serving path — no new profile kind.
    python-doctr (and its torch/tensorflow stack) is lazy-imported at extraction
    time, mirroring the tesseract backend, so importing this module never needs
    the optional extra.
    """

    name = "doctr"

    def __init__(self, language: str | None = None) -> None:
        self.language = language

    def extract(self, path: Path) -> list[OCRBlock]:
        if path.suffix.lower() == ".txt":
            return text_to_blocks(
                path.read_text(encoding="utf-8", errors="replace"), source="manual"
            )
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
        except ImportError as exc:
            raise RuntimeError(
                "Install optional dependency: pip install small-doc-ie-bench[doctr]"
            ) from exc

        if path.suffix.lower() == ".pdf":
            document = DocumentFile.from_pdf(str(path))
        else:
            document = DocumentFile.from_images(str(path))
        predictor = ocr_predictor(pretrained=True)
        result = predictor(document)
        lines: list[str] = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    text = " ".join(word.value for word in line.words)
                    if text.strip():
                        lines.append(text)
        return text_to_blocks("\n".join(lines), source="doctr")
