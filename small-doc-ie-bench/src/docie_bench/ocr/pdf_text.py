from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pdfplumber

from docie_bench.ocr.base import OCRBackend, stable_block_id, text_to_blocks
from docie_bench.schemas.common import BoundingBox, OCRBlock


class PdfTextBackend(OCRBackend):
    name = "pdf_text"

    def version(self) -> str:
        try:
            return f"1:pdfplumber-{version('pdfplumber')}"
        except PackageNotFoundError:
            return "1:pdfplumber-unknown"

    def extract(self, path: Path) -> list[OCRBlock]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return text_to_blocks(
                path.read_text(encoding="utf-8", errors="replace"), source="manual"
            )
        if suffix != ".pdf":
            raise ValueError(f"pdf_text backend supports .pdf and .txt only, got {path.suffix}")

        blocks: list[OCRBlock] = []
        with pdfplumber.open(str(path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
                if not words:
                    text = page.extract_text() or ""
                    for idx, line in enumerate(text.splitlines()):
                        line = line.strip()
                        if line:
                            blocks.append(
                                OCRBlock(
                                    id=stable_block_id(page_idx, idx, line),
                                    text=line,
                                    page=page_idx,
                                    source="pdf_text",
                                )
                            )
                    continue
                # Group words into approximate lines by top coordinate.
                rows: list[list[dict[str, Any]]] = []
                for word in words:
                    placed = False
                    for row in rows:
                        if abs(row[0]["top"] - word["top"]) < 3:
                            row.append(word)
                            placed = True
                            break
                    if not placed:
                        rows.append([word])
                for idx, row in enumerate(rows):
                    row = sorted(row, key=lambda item: item["x0"])
                    text = " ".join(item["text"] for item in row).strip()
                    if not text:
                        continue
                    bbox = BoundingBox(
                        x0=min(float(item["x0"]) for item in row),
                        y0=min(float(item["top"]) for item in row),
                        x1=max(float(item["x1"]) for item in row),
                        y1=max(float(item["bottom"]) for item in row),
                    )
                    blocks.append(
                        OCRBlock(
                            id=stable_block_id(page_idx, idx, text),
                            text=text,
                            page=page_idx,
                            bbox=bbox,
                            source="pdf_text",
                        )
                    )
        return blocks
