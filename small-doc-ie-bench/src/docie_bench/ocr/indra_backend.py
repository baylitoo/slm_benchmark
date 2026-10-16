from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from docie_bench.ocr.base import OCRBackend, stable_block_id, text_to_blocks
from docie_bench.schemas.common import BoundingBox, OCRBlock


class IndraOCRBackend(OCRBackend):
    """datalab-to/indra-ocr: CPU-only ONNX line OCR with reading order.

    ``mode="auto"`` reads a PDF's text layer when it is usable; ``"ocr"``
    always runs recognition.
    """

    name = "indra"

    def __init__(self, threads: int = 4, mode: str = "auto", dpi: float = 200.0) -> None:
        self.threads = threads
        self.mode = mode
        self.dpi = dpi
        self._engine: Any | None = None

    def configuration(self) -> dict[str, Any]:
        return {"mode": self.mode, "dpi": self.dpi}

    def version(self) -> str:
        try:
            return f"1:indra-ocr-{version('indra-ocr')}"
        except PackageNotFoundError:
            return "1:indra-ocr-unknown"

    @property
    def engine(self) -> Any:
        if self._engine is None:
            try:
                import indra_ocr
            except ImportError as exc:
                raise RuntimeError(
                    "Install optional dependency: pip install small-doc-ie-bench[indra]"
                ) from exc
            self._engine = indra_ocr.load(threads=self.threads, mode=self.mode, dpi=self.dpi)
        return self._engine

    def extract(self, path: Path) -> list[OCRBlock]:
        if path.suffix.lower() == ".txt":
            return text_to_blocks(
                path.read_text(encoding="utf-8", errors="replace"), source="manual"
            )

        blocks: list[OCRBlock] = []
        idx = 0
        for page_index, page in self.engine.convert(path):
            if page.source == "error":
                raise RuntimeError(
                    f"indra-ocr failed on {path.name} page {page_index + 1}: {page.note}"
                )
            page_no = page_index + 1
            for line in page.lines:
                clean = str(line.text).strip()
                if not clean:
                    continue
                x0, y0, x1, y1 = (float(v) for v in line.box)
                blocks.append(
                    OCRBlock(
                        id=stable_block_id(page_no, idx, clean),
                        text=clean,
                        page=page_no,
                        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                        source="indra",
                        confidence=min(max(float(line.conf), 0.0), 1.0),
                    )
                )
                idx += 1
        return blocks
