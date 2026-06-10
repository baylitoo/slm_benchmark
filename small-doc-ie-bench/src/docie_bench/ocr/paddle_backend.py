from __future__ import annotations

from pathlib import Path
from typing import Any

from docie_bench.ocr.base import OCRBackend, stable_block_id, text_to_blocks
from docie_bench.schemas.common import BoundingBox, OCRBlock


class PaddleOCRBackend(OCRBackend):
    name = "paddleocr"

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang
        self._engine: Any | None = None

    @property
    def engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("Install optional dependency: pip install small-doc-ie-bench[paddle]") from exc
            self._engine = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        return self._engine

    def extract(self, path: Path) -> list[OCRBlock]:
        if path.suffix.lower() == ".txt":
            return text_to_blocks(path.read_text(encoding="utf-8", errors="replace"), source="manual")

        result = self.engine.ocr(str(path), cls=True)
        blocks: list[OCRBlock] = []
        idx = 0
        for page_idx, page in enumerate(result or [], start=1):
            for item in page or []:
                box = item[0]
                text, conf = item[1]
                if not text or not str(text).strip():
                    continue
                xs = [float(point[0]) for point in box]
                ys = [float(point[1]) for point in box]
                clean = str(text).strip()
                blocks.append(
                    OCRBlock(
                        id=stable_block_id(page_idx, idx, clean),
                        text=clean,
                        page=page_idx,
                        bbox=BoundingBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys)),
                        source="paddleocr",
                        confidence=float(conf) if conf is not None else None,
                    )
                )
                idx += 1
        return blocks
