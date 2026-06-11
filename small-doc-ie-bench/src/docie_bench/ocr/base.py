from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from docie_bench.schemas.common import OCRBlock


class OCRBackend(ABC):
    name: str

    def version(self) -> str:
        """Return a cache-relevant backend/runtime version."""
        return "1"

    def configuration(self) -> dict[str, Any]:
        """Return cache-relevant backend configuration."""
        return {}

    @abstractmethod
    def extract(self, path: Path) -> list[OCRBlock]:
        raise NotImplementedError


def stable_block_id(page: int, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{page}:{index}:{text}".encode()).hexdigest()[:12]
    return f"b{page}_{index}_{digest}"


def text_to_blocks(
    text: str,
    source: Literal["pdf_text", "tesseract", "paddleocr", "manual", "unknown"] = "manual",
) -> list[OCRBlock]:
    blocks: list[OCRBlock] = []
    for idx, line in enumerate(line.strip() for line in text.splitlines()):
        if not line:
            continue
        blocks.append(OCRBlock(id=stable_block_id(1, idx, line), text=line, page=1, source=source))
    if not blocks and text.strip():
        blocks.append(
            OCRBlock(
                id=stable_block_id(1, 0, text.strip()),
                text=text.strip(),
                page=1,
                source=source,
            )
        )
    return blocks
