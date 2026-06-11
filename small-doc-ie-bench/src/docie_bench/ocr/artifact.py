from __future__ import annotations

import base64
import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from docie_bench.schemas.common import OCRBlock

ARTIFACT_FORMAT_VERSION = 1


class OCRPageImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1)
    media_type: str
    sha256: str
    data_base64: str | None = None

    @classmethod
    def from_bytes(
        cls, *, page: int, media_type: str, data: bytes, embed: bool = True
    ) -> OCRPageImage:
        return cls(
            page=page,
            media_type=media_type,
            sha256=hashlib.sha256(data).hexdigest(),
            data_base64=base64.b64encode(data).decode("ascii") if embed else None,
        )


class OCRQualitySignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_count: int = Field(ge=0)
    block_count: int = Field(ge=0)
    page_count: int = Field(ge=0)
    mean_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    low_confidence_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    empty: bool
    low_quality: bool
    reasons: list[str] = Field(default_factory=list)


class OCRArtifact(BaseModel):
    """Portable, versioned output shared by OCR backends, cache, and benchmarks."""

    model_config = ConfigDict(extra="forbid")

    format_version: int = ARTIFACT_FORMAT_VERSION
    document_hash: str
    backend: str
    backend_version: str
    language: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    blocks: list[OCRBlock] = Field(default_factory=list)
    page_images: list[OCRPageImage] = Field(default_factory=list)
    quality: OCRQualitySignals
    latency_ms: int = Field(ge=0)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks)


def quality_signals(
    blocks: list[OCRBlock],
    *,
    low_confidence_threshold: float = 0.5,
    low_quality_confidence: float = 0.65,
) -> OCRQualitySignals:
    confidences = [block.confidence for block in blocks if block.confidence is not None]
    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    low_fraction = (
        sum(value < low_confidence_threshold for value in confidences) / len(confidences)
        if confidences
        else None
    )
    characters = sum(len(block.text.strip()) for block in blocks)
    pages = {block.page for block in blocks}
    reasons: list[str] = []
    if characters == 0:
        reasons.append("empty")
    if mean_confidence is not None and mean_confidence < low_quality_confidence:
        reasons.append("low_mean_confidence")
    if low_fraction is not None and low_fraction > 0.5:
        reasons.append("many_low_confidence_blocks")
    return OCRQualitySignals(
        character_count=characters,
        block_count=len(blocks),
        page_count=len(pages),
        mean_confidence=mean_confidence,
        low_confidence_fraction=low_fraction,
        empty=characters == 0,
        low_quality=bool(reasons),
        reasons=reasons,
    )
