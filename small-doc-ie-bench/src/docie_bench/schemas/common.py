from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class OCRBlock(BaseModel):
    id: str
    text: str
    page: int = 1
    bbox: BoundingBox | None = None
    source: Literal["pdf_text", "tesseract", "paddleocr", "manual", "unknown"] = "unknown"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TextField(BaseModel):
    value: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class DateField(BaseModel):
    value: str = Field(description="ISO-8601 date when possible: YYYY-MM-DD")
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class NumberField(BaseModel):
    value: Decimal
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class MoneyField(BaseModel):
    amount: Decimal
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class ExtractionValidation(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    schema_name: str
    model_profile: str
    document_hash: str | None
    result: dict
    validation: ExtractionValidation
    usage: Usage | None = None
    latency_ms: int
    dynamic_schema: dict[str, Any] | None = None
