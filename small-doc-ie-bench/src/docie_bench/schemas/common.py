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
    source: Literal["pdf_text", "tesseract", "paddleocr", "doctr", "manual", "unknown"] = "unknown"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


# A field wrapper may be *present with a null value*: models (notably template-based
# VLMs like NuExtract3) emit `{"value": null}` for an absent optional field rather
# than omitting it. A null value is therefore valid — it means "field seen, no value".


class TextField(BaseModel):
    value: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class DateField(BaseModel):
    value: str | None = Field(default=None, description="ISO-8601 date when possible: YYYY-MM-DD")
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class NumberField(BaseModel):
    value: Decimal | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class MoneyField(BaseModel):
    amount: Decimal | None = None
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
    result: dict[str, Any]
    validation: ExtractionValidation
    usage: Usage | None = None
    latency_ms: int
    dynamic_schema: dict[str, Any] | None = None
    routing: dict[str, Any] | None = None
    # The response-format style the runtime actually honoured for this
    # extraction (after any negotiation downgrade); distinguishes constrained
    # from unconstrained decoding in predictions. None for non-LLM adapters.
    response_format_style: str | None = None
