from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from docie_bench.schemas.common import OCRBlock


class ExtractTextRequest(BaseModel):
    text: str | None = None
    ocr_blocks: list[OCRBlock] | None = None
    schema_name: str = "invoice"
    schema_mode: Literal["static", "dynamic"] = "static"
    dynamic_schema: dict[str, Any] | None = None
    schema_proposer_profile: str | None = None
    model_profile: str | None = None
    language: str | None = None
    document_id: str | None = None
    document_hash: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class BenchmarkRunRequest(BaseModel):
    dataset: str
    models_config: str = "configs/models.yaml"
    model_profile: str | None = None
    output_dir: str | None = None
    concurrency: int = Field(default=1, ge=1, le=8)
    resume: bool = False
