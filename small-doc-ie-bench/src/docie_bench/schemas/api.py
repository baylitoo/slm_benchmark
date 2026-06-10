from __future__ import annotations

from pydantic import BaseModel, Field

from docie_bench.schemas.common import OCRBlock


class ExtractTextRequest(BaseModel):
    text: str | None = None
    ocr_blocks: list[OCRBlock] | None = None
    schema_name: str = "invoice"
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
