from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from docie_bench.schemas.common import OCRBlock


class DatasetItem(BaseModel):
    doc_id: str
    file_path: str
    schema_name: str = "invoice"
    schema_mode: Literal["static", "dynamic"] = "static"
    dynamic_schema: dict[str, Any] | None = None
    language: str | None = None
    ocr_reference_text: str | None = None
    ocr_reference_path: str | None = None
    ocr_reference_blocks: list[OCRBlock] | None = None
    split: str = "unspecified"
    ground_truth: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)


def load_dataset(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    base = path.parent
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            item = DatasetItem.model_validate(raw)
        except Exception as exc:
            raise ValueError(f"Invalid dataset row {line_no} in {path}: {exc}") from exc
        file_path = Path(item.file_path)
        if not file_path.is_absolute():
            item.file_path = str((base / file_path).resolve())
        if item.ocr_reference_path:
            reference_path = Path(item.ocr_reference_path)
            if not reference_path.is_absolute():
                item.ocr_reference_path = str((base / reference_path).resolve())
        items.append(item)
    return items
