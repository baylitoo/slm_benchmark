from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from docie_bench.schemas.common import ExtractionValidation, OCRBlock
from docie_bench.schemas.extraction import get_schema_model


def _collect_evidence_ids(obj: Any) -> list[str]:
    ids: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "evidence_ids" and isinstance(value, list):
                ids.extend(str(item) for item in value)
            else:
                ids.extend(_collect_evidence_ids(value))
    elif isinstance(obj, list):
        for item in obj:
            ids.extend(_collect_evidence_ids(item))
    return ids


def validate_extraction(schema_name: str, payload: dict[str, Any], blocks: list[OCRBlock]) -> tuple[dict[str, Any], ExtractionValidation]:
    model_cls = get_schema_model(schema_name)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        parsed = model_cls.model_validate(payload)
    except ValidationError as exc:
        return payload, ExtractionValidation(valid=False, errors=[str(exc)], warnings=[])

    block_ids = {block.id for block in blocks}
    for evidence_id in _collect_evidence_ids(parsed.model_dump(mode="json")):
        if evidence_id not in block_ids:
            warnings.append(f"Unknown evidence_id referenced by model: {evidence_id}")

    normalized = parsed.model_dump(mode="json")
    # Basic invoice consistency checks.
    if schema_name == "invoice":
        subtotal = _money_amount(normalized.get("subtotal"))
        vat = _money_amount(normalized.get("vat_amount"))
        total = _money_amount(normalized.get("total_ttc"))
        if subtotal is not None and vat is not None and total is not None:
            if abs((subtotal + vat) - total) > Decimal("0.05"):
                warnings.append("subtotal + vat_amount does not match total_ttc within 0.05")
    return normalized, ExtractionValidation(valid=not errors, errors=errors, warnings=warnings)


def _money_amount(value: Any) -> Decimal | None:
    if not isinstance(value, dict) or value.get("amount") is None:
        return None
    try:
        return Decimal(str(value["amount"]))
    except Exception:
        return None
