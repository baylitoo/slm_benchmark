from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ValidationError

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


def validate_extraction(
    schema_name: str,
    payload: dict[str, Any],
    blocks: list[OCRBlock],
    model_cls: type[BaseModel] | None = None,
) -> tuple[dict[str, Any], ExtractionValidation]:
    model_cls = model_cls or get_schema_model(schema_name)
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
    if schema_name == "invoice":
        warnings.extend(_validate_invoice_arithmetic(normalized))
    return normalized, ExtractionValidation(valid=not errors, errors=errors, warnings=warnings)


def _validate_invoice_arithmetic(invoice: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    tolerance = Decimal("0.05")
    subtotal = _money_amount(invoice.get("subtotal"))
    vat = _money_amount(invoice.get("vat_amount"))
    total = _money_amount(invoice.get("total_ttc"))
    if (
        subtotal is not None
        and vat is not None
        and total is not None
        and abs((subtotal + vat) - total) > tolerance
    ):
        warnings.append("subtotal + vat_amount does not match total_ttc within 0.05")

    line_totals: list[Decimal] = []
    for index, item in enumerate(invoice.get("line_items", [])):
        if not isinstance(item, dict):
            continue
        quantity = _number_value(item.get("quantity"))
        unit_price = _money_amount(item.get("unit_price"))
        line_total = _money_amount(item.get("line_total"))
        if line_total is not None:
            line_totals.append(line_total)
        if (
            quantity is not None
            and unit_price is not None
            and line_total is not None
            and abs((quantity * unit_price) - line_total) > tolerance
        ):
            warnings.append(
                f"line_items[{index}].quantity * unit_price does not match "
                "line_total within 0.05"
            )
        currencies = {
            value.get("currency")
            for value in (item.get("unit_price"), item.get("line_total"))
            if isinstance(value, dict) and value.get("currency")
        }
        if len(currencies) > 1:
            warnings.append(f"line_items[{index}] contains inconsistent currencies")
    if subtotal is not None and line_totals and abs(sum(line_totals) - subtotal) > tolerance:
        warnings.append("sum(line_items.line_total) does not match subtotal within 0.05")
    return warnings


def _money_amount(value: Any) -> Decimal | None:
    if not isinstance(value, dict) or value.get("amount") is None:
        return None
    try:
        return Decimal(str(value["amount"]))
    except Exception:
        return None


def _number_value(value: Any) -> Decimal | None:
    if not isinstance(value, dict) or value.get("value") is None:
        return None
    try:
        return Decimal(str(value["value"]))
    except Exception:
        return None
