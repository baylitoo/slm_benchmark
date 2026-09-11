from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ValidationError

from docie_bench.schemas.common import ExtractionValidation, OCRBlock
from docie_bench.schemas.extraction import get_schema_model

# How far an invoice's stated total may sit from the sum of its parts before it
# is reported. One constant: the agent surface's sum_check answers the same
# question on the same document, and two thresholds made the two disagree.
INVOICE_TOLERANCE = Decimal("0.05")


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


MAX_SALVAGE_PASSES = 20


def _describe(loc: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for segment in loc:
        if isinstance(segment, int):
            parts.append(f"[{segment}]")
        elif isinstance(segment, str) and not segment.endswith("]"):
            parts.append(f".{segment}" if parts else segment)
    return "".join(parts) or "<root>"


def _prune(payload: Any, loc: tuple[Any, ...]) -> bool:
    """Drop the deepest reachable value on an error path.

    A list index means the element itself is the wrong shape, so it is removed;
    nulling it would fail validation again on the very same location.
    """
    parent: Any = None
    key: Any = None
    node: Any = payload
    for segment in loc:
        if isinstance(node, dict):
            reachable = segment in node
        elif isinstance(node, list):
            reachable = isinstance(segment, int) and -len(node) <= segment < len(node)
        else:
            reachable = False
        if not reachable:
            break
        parent, key, node = node, segment, node[segment]
    if parent is None:
        return False
    if isinstance(parent, list):
        del parent[key]
        return True
    if parent[key] is None:
        return False
    parent[key] = None
    return True


def _salvage(
    model_cls: type[BaseModel], payload: dict[str, Any]
) -> tuple[BaseModel | None, dict[str, Any], list[str], list[str]]:
    """Validate, dropping one offending leaf per pass until the rest parses.

    A single unparsable scalar used to invalidate the whole document and return
    the raw payload, so a correct 40-field extraction was lost to one bad year.
    Every dropped value is reported as a warning.
    """
    payload = copy.deepcopy(payload)
    warnings: list[str] = []
    for _ in range(MAX_SALVAGE_PASSES):
        try:
            return model_cls.model_validate(payload), payload, warnings, []
        except ValidationError as exc:
            error = exc.errors()[0]
            if not _prune(payload, tuple(error.get("loc", ()))):
                return None, payload, warnings, [str(exc)]
            warnings.append(
                f"{_describe(tuple(error.get('loc', ())))}: "
                f"{error.get('msg', 'invalid value')}; dropped"
            )
    try:
        return model_cls.model_validate(payload), payload, warnings, []
    except ValidationError as exc:
        return None, payload, warnings, [str(exc)]


def validate_extraction(
    schema_name: str,
    payload: dict[str, Any],
    blocks: list[OCRBlock],
    model_cls: type[BaseModel] | None = None,
) -> tuple[dict[str, Any], ExtractionValidation]:
    model_cls = model_cls or get_schema_model(schema_name)
    parsed, payload, warnings, errors = _salvage(model_cls, payload)
    if parsed is None:
        return payload, ExtractionValidation(valid=False, errors=errors, warnings=warnings)

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
    tolerance = INVOICE_TOLERANCE
    subtotal = money_amount(invoice.get("subtotal"))
    vat = money_amount(invoice.get("vat_amount"))
    total = money_amount(invoice.get("total_ttc"))
    if (
        subtotal is not None
        and vat is not None
        and total is not None
        and abs((subtotal + vat) - total) > tolerance
    ):
        warnings.append(
            f"subtotal + vat_amount ({subtotal} + {vat} = {subtotal + vat}) does not match "
            f"total_ttc ({total}) within {tolerance}"
        )

    line_totals: list[Decimal] = []
    for index, item in enumerate(invoice.get("line_items", [])):
        if not isinstance(item, dict):
            continue
        quantity = number_value(item.get("quantity"))
        unit_price = money_amount(item.get("unit_price"))
        line_total = money_amount(item.get("line_total"))
        if line_total is not None:
            line_totals.append(line_total)
        if (
            quantity is not None
            and unit_price is not None
            and line_total is not None
            and abs((quantity * unit_price) - line_total) > tolerance
        ):
            warnings.append(
                f"line_items[{index}].quantity * unit_price "
                f"({quantity} * {unit_price} = {quantity * unit_price}) does not match "
                f"line_total ({line_total}) within {tolerance}"
            )
        currencies = {
            value.get("currency")
            for value in (item.get("unit_price"), item.get("line_total"))
            if isinstance(value, dict) and value.get("currency")
        }
        if len(currencies) > 1:
            warnings.append(f"line_items[{index}] contains inconsistent currencies")
    if subtotal is not None and line_totals and abs(sum(line_totals) - subtotal) > tolerance:
        warnings.append(
            f"sum(line_items.line_total) ({sum(line_totals)}) does not match "
            f"subtotal ({subtotal}) within {tolerance}"
        )
    return warnings


def money_amount(value: Any) -> Decimal | None:
    """A MoneyField's amount as a Decimal, or None when it is absent or unreadable.

    Decimal, not float: these values are compared against a tolerance of a few
    cents, and a float sum of amounts drifts below that.
    """
    if not isinstance(value, dict) or value.get("amount") is None:
        return None
    try:
        return Decimal(str(value["amount"]))
    except Exception:
        return None


def number_value(value: Any) -> Decimal | None:
    if not isinstance(value, dict) or value.get("value") is None:
        return None
    try:
        return Decimal(str(value["value"]))
    except Exception:
        return None
