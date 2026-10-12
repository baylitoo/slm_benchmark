"""Text the model wrote that no typed leaf could hold is reported, not dropped."""

from typing import Any

import pytest

from docie_bench.extract.postprocess import (
    coerce_scalars,
    normalize_by_schema,
    report_lost_text,
)
from docie_bench.extract.validators import validate_extraction
from docie_bench.schemas.dynamic import DynamicSchemaSpec, DynamicTemplateBuilder
from docie_bench.schemas.extraction import rehydrate_extraction_result

SPEC = DynamicSchemaSpec.model_validate(
    {
        "document_type": "invoice_like",
        "fields": [
            {"name": "title", "type": "string"},
            {"name": "amount", "type": "money"},
            {"name": "count", "type": "number"},
            {
                "name": "rows",
                "type": "list",
                "fields": [
                    {"name": "label", "type": "string"},
                    {"name": "price", "type": "money"},
                ],
            },
        ],
    }
)
MODEL = DynamicTemplateBuilder.build_model(SPEC)
ROOT = MODEL.model_json_schema()


def _pipeline(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    result = normalize_by_schema(rehydrate_extraction_result(raw, ROOT), ROOT)
    result, coercion = coerce_scalars(result, ROOT)
    normalized, _validation = validate_extraction("invoice_like", result, [], model_cls=MODEL)
    return normalized, coercion + report_lost_text(result, normalized)


def test_a_money_field_answered_with_the_wrong_key_is_reported() -> None:
    # The failure this closes: validation builds the declared shape and drops
    # what does not fit WITHOUT raising, so the field arrived null and nothing
    # said the model had written anything.
    normalized, warnings = _pipeline({"amount": {"value": "5 ans"}})
    assert normalized["amount"]["amount"] is None
    assert any("amount" in warning and "5 ans" in warning for warning in warnings)


def test_the_path_names_the_row_and_the_field() -> None:
    _, warnings = _pipeline(
        {"rows": [{"label": {"value": "Conseil"}, "price": {"value": "12,00"}}]}
    )
    assert any(warning.startswith("rows[0].price:") for warning in warnings)


def test_a_well_formed_field_is_silent() -> None:
    _, warnings = _pipeline({"amount": {"amount": "1 234,56", "currency": "EUR"}})
    assert warnings == []


def test_an_absent_field_is_silent() -> None:
    _, warnings = _pipeline({"amount": None, "count": None})
    assert warnings == []


def test_an_unparseable_number_is_reported_once_not_twice() -> None:
    # coerce_scalars already names this one; the comparison must not repeat it.
    _, warnings = _pipeline({"count": {"value": "5 ans"}})
    assert len(warnings) == 1


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_text_is_not_worth_a_warning(blank: str) -> None:
    _, warnings = _pipeline({"amount": {"value": blank}})
    assert warnings == []


def test_a_field_named_amount_does_not_make_its_container_look_like_a_leaf() -> None:
    # The schema has a field called "amount"; testing for the mere PRESENCE of
    # that key read the whole document as one leaf and reported nothing at all.
    _, warnings = _pipeline({"amount": {"value": "5 ans"}, "rows": [{"price": {"value": "3,00"}}]})
    assert len(warnings) == 2
