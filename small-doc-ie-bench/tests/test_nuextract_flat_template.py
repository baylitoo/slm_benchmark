"""NuExtract templates use bare type strings, and plain values still get cleaned."""

from typing import Any

from docie_bench.extract.service import _COUNTRY_ISO, _normalize_nuextract_raw
from docie_bench.llm.prompts import nuextract_template_for
from docie_bench.schemas.dynamic import DynamicSchemaSpec, DynamicTemplateBuilder


def _wrapped_leaves(node: Any) -> list[Any]:
    if isinstance(node, list):
        return [leaf for item in node for leaf in _wrapped_leaves(item)]
    if not isinstance(node, dict):
        return []
    if set(node) == {"value"}:
        return [node]
    return [leaf for value in node.values() for leaf in _wrapped_leaves(value)]


def test_static_templates_send_bare_type_strings() -> None:
    for schema_name in ("invoice", "identity_card"):
        template = nuextract_template_for(schema_name)
        assert template
        assert _wrapped_leaves(template) == []
    invoice = nuextract_template_for("invoice")
    assert invoice["issue_date"] == "date"
    assert invoice["line_items"][0]["description"] == "verbatim-string"
    # Money is a genuine two-field object, not a wrapper.
    assert invoice["total_ttc"] == {"amount": "number", "currency": "currency"}


def test_a_saved_schema_sends_bare_type_strings_at_every_depth() -> None:
    spec = DynamicSchemaSpec.model_validate(
        {
            "document_type": "adbi_resume",
            "fields": [
                {"name": "name", "type": "string"},
                {"name": "salary", "type": "money"},
                {
                    "name": "skills",
                    "type": "list",
                    "fields": [
                        {"name": "category", "type": "string"},
                        {
                            "name": "items",
                            "type": "list",
                            "fields": [{"name": "item", "type": "string"}],
                        },
                    ],
                },
            ],
        }
    )
    template = DynamicTemplateBuilder.build_nuextract_template(spec)
    assert template == {
        "name": "verbatim-string",
        "salary": {"amount": "number", "currency": "currency"},
        "skills": [{"category": "verbatim-string", "items": [{"item": "verbatim-string"}]}],
    }


def test_field_clean_ups_apply_to_bare_values() -> None:
    country_name, country_iso = next(iter(_COUNTRY_ISO.items()))
    result, _ = _normalize_nuextract_raw(
        {
            "iban": "FR76 3000 6000",
            "country": country_name.title(),
            "document_number": "N° 12AB34",
        },
        "identity_card",
    )
    assert result["iban"] == "FR7630006000"
    assert result["country"] == country_iso
    assert result["document_number"] == "12AB34"


def test_field_clean_ups_still_apply_to_wrapped_values() -> None:
    result, _ = _normalize_nuextract_raw(
        {"iban": {"value": "FR76 3000"}, "document_number": {"value": "N° 9"}},
        "identity_card",
    )
    assert result["iban"] == {"value": "FR763000"}
    assert result["document_number"] == {"value": "9"}
