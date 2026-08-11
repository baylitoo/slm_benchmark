"""flat_schema_json: a llama.cpp-GBNF-compatible, flattened extraction schema."""

from __future__ import annotations

import json

import pytest

from docie_bench.schemas.extraction import flat_schema_json


@pytest.mark.parametrize("name", ["invoice", "identity_card"])
def test_flat_schema_drops_grammar_breakers(name: str) -> None:
    blob = json.dumps(flat_schema_json(name))
    # $ref/$defs must be inlined; pattern (negative lookahead) + numeric bounds
    # are exactly what llama.cpp's schema->GBNF can't compile.
    for banned in ("$ref", "$defs", "pattern", "minimum", "maximum"):
        assert banned not in blob, f"{name} flat schema still has {banned}"


def test_invoice_wrappers_unwrapped_to_values() -> None:
    inv = flat_schema_json("invoice")
    blob = json.dumps(inv)
    # The {value, evidence_ids, confidence} wrapper is gone everywhere.
    assert "evidence_ids" not in blob
    assert "confidence" not in blob
    # Fields are preserved and flattened to plain nullable values.
    props = inv["properties"]
    assert "invoice_number" in props
    assert props["invoice_number"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    # A richer wrapper (MoneyField) keeps its data (amount/currency), drops meta.
    money = props["total_ttc"]["anyOf"][0]
    assert money["type"] == "object"
    assert set(money["properties"]) == {"amount", "currency"}


def test_flat_schema_forbids_extra_keys() -> None:
    # additionalProperties:false is KEPT so the compiled grammar can't emit the
    # extra/duplicate keys a shapeless json_object produced.
    assert flat_schema_json("invoice").get("additionalProperties") is False


def test_unknown_schema_raises() -> None:
    with pytest.raises(ValueError, match="Unknown schema_name"):
        flat_schema_json("not-a-schema")


def test_flat_schema_omits_meta_fields_so_data_lands_in_typed_fields() -> None:
    # document_type (a const) and extraction_notes (the free-text catch-all) are
    # dropped from the GRAMMAR so a small model can't dump the whole document into
    # notes; Pydantic fills both from defaults on parse, so the model still
    # validates. The real typed fields stay.
    props = flat_schema_json("invoice")["properties"]
    assert "extraction_notes" not in props
    assert "document_type" not in props
    assert "total_ttc" in props
    assert "invoice_number" in props
    required = flat_schema_json("invoice").get("required", [])
    assert "extraction_notes" not in required
    assert "document_type" not in required


def test_flat_schema_result_still_validates_the_basemodel() -> None:
    # A grammar-shaped object (no meta fields) must still build the BaseModel:
    # document_type defaults to the const, extraction_notes to [].
    from docie_bench.schemas.extraction import get_schema_model

    model = get_schema_model("invoice")
    obj = model.model_validate({"invoice_number": {"value": "F-1"}})
    assert obj.document_type == "invoice"
    assert obj.extraction_notes == []
