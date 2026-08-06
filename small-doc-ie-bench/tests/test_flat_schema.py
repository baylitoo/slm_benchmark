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
