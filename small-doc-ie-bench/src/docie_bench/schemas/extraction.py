from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from docie_bench.schemas.common import DateField, MoneyField, NumberField, TextField


class InvoiceLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: TextField | None = None
    sku: TextField | None = None
    quantity: NumberField | None = None
    unit_price: MoneyField | None = None
    line_total: MoneyField | None = None
    tax_rate: NumberField | None = None


class InvoiceExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["invoice"] = "invoice"
    invoice_number: TextField | None = None
    vendor_name: TextField | None = None
    vendor_tax_id: TextField | None = None
    customer_name: TextField | None = None
    customer_tax_id: TextField | None = None
    issue_date: DateField | None = None
    due_date: DateField | None = None
    purchase_order_number: TextField | None = None
    subtotal: MoneyField | None = None
    vat_amount: MoneyField | None = None
    vat_rate: NumberField | None = None
    total_ttc: MoneyField | None = Field(default=None, description="Total amount including taxes")
    currency: TextField | None = Field(
        default=None, description="ISO-4217 currency code if explicit"
    )
    iban: TextField | None = None
    payment_terms: TextField | None = None
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    extraction_notes: list[str] = Field(default_factory=list)


class IdentityCardExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Literal["identity_card"] = "identity_card"
    country: TextField | None = None
    document_number: TextField | None = None
    surname: TextField | None = None
    given_names: TextField | None = None
    birth_date: DateField | None = None
    birth_place: TextField | None = None
    nationality: TextField | None = None
    sex: TextField | None = None
    issue_date: DateField | None = None
    expiry_date: DateField | None = None
    issuing_authority: TextField | None = None
    mrz_line_1: TextField | None = None
    mrz_line_2: TextField | None = None
    extraction_notes: list[str] = Field(default_factory=list)


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "invoice": InvoiceExtraction,
    "identity_card": IdentityCardExtraction,
}


def get_schema_model(schema_name: str) -> type[BaseModel]:
    try:
        return SCHEMA_REGISTRY[schema_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown schema_name={schema_name!r}. Available: {sorted(SCHEMA_REGISTRY)}"
        ) from exc


def schema_json(schema_name: str) -> dict:
    return get_schema_model(schema_name).model_json_schema()


# Keys stripped when flattening for a grammar target: presentation noise and the
# constraints llama.cpp's schema->GBNF can't express (notably `pattern`, which
# carries a negative lookahead on the money/number fields — non-context-free, so
# the grammar fails to compile). Validated downstream, never by the sampler.
_FLATTEN_DROP = frozenset(
    {
        "title",
        "description",
        "default",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "format",
    }
)
# A field wrapper is `{value, evidence_ids, confidence}` — detected by these
# metadata siblings of `value`.
_WRAPPER_MARKERS = frozenset({"evidence_ids", "confidence"})

# Meta fields OMITTED from the GRAMMAR (not the model): the const document_type
# and the free-text extraction_notes catch-all. Keeping them in the grammar lets
# a small model route the data into notes instead of the typed fields (observed:
# a whole invoice dumped into extraction_notes). Pydantic fills both from their
# defaults on parse (document_type is a const, extraction_notes defaults to []),
# so the BaseModel still validates — and the model is forced to populate the
# typed fields. Mirrors the NuExtract templates, which omit the same two.
_GRAMMAR_OMIT_FIELDS = frozenset({"document_type", "extraction_notes"})


def flat_schema_json(schema_name: str) -> dict:
    """A llama.cpp-GBNF-compatible, FLATTENED copy of the extraction schema.

    The raw schema wraps every field as ``{value, evidence_ids, confidence}`` and
    puts a negative-lookahead regex on the money/number fields. GBNF is
    context-free, so llama.cpp can't compile a grammar for the lookahead and the
    strict ``response_format`` is dropped to ``json_object`` — which enforces no
    shape, so a model emits extra + duplicate keys.

    This resolves ``$ref``s, UNWRAPS each field to its plain value, and strips the
    grammar-breaking constraints, while REQUIRING every top-level typed field and
    keeping ``additionalProperties: false`` — so the compiled grammar yields a
    clean flat ``{field: value | null}`` object with no missing, extra, or
    duplicate keys. Optional scalar fields remain nullable and optional list
    fields can be empty; requiring their keys prevents the grammar from accepting
    an empty ``{}`` as a successful extraction.
    """
    return flatten_schema_json(schema_json(schema_name))


def flatten_schema_json(root: dict) -> dict:
    """Flatten an extraction model's JSON schema for a grammar target.

    Unlike :func:`flat_schema_json`, this accepts an already-built schema. It is
    used for saved dynamic schemas whose Pydantic model is created at runtime.
    """
    defs = root.get("$defs", {})

    def resolve(node: object) -> object:
        seen = 0
        while isinstance(node, dict) and "$ref" in node and seen < 100:
            node = defs.get(str(node["$ref"]).rsplit("/", 1)[-1], {})
            seen += 1
        return node

    def transform(node: object) -> object:
        node = resolve(node)
        if isinstance(node, list):
            return [transform(item) for item in node]
        if not isinstance(node, dict):
            return node
        props = node.get("properties")
        if isinstance(props, dict) and (props.keys() & _WRAPPER_MARKERS):
            # A field wrapper. A simple one (`value`) unwraps to that value; a
            # richer one (MoneyField: amount/currency) keeps its DATA properties
            # and drops only the evidence/confidence metadata.
            if "value" in props:
                return transform(props["value"])
            data = {
                k: transform(v) for k, v in props.items() if k not in _WRAPPER_MARKERS
            }
            return {"type": "object", "additionalProperties": False, "properties": data}
        out: dict[str, object] = {}
        for key, value in node.items():
            if key in _FLATTEN_DROP or key in ("$defs", "$ref"):
                continue
            if key == "properties" and isinstance(value, dict):
                out[key] = {k: transform(v) for k, v in value.items()}
            elif key == "items":
                out[key] = transform(value)
            elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
                out[key] = _collapse_union([transform(item) for item in value])
            else:
                out[key] = value
        return out

    result = transform(root)
    if isinstance(result, dict):
        props = result.get("properties")
        if isinstance(props, dict):
            for meta in _GRAMMAR_OMIT_FIELDS:
                props.pop(meta, None)
            # Pydantic omits ``required`` for fields with defaults, even when the
            # field's value type is nullable. That makes the strict llama.cpp
            # grammar accept ``{}``, which a small extractor can choose as the
            # shortest valid answer. Require the typed keys while preserving
            # null/empty values as the representation for genuinely absent data.
            result["required"] = list(props)
    return result if isinstance(result, dict) else {}


def _collapse_union(items: list) -> list:
    """Flatten one level of nested ``anyOf`` and drop exact duplicates, so an
    unwrapped nullable field is ``[string, null]`` not ``[[string, null], null]``."""
    flat: list = []
    for item in items:
        if isinstance(item, dict) and set(item.keys()) == {"anyOf"}:
            flat.extend(item["anyOf"])
        else:
            flat.append(item)
    seen: set[str] = set()
    unique: list = []
    for item in flat:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
