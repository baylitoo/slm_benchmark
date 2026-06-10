from __future__ import annotations

import json

from docie_bench.schemas.common import OCRBlock


SYSTEM_PROMPT = """You are a deterministic document information extraction engine.
Extract only fields that are explicitly present in the OCR evidence.
Return only JSON matching the provided schema.
Use null for missing fields.
Every non-null field with evidence_ids must reference OCR block ids from the input.
Never invent values, never infer from world knowledge, and never output markdown.
Normalize dates to YYYY-MM-DD when the OCR evidence is unambiguous.
Normalize currency to ISO-4217 when explicit or strongly indicated by a symbol in the evidence.
"""

SCHEMA_PROPOSER_SYSTEM_PROMPT = """You design compact schemas for document information extraction.
Return only JSON matching the provided schema.
Include only useful fields explicitly supported by the document.
Use stable lower_snake_case names and one of: string, date, number, money.
Do not include document_type or extraction_notes as fields.
"""

VISION_SYSTEM_PROMPT = """You are a deterministic document information extraction engine.
Extract only fields that are explicitly visible in the supplied document images.
Return only JSON matching the provided schema.
Use null for missing fields and use an empty evidence_ids list for extracted fields.
Never invent values, never infer from world knowledge, and never output markdown.
Normalize dates to YYYY-MM-DD when the document is unambiguous.
Normalize currency to ISO-4217 when explicit or strongly indicated by a symbol.
"""

# Per-schema JSON templates for NuExtract3.
# Leaf values use NuExtract's semantic type system:
#   "verbatim-string" → extract text exactly as it appears
#   "date"            → output ISO-8601 date (YYYY-MM-DD)
#   "number"          → output clean decimal (no symbols, no locale separators)
#   "currency"        → output ISO-4217 code (EUR, GBP, USD …)
#   ["A", "B", ...]   → enum, model picks one value
# document_type and extraction_notes are omitted — Pydantic fills them from defaults.
# evidence_ids and confidence are omitted — Pydantic defaults them to [] and 0.0.
_NUEXTRACT_TEMPLATES: dict[str, dict] = {
    "invoice": {
        "invoice_number": {"value": "verbatim-string"},
        "vendor_name": {"value": "verbatim-string"},
        "vendor_tax_id": {"value": "verbatim-string"},
        "customer_name": {"value": "verbatim-string"},
        "customer_tax_id": {"value": "verbatim-string"},
        "issue_date": {"value": "date"},
        "due_date": {"value": "date"},
        "purchase_order_number": {"value": "verbatim-string"},
        "subtotal": {"amount": "number", "currency": "currency"},
        "vat_amount": {"amount": "number", "currency": "currency"},
        "vat_rate": {"value": "number"},
        "total_ttc": {"amount": "number", "currency": "currency"},
        "iban": {"value": "verbatim-string"},
        "payment_terms": {"value": "string"},
    },
    "identity_card": {
        "country": {"value": "country"},
        "document_number": {"value": "verbatim-string"},
        "surname": {"value": "verbatim-string"},
        "given_names": {"value": "verbatim-string"},
        "birth_date": {"value": "date"},
        "birth_place": {"value": "verbatim-string"},
        "nationality": {"value": "verbatim-string"},
        "sex": {"value": "verbatim-string"},
        "issue_date": {"value": "date"},
        "expiry_date": {"value": "date"},
        "issuing_authority": {"value": "verbatim-string"},
        "mrz_line_1": {"value": "verbatim-string"},
        "mrz_line_2": {"value": "verbatim-string"},
    },
}


def render_ocr_blocks(blocks: list[OCRBlock], max_blocks: int = 800) -> str:
    compact = []
    for block in blocks[:max_blocks]:
        compact.append(
            {
                "id": block.id,
                "page": block.page,
                "text": block.text,
                "bbox": block.bbox.model_dump() if block.bbox else None,
                "ocr_confidence": block.confidence,
            }
        )
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def build_user_prompt(
    *,
    schema_name: str,
    schema: dict,
    blocks: list[OCRBlock],
    language: str | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    metadata = metadata or {}
    return (
        f"Task: extract structured fields for schema_name={schema_name!r}.\n"
        f"Language hint: {language or 'unknown'}.\n"
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
        "JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
        "OCR evidence blocks as JSON array:\n"
        f"{render_ocr_blocks(blocks)}\n"
        "Return the extraction JSON only."
    )


def build_vision_user_prompt(
    *,
    schema_name: str,
    schema: dict,
    page_count: int,
    language: str | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    metadata = metadata or {}
    return (
        f"Task: extract structured fields for schema_name={schema_name!r} from the attached "
        f"{page_count} document page image(s).\n"
        f"Language hint: {language or 'unknown'}.\n"
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
        "JSON Schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
        "Return the extraction JSON only."
    )


def build_nuextract_prompts(
    *,
    schema_name: str,
    blocks: list[OCRBlock],
    language: str | None = None,
    template: dict | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) in NuExtract3 format.

    NuExtract3 uses a special input format: a JSON template with empty values
    followed by the document text. It doesn't use a system prompt.
    """
    template = template if template is not None else _NUEXTRACT_TEMPLATES.get(schema_name, {})
    template_json = json.dumps(template, ensure_ascii=False, indent=2)
    document_text = "\n".join(b.text for b in blocks)
    user_prompt = (
        "<|input|>\n"
        "### Template:\n"
        f"{template_json}\n\n"
        "### Document:\n"
        f"{document_text}\n"
        "<|output|>"
    )
    return "", user_prompt


def build_schema_proposer_prompt(*, blocks: list[OCRBlock], language: str | None = None) -> str:
    return (
        f"Language hint: {language or 'unknown'}.\n"
        "Propose a reusable extraction schema for documents of this type.\n"
        "OCR evidence blocks as JSON array:\n"
        f"{render_ocr_blocks(blocks)}\n"
        "Return the schema specification JSON only."
    )
