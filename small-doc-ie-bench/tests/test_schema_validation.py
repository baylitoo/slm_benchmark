from docie_bench.extract.validators import validate_extraction
from docie_bench.schemas.common import OCRBlock


def test_invoice_schema_validation():
    blocks = [OCRBlock(id="b1", text="Facture INV-1", source="manual")]
    payload = {
        "document_type": "invoice",
        "invoice_number": {"value": "INV-1", "evidence_ids": ["b1"], "confidence": 0.9},
    }
    normalized, validation = validate_extraction("invoice", payload, blocks)
    assert validation.valid
    assert normalized["invoice_number"]["value"] == "INV-1"


def test_invoice_line_items_validate_arithmetic_and_evidence_ids():
    blocks = [OCRBlock(id="row-1", text="Consulting 2 x 100.00 = 200.00", source="manual")]
    payload = {
        "document_type": "invoice",
        "subtotal": {"amount": "210.00", "currency": "EUR"},
        "line_items": [
            {
                "description": {"value": "Consulting", "evidence_ids": ["row-1"]},
                "quantity": {"value": "2", "evidence_ids": ["row-1"]},
                "unit_price": {"amount": "100.00", "currency": "EUR", "evidence_ids": ["row-1"]},
                "line_total": {"amount": "190.00", "currency": "EUR", "evidence_ids": ["missing"]},
            }
        ],
    }

    normalized, validation = validate_extraction("invoice", payload, blocks)

    assert validation.valid
    assert normalized["line_items"][0]["quantity"]["value"] == "2"
    assert "Unknown evidence_id referenced by model: missing" in validation.warnings
    # The computed value rides in the warning text itself — a reviewer sees the
    # actual mismatch (2 * 100.00 = 200.00 vs the model's 190.00) without doing
    # the arithmetic by hand.
    assert any(
        "quantity * unit_price (2 * 100.00 = 200.00) does not match line_total (190.00)" in w
        for w in validation.warnings
    )
    assert any(
        "sum(line_items.line_total) (190.00) does not match subtotal (210.00)" in w
        for w in validation.warnings
    )


def test_invoice_total_mismatch_warning_shows_computed_sum():
    # The documented NuExtract3 failure mode: text/quantities read correctly,
    # but the model's own addition (subtotal + vat) is wrong. The warning must
    # surface the correct computed total so a reviewer doesn't have to add it
    # up themselves.
    blocks = [OCRBlock(id="b1", text="Total", source="manual")]
    payload = {
        "document_type": "invoice",
        "subtotal": {"amount": "100.00", "currency": "EUR"},
        "vat_amount": {"amount": "20.00", "currency": "EUR"},
        "total_ttc": {"amount": "115.00", "currency": "EUR"},
    }
    _normalized, validation = validate_extraction("invoice", payload, blocks)
    assert validation.valid
    assert any(
        "subtotal + vat_amount (100.00 + 20.00 = 120.00) does not match total_ttc (115.00)" in w
        for w in validation.warnings
    )


def test_optional_fields_with_null_value_are_valid():
    # Template VLMs (NuExtract3) emit {"value": null} for an absent optional field
    # rather than omitting it; that must validate, not fail with a string_type error.
    blocks = [OCRBlock(id="b1", text="Invoice", source="manual")]
    payload = {
        "document_type": "invoice",
        "vendor_name": {"value": "Acme"},
        "due_date": {"value": None},
        "payment_terms": {"value": None},
        "line_items": [{"sku": {"value": None}, "quantity": {"value": None}}],
    }
    _normalized, validation = validate_extraction("invoice", payload, blocks)
    assert validation.valid, validation.errors
