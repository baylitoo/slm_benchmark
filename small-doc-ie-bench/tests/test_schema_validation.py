from docie_bench.extract.validators import validate_extraction
from docie_bench.schemas.common import OCRBlock


def test_invoice_schema_validation():
    blocks = [OCRBlock(id="b1", text="Facture INV-1", source="manual")]
    payload = {"document_type": "invoice", "invoice_number": {"value": "INV-1", "evidence_ids": ["b1"], "confidence": 0.9}}
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
    assert any("quantity * unit_price" in warning for warning in validation.warnings)
    assert any("sum(line_items.line_total)" in warning for warning in validation.warnings)
