from docie_bench.extract.validators import validate_extraction
from docie_bench.schemas.common import OCRBlock


def test_invoice_schema_validation():
    blocks = [OCRBlock(id="b1", text="Facture INV-1", source="manual")]
    payload = {"document_type": "invoice", "invoice_number": {"value": "INV-1", "evidence_ids": ["b1"], "confidence": 0.9}}
    normalized, validation = validate_extraction("invoice", payload, blocks)
    assert validation.valid
    assert normalized["invoice_number"]["value"] == "INV-1"
