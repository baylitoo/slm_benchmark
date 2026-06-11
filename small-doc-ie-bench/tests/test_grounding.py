from docie_bench.extract.grounding import ground_evidence
from docie_bench.schemas.common import OCRBlock


def test_ground_evidence_links_normalized_values_to_blocks():
    blocks = [
        OCRBlock(id="vendor", text="Fournisseur: ACME SAS", source="manual"),
        OCRBlock(id="date", text="Date: 21/05/2026", source="manual"),
        OCRBlock(id="total", text="Total TTC: 5 400,00 EUR", source="manual"),
    ]
    payload = {
        "vendor_name": {"value": "ACME SAS"},
        "issue_date": {"value": "2026-05-21"},
        "total_ttc": {"amount": "5400.00", "currency": "EUR"},
    }

    grounded = ground_evidence(payload, blocks)

    assert grounded["vendor_name"]["evidence_ids"] == ["vendor"]
    assert grounded["issue_date"]["evidence_ids"] == ["date"]
    assert grounded["total_ttc"]["evidence_ids"] == ["total"]
    assert grounded["vendor_name"]["confidence"] == 1.0


def test_ground_evidence_marks_unmatched_field_as_ungrounded():
    payload = {
        "vendor_name": {
            "value": "Invented Corp",
            "evidence_ids": ["unknown"],
            "confidence": 0.9,
        }
    }

    grounded = ground_evidence(
        payload,
        [OCRBlock(id="source", text="ACME SAS", source="manual")],
    )

    assert grounded["vendor_name"]["evidence_ids"] == []
    assert grounded["vendor_name"]["confidence"] == 0.0


def test_ground_evidence_links_value_split_across_adjacent_blocks():
    blocks = [
        OCRBlock(id="name-1", text="Vendor: NORTHFIELD", source="manual"),
        OCRBlock(id="name-2", text="CONSULTING LTD", source="manual"),
    ]

    grounded = ground_evidence(
        {"vendor_name": {"value": "NORTHFIELD CONSULTING LTD"}},
        blocks,
    )

    assert grounded["vendor_name"]["evidence_ids"] == ["name-1", "name-2"]
