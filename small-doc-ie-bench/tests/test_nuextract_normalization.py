import pytest

from docie_bench.extract.service import _normalize_nuextract_raw


@pytest.mark.parametrize(
    ("vat_amount", "total_amount", "expected_subtotal"),
    [
        ("900,00", "5 400,00", "4500.00"),
        ("3.383,10", "19.493,10", "16110.00"),
        ("366,00", "4 026,00", "3660.00"),
    ],
)
def test_derives_missing_invoice_subtotal_from_total_and_vat(
    vat_amount, total_amount, expected_subtotal
):
    raw = {
        "subtotal": None,
        "vat_amount": {"amount": vat_amount, "currency": "EUR"},
        "total_ttc": {"amount": total_amount, "currency": "EUR"},
    }

    result = _normalize_nuextract_raw(raw, "invoice")

    assert result["subtotal"] == {"amount": expected_subtotal, "currency": "EUR"}


def test_preserves_subtotal_extracted_by_nuextract():
    raw = {
        "subtotal": {"amount": "4500.00", "currency": "EUR"},
        "vat_amount": {"amount": "900.00", "currency": "EUR"},
        "total_ttc": {"amount": "5400.00", "currency": "EUR"},
    }

    result = _normalize_nuextract_raw(raw, "invoice")

    assert result["subtotal"] == {"amount": "4500.00", "currency": "EUR"}


def test_does_not_derive_subtotal_when_currencies_conflict():
    raw = {
        "subtotal": None,
        "vat_amount": {"amount": "20.00", "currency": "EUR"},
        "total_ttc": {"amount": "120.00", "currency": "GBP"},
    }

    result = _normalize_nuextract_raw(raw, "invoice")

    assert result["subtotal"] is None


def test_does_not_derive_subtotal_for_other_schemas():
    raw = {
        "vat_amount": {"amount": "20.00", "currency": "EUR"},
        "total_ttc": {"amount": "120.00", "currency": "EUR"},
    }

    result = _normalize_nuextract_raw(raw, "identity_card")

    assert "subtotal" not in result


def test_normalizes_nested_line_item_numbers_and_money():
    result = _normalize_nuextract_raw(
        {
            "line_items": [
                {
                    "description": {"value": "Consulting"},
                    "quantity": {"value": "2,5"},
                    "unit_price": {"amount": "1 200,00", "currency": "€"},
                    "line_total": {"amount": "3.000,00", "currency": "EUR"},
                }
            ]
        },
        "invoice",
    )

    item = result["line_items"][0]
    assert item["quantity"]["value"] == "2.5"
    assert item["unit_price"] == {"amount": "1200.00", "currency": "EUR"}
    assert item["line_total"] == {"amount": "3000.00", "currency": "EUR"}
