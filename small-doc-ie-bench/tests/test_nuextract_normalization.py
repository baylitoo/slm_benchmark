from typing import Any

import pytest

from docie_bench.extract.service import ExtractionService, _normalize_nuextract_raw
from docie_bench.llm.model_profiles import ModelProfile


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

    result, derived = _normalize_nuextract_raw(raw, "invoice")

    assert result["subtotal"] == {"amount": expected_subtotal, "currency": "EUR"}
    assert derived is True


def test_preserves_subtotal_extracted_by_nuextract():
    raw = {
        "subtotal": {"amount": "4500.00", "currency": "EUR"},
        "vat_amount": {"amount": "900.00", "currency": "EUR"},
        "total_ttc": {"amount": "5400.00", "currency": "EUR"},
    }

    result, derived = _normalize_nuextract_raw(raw, "invoice")

    assert result["subtotal"] == {"amount": "4500.00", "currency": "EUR"}
    assert derived is False


def test_does_not_derive_subtotal_when_currencies_conflict():
    raw = {
        "subtotal": None,
        "vat_amount": {"amount": "20.00", "currency": "EUR"},
        "total_ttc": {"amount": "120.00", "currency": "GBP"},
    }

    result, derived = _normalize_nuextract_raw(raw, "invoice")

    assert result["subtotal"] is None
    assert derived is False


def test_does_not_derive_subtotal_for_other_schemas():
    raw = {
        "vat_amount": {"amount": "20.00", "currency": "EUR"},
        "total_ttc": {"amount": "120.00", "currency": "EUR"},
    }

    result, derived = _normalize_nuextract_raw(raw, "identity_card")

    assert "subtotal" not in result
    assert derived is False


def test_normalizes_nested_line_item_numbers_and_money():
    result, _derived = _normalize_nuextract_raw(
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


@pytest.mark.asyncio
async def test_extraction_response_flags_a_derived_subtotal(monkeypatch) -> None:
    # End-to-end through ExtractionService, not just the normalization helper:
    # a subtotal synthesized from total_ttc - vat_amount must be visible on the
    # actual response a caller (API, review queue, benchmark) receives, not
    # just provable by calling the private helper directly.
    class FakeClient:
        def __init__(self, profile: ModelProfile) -> None:
            self.profile = profile

        async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], None, dict[str, Any]]:
            return (
                {
                    "document_type": "invoice",
                    "subtotal": None,
                    "vat_amount": {"amount": "20.00", "currency": "EUR"},
                    "total_ttc": {"amount": "120.00", "currency": "EUR"},
                },
                None,
                {},
            )

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("docie_bench.extract.service.OpenAICompatibleClient", FakeClient)
    profile = ModelProfile(
        name="test-nuextract3",
        model="test-model",
        base_url="http://test",
        api_key="test",
        prompt_profile="nuextract3",
    )

    response = await ExtractionService(profile).extract_from_text(
        text="Some invoice text", ocr_blocks=None, schema_name="invoice"
    )

    assert response.result["subtotal"]["amount"] == "100.00"
    assert response.result["subtotal"]["currency"] == "EUR"
    assert any("subtotal.amount was derived" in w for w in response.validation.warnings)


@pytest.mark.asyncio
async def test_extraction_response_has_no_warning_when_subtotal_is_extracted(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, profile: ModelProfile) -> None:
            self.profile = profile

        async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], None, dict[str, Any]]:
            return (
                {
                    "document_type": "invoice",
                    "subtotal": {"amount": "100.00", "currency": "EUR"},
                    "vat_amount": {"amount": "20.00", "currency": "EUR"},
                    "total_ttc": {"amount": "120.00", "currency": "EUR"},
                },
                None,
                {},
            )

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("docie_bench.extract.service.OpenAICompatibleClient", FakeClient)
    profile = ModelProfile(
        name="test-nuextract3",
        model="test-model",
        base_url="http://test",
        api_key="test",
        prompt_profile="nuextract3",
    )

    response = await ExtractionService(profile).extract_from_text(
        text="Some invoice text", ocr_blocks=None, schema_name="invoice"
    )

    assert not any("derived" in w for w in response.validation.warnings)
