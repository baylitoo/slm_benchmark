from __future__ import annotations

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
    currency: TextField | None = Field(default=None, description="ISO-4217 currency code if explicit")
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
        raise ValueError(f"Unknown schema_name={schema_name!r}. Available: {sorted(SCHEMA_REGISTRY)}") from exc


def schema_json(schema_name: str) -> dict:
    return get_schema_model(schema_name).model_json_schema()
