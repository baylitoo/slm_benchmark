from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from dateutil import parser as date_parser

CURRENCY_SYMBOLS = {
    "€": "EUR",
    "$": "USD",
    "£": "GBP",
    "CHF": "CHF",
    "EUR": "EUR",
    "USD": "USD",
    "GBP": "GBP",
}


def normalize_date(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    try:
        dt = date_parser.parse(value, dayfirst=True, fuzzy=True)
    except (ValueError, OverflowError):
        return None
    return dt.date().isoformat()


def normalize_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    text = str(value).strip()
    text = re.sub(r"[^0-9,.-]", "", text)
    if not text:
        return None
    # European amount: 1 234,56 or 1.234,56
    if "," in text and text.rfind(",") > text.rfind("."):
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", "")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def normalize_currency(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.strip().upper()
    return CURRENCY_SYMBOLS.get(upper, CURRENCY_SYMBOLS.get(value.strip(), upper if len(upper) == 3 else None))
