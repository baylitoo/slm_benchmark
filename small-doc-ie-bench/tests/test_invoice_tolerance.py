"""The agent surface and the validator answer the same question the same way."""

from decimal import Decimal
from typing import Any

import pytest

from docie_bench.agents.runtime import _invoice_sum_check
from docie_bench.extract.validators import (
    INVOICE_TOLERANCE,
    _validate_invoice_arithmetic,
    money_amount,
    number_value,
)


def _invoice(subtotal: str, line_totals: tuple[str, ...] = ("50.00", "50.00")) -> dict[str, Any]:
    return {
        "subtotal": {"amount": subtotal, "currency": "EUR"},
        "line_items": [
            {"line_total": {"amount": amount, "currency": "EUR"}} for amount in line_totals
        ],
    }


@pytest.mark.parametrize(
    ("subtotal", "within"),
    [
        ("100.00", True),
        ("100.03", True),
        ("100.05", True),
        ("100.06", False),
        ("99.90", False),
    ],
)
def test_the_two_invoice_checks_never_disagree(subtotal: str, within: bool) -> None:
    # Before: the agent's sum_check used the calculator's default 0.01 while the
    # validator used 0.05, so a subtotal three cents out was reported as a
    # mismatch by one and as fine by the other, on the same document.
    invoice = _invoice(subtotal)
    agent = _invoice_sum_check(invoice)
    assert agent is not None
    warnings = _validate_invoice_arithmetic(invoice)
    assert agent["matches"] is within
    assert (not warnings) is within


def test_the_threshold_is_named_once() -> None:
    assert Decimal("0.05") == INVOICE_TOLERANCE


def test_amounts_are_read_as_decimals_not_floats() -> None:
    # 0.1 + 0.2 is not 0.3 in binary floating point; these values are compared
    # against a few cents, so the reader must not introduce that drift.
    assert money_amount({"amount": "0.1"}) + money_amount({"amount": "0.2"}) == Decimal("0.3")
    assert isinstance(money_amount({"amount": "12.50"}), Decimal)
    assert isinstance(number_value({"value": "2.5"}), Decimal)


@pytest.mark.parametrize("value", [None, {}, {"amount": None}, {"amount": "n/a"}, "12.50"])
def test_an_unreadable_amount_is_none_rather_than_an_error(value: Any) -> None:
    assert money_amount(value) is None


def test_a_long_invoice_agrees_across_many_line_items() -> None:
    # Ten amounts whose float sum drifts from their decimal sum.
    invoice = _invoice("1.00", tuple(["0.10"] * 10))
    agent = _invoice_sum_check(invoice)
    assert agent is not None
    assert agent["matches"] is True
    assert _validate_invoice_arithmetic(invoice) == []


def test_an_invoice_with_no_line_items_is_left_alone() -> None:
    assert _invoice_sum_check({"subtotal": {"amount": "10.00"}}) is None
