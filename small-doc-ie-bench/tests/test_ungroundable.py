"""A value with nothing to match on is ungrounded, not a crash."""

import pytest

from docie_bench.extract.grounding import _candidate_variants, ground_evidence
from docie_bench.schemas.common import OCRBlock

BLOCKS = [
    OCRBlock(id="b0", text="Facture # BEEZ-FACT-001925", page=1, source="manual"),
    OCRBlock(id="b1", text="Sous-total 8.820,00", page=1, source="manual"),
    OCRBlock(id="b2", text="TVA (20%): 1.764,00", page=1, source="manual"),
]

# Normalisation keeps letters and digits only, so each of these reduces to the
# empty string and leaves nothing to compare against a block.
SYMBOL_ONLY = ["€", "$", "—", "...", "-", "%", "•", "«»", "/"]


@pytest.mark.parametrize("value", SYMBOL_ONLY)
def test_a_symbol_only_value_is_reported_ungrounded(value: str) -> None:
    # A bare "€" in a currency field is what small models emit routinely; this
    # raised ValueError from max() over no variants and failed the whole
    # extraction, after the model had already answered.
    grounded = ground_evidence({"currency": {"value": value}}, BLOCKS)
    assert grounded["currency"]["evidence_ids"] == []
    assert grounded["currency"]["confidence"] == 0.0


@pytest.mark.parametrize("value", SYMBOL_ONLY)
def test_such_a_value_has_no_variants_to_match_on(value: str) -> None:
    assert _candidate_variants(value) == set()


def test_the_rest_of_the_document_still_grounds_around_it() -> None:
    # The failure took the whole extraction with it, including the fields that
    # would have matched perfectly well.
    grounded = ground_evidence(
        {
            "currency": {"value": "€"},
            "subtotal": {"amount": "8.820,00"},
            "invoice_number": {"value": "BEEZ-FACT-001925"},
        },
        BLOCKS,
    )
    assert grounded["currency"]["evidence_ids"] == []
    assert grounded["subtotal"]["evidence_ids"] == ["b1"]
    assert grounded["invoice_number"]["evidence_ids"] == ["b0"]


def test_a_symbol_only_value_inside_a_list_row_does_not_fail_the_row() -> None:
    grounded = ground_evidence(
        {"line_items": [{"unit_price": {"amount": "€"}, "line_total": {"amount": "1.764,00"}}]},
        BLOCKS,
    )
    row = grounded["line_items"][0]
    assert row["unit_price"]["evidence_ids"] == []
    assert row["line_total"]["evidence_ids"] == ["b2"]


def test_a_blank_value_was_never_in_the_crash_class() -> None:
    # An empty or whitespace-only value is not a grounding candidate at all, so
    # it is left unannotated rather than reaching the comparison. Only a
    # NON-empty value that normalises away ever raised.
    grounded = ground_evidence({"currency": {"value": "   "}}, BLOCKS)
    assert grounded["currency"] == {"value": "   "}


def test_an_empty_document_still_grounds_nothing_rather_than_failing() -> None:
    grounded = ground_evidence({"currency": {"value": "€"}}, [])
    assert grounded["currency"]["confidence"] == 0.0
