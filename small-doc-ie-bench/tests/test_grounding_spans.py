"""The candidate spans are built once per block list, not once per field."""

from typing import Any

from docie_bench.extract import grounding
from docie_bench.schemas.common import OCRBlock

LINES = [
    "Conception et developpement d'une plateforme SaaS de gestion de stocks.",
    "Mise en place de l'API REST et du frontend React.",
    "Deploiement Docker sur AWS et supervision Prometheus.",
    "Ada Lovelace",
    "Ingenieur logiciel",
]


def _blocks() -> list[OCRBlock]:
    return [
        OCRBlock(id=f"b{i}", text=text, page=1, source="manual") for i, text in enumerate(LINES)
    ]


def _count_normalisations(payload: dict[str, Any], blocks: list[OCRBlock], monkeypatch: Any) -> int:
    calls = 0
    original = grounding._normalize

    def counting(value: str) -> str:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(grounding, "_normalize", counting)
    grounding.ground_evidence(payload, blocks)
    return calls


def test_the_work_scales_with_the_page_not_with_the_field_count(monkeypatch: Any) -> None:
    blocks = _blocks()
    one = _count_normalisations({"a": {"value": "Ada Lovelace"}}, blocks, monkeypatch)
    many = _count_normalisations(
        {f"f{i}": {"value": "Ada Lovelace"} for i in range(20)}, blocks, monkeypatch
    )
    # Twenty fields over the same page cost twenty candidate normalisations
    # more, not twenty times the span index.
    assert many - one <= 20 + 5


def test_a_value_spanning_several_lines_still_grounds_to_its_window() -> None:
    value = " ".join(LINES[:3])
    grounded = grounding.ground_evidence({"description": {"value": value}}, _blocks())
    assert grounded["description"]["evidence_ids"] == ["b0", "b1", "b2"]
    assert grounded["description"]["confidence"] >= 0.9


def test_a_single_line_value_still_grounds_to_its_single_block() -> None:
    grounded = grounding.ground_evidence({"name": {"value": "Ada Lovelace"}}, _blocks())
    assert grounded["name"]["evidence_ids"] == ["b3"]


def test_an_absent_value_is_reported_ungrounded() -> None:
    grounded = grounding.ground_evidence({"name": {"value": "Grace Hopper"}}, _blocks())
    assert grounded["name"]["evidence_ids"] == []
    assert grounded["name"]["confidence"] == 0.0


def test_a_row_restricts_its_leaves_to_the_blocks_the_row_matched() -> None:
    # The row matches b3+b4; its leaves must ground inside that window, not
    # anywhere on the page.
    payload = {
        "people": [{"name": {"value": "Ada Lovelace"}, "role": {"value": "Ingenieur logiciel"}}]
    }
    grounded = grounding.ground_evidence(payload, _blocks())
    row = grounded["people"][0]
    assert row["name"]["evidence_ids"] == ["b3"]
    assert row["role"]["evidence_ids"] == ["b4"]


def test_restricting_keeps_only_the_named_blocks() -> None:
    evidence = grounding._Evidence.of(_blocks())
    restricted = evidence.restrict(["b1", "b3"])
    assert [block.id for block in restricted.blocks] == ["b1", "b3"]
    assert all(set(ids) <= {"b1", "b3"} for ids, _ in restricted.spans)
