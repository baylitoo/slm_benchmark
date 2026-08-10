"""fix_mojibake: repair model-emitted double-encoded UTF-8 (accented OCR)."""

from __future__ import annotations

from docie_bench.llm.mojibake import fix_completion_content, fix_mojibake


def test_fixes_french_mojibake() -> None:
    # The exact pattern a small vision model emits OCR'ing a French invoice.
    assert fix_mojibake("universitÃ© thÃ¨se de LambachÃ®ri") == "université thèse de Lambachîri"
    assert fix_mojibake("pÃ©nalitÃ© de retard") == "pénalité de retard"


def test_noop_on_clean_and_empty() -> None:
    # Already-correct text is untouched; None/empty pass through.
    assert fix_mojibake("université thèse") == "université thèse"
    assert fix_mojibake("plain ASCII invoice") == "plain ASCII invoice"
    assert fix_mojibake("") == ""
    assert fix_mojibake(None) is None


def test_idempotent() -> None:
    once = fix_mojibake("MontbÃ©liard")
    assert fix_mojibake(once) == once == "Montbéliard"


def test_fix_completion_content_string() -> None:
    completion = {
        "choices": [{"message": {"role": "assistant", "content": "FACTURE universitÃ©"}}]
    }
    fixed = fix_completion_content(completion)
    assert fixed["choices"][0]["message"]["content"] == "FACTURE université"


def test_fix_completion_content_multimodal_parts() -> None:
    completion = {
        "choices": [
            {"message": {"content": [{"type": "text", "text": "thÃ¨se"}, {"type": "text"}]}}
        ]
    }
    parts = fix_completion_content(completion)["choices"][0]["message"]["content"]
    assert parts[0]["text"] == "thèse"


def test_fix_completion_content_tolerates_odd_shapes() -> None:
    # Must never raise on an unexpected shape (passthrough responses).
    assert fix_completion_content({}) == {}
    assert fix_completion_content({"choices": [None, {"message": None}]}) is not None
    assert fix_completion_content("not a dict") == "not a dict"
