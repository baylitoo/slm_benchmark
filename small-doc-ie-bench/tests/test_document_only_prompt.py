"""A model that receives its schema out-of-band is sent the document alone."""

from docie_bench.llm.prompts import (
    SYSTEM_PROMPT,
    build_document_only_prompts,
    build_nuextract3_prompts,
    build_user_prompt,
)
from docie_bench.schemas.common import OCRBlock
from docie_bench.schemas.extraction import flat_schema_json
from docie_bench.serving.model_store import get_family

LINES = [
    "Facture # BEEZ-FACT-001925",
    "Date de facture : 31/12/2019",
    "Sous-total 8.820,00",
    "TVA (20%): 1.764,00",
    "Total €10.584,00",
]
BLOCKS = [OCRBlock(id=f"b{i}", text=text, page=1, source="manual") for i, text in enumerate(LINES)]


def test_the_prompt_is_the_document_and_nothing_else() -> None:
    system, user = build_document_only_prompts(BLOCKS)
    assert system == ""
    assert user == "\n".join(LINES)


def test_none_of_the_generic_scaffolding_survives() -> None:
    # Each of these is text the model can only mistake for the document.
    _, user = build_document_only_prompts(BLOCKS)
    for scaffolding in (
        "UNTRUSTED OCR EVIDENCE",
        "Document type:",
        "Metadata:",
        "Fields (null when absent",
        "Return the extraction JSON only",
    ):
        assert scaffolding not in user


def test_the_type_hint_the_model_echoed_is_gone() -> None:
    # A GLiFormer run returned issue_date as the literal "YYYY-MM-DD", which
    # appears nowhere in the invoice and only in the rendered field list.
    generic = build_user_prompt(
        schema_name="invoice",
        schema=flat_schema_json("invoice"),
        blocks=BLOCKS,
        language=None,
        metadata={},
    )
    _, user = build_document_only_prompts(BLOCKS)
    assert "YYYY-MM-DD" in generic
    assert "YYYY-MM-DD" not in user


def test_it_is_far_shorter_than_the_generic_prompt() -> None:
    generic = build_user_prompt(
        schema_name="invoice",
        schema=flat_schema_json("invoice"),
        blocks=BLOCKS,
        language=None,
        metadata={"source": "playground_stream", "filename": "x.pdf"},
    )
    _, user = build_document_only_prompts(BLOCKS)
    assert len(user) < len(generic) + len(SYSTEM_PROMPT)


def test_nuextract3_now_shares_the_same_builder() -> None:
    # It already sent the document alone; this only removes the duplicate.
    assert build_nuextract3_prompts(blocks=BLOCKS, has_images=False) == (
        build_document_only_prompts(BLOCKS)
    )


def test_nuextract3_still_sends_nothing_when_the_document_is_an_image() -> None:
    assert build_nuextract3_prompts(blocks=BLOCKS, has_images=True) == ("", "")


def test_the_gliformer_family_asks_for_it() -> None:
    assert get_family("encoder_gliformer").prompt_profile == "document_only"


def test_a_generative_family_still_gets_the_full_prompt() -> None:
    # The generic path is what a model that reads instructions needs; this
    # change must not reach it.
    assert get_family("lfm2").prompt_profile != "document_only"


def test_an_empty_document_produces_an_empty_prompt_rather_than_failing() -> None:
    assert build_document_only_prompts([]) == ("", "")
