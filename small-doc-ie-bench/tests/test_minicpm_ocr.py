"""MiniCPM-OCR-1.0 and MiniCPM-V 4.7.

Three things here are specific to these models rather than shared with the
other served vision backends: the inline-field extraction prompt, the ``###``
sentinel it declares for absent fields, and the bracketed reasoning trace the
4.7 GGUF emits instead of ``<think>`` tags. Nothing contacts a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from docie_bench.extract.postprocess import is_placeholder, normalize_placeholders
from docie_bench.llm.openai_client import _clean_content
from docie_bench.llm.prompts import build_minicpm_kie_prompts
from docie_bench.ocr.factory import available_ocr_backends, get_ocr_backend
from docie_bench.ocr.minicpm_backend import (
    MARKUP_MODES,
    MODE_INSTRUCTIONS,
    MiniCPMOCRBackend,
)
from docie_bench.schemas.common import OCRBlock
from docie_bench.serving.model_store import FAMILIES

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total_amount": {"type": "number"},
    },
}


# --- the extraction prompt -------------------------------------------------


def test_the_prompt_names_the_fields_inline_and_declares_the_sentinel() -> None:
    """The model card's own phrasing: fields twice, then the absent marker.

    Not a key skeleton — that is Xiaomi-OCR-0's convention, and sending one
    here would be prompting this model in a shape it was not trained on.
    """
    _system, user = build_minicpm_kie_prompts(
        schema=RECEIPT_SCHEMA, blocks=[], has_images=True
    )

    assert user == (
        'Extract the "invoice_number", "total_amount". '
        'Format the output as a JSON object with keys: "invoice_number", "total_amount". '
        'If an item is not in the image, use "###".'
    )


def test_there_is_no_system_prompt() -> None:
    system, _user = build_minicpm_kie_prompts(
        schema=RECEIPT_SCHEMA, blocks=[], has_images=True
    )

    assert system == ""


def test_without_images_the_text_arrives_fenced_as_untrusted() -> None:
    blocks = [OCRBlock(id="b1_0_x", text="INVOICE 42", page=1, source="manual")]

    _system, user = build_minicpm_kie_prompts(
        schema=RECEIPT_SCHEMA, blocks=blocks, has_images=False
    )

    assert "BEGIN UNTRUSTED OCR EVIDENCE" in user
    assert "INVOICE 42" in user


def test_a_schema_with_no_properties_still_produces_a_usable_instruction() -> None:
    _system, user = build_minicpm_kie_prompts(schema={}, blocks=[], has_images=True)

    assert user.startswith("Extract the .")


# --- the "###" sentinel ----------------------------------------------------


def test_the_absent_marker_normalises_to_none() -> None:
    """Otherwise every field the model could not see is stored as "###".

    A grammar-constrained string field has no way to be absent, so the model
    does exactly what it was told and the sentinel reaches the record.
    """
    assert is_placeholder("###") is True
    assert normalize_placeholders({"invoice_number": "42", "total_amount": "###"}) == {
        "invoice_number": "42",
        "total_amount": None,
    }


def test_the_marker_is_normalised_inside_nested_shapes_too() -> None:
    payload = {"lines": [{"sku": "###", "qty": "2"}], "vendor": {"name": "###"}}

    assert normalize_placeholders(payload) == {
        "lines": [{"sku": None, "qty": "2"}],
        "vendor": {"name": None},
    }


def test_a_value_that_merely_contains_hashes_is_untouched() -> None:
    """Only the exact sentinel. A reference like "###42" is real content."""
    assert normalize_placeholders({"ref": "###42"}) == {"ref": "###42"}


# --- the bracketed reasoning trace -----------------------------------------


def test_a_bracketed_thinking_trace_is_stripped() -> None:
    """MiniCPM-V 4.7's GGUF delimits reasoning with brackets, not tags.

    The tag-only strip would have left the trace in front of the JSON, so the
    parse fails on a model that answered correctly.
    """
    raw = '[Start thinking] the total looks like 5.90 [End thinking]{"total": "5.90"}'

    assert _clean_content(raw) == '{"total": "5.90"}'


def test_an_unclosed_bracketed_trace_fails_cleanly_instead_of_guessing() -> None:
    """Cut off mid-reasoning there is no answer yet.

    Falling through to bracket-balance extraction would find the braces in the
    reasoning prose and return syntactically valid, semantically wrong JSON —
    the same silent corruption the tag path documents.
    """
    raw = '[Start thinking] planning {"total": "string"} and more'

    assert _clean_content(raw) == raw


def test_the_tag_form_still_works() -> None:
    assert _clean_content('<think>musing</think>{"a": 1}') == '{"a": 1}'


# --- modes ------------------------------------------------------------------


def test_every_documented_task_has_its_verbatim_prompt() -> None:
    """The trailing colon is part of the convention, not formatting."""
    assert MODE_INSTRUCTIONS == {
        "text": "Text Recognition:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "seal": "Seal Recognition:",
        "chart": "Chart to Table:",
    }


def test_an_unknown_mode_is_refused_where_the_typo_is_readable() -> None:
    with pytest.raises(ValueError, match="unknown MiniCPM OCR mode"):
        MiniCPMOCRBackend(mode="tabel").instruction()


def test_the_mode_is_part_of_the_cache_key() -> None:
    """Two modes against one deployment produce different output."""
    text = MiniCPMOCRBackend(mode="text", model="m").version()
    table = MiniCPMOCRBackend(mode="table", model="m").version()

    assert text != table


@pytest.mark.parametrize("mode", sorted(MARKUP_MODES))
def test_a_markup_mode_returns_one_block_per_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    """An HTML table split on newlines is fragments of nothing.

    Line-per-block is right for a transcript and destroys markup, so the
    structured modes keep their answer whole.
    """
    blocks = _transcribe_with(monkeypatch, tmp_path, mode=mode, answer="<table>\n<tr>\n</table>")

    assert len(blocks) == 1
    assert blocks[0].text == "<table>\n<tr>\n</table>"
    assert blocks[0].source == "minicpm"


def test_the_text_mode_still_splits_into_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocks = _transcribe_with(monkeypatch, tmp_path, mode="text", answer="one\ntwo")

    assert [block.text for block in blocks] == ["one", "two"]


def _transcribe_with(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, mode: str, answer: str
) -> list[OCRBlock]:
    from docie_bench.ocr.served_vlm import ServedVLMBackend
    from docie_bench.vision import DocumentImage

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": answer}}]}

    class _Client:
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def post(self, _url: str, **_kwargs: Any) -> _Response:
            return _Response()

    profile = type(
        "P",
        (),
        {
            "base_url": "http://node:8088/v1",
            "api_key": "k",
            "model": "minicpm-ocr-1.0",
            "max_tokens": 4096,
            "timeout_seconds": 600.0,
            "vision_pdf_dpi": 150,
            "vision_max_pages": 8,
        },
    )()

    monkeypatch.setattr(
        "docie_bench.vision.load_document_images",
        lambda _path, pdf_dpi: [DocumentImage(page=1, media_type="image/png", data=b"x")],
    )
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: _Client())
    monkeypatch.setattr(ServedVLMBackend, "_profile", lambda self: profile)

    document = tmp_path / "page.png"
    document.write_bytes(b"not really a png")
    return MiniCPMOCRBackend(mode=mode).extract(document)


# --- families ---------------------------------------------------------------


def test_the_ocr_family_sends_a_grammar_and_the_trained_wording() -> None:
    family = FAMILIES["minicpm_ocr"]

    assert family.response_format_style == "openai_json_schema"
    assert family.prompt_profile == "minicpm_kie_v1"
    assert family.needs_mmproj is True
    assert family.vision is True


def test_the_general_vision_family_does_not_invent_a_convention() -> None:
    """MiniCPM-V 4.7 documents no extraction prompt of its own.

    Giving it a bespoke profile would be a guess dressed as a contract, so it
    takes the ordinary vision-extraction one.
    """
    assert FAMILIES["minicpm_v"].prompt_profile == "strict_extraction_v1"


@pytest.mark.parametrize("family", ["minicpm_ocr", "minicpm_v"])
def test_reasoning_is_turned_off_at_the_server_too(family: str) -> None:
    """Belt and braces with the enable_thinking the extraction path sends.

    A template that ignores the request-level flag is still covered by the
    server-side setting, and the reverse.
    """
    args = FAMILIES[family].llama_server_args

    assert "--reasoning" in args
    assert args[args.index("--reasoning") + 1] == "off"


# --- the backend is selectable ---------------------------------------------


def test_the_backend_is_listed_and_declares_no_library() -> None:
    entry = next(row for row in available_ocr_backends() if row["name"] == "minicpm")

    assert entry["aliases"] == []
    assert isinstance(entry["available"], bool)
    assert isinstance(get_ocr_backend("minicpm"), MiniCPMOCRBackend)


def test_an_undeployed_model_reads_as_unavailable_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docie_bench.ocr import factory

    def _no_deployment(self: MiniCPMOCRBackend) -> Any:
        raise RuntimeError("not deployed")

    monkeypatch.setattr(MiniCPMOCRBackend, "_profile", _no_deployment)
    entry = next(row for row in factory.available_ocr_backends() if row["name"] == "minicpm")

    assert entry["available"] is False
