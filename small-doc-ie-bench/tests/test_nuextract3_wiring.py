from __future__ import annotations

import json

from docie_bench.llm.openai_client import _clean_content
from docie_bench.llm.prompts import build_nuextract3_prompts, nuextract_template_for
from docie_bench.llm.response_format import build_response_format
from docie_bench.schemas.common import OCRBlock


def test_nuextract3_response_style_delivers_template_via_chat_template_kwargs() -> None:
    response_format, extra_body = build_response_format("nuextract3", "invoice", {"unused": True})
    assert response_format is None
    kwargs = extra_body["chat_template_kwargs"]
    assert kwargs["enable_thinking"] is False
    # The template is the NuExtract type-string template for the schema.
    assert json.loads(kwargs["template"]) == nuextract_template_for("invoice")


def test_nuextract3_think_style_enables_reasoning() -> None:
    _, extra_body = build_response_format("nuextract3_think", "identity_card", {})
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is True


def test_clean_content_strips_reasoning_block() -> None:
    answer = '{"store":"Trader Joe\'s","total":12.4}'
    raw = f"<think>The store is Trader Joe's and the total is 12.40.</think>{answer}"
    assert _clean_content(raw) == answer


def test_clean_content_without_think_is_unchanged() -> None:
    assert _clean_content('{"store":"X"}') == '{"store":"X"}'


def test_build_nuextract3_prompts_image_mode_is_empty() -> None:
    system, user = build_nuextract3_prompts(blocks=[], has_images=True)
    assert system == ""
    assert user == ""


def test_build_nuextract3_prompts_text_mode_carries_document_only() -> None:
    blocks = [
        OCRBlock(id="b1", text="Trader Joe's", page=1, source="manual"),
        OCRBlock(id="b2", text="Total $12.40", page=1, source="manual"),
    ]
    system, user = build_nuextract3_prompts(blocks=blocks, has_images=False)
    assert system == ""
    assert user == "Trader Joe's\nTotal $12.40"
