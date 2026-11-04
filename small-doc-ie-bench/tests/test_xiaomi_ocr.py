"""Xiaomi-OCR-0: the key-extraction prompt, and the served OCR backend.

One deployment answers two ways, so the two shapes are pinned separately: the
extraction path must send the trained key skeleton, and the OCR path must send
no grammar at all. Nothing here contacts a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from docie_bench.llm.prompts import build_xiaomi_kie_prompts, schema_key_skeleton
from docie_bench.ocr.factory import available_ocr_backends, get_ocr_backend
from docie_bench.ocr.xiaomi_backend import TRANSCRIBE_INSTRUCTION, XiaomiOCRBackend
from docie_bench.schemas.common import OCRBlock
from docie_bench.serving.model_store import FAMILIES

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "seller_name": {"type": "string"},
        "total_amount": {"type": "number"},
    },
}


# --- the key skeleton ------------------------------------------------------


def test_the_skeleton_names_fields_and_states_no_types() -> None:
    """The model was trained on empty values, not on a type declaration.

    Types are the grammar's job. Restating them in the prompt would be a
    second contract that can disagree with the first.
    """
    assert schema_key_skeleton(RECEIPT_SCHEMA) == {
        "date": "",
        "seller_name": "",
        "total_amount": "",
    }


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"type": "object", "properties": {"city": {"type": "string"}}}, {"city": ""}),
        ({"type": "array", "items": {"type": "string"}}, []),
        (
            {"type": "array", "items": {"type": "object", "properties": {"sku": {}}}},
            [{"sku": ""}],
        ),
    ],
)
def test_nested_shapes_keep_their_structure(spec: dict[str, Any], expected: Any) -> None:
    """A nested field collapsed to "" would ask for the wrong shape.

    The skeleton is the only place the nesting is communicated, so it has to
    survive -- a schema this cannot express flat still tells the model what to
    produce.
    """
    skeleton = schema_key_skeleton({"type": "object", "properties": {"field": spec}})

    assert skeleton == {"field": expected}


def test_a_schema_without_properties_yields_an_empty_skeleton() -> None:
    assert schema_key_skeleton({"type": "object"}) == {}
    assert schema_key_skeleton({}) == {}


# --- the extraction prompt -------------------------------------------------


def test_the_image_prompt_is_the_documented_phrasing_and_nothing_else() -> None:
    """A vision request carries the instruction and the skeleton only.

    No system prompt and no OCR text: the document IS the attached image, and
    the card's example is a single user turn.
    """
    system, user = build_xiaomi_kie_prompts(
        schema=RECEIPT_SCHEMA, blocks=[], has_images=True
    )

    assert system == ""
    assert user.startswith("Extract key information in the image")
    assert "Please output the key information in JSON format" in user
    assert json.dumps(schema_key_skeleton(RECEIPT_SCHEMA), indent=4) in user
    assert "UNTRUSTED" not in user


def test_without_images_the_text_arrives_fenced_as_untrusted() -> None:
    """The text path still fences OCR evidence, like every other text path."""
    blocks = [OCRBlock(id="b1_0_x", text="TOTAL 5.90", page=1, source="manual")]

    _system, user = build_xiaomi_kie_prompts(
        schema=RECEIPT_SCHEMA, blocks=blocks, has_images=False
    )

    assert "BEGIN UNTRUSTED OCR EVIDENCE" in user
    assert "END UNTRUSTED OCR EVIDENCE" in user
    assert "TOTAL 5.90" in user


# --- the family ------------------------------------------------------------


def test_the_family_sends_a_grammar_and_the_trained_wording() -> None:
    """Both halves, because either alone is the wrong contract.

    The grammar is what guarantees the answer parses and carries the requested
    keys; the prompt profile is what makes the values accurate. A family with
    the grammar but the generic vision prompt would be well-formed and worse.
    """
    family = FAMILIES["xiaomi_ocr"]

    assert family.response_format_style == "openai_json_schema"
    assert family.prompt_profile == "xiaomi_kie_v1"
    assert family.vision is True
    assert family.needs_mmproj is True
    assert "--jinja" in family.llama_server_args


def test_the_family_is_not_the_free_text_ocr_contract() -> None:
    """``vision_ocr`` exists for transcription and suppresses the grammar.

    Deploying this model under that family would silence the extraction mode
    entirely, which is the mistake the two contracts exist to keep apart.
    """
    assert FAMILIES["vision_ocr"].response_format_style == "none"
    assert FAMILIES["xiaomi_ocr"].response_format_style != "none"


# --- the served OCR backend ------------------------------------------------


def test_the_backend_is_selectable_and_declares_no_library() -> None:
    """Its availability cannot be an import check -- it is a model."""
    entry = next(row for row in available_ocr_backends() if row["name"] == "xiaomi")

    assert entry["aliases"] == []
    assert isinstance(entry["available"], bool)
    assert isinstance(get_ocr_backend("xiaomi"), XiaomiOCRBackend)


def test_an_undeployed_model_reads_as_unavailable_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Studio lists backends to populate a dropdown.

    A fleet that has never deployed this model must see the option hidden, not
    a failed listing.
    """
    from docie_bench.ocr import factory

    def _no_deployment(self: XiaomiOCRBackend) -> Any:
        raise RuntimeError("not deployed")

    monkeypatch.setattr(XiaomiOCRBackend, "_profile", _no_deployment)
    entry = next(row for row in factory.available_ocr_backends() if row["name"] == "xiaomi")

    assert entry["available"] is False


def test_a_resolution_failure_names_the_model_and_the_way_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docie_bench.serving import profile_resolver

    def _unresolvable(**_kwargs: Any) -> Any:
        raise profile_resolver.ProfileResolutionError("no live deployment")

    monkeypatch.setattr(profile_resolver, "resolve_extraction_profile", _unresolvable)
    backend = XiaomiOCRBackend(model="xiaomi-ocr-0")

    with pytest.raises(RuntimeError, match="DOCIE_OCR_XIAOMI_MODEL"):
        backend._profile()


def test_transcription_sends_no_grammar_and_keeps_page_numbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The OCR mode's two invariants, on one two-page document.

    No ``response_format``: a JSON grammar would corrupt a transcript, which is
    the whole reason this path does not reuse the family's contract. And real
    page numbers: a transcript that collapses onto page 1 cannot be grounded
    back to the document.
    """
    from docie_bench.vision import DocumentImage

    sent: list[dict[str, Any]] = []

    class _Response:
        def __init__(self, text: str) -> None:
            self._text = text

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": self._text}}]}

    class _Client:
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def post(self, _url: str, *, headers: dict[str, str], json: dict[str, Any]) -> _Response:
            sent.append(json)
            return _Response(f"line for page {len(sent)}")

    profile = type(
        "P",
        (),
        {
            "base_url": "http://node:8088/v1",
            "api_key": "k",
            "model": "xiaomi-ocr-0",
            "max_tokens": 4096,
            "timeout_seconds": 600.0,
            "vision_pdf_dpi": 150,
            "vision_max_pages": 8,
        },
    )()

    monkeypatch.setattr(
        "docie_bench.vision.load_document_images",
        lambda _path, pdf_dpi: [
            DocumentImage(page=1, media_type="image/png", data=b"a"),
            DocumentImage(page=2, media_type="image/png", data=b"b"),
        ],
    )
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: _Client())
    monkeypatch.setattr(XiaomiOCRBackend, "_profile", lambda self: profile)

    document = tmp_path / "receipt.png"
    document.write_bytes(b"not really a png")
    blocks = XiaomiOCRBackend().extract(document)

    assert [block.page for block in blocks] == [1, 2]
    assert all(block.source == "xiaomi" for block in blocks)
    assert len(sent) == 2
    for request in sent:
        assert "response_format" not in request
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
        text_parts = [
            part["text"] for part in request["messages"][0]["content"] if part["type"] == "text"
        ]
        assert text_parts == [TRANSCRIBE_INSTRUCTION]


def test_a_long_document_is_capped_at_the_profiles_page_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each page is a sequential request to a CPU-bound VLM.

    Uncapped, a long PDF is an unbounded transcription rather than a slow one.
    """
    from docie_bench.vision import DocumentImage

    calls: list[int] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "x"}}]}

    class _Client:
        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def post(self, _url: str, **_kwargs: Any) -> _Response:
            calls.append(1)
            return _Response()

    profile = type(
        "P",
        (),
        {
            "base_url": "http://node:8088/v1",
            "api_key": "k",
            "model": "xiaomi-ocr-0",
            "max_tokens": 4096,
            "timeout_seconds": 600.0,
            "vision_pdf_dpi": 150,
            "vision_max_pages": 3,
        },
    )()

    monkeypatch.setattr(
        "docie_bench.vision.load_document_images",
        lambda _path, pdf_dpi: [
            DocumentImage(page=page, media_type="image/png", data=b"x") for page in range(1, 21)
        ],
    )
    monkeypatch.setattr("httpx.Client", lambda **_kwargs: _Client())
    monkeypatch.setattr(XiaomiOCRBackend, "_profile", lambda self: profile)

    document = tmp_path / "long.pdf"
    document.write_bytes(b"%PDF-1.4")
    XiaomiOCRBackend().extract(document)

    assert len(calls) == 3
