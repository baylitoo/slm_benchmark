"""The flat record form: what the model is sent, and how the answer reads back."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from docie_bench.encoders.server import create_encoder_app
from docie_bench.llm.response_format import build_response_format
from docie_bench.schemas.dynamic import (
    DynamicSchemaSpec,
    DynamicTemplateBuilder,
    document_from_gliformer_records,
    gliformer_records_plan,
)
from docie_bench.schemas.extraction import flat_schema_json, flatten_schema_json


def _resume_flat() -> dict[str, Any]:
    spec = DynamicSchemaSpec.model_validate(
        {
            "document_type": "adbi_resume",
            "fields": [
                {"name": "full_name", "type": "string"},
                {
                    "name": "contact",
                    "type": "object",
                    "fields": [
                        {"name": "email", "type": "string"},
                        {"name": "phone", "type": "string"},
                    ],
                },
                {
                    "name": "skills",
                    "type": "list",
                    "fields": [
                        {"name": "category", "type": "string"},
                        {
                            "name": "items",
                            "type": "list",
                            "fields": [{"name": "skill", "type": "string"}],
                        },
                    ],
                },
                {
                    "name": "experience",
                    "type": "list",
                    "fields": [
                        {"name": "title", "type": "string"},
                        {
                            "name": "location",
                            "type": "object",
                            "fields": [{"name": "city", "type": "string"}],
                        },
                    ],
                },
            ],
        }
    )
    return flatten_schema_json(DynamicTemplateBuilder.build_model(spec).model_json_schema())


def test_a_list_inside_a_list_contributes_fields_rather_than_a_record() -> None:
    # The rule that makes the answer reassemble: every leaf joins the outermost
    # list above it. Nothing has to guess which "items" row belongs to which
    # skill, because the model returns both in one record.
    plan = gliformer_records_plan(_resume_flat(), name="adbi_resume")
    assert plan.fields == {
        "adbi_resume": ["full_name", "email", "phone"],
        "skills": ["category", "skill"],
        "experience": ["title", "city"],
    }


def test_the_answer_reads_back_into_the_shape_the_schema_describes() -> None:
    plan = gliformer_records_plan(_resume_flat(), name="adbi_resume")
    document = document_from_gliformer_records(
        {
            "adbi_resume": [{"full_name": "Jane Doe", "email": "j@x.fr", "phone": "0600"}],
            "skills": [
                {"category": "LLM", "skill": "LangGraph"},
                {"category": "LLM", "skill": "vLLM"},
            ],
            "experience": [{"title": "Dev", "city": "Paris"}],
        },
        plan,
        name="adbi_resume",
    )
    assert document == {
        "full_name": "Jane Doe",
        "contact": {"email": "j@x.fr", "phone": "0600"},
        # One parent row per item: this is the cost of the flat form, and it is
        # what the model returned rather than a regrouping of it.
        "skills": [
            {"category": "LLM", "items": [{"skill": "LangGraph"}]},
            {"category": "LLM", "items": [{"skill": "vLLM"}]},
        ],
        "experience": [{"title": "Dev", "location": {"city": "Paris"}}],
    }


def test_the_reassembled_document_validates_against_its_own_schema() -> None:
    from docie_bench.extract.validators import validate_extraction
    from docie_bench.schemas.extraction import rehydrate_extraction_result

    spec = DynamicSchemaSpec.model_validate(
        {
            "document_type": "adbi_resume",
            "fields": [
                {"name": "full_name", "type": "string"},
                {
                    "name": "skills",
                    "type": "list",
                    "fields": [
                        {"name": "category", "type": "string"},
                        {
                            "name": "items",
                            "type": "list",
                            "fields": [{"name": "skill", "type": "string"}],
                        },
                    ],
                },
            ],
        }
    )
    model = DynamicTemplateBuilder.build_model(spec)
    plan = gliformer_records_plan(
        flatten_schema_json(model.model_json_schema()), name="adbi_resume"
    )
    document = document_from_gliformer_records(
        {
            "adbi_resume": [{"full_name": "Jane Doe"}],
            "skills": [{"category": "LLM", "skill": "LangGraph"}],
        },
        plan,
        name="adbi_resume",
    )
    payload, validation = validate_extraction(
        "adbi_resume",
        rehydrate_extraction_result(document, model.model_json_schema()),
        [],
        model_cls=model,
    )
    assert validation.valid
    assert validation.warnings == []
    assert payload["skills"][0]["items"][0]["skill"]["value"] == "LangGraph"


def test_a_field_name_used_twice_in_one_record_keeps_both() -> None:
    schema = {
        "type": "object",
        "properties": {
            "vendor": {"type": "object", "properties": {"name": {"type": "string"}}},
            "name": {"type": "string"},
        },
    }
    plan = gliformer_records_plan(schema, name="invoice")
    assert sorted(plan.fields["invoice"]) == ["name", "vendor_name"]
    assert document_from_gliformer_records(
        {"invoice": [{"name": "ACME SARL", "vendor_name": "ACME"}]}, plan, name="invoice"
    ) == {"name": "ACME SARL", "vendor": {"name": "ACME"}}


def test_a_static_schema_converts_too() -> None:
    plan = gliformer_records_plan(flat_schema_json("invoice"), name="invoice")
    assert "line_items" in plan.fields
    assert "invoice" in plan.fields


def test_a_schema_that_refers_to_itself_is_rejected_rather_than_looping() -> None:
    schema = {
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }
    with pytest.raises(ValueError, match="refers to itself"):
        gliformer_records_plan(schema, name="tree")


def test_a_schema_with_no_fields_is_rejected() -> None:
    with pytest.raises(ValueError, match="no fields"):
        gliformer_records_plan({"type": "object", "properties": {}}, name="empty")


def test_the_style_asks_for_the_mode_through_extra_body() -> None:
    # The knob is a named response-format style, like every other runtime quirk.
    response_format, extra_body = build_response_format(
        "gliformer_records", "invoice", {"type": "object"}
    )
    assert response_format["json_schema"]["name"] == "invoice_extraction"
    assert extra_body == {"structure_mode": "records"}


class _FakeGliformer:
    """Records the schema it was handed, and answers in the flat form."""

    def __init__(self) -> None:
        self.seen: Any = None
        self.reply: Any = {}

    def structure(self, text: str, schema: Any, *, validate_output: bool = True) -> Any:
        self.seen = schema
        return self.reply

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        return []


def test_the_server_switches_form_on_structure_mode() -> None:
    backend = _FakeGliformer()
    backend.reply = {
        "adbi_resume": [{"full_name": "Jane Doe"}],
        "skills": [{"category": "LLM", "skill": "LangGraph"}],
    }
    schema = {
        "type": "object",
        "properties": {
            "full_name": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"skill": {"type": "string"}},
                            },
                        },
                    },
                },
            },
        },
    }
    app = create_encoder_app(model_id="knowledgator/gliformer-base-v1", backend=backend)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gliformer-base-v1",
                "messages": [{"role": "user", "content": "CV ..."}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "adbi_resume", "schema": schema},
                },
                "structure_mode": "records",
            },
        )
    assert response.status_code == 200
    assert backend.seen == {"adbi_resume": ["full_name"], "skills": ["category", "skill"]}
    import json

    content = json.loads(response.json()["choices"][0]["message"]["content"])
    assert content == {
        "full_name": "Jane Doe",
        "skills": [{"category": "LLM", "items": [{"skill": "LangGraph"}]}],
    }


def test_without_the_mode_the_nested_form_is_still_used() -> None:
    backend = _FakeGliformer()
    backend.reply = {"invoice": [{"invoice_number": "F-1"}]}
    app = create_encoder_app(model_id="knowledgator/gliformer-base-v1", backend=backend)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gliformer-base-v1",
                "messages": [{"role": "user", "content": "Facture ..."}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "invoice", "schema": flat_schema_json("invoice")},
                },
            },
        )
    assert response.status_code == 200
    # A Pydantic model per record, not a field list.
    assert not isinstance(backend.seen["invoice"], list)


class _ChunkedModel:
    """Answers each chunk with its own rows, for every record type."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def structure(self, text: str, schema: Any, validate_output: bool = True) -> Any:
        self.seen.append(text)
        index = len(self.seen)
        return {
            "adbi_resume": [{"full_name": "Ada"}],
            "skills": [{"category": f"cat {index}", "skill": f"skill {index}"}],
            "experience": [{"title": f"Job {index}"}],
        }


def test_a_long_document_keeps_every_record_type_not_just_the_first() -> None:
    from docie_bench.encoders.server import GliformerBackend

    backend = GliformerBackend.__new__(GliformerBackend)
    backend.model_id = "knowledgator/gliformer-base-v1"
    backend._model = _ChunkedModel()
    backend.input_window = 6
    text = "\n".join(f"ligne {i} du cv" for i in range(10))
    result = backend.structure(
        text,
        {
            "adbi_resume": ["full_name"],
            "skills": ["category", "skill"],
            "experience": ["title"],
        },
    )
    chunks = len(backend._model.seen)
    assert chunks > 1
    # Every record type survives every chunk; the single-key fold would have
    # returned only the first one.
    assert len(result["skills"]) == chunks
    assert len(result["experience"]) == chunks
    assert len(result["adbi_resume"]) == chunks


def test_a_single_record_type_is_not_folded_in_the_plain_form() -> None:
    # Folding on key count would read one record type as "the document is one
    # record" and keep only the first row of every chunk.
    from docie_bench.encoders.server import GliformerBackend

    class _OneType:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def structure(self, text: str, schema: Any, validate_output: bool = True) -> Any:
            self.seen.append(text)
            return {"experience": [{"title": f"Job {len(self.seen)}"}]}

    backend = GliformerBackend.__new__(GliformerBackend)
    backend.model_id = "knowledgator/gliformer-base-v1"
    backend._model = _OneType()
    backend.input_window = 6
    result = backend.structure(
        "\n".join(f"ligne {i} du cv" for i in range(10)), {"experience": ["title"]}
    )
    assert len(result["experience"]) == len(backend._model.seen) > 1
