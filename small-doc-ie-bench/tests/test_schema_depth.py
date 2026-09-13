"""A schema arrives on a request, so its depth is a caller's choice."""

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docie_bench.encoders.server import create_encoder_app
from docie_bench.schemas.dynamic import (
    DynamicSchemaSpec,
    DynamicTemplateBuilder,
    gliformer_model_from_json_schema,
)
from docie_bench.schemas.extraction import flat_schema_json, flatten_schema_json

SELF_REFERENTIAL: dict[str, Any] = {
    "$defs": {"Node": {"properties": {"child": {"$ref": "#/$defs/Node"}}}},
    "properties": {"root": {"$ref": "#/$defs/Node"}},
}


def _deeply_nested(levels: int) -> dict[str, Any]:
    root: dict[str, Any] = {"type": "object"}
    node = root
    for _ in range(levels):
        node["properties"] = {"n": {"type": "object"}}
        node = node["properties"]["n"]
    return root


# ── refused, as a 400 rather than a stack overflow ───────────────────────────


def test_a_definition_that_refers_to_itself_is_refused() -> None:
    with pytest.raises(ValueError, match="refers to itself"):
        gliformer_model_from_json_schema(SELF_REFERENTIAL, name="probe")


def test_a_schema_nested_past_the_ceiling_is_refused() -> None:
    with pytest.raises(ValueError, match="nests deeper than"):
        gliformer_model_from_json_schema(_deeply_nested(2000), name="probe")


def test_the_refusal_is_a_value_error_so_the_endpoint_can_answer_400() -> None:
    # RecursionError is not a ValueError, and the endpoint only catches the
    # latter, so an unguarded schema reached the client as a 500.
    for schema in (SELF_REFERENTIAL, _deeply_nested(2000)):
        with pytest.raises(ValueError, match="refers to itself|nests deeper"):
            gliformer_model_from_json_schema(schema, name="probe")


class _Structuring:
    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        return []

    def structure(self, text: str, schema: Any, *, validate_output: bool = True) -> Any:
        return {"probe": [{}]}


@pytest.mark.parametrize("schema", [SELF_REFERENTIAL, _deeply_nested(2000)])
def test_the_endpoint_answers_400_not_500(schema: dict[str, Any]) -> None:
    app = create_encoder_app(model_id="knowledgator/gliformer-base-v1", backend=_Structuring())
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gliformer",
                "messages": [{"role": "user", "content": "text"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "probe", "schema": schema},
                },
            },
        )
    assert response.status_code == 400
    assert "cannot be structured" in response.json()["error"]["message"]


# ── still accepted: every schema this platform actually builds ───────────────


def _resume_root() -> dict[str, Any]:
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
    return DynamicTemplateBuilder.build_model(spec).model_json_schema()


def test_a_saved_schema_with_real_refs_still_converts() -> None:
    # The rich schema uses $defs for every nested model. An over-eager cycle
    # guard rejected a single, non-cyclic reference.
    model = gliformer_model_from_json_schema(_resume_root(), name="adbi_resume")
    assert "skills" in model.model_fields


def test_the_flattened_form_still_converts() -> None:
    model = gliformer_model_from_json_schema(
        flatten_schema_json(_resume_root()), name="adbi_resume"
    )
    parsed = model.model_validate(
        {"full_name": "Ada", "skills": [{"category": "LLM", "items": [{"skill": "LangGraph"}]}]}
    )
    assert parsed.model_dump()["skills"][0]["items"][0]["skill"] == "LangGraph"


def test_a_static_schema_still_converts() -> None:
    model = gliformer_model_from_json_schema(flat_schema_json("invoice"), name="invoice")
    assert "line_items" in model.model_fields


def test_a_schema_just_inside_the_ceiling_is_accepted() -> None:
    model = gliformer_model_from_json_schema(_deeply_nested(20), name="probe")
    assert json.dumps(model.model_json_schema())
