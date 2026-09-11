"""liteparse text -> GLiFormer structure() -> the JSON the extraction path reads.

The schema travels on the ordinary ``response_format.json_schema`` every model
gets, so a served GLiFormer is reached through the Playground and the agents
endpoint without either knowing it is not a generative model.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from docie_bench.encoders.server import create_encoder_app
from docie_bench.extract.postprocess import coerce_scalars, normalize_by_schema
from docie_bench.schemas.dynamic import (
    DynamicSchemaSpec,
    DynamicTemplateBuilder,
    gliformer_model_from_json_schema,
)
from docie_bench.schemas.extraction import (
    flat_schema_json,
    flatten_schema_json,
    rehydrate_extraction_result,
    schema_json,
)
from docie_bench.serving.model_store import get_family


class FakeGliformer:
    """Records the schema it was handed and answers with spans, as the real one
    does: values copied off the page, no offsets, one list of records per key."""

    def __init__(self) -> None:
        self.schemas: list[Any] = []

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        return []

    reply: dict[str, Any] | None = None

    def structure(self, text: str, schema: Any, *, validate_output: bool = True) -> Any:
        self.schemas.append(schema)
        name, model = next(iter(schema.items()))
        assert isinstance(model, type)
        assert issubclass(model, BaseModel)
        if self.reply is not None:
            return {name: [self.reply]}
        return {
            name: [
                {
                    "invoice_number": "INV-2026-014",
                    "issue_date": "28/02/2026",
                    "total_ttc": {"amount": "1 234,56 EUR", "currency": None},
                    "line_items": [{"description": "Conseil", "quantity": "2,5"}],
                }
            ]
        }


@pytest.fixture
def backend() -> FakeGliformer:
    return FakeGliformer()


@pytest.fixture
def client(backend: FakeGliformer) -> TestClient:
    return TestClient(
        create_encoder_app(model_id="knowledgator/gliformer-base-v1", backend=backend)
    )


def _request(schema_name: str = "invoice") -> dict[str, Any]:
    return {
        "model": "gliformer-base-v1",
        "messages": [{"role": "user", "content": "FACTURE INV-2026-014 ... Conseil 2,5"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": flat_schema_json(schema_name)},
        },
    }


def test_a_schema_on_the_request_returns_records_not_entities(
    client: TestClient, backend: FakeGliformer
) -> None:
    response = client.post("/v1/chat/completions", json=_request())
    assert response.status_code == 200
    content = json.loads(response.json()["choices"][0]["message"]["content"])
    assert content["invoice_number"] == "INV-2026-014"
    assert "entities" not in content


def test_the_schema_reaches_the_model_as_a_pydantic_record(
    client: TestClient, backend: FakeGliformer
) -> None:
    client.post("/v1/chat/completions", json=_request())
    schema = backend.schemas[0]
    assert list(schema) == ["invoice"]
    model = schema["invoice"]
    assert "line_items" in model.model_fields
    # A span off the page validates: every leaf is a string.
    model.model_validate({"total_ttc": {"amount": "1 234,56", "currency": "EUR"}})


def test_without_a_schema_the_encoder_still_answers_with_entities(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "Jean Dupont"}]},
    )
    assert response.status_code == 200
    content = json.loads(response.json()["choices"][0]["message"]["content"])
    assert "entities" in content


def test_a_backend_without_a_structuring_head_says_so(client: TestClient) -> None:
    app = create_encoder_app(model_id="urchade/gliner_multi_pii-v1", backend=_NerOnly())
    with TestClient(app) as ner_client:
        response = ner_client.post("/v1/chat/completions", json=_request())
    assert response.status_code == 400
    assert "structuring head" in response.json()["error"]["message"]


class _NerOnly:
    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        return []


def test_the_returned_spans_are_typed_by_the_pass_that_already_runs(
    client: TestClient,
) -> None:
    # The point of the whole chain: GLiFormer copies "1 234,56 EUR" and
    # "28/02/2026" verbatim, and the existing post-extraction pass turns them
    # into a Decimal-shaped amount, a currency and an ISO date.
    raw = json.loads(
        client.post("/v1/chat/completions", json=_request()).json()["choices"][0]["message"][
            "content"
        ]
    )
    root = schema_json("invoice")
    result = normalize_by_schema(rehydrate_extraction_result(raw, root), root)
    result, warnings = coerce_scalars(result, root)
    assert result["total_ttc"]["amount"] == "1234.56"
    assert result["total_ttc"]["currency"] == "EUR"
    assert result["issue_date"]["value"] == "2026-02-28"
    assert result["line_items"][0]["quantity"]["value"] == "2.5"
    assert warnings == []


def test_the_family_asks_for_the_schema_it_needs() -> None:
    # "none" would send no schema at all and the server would fall back to
    # entity recognition.
    assert get_family("encoder_gliformer").response_format_style == "openai_json_schema"


def test_a_schema_with_nothing_to_extract_is_refused(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "text"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "empty", "schema": {"type": "string"}},
            },
        },
    )
    assert response.status_code == 400
    assert "cannot be structured" in response.json()["error"]["message"]


def _adbi_resume_spec() -> DynamicSchemaSpec:
    """The saved schema's real shape: skills is a list of objects each holding
    its own list of strings, which is the deepest nesting in use."""
    return DynamicSchemaSpec.model_validate(
        {
            "document_type": "adbi_resume",
            "fields": [
                {"name": "full_name", "type": "string"},
                {"name": "years_experience", "type": "number"},
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
                        {"name": "start_date", "type": "date"},
                    ],
                },
            ],
        }
    )


def _adbi_resume_flat() -> dict[str, Any]:
    root = DynamicTemplateBuilder.build_model(_adbi_resume_spec()).model_json_schema()
    return flatten_schema_json(root)


def test_a_saved_dynamic_schema_converts_with_its_nesting_intact() -> None:
    model = gliformer_model_from_json_schema(_adbi_resume_flat(), name="adbi_resume")
    assert "skills" in model.model_fields
    parsed = model.model_validate(
        {
            "full_name": "Ada Lovelace",
            "years_experience": "5",
            "skills": [{"category": "Agents / LLM Systems", "items": [{"skill": "LangGraph"}]}],
            "experience": [{"title": "Dev", "start_date": "28/02/2026"}],
        }
    )
    dumped = parsed.model_dump()
    assert dumped["skills"][0]["items"][0]["skill"] == "LangGraph"
    assert dumped["experience"][0]["start_date"] == "28/02/2026"


def test_a_dynamic_schema_reaches_the_model_and_its_spans_are_typed(
    backend: FakeGliformer,
) -> None:
    # End to end on the saved-schema path: the schema travels on the same
    # response_format a static one does, so nothing distinguishes the two.
    flat = _adbi_resume_flat()
    app = create_encoder_app(model_id="knowledgator/gliformer-base-v1", backend=backend)
    backend.reply = {
        "full_name": "Ada Lovelace",
        "years_experience": "5 ans",
        "skills": [{"category": "Agents / LLM Systems", "items": [{"skill": "LangGraph"}]}],
        "experience": [{"title": "Dev", "start_date": "28/02/2026"}],
    }
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gliformer-base-v1",
                "messages": [{"role": "user", "content": "CV ... LangGraph ..."}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "adbi_resume", "schema": flat},
                },
            },
        )
    raw = json.loads(response.json()["choices"][0]["message"]["content"])
    assert list(backend.schemas[0]) == ["adbi_resume"]

    root = DynamicTemplateBuilder.build_model(_adbi_resume_spec()).model_json_schema()
    result = normalize_by_schema(rehydrate_extraction_result(raw, root), root)
    result, warnings = coerce_scalars(result, root)
    assert result["experience"][0]["start_date"]["value"] == "2026-02-28"
    assert result["skills"][0]["items"][0]["skill"]["value"] == "LangGraph"
    # "5 ans" is not a number: dropped with a warning naming the field, never
    # guessed at.
    assert result["years_experience"]["value"] is None
    assert any("years_experience" in warning for warning in warnings)
