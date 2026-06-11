from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.schemas.dynamic import DynamicSchemaSpec, DynamicTemplateBuilder


def _profile(prompt_profile: str = "strict_extraction_v1") -> ModelProfile:
    return ModelProfile(
        name="test",
        model="test-model",
        base_url="http://test",
        api_key="test",
        prompt_profile=prompt_profile,
    )


def _purchase_order_spec() -> dict[str, Any]:
    return {
        "document_type": "purchase_order",
        "fields": [
            {"name": "order_number", "type": "string"},
            {"name": "order_date", "type": "date"},
            {"name": "total_amount", "type": "money"},
        ],
    }


def test_dynamic_schema_is_serializable_and_builds_runtime_artifacts() -> None:
    spec = DynamicSchemaSpec.model_validate(_purchase_order_spec())
    restored = DynamicSchemaSpec.model_validate_json(spec.model_dump_json())
    model = DynamicTemplateBuilder.build_model(restored)

    parsed = model.model_validate(
        {
            "order_number": {"value": "PO-42"},
            "total_amount": {"amount": "125.50", "currency": "EUR"},
        }
    )

    assert parsed.document_type == "purchase_order"
    assert model.model_json_schema()["additionalProperties"] is False
    assert DynamicTemplateBuilder.build_nuextract_template(restored)["order_date"] == {
        "value": "date"
    }


def test_dynamic_schema_rejects_duplicate_or_unsafe_fields() -> None:
    raw = _purchase_order_spec()
    raw["fields"].append({"name": "order_number", "type": "string"})
    with pytest.raises(ValidationError, match="unique"):
        DynamicSchemaSpec.model_validate(raw)


def test_dynamic_schema_builds_reusable_nested_list_schema() -> None:
    spec = DynamicSchemaSpec.model_validate(
        {
            "document_type": "delivery_note",
            "fields": [
                {"name": "delivery_number", "type": "string"},
                {
                    "name": "items",
                    "type": "list",
                    "fields": [
                        {"name": "description", "type": "string"},
                        {"name": "quantity", "type": "number"},
                        {
                            "name": "dimensions",
                            "type": "object",
                            "fields": [{"name": "weight", "type": "number"}],
                        },
                    ],
                },
            ],
        }
    )

    model = DynamicTemplateBuilder.build_model(spec)
    parsed = model.model_validate(
        {
            "delivery_number": {"value": "DN-1"},
            "items": [
                {
                    "description": {"value": "Steel plate"},
                    "quantity": {"value": "2"},
                    "dimensions": {"weight": {"value": "10.5"}},
                }
            ],
        }
    )
    template = DynamicTemplateBuilder.build_nuextract_template(spec)

    assert parsed.items[0].quantity.value == 2
    assert parsed.items[0].dimensions.weight.value == 10.5
    assert template["items"] == [
        {
            "description": {"value": "verbatim-string"},
            "quantity": {"value": "number"},
            "dimensions": {"weight": {"value": "number"}},
        }
    ]


def test_dynamic_schema_rejects_unconfigured_container() -> None:
    with pytest.raises(ValidationError, match="must define nested fields"):
        DynamicSchemaSpec.model_validate(
            {
                "document_type": "delivery_note",
                "fields": [{"name": "items", "type": "list"}],
            }
        )

    raw = _purchase_order_spec()
    raw["fields"][0]["name"] = "Document Type"
    with pytest.raises(ValidationError, match="pattern"):
        DynamicSchemaSpec.model_validate(raw)


@pytest.mark.asyncio
async def test_dynamic_schema_inference_extracts_unseen_type_and_can_be_reused(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    responses = [
        _purchase_order_spec(),
        {
            "document_type": "purchase_order",
            "order_number": {"value": "PO-42", "evidence_ids": ["b1"], "confidence": 0.9},
            "order_date": {"value": "2026-06-10", "evidence_ids": ["b1"], "confidence": 0.8},
            "total_amount": {
                "amount": "125.50",
                "currency": "EUR",
                "evidence_ids": ["b1"],
                "confidence": 0.8,
            },
        },
    ]

    class FakeClient:
        def __init__(self, profile: ModelProfile) -> None:
            self.profile = profile

        async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], None, dict[str, Any]]:
            calls.append(kwargs)
            return responses.pop(0), None, {}

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("docie_bench.extract.service.OpenAICompatibleClient", FakeClient)
    service = ExtractionService(_profile())

    inferred = await service.extract_from_text(
        text="PURCHASE ORDER PO-42 | Date 2026-06-10 | Total EUR 125.50",
        ocr_blocks=None,
        schema_name="unknown",
        schema_mode="dynamic",
    )

    assert inferred.validation.valid
    assert inferred.schema_name == "purchase_order"
    assert inferred.result["order_number"]["value"] == "PO-42"
    assert inferred.dynamic_schema == _purchase_order_spec()
    assert calls[0]["schema_name"] == "dynamic_schema_spec"

    responses.append(
        {
            "document_type": "purchase_order",
            "order_number": {"value": "PO-43"},
            "order_date": None,
            "total_amount": None,
        }
    )
    calls.clear()
    serialized = json.loads(json.dumps(inferred.dynamic_schema))
    reused = await service.extract_from_text(
        text="PURCHASE ORDER PO-43",
        ocr_blocks=None,
        schema_name="ignored",
        schema_mode="dynamic",
        dynamic_schema=serialized,
    )

    assert reused.validation.valid
    assert reused.result["order_number"]["value"] == "PO-43"
    assert len(calls) == 1
    assert calls[0]["schema_name"] == "purchase_order"


@pytest.mark.asyncio
async def test_nuextract_dynamic_inference_requires_proposer_or_reused_schema() -> None:
    service = ExtractionService(_profile(prompt_profile="nuextract_v1"))
    with pytest.raises(ValueError, match="instruction-following proposer"):
        await service.extract_from_text(
            text="Unknown document",
            ocr_blocks=None,
            schema_name="unknown",
            schema_mode="dynamic",
        )
