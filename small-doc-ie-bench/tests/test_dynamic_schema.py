from __future__ import annotations

import asyncio
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
async def test_nuextract3_dynamic_schema_sends_its_real_template_not_an_empty_one(
    monkeypatch,
) -> None:
    """Regression: build_response_format's "nuextract3" branch resolves its
    out-of-band template PURELY from schema_name, via a static lookup that
    only ever knew about the built-in schemas (llm.prompts._NUEXTRACT_TEMPLATES).
    A dynamic (user-defined) schema's freshly-built template
    (DynamicTemplateBuilder.build_nuextract_template) never reached that
    lookup, so every dynamic-schema extraction through nuextract3 silently
    sent an EMPTY template -- nothing telling the model what to extract, so
    every field came back null. Fixed by threading the dynamic template
    through chat_json's chat_template_kwargs, which merges on top of
    build_response_format's own (empty) one."""
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, profile: ModelProfile) -> None:
            self.profile = profile

        async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], None, dict[str, Any]]:
            calls.append(kwargs)
            return {
                "document_type": "adbi_resume",
                "name": {"value": "Jane Doe", "evidence_ids": ["b1"], "confidence": 0.9},
            }, None, {}

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("docie_bench.extract.service.OpenAICompatibleClient", FakeClient)
    service = ExtractionService(_profile(prompt_profile="nuextract3"))

    await service.extract_from_text(
        text="Jane Doe, Senior Engineer",
        ocr_blocks=None,
        schema_name="adbi_resume",
        schema_mode="dynamic",
        dynamic_schema={
            "document_type": "adbi_resume",
            "fields": [{"name": "name", "type": "string"}],
        },
    )

    sent_template = json.loads(calls[0]["chat_template_kwargs"]["template"])
    assert sent_template == {"name": {"value": "verbatim-string"}}


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


@pytest.mark.asyncio
async def test_parallel_extraction_bounds_fanout_to_deployment_slots(monkeypatch) -> None:
    in_flight = 0
    peak = 0

    class FakeClient:
        last_response_format_style = "json_schema"
        last_queue_wait_ms = 0

        def __init__(self, profile: ModelProfile) -> None:
            self.profile = profile

        async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], None, dict[str, Any]]:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            props = kwargs["schema"]["properties"]
            out = {k: ([] if "items" in str(v) else {"value": "x"}) for k, v in props.items()}
            return out, None, {}

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("docie_bench.extract.service.OpenAICompatibleClient", FakeClient)
    profile = ModelProfile(
        name="t", model="m", base_url="http://t", api_key="k",
        max_concurrency=4, deployment_slot_count=1,
    )
    service = ExtractionService(profile)
    fields = [{"name": "name", "type": "string"}] + [
        {"name": f"list{i}", "type": "list", "fields": [{"name": "v", "type": "string"}]}
        for i in range(6)
    ]
    response = await service.extract_from_text(
        text="x", ocr_blocks=None, schema_name="doc", schema_mode="dynamic",
        dynamic_schema={"document_type": "doc", "fields": fields}, parallel_extraction=True,
    )
    assert response.validation.valid
    assert peak == 1


@pytest.mark.asyncio
async def test_parallel_extraction_group_failure_surfaces_as_plain_exception(monkeypatch) -> None:
    from docie_bench.llm.model_gateway import ModelQueueFullError

    class FakeClient:
        def __init__(self, profile: ModelProfile) -> None:
            self.profile = profile

        async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], None, dict[str, Any]]:
            raise ModelQueueFullError("Queue wait timed out")

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("docie_bench.extract.service.OpenAICompatibleClient", FakeClient)
    service = ExtractionService(_profile())
    with pytest.raises(ModelQueueFullError):
        await service.extract_from_text(
            text="x", ocr_blocks=None, schema_name="doc", schema_mode="dynamic",
            dynamic_schema={"document_type": "doc", "fields": [
                {"name": "a", "type": "string"},
                {"name": "b", "type": "list", "fields": [{"name": "v", "type": "string"}]},
            ]},
            parallel_extraction=True,
        )
