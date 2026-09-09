from typing import Any

import pytest

from docie_bench.extract.postprocess import (
    dedupe_lists,
    normalize_by_schema,
    normalize_date,
    normalize_placeholders,
)
from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.schemas.dynamic import DynamicSchemaSpec, DynamicTemplateBuilder


@pytest.mark.parametrize(
    ("value", "field", "expected"),
    [
        ("Mars 2022", "start_date", "2022-03"),
        ("Septembre 2019 - F\u00e9vrier 2022", "start_date", "2019-09"),
        ("Septembre 2019 - F\u00e9vrier 2022", "end_date", "2022-02"),
        ("2023-2025", "start_date", "2023"),
        ("2023-2025", "end_date", "2025"),
        ("2025-2026", "year", "2025"),
        ("Aujourd'hui", "end_date", None),
        ("pr\u00e9sent", "end_date", None),
        ("2022-03", "start_date", "2022-03"),
        ("2022-03-01", "start_date", "2022-03-01"),
        ("12/03/2021", "issue_date", "2021-03-12"),
        ("03/2021", "issue_date", "2021-03"),
        ("1er janvier 2020", "start_date", "2020-01-01"),
        ("N/A", "end_date", None),
        ("Licence EEA", "year", "Licence EEA"),
    ],
)
def test_normalize_date(value: str, field: str, expected: str | None) -> None:
    assert normalize_date(value, field_name=field) == expected


def test_placeholders_and_duplicates() -> None:
    raw = {
        "location": "N/A",
        "phone": "-",
        "name": "Amine",
        "experience": [
            {"company": "A", "title": "x"},
            {"company": "A", "title": "x"},
            {"company": "B"},
        ],
    }
    cleaned = dedupe_lists(normalize_placeholders(raw))
    assert cleaned["location"] is None
    assert cleaned["phone"] is None
    assert cleaned["name"] == "Amine"
    assert cleaned["experience"] == [{"company": "A", "title": "x"}, {"company": "B"}]


def test_null_lists_become_empty_lists_but_null_scalars_stay() -> None:
    spec = DynamicSchemaSpec.model_validate(
        {
            "document_type": "doc",
            "fields": [
                {"name": "title", "type": "string"},
                {
                    "name": "interests",
                    "type": "list",
                    "fields": [{"name": "interest", "type": "string"}],
                },
            ],
        }
    )
    root = DynamicTemplateBuilder.build_model(spec).model_json_schema()
    out = normalize_by_schema({"title": None, "interests": None}, root)
    assert out == {"title": None, "interests": []}


@pytest.mark.asyncio
async def test_service_normalizes_dates_by_schema_type(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, profile: ModelProfile) -> None:
            self.profile = profile

        async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], None, dict[str, Any]]:
            raw = {
                "title": "Mars 2022",
                "experience": [
                    {"start_date": "Mars 2022 - Aujourd'hui", "end_date": "Aujourd'hui"},
                    {"start_date": "2019-2022", "end_date": "2019-2022"},
                ],
            }
            return raw, None, {}

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr("docie_bench.extract.service.OpenAICompatibleClient", FakeClient)
    profile = ModelProfile(name="t", model="m", base_url="http://t", api_key="k")
    response = await ExtractionService(profile).extract_from_text(
        text="Mars 2022 Aujourd'hui 2019-2022",
        ocr_blocks=None,
        schema_name="doc",
        schema_mode="dynamic",
        dynamic_schema={
            "document_type": "doc",
            "fields": [
                {"name": "title", "type": "string"},
                {"name": "experience", "type": "list", "fields": [
                    {"name": "start_date", "type": "date"},
                    {"name": "end_date", "type": "date"},
                ]},
            ],
        },
    )
    result = response.result
    assert result["title"]["value"] == "Mars 2022"
    first, second = result["experience"]
    assert first["start_date"]["value"] == "2022-03"
    assert first["end_date"]["value"] is None
    assert second["start_date"]["value"] == "2019"
    assert second["end_date"]["value"] == "2022"
