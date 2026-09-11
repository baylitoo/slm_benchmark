from typing import Any

from docie_bench.extract.validators import validate_extraction
from docie_bench.schemas.dynamic import DynamicSchemaSpec, DynamicTemplateBuilder


def _model() -> Any:
    spec = DynamicSchemaSpec.model_validate(
        {
            "document_type": "resume",
            "fields": [
                {"name": "full_name", "type": "string"},
                {"name": "years_experience", "type": "number"},
                {
                    "name": "education",
                    "type": "list",
                    "fields": [{"name": "school", "type": "string"}],
                },
            ],
        }
    )
    return DynamicTemplateBuilder.build_model(spec)


def test_one_unparsable_scalar_no_longer_invalidates_the_whole_document() -> None:
    payload = {
        "full_name": {"value": "Ada Lovelace", "evidence_ids": [], "confidence": 0.9},
        "years_experience": {"value": "Septembre 2019", "evidence_ids": [], "confidence": 0.2},
    }
    normalized, validation = validate_extraction("resume", payload, [], model_cls=_model())
    assert validation.valid is True
    assert normalized["full_name"]["value"] == "Ada Lovelace"
    assert normalized["years_experience"]["value"] is None
    assert any("years_experience" in warning for warning in validation.warnings)


def test_a_structurally_wrong_list_element_is_removed_not_nulled() -> None:
    payload = {
        "full_name": {"value": "Ada Lovelace", "evidence_ids": [], "confidence": 0.9},
        "education": [
            {"school": {"value": "MIT", "evidence_ids": [], "confidence": 0.9}},
            "Oxford",
        ],
    }
    normalized, validation = validate_extraction("resume", payload, [], model_cls=_model())
    assert validation.valid is True
    assert len(normalized["education"]) == 1
    assert normalized["education"][0]["school"]["value"] == "MIT"
    assert validation.warnings
