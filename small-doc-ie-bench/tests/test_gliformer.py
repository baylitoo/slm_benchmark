"""GLiFormer support: family detection, backend selection, schema building.

No model is loaded — the backend is exercised through a stand-in with the API
the model card documents.
"""

from typing import Any

import pytest

from docie_bench.encoders.server import GliformerBackend, build_backend
from docie_bench.extract.postprocess import coerce_scalars, normalize_by_schema
from docie_bench.schemas.dynamic import DynamicSchemaSpec, DynamicTemplateBuilder
from docie_bench.schemas.extraction import rehydrate_extraction_result
from docie_bench.serving.arch_registry import resolve_family
from docie_bench.serving.model_store import FAMILIES

BASE = "knowledgator/gliformer-base-v1"
LARGE = "knowledgator/gliformer-large-v1"


def _spec() -> DynamicSchemaSpec:
    return DynamicSchemaSpec.model_validate(
        {
            "document_type": "resume",
            "fields": [
                {"name": "full_name", "type": "string"},
                {"name": "salary", "type": "money"},
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


# --- family detection --------------------------------------------------------


@pytest.mark.parametrize("repo_id", [BASE, LARGE])
def test_a_gliformer_repo_resolves_to_its_own_family(repo_id: str) -> None:
    # These repos ship no config.json at all (weights are described by
    # gliner_config.json), so the architecture is empty and the repo has to be
    # identified by how it advertises itself.
    verdict = resolve_family(
        None,
        has_gguf=False,
        has_safetensors=False,
        has_mmproj=False,
        repo_id=repo_id,
        tags=("gliformer", "gliner", "deberta", "token-classification"),
        pipeline_tag="token-classification",
        library_name="gliformer",
    )
    assert verdict.verdict == "supported"
    assert verdict.family == "encoder_gliformer"


def test_the_gliner_tag_a_gliformer_repo_carries_does_not_claim_it() -> None:
    # The Hub tags include "gliner"; detection keys on the architecture for
    # that family, never on tags, so the two do not collide.
    verdict = resolve_family(
        None,
        has_gguf=False,
        has_safetensors=False,
        has_mmproj=False,
        repo_id=BASE,
        tags=("gliner",),
        library_name="gliformer",
    )
    assert verdict.family == "encoder_gliformer"


def test_a_gliner_repo_still_resolves_to_the_gliner_family() -> None:
    verdict = resolve_family(
        "gliner",
        has_gguf=False,
        has_safetensors=True,
        has_mmproj=False,
        repo_id="urchade/gliner_multi_pii-v1",
    )
    assert verdict.family == "encoder_gliner"


def test_the_family_is_an_analyzer_served_by_the_gliformer_backend() -> None:
    contract = FAMILIES["encoder_gliformer"]
    assert contract.analyzer is True
    assert contract.encoder_backend == "gliformer"


# --- backend selection -------------------------------------------------------


@pytest.mark.parametrize("model_id", [BASE, LARGE, "KNOWLEDGATOR/GLiFormer-Base-V1"])
def test_auto_selects_the_gliformer_backend(monkeypatch: pytest.MonkeyPatch, model_id: str) -> None:
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        "docie_bench.encoders.server.GliformerBackend",
        lambda mid: seen.setdefault("model_id", mid),
    )
    build_backend(model_id)
    assert seen["model_id"] == model_id


def test_auto_still_selects_gliner_for_a_gliner_model(monkeypatch: pytest.MonkeyPatch) -> None:
    picked: list[str] = []
    monkeypatch.setattr(
        "docie_bench.encoders.server.GlinerBackend", lambda mid: picked.append("gliner")
    )
    build_backend("urchade/gliner_multi_pii-v1")
    assert picked == ["gliner"]


def test_an_unknown_backend_name_is_refused() -> None:
    with pytest.raises(ValueError, match="gliformer"):
        build_backend(BASE, kind="nonsense")


# --- the backend, against the documented API ---------------------------------


class _FakeGliformer:
    """The API the model card documents, nothing more."""

    def predict_entities(
        self, text: str, labels: list[str], threshold: float
    ) -> list[dict[str, Any]]:
        assert threshold == 0.5
        return [{"text": "Alice", "label": "person", "start": 0, "end": 5, "score": 0.91}]

    def structure(self, text: str, schema: Any, validate_output: bool = True) -> Any:
        assert validate_output is True
        return {"employee": [{"name": "Alice", "company": "Acme"}]}


def _backend() -> GliformerBackend:
    backend = GliformerBackend.__new__(GliformerBackend)
    backend.model_id = BASE
    backend._model = _FakeGliformer()
    return backend


def test_entities_are_normalised_to_the_shape_every_encoder_returns() -> None:
    assert _backend().predict("Alice works at Acme.", ["person"], 0.5) == [
        {"type": "person", "value": "Alice", "start": 0, "end": 5, "score": 0.91}
    ]


def test_the_normalised_entity_shape_matches_the_gliner_backend() -> None:
    assert set(_backend().predict("Alice", ["person"], 0.5)[0]) == {
        "type",
        "value",
        "start",
        "end",
        "score",
    }


def test_structuring_passes_the_schema_through_and_validates() -> None:
    result = _backend().structure("Alice works at Acme.", {"employee": ["name", "company"]})
    assert result == {"employee": [{"name": "Alice", "company": "Acme"}]}


# --- schema building ---------------------------------------------------------


def test_the_pydantic_schema_is_keyed_by_document_type() -> None:
    schema = DynamicTemplateBuilder.build_gliformer_schema(_spec())
    assert list(schema) == ["resume"]
    assert list(schema["resume"].model_fields) == ["full_name", "salary", "experience"]


def test_every_leaf_is_a_string_so_a_span_off_the_page_validates() -> None:
    model = DynamicTemplateBuilder.build_gliformer_schema(_spec())["resume"]
    # Written as the document has them: a European amount and a French date.
    parsed = model.model_validate(
        {
            "full_name": "Ada Lovelace",
            "salary": {"amount": "1 234,56", "currency": "€"},
            "experience": [{"title": "Engineer", "start_date": "28/02/2026"}],
        }
    )
    assert parsed.model_dump()["salary"]["amount"] == "1 234,56"


def test_the_spans_are_typed_by_the_pipeline_that_already_exists() -> None:
    spec = _spec()
    root = DynamicTemplateBuilder.build_model(spec).model_json_schema()
    raw = {
        "full_name": "Ada Lovelace",
        "salary": {"amount": "1 234,56 €", "currency": None},
        "experience": [{"title": "Engineer", "start_date": "28/02/2026"}],
    }
    result = normalize_by_schema(rehydrate_extraction_result(raw, root), root)
    result, warnings = coerce_scalars(result, root)
    assert result["salary"]["amount"] == "1234.56"
    assert result["salary"]["currency"] == "EUR"
    assert result["experience"][0]["start_date"]["value"] == "2026-02-28"
    assert warnings == []


def test_the_field_list_form_records_nested_fields_under_their_own_name() -> None:
    assert DynamicTemplateBuilder.build_gliformer_fields(_spec()) == {
        "resume": ["full_name", "salary"],
        "experience": ["title", "start_date"],
    }
