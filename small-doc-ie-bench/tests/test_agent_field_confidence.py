"""The Agent endpoint's flat value contract and its per-field review signal."""

from typing import Any

from docie_bench.agents.runtime import _field_confidence, _flatten_agent_result


def _result() -> dict[str, Any]:
    return {
        "document_type": {"value": "resume"},
        "full_name": {"value": "Ada Lovelace", "evidence_ids": ["b1"], "confidence": 0.93},
        "interests": [],
        "experience": [
            {
                "title": {"value": "Engineer", "evidence_ids": ["b4"], "confidence": 0.5},
                "start": {"value": "2020-01", "evidence_ids": [], "confidence": 0.0},
            }
        ],
    }


def test_a_wrapper_is_unwrapped_even_when_it_carries_logprob_confidence() -> None:
    # model_confidence is attached only when logprob confidence is enabled.
    # Before, its presence defeated the exact key-set check and the consumer
    # received {"value": ..., "model_confidence": ...} instead of the value.
    wrapped = {
        "full_name": {
            "value": "Ada Lovelace",
            "evidence_ids": ["b1"],
            "confidence": 0.93,
            "model_confidence": 0.82,
        }
    }
    assert _flatten_agent_result(wrapped) == {"full_name": "Ada Lovelace"}


def test_flattening_is_unchanged_without_logprob_confidence() -> None:
    assert _flatten_agent_result(_result()) == {
        "full_name": "Ada Lovelace",
        "interests": [],
        "experience": [{"title": "Engineer", "start": "2020-01"}],
    }


def test_every_leaf_is_reported_at_the_path_its_value_sits_at() -> None:
    confidence = _field_confidence(_result())
    assert set(confidence) == {"full_name", "experience[0].title", "experience[0].start"}
    assert confidence["full_name"]["confidence"] == 0.93
    assert confidence["full_name"]["evidence_ids"] == ["b1"]


def test_a_truncated_field_is_visible_below_the_review_threshold() -> None:
    # A repetition-loop truncation caps the field to 0.5, which is what makes
    # the map usable as a review trigger.
    confidence = _field_confidence(_result())
    flagged = {path for path, entry in confidence.items() if entry["confidence"] <= 0.5}
    assert flagged == {"experience[0].title", "experience[0].start"}


def test_an_ungrounded_field_is_reported_rather_than_omitted() -> None:
    confidence = _field_confidence(_result())
    assert confidence["experience[0].start"] == {"confidence": 0.0}


def test_the_logprob_is_reported_under_a_name_that_cannot_be_read_as_a_score() -> None:
    result = {
        "full_name": {
            "value": "Ada Lovelace",
            "evidence_ids": ["b1"],
            "confidence": 0.93,
            "model_confidence": -7.5,
        }
    }
    # A natural-log probability, never a 0..1 score: -7.5 under a key called
    # "model_confidence" would fail any "below 0.5 needs review" rule.
    assert _field_confidence(result) == {
        "full_name": {"confidence": 0.93, "model_logprob": -7.5, "evidence_ids": ["b1"]}
    }


def test_the_reserved_root_fields_are_not_reported() -> None:
    assert "document_type" not in _field_confidence(_result())


def test_the_map_keys_address_the_flat_content() -> None:
    result = _result()
    flat = _flatten_agent_result(result)
    for path in _field_confidence(result):
        node: Any = flat
        for segment in path.replace("[", ".[").split("."):
            node = node[int(segment[1:-1])] if segment.startswith("[") else node[segment]
        assert node is not None


def _profile(name: str, prompt_profile: str) -> Any:
    from docie_bench.llm.model_profiles import ModelProfile

    return ModelProfile(
        name=name,
        model=name,
        base_url="http://x",
        api_key=None,
        prompt_profile=prompt_profile,
    )


def test_the_prompt_profile_of_the_profile_that_served_is_reported() -> None:
    from docie_bench.agents.runtime import _prompt_profile

    single = _profile("lfm2.5-2.6b", "strict_extraction_v1")
    assert _prompt_profile("lfm2.5-2.6b", single, None) == "strict_extraction_v1"


def test_a_router_reports_the_profile_that_actually_ran_not_the_one_requested() -> None:
    from docie_bench.agents.runtime import _prompt_profile

    requested = _profile("lfm2.5-2.6b", "strict_extraction_v1")
    served = _profile("nuextract3", "nuextract3")
    assert _prompt_profile("nuextract3", requested, {"nuextract3": served}) == "nuextract3"
