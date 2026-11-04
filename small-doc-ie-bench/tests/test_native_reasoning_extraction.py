"""A checkpoint trained to reason keeps reasoning, extraction included.

Extraction used to send ``enable_thinking: False`` unconditionally. For a model
whose template always opens a reasoning block that is not a shorter answer but
a wrong one: the reasoning ends up INSIDE the JSON rather than before it,
because suppressing it leaves the template nowhere else to put it.

Two mechanisms, and they must not both fire. ``enable_thinking`` is the ask;
the assistant prefill is the ENFORCEMENT for templates that ignore the ask, and
sending it while reasoning is wanted truncates the reasoning.
"""

from __future__ import annotations

from typing import Any

import pytest

from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.reasoning import uses_native_reasoning
from docie_bench.serving.model_store import FAMILIES


def profile(**overrides: Any) -> ModelProfile:
    base = {
        "name": "lfm3-1.7b",
        "model": "lfm3-1.7b",
        "base_url": "http://node:8088/v1",
        "api_key": "k",
        "prompt_profile": "strict_extraction_v1",
    }
    base.update(overrides)
    return ModelProfile(**base)  # type: ignore[arg-type]


# --- the family declares it -------------------------------------------------


@pytest.mark.parametrize("family", ["lfm3", "lfm3_vl"])
def test_the_lfm3_families_declare_native_reasoning(family: str) -> None:
    """The card warns against disabling it, so the default must not disable it."""
    assert FAMILIES[family].native_reasoning is True


@pytest.mark.parametrize("family", ["lfm2", "nuextract3", "vision_ocr"])
def test_other_families_are_unchanged(family: str) -> None:
    assert FAMILIES[family].native_reasoning is False


def test_a_family_declaration_reaches_the_profile_as_the_existing_option() -> None:
    """A store-deployed model has no models.yaml entry to carry the option.

    Reusing the option `uses_native_reasoning` already honours means a
    family-declared checkpoint and a yaml-declared one reach every caller
    through one path, not two.
    """
    assert uses_native_reasoning(profile(options={"native_reasoning": True})) is True
    assert uses_native_reasoning(profile(options={})) is False


# --- what extraction now sends ---------------------------------------------


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture the kwargs each extraction call hands the client."""
    captured: list[dict[str, Any]] = []

    async def _chat_json(self: Any, **kwargs: Any) -> tuple[Any, dict[str, Any], Any]:
        captured.append(kwargs)
        # raw_response is read for loop/telemetry flags, so an empty dict
        # rather than None -- the stub stands in for a response, not for
        # the absence of one.
        return {}, {}, {}

    monkeypatch.setattr(
        "docie_bench.llm.openai_client.OpenAICompatibleClient.chat_json", _chat_json
    )
    return captured


async def _extract(service: ExtractionService) -> None:
    await service._extract_group(
        blocks=[],
        images=[],
        image_urls=[],
        schema_name="receipt",
        generation_schema={"type": "object", "properties": {"total": {"type": "string"}}},
        nuextract_template=None,
        language=None,
        metadata=None,
        field_names=None,
        want_logprobs=False,
    )


@pytest.mark.anyio
async def test_a_native_reasoning_model_is_allowed_to_think(sent: list[dict[str, Any]]) -> None:
    service = ExtractionService(profile(options={"native_reasoning": True}))

    await _extract(service)

    assert sent[0]["chat_template_kwargs"]["enable_thinking"] is True


@pytest.mark.anyio
async def test_and_is_not_prefilled_while_it_thinks(sent: list[dict[str, Any]]) -> None:
    """The prefill is what skips the reasoning block.

    Sending it here would truncate exactly the reasoning the model was just
    told to produce — the two mechanisms are opposites, not companions.
    """
    service = ExtractionService(profile(options={"native_reasoning": True}))

    await _extract(service)

    assert sent[0]["assistant_prefill"] is None


@pytest.mark.anyio
async def test_an_explicit_request_to_stop_thinking_still_wins(
    sent: list[dict[str, Any]],
) -> None:
    """The operator asked. The prefill comes back to enforce it."""
    service = ExtractionService(
        profile(options={"native_reasoning": True}), disable_thinking=True
    )

    await _extract(service)

    assert sent[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert sent[0]["assistant_prefill"] == "{"


@pytest.mark.anyio
async def test_an_ordinary_model_still_never_thinks(sent: list[dict[str, Any]]) -> None:
    """Unchanged for everything that was not trained to reason."""
    service = ExtractionService(profile(options={}))

    await _extract(service)

    assert sent[0]["chat_template_kwargs"]["enable_thinking"] is False
