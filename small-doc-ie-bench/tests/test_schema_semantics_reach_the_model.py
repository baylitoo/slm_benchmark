"""Every extraction prompt carries what the schema's fields MEAN, or says why not.

A grammar constrains shape. It cannot say that ``end_client`` is the client and
``company`` is the employer -- those are three strings in one object, and the
only thing that distinguishes them is ``description``. Drop the descriptions and
a model fills each slot with whatever text sits nearest, which reads as a model
failure and is a prompt failure.

The two paths that did drop them were added for models with their own documented
key formats, and the formats have no slot for prose. That is easy to do again, so
this file enumerates every prompt profile the families use and requires each one
either to deliver descriptions or to appear below with a reason.
"""

from __future__ import annotations

from typing import Any

import pytest

from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.model_store import FAMILIES

#: A description no other text could produce, so finding it proves it travelled.
SENTINEL = "ZZQX the employer, never the end client"
NESTED_SENTINEL = "ZZQX the role on this mission only"

PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company": {"type": "string", "description": SENTINEL},
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string", "description": NESTED_SENTINEL}},
            },
        },
    },
}

#: Profiles that deliberately do not put the schema in the prompt, and why.
#: Adding a profile here is a decision; leaving one out is caught by the test.
EXEMPT: dict[str, str] = {
    "nuextract3": (
        "the template is delivered out-of-band via chat_template_kwargs, in the "
        "format the checkpoint was trained on; that format has no description slot"
    ),
    "nuextract_v1": (
        "same out-of-band template contract as nuextract3, with its own token "
        "convention"
    ),
    "document_only": (
        "GLiFormer is an encoder, not a model reading a prompt -- its labels ARE "
        "the semantics, and it receives no instruction text at all"
    ),
}


def profiles_in_use() -> set[str]:
    return {family.prompt_profile for family in FAMILIES.values()}


def test_every_profile_is_either_covered_or_explicitly_exempt() -> None:
    """A new family with a new prompt profile has to make this choice.

    The failure mode this guards is silent: the extraction still returns valid
    JSON, so nothing errors -- the fields are just filled with the wrong text.
    """
    unaccounted = {
        profile
        for profile in profiles_in_use()
        if profile not in EXEMPT and profile not in _COVERED
    }

    assert unaccounted == set(), (
        f"prompt profile(s) {sorted(unaccounted)} neither deliver schema "
        "descriptions nor declare why they cannot -- add them to EXEMPT with a "
        "reason, or send the field notes"
    )


#: Profiles asserted below to carry descriptions.
_COVERED = {"strict_extraction_v1", "minicpm_kie_v1", "xiaomi_kie_v1"}


def test_the_exemptions_are_still_real_profiles() -> None:
    """An exemption for a profile nobody uses is stale, not safe."""
    assert set(EXEMPT) <= profiles_in_use()


@pytest.fixture
def user_prompt(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    captured: list[str] = []

    async def _chat_json(self: Any, **kwargs: Any) -> tuple[Any, dict[str, Any], Any]:
        captured.append(str(kwargs.get("user_prompt") or ""))
        return {}, {}, {}

    monkeypatch.setattr(
        "docie_bench.llm.openai_client.OpenAICompatibleClient.chat_json", _chat_json
    )
    return captured


async def _run(prompt_profile: str, *, images: list[Any]) -> None:
    service = ExtractionService(
        ModelProfile(
            name="probe",
            model="probe",
            base_url="http://node:8088/v1",
            api_key="k",
            prompt_profile=prompt_profile,
            vision=bool(images),
        )
    )
    await service._extract_group(
        blocks=[],
        images=images,
        image_urls=["data:image/png;base64,AA"] if images else [],
        schema_name="probe",
        generation_schema=PROBE_SCHEMA,
        nuextract_template=None,
        language=None,
        metadata=None,
        field_names=None,
        want_logprobs=False,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("prompt_profile", sorted(_COVERED))
async def test_a_covered_profile_carries_descriptions_with_images(
    user_prompt: list[str], prompt_profile: str
) -> None:
    """The vision path is where the KIE profiles live, so it is the one that
    mattered: a vision model receives the page and the field list, and nothing
    else tells it what the fields are for."""
    await _run(prompt_profile, images=[object()])

    assert SENTINEL in user_prompt[0]


@pytest.mark.anyio
@pytest.mark.parametrize("prompt_profile", sorted(_COVERED))
async def test_a_covered_profile_qualifies_a_nested_description(
    user_prompt: list[str], prompt_profile: str
) -> None:
    """A nested name can repeat with a different meaning.

    ``title`` at the top of a resume is the headline role; ``experience[].title``
    is the role on one mission. Unqualified, the two descriptions collide and the
    model gets a contradiction instead of a distinction.
    """
    await _run(prompt_profile, images=[object()])

    assert NESTED_SENTINEL in user_prompt[0]


@pytest.mark.anyio
async def test_a_schema_with_no_descriptions_leaves_the_trained_format_alone(
    user_prompt: list[str],
) -> None:
    """Nothing to say, nothing appended.

    The key skeleton and the inline key list are what these checkpoints were
    trained on. Adding an empty annotation block to every prompt would trade a
    known-good framing for noise on the schemas that never needed it.
    """
    service = ExtractionService(
        ModelProfile(
            name="probe",
            model="probe",
            base_url="http://node:8088/v1",
            api_key="k",
            prompt_profile="minicpm_kie_v1",
            vision=True,
        )
    )
    await service._extract_group(
        blocks=[],
        images=[object()],
        image_urls=["data:image/png;base64,AA"],
        schema_name="probe",
        generation_schema={"type": "object", "properties": {"total": {"type": "string"}}},
        nuextract_template=None,
        language=None,
        metadata=None,
        field_names=None,
        want_logprobs=False,
    )

    assert "What each field means" not in user_prompt[0]
