"""An extraction says which serving replica answered it (#533 check 2).

A model scaled across a fleet has one placement row per replica, all sharing
`model_name`. The resolved profile deliberately drops the distinction -- `name`
is `store:<base>` and `model` is the llama-server alias, identical for every
replica, because routing to one by its record name would ask for a model the
server does not answer to.

The consequence was that no part of an extraction response could tell which
machine produced it, so #533's "requests spread" check had to be verified at
the node rather than from the response, and a slow reply named no suspect.
"""

from __future__ import annotations

from typing import Any

import pytest

from docie_bench.extract.service import ExtractionService, _served_by
from docie_bench.llm.model_profiles import ModelProfile


def _profile(**overrides: Any) -> ModelProfile:
    return ModelProfile(
        name="store:lfm2.5-350m",
        model="lfm2.5-350m",
        base_url="http://10.8.0.3:8089/v1",
        api_key="local-not-used",
        **overrides,
    )


class _Client:
    last_response_format_style = "json_schema"

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    async def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], None, dict[str, Any]]:
        return {"document_type": "invoice"}, None, {}

    async def aclose(self) -> None:
        return None


# --- the pure part ---------------------------------------------------------


def test_a_resolved_profile_reports_its_replica() -> None:
    assert _served_by(
        _profile(served_by_deployment="lfm2.5-350m-2", served_by_node="ovh-cpu-02")
    ) == {"deployment": "lfm2.5-350m-2", "node": "ovh-cpu-02"}


def test_a_configured_profile_reports_nothing_rather_than_nulls() -> None:
    """A configured profile names one endpoint, so "which replica" does not
    apply to it. Reporting `{"deployment": null}` would read as a measurement
    that failed rather than a question that does not arise."""
    assert _served_by(_profile()) is None


def test_a_replica_with_no_owning_node_still_names_itself() -> None:
    """A single-node stack records no node id, and inventing one would be
    worse than leaving it absent."""
    assert _served_by(_profile(served_by_deployment="lfm2.5-350m")) == {
        "deployment": "lfm2.5-350m",
        "node": None,
    }


# --- the wiring ------------------------------------------------------------


@pytest.mark.asyncio
async def test_served_by_surfaces_on_the_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("docie_bench.extract.service.OpenAICompatibleClient", _Client)
    service = ExtractionService(
        _profile(served_by_deployment="lfm2.5-350m-2", served_by_node="ovh-cpu-02")
    )

    response = await service.extract_from_text(
        text="irrelevant", ocr_blocks=None, schema_name="invoice"
    )

    assert response.served_by == {
        "deployment": "lfm2.5-350m-2",
        "node": "ovh-cpu-02",
    }


@pytest.mark.asyncio
async def test_a_configured_profile_leaves_the_field_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every existing caller: no placement behind the profile, no claim made."""
    monkeypatch.setattr("docie_bench.extract.service.OpenAICompatibleClient", _Client)
    service = ExtractionService(_profile())

    response = await service.extract_from_text(
        text="irrelevant", ocr_blocks=None, schema_name="invoice"
    )

    assert response.served_by is None
