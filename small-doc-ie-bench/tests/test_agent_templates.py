"""The agent catalog's templates.

The guard that matters here is not that a template exists but that it is
HONEST: every option it prefills has to be one the runtime actually reads. A
template offering a knob nothing consumes looks like a feature and silently
does nothing, which is worse than omitting it.
"""

from __future__ import annotations

import pytest

from docie_bench.agents.templates import AGENT_TEMPLATES, template_by_id

LOW_LATENCY_IDS = ("low-latency-extraction", "document-triage", "low-latency-pii-guard")

# The kinds the runtime dispatches on (agents/runtime.py). A template naming
# anything else would render with a fallback icon and then fail to run.
KNOWN_KINDS = {"ocr", "proxy_security", "custom", "workflow"}


def test_ids_are_unique() -> None:
    ids = [template["id"] for template in AGENT_TEMPLATES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("template", AGENT_TEMPLATES, ids=lambda t: str(t["id"]))
def test_every_template_is_well_formed(template: dict) -> None:
    assert template["kind"] in KNOWN_KINDS
    assert template["display_name"].strip()
    assert template["description"].strip()
    assert "options" in template["defaults"]
    assert isinstance(template["defaults"]["options"], dict)


@pytest.mark.parametrize("template_id", LOW_LATENCY_IDS)
def test_the_encoder_templates_are_reachable_by_id(template_id: str) -> None:
    assert template_by_id(template_id) is not None


def test_low_latency_extraction_prefills_only_options_the_ocr_path_reads() -> None:
    options = template_by_id("low-latency-extraction")["defaults"]["options"]

    # These four are read by the ocr kind in agents/runtime.py. `extractor`
    # plus `schema` is what turns OCR into structured extraction; without both
    # the agent returns text and the template would be misleading.
    assert set(options) == {"backend", "language", "extractor", "schema"}
    assert options["schema"] == "invoice"
    # Empty, not a made-up deployment name: the create form should require the
    # operator to pick an encoder that exists on this node.
    assert options["extractor"] == ""


def test_low_latency_extraction_is_the_ocr_kind() -> None:
    # Not `custom`: the extraction pipeline lives behind the ocr kind, which
    # is what reads `extractor`/`schema`. A custom agent would ignore both.
    assert template_by_id("low-latency-extraction")["kind"] == "ocr"


def test_pii_guard_prefills_the_encoder_analyzer_knobs() -> None:
    options = template_by_id("low-latency-pii-guard")["defaults"]["options"]
    baseline = template_by_id("proxy-security")["defaults"]["options"]

    # Same option surface as the generic security proxy -- this template is
    # that one pointed at an encoder, not a different feature.
    assert set(options) == set(baseline)
    # The difference: it prompts for a guard model instead of defaulting to
    # the regex analyzer.
    assert options["guard_model"] == ""
    assert baseline["guard_model"] is None


def test_triage_is_a_plain_custom_agent() -> None:
    template = template_by_id("document-triage")

    # Classification happens in the encoder shim from the request's labels;
    # there is no agent-side option for it, so prefilling one would be a lie.
    assert template["kind"] == "custom"
    assert template["defaults"]["options"] == {}


def test_descriptions_state_the_span_limitation() -> None:
    # The single most expensive misunderstanding about these models is that
    # they can compute or normalise. The extraction template has to say
    # otherwise, or an operator will point it at a total and trust the answer.
    description = template_by_id("low-latency-extraction")["description"].lower()

    assert "span" in description
    assert "computed" in description or "normalised" in description
