"""Routing policies on the LIVE extraction routes (/v1/extract/text|file).

The ExtractionRouter (confidence-gated escalation across profiles, budgets,
audit) existed and was tested -- but only the benchmark could drive it. These
tests pin the seam that makes a saved policy USABLE on a real document:
``routing_policy`` on the request -> ``resolve_extraction_executor`` builds a
router whose stages resolve through the same ``resolve_profile`` a single-
model request uses -> the audit lands in ``response.routing`` (per-stage
``output`` stripped) -> ``finalize_response`` runs unchanged.

Model calls are faked at the ExtractionService boundary (each stage's
``extract_from_text`` returns a canned response), so no model is loaded and
the router's own decision logic is exercised for real.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from docie_bench import api
from docie_bench.extract.routing import (
    RouteDecision,
    RoutingPolicy,
    RoutingRule,
    RuleCondition,
    StagePolicy,
)
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation, Usage


def _response(profile: str, *, confidence: float) -> ExtractionResponse:
    return ExtractionResponse(
        request_id=f"request-{profile}",
        schema_name="invoice",
        model_profile=profile,
        document_hash=None,
        result={
            "invoice_number": {"value": "INV-1", "confidence": confidence, "evidence_ids": ["b1"]}
        },
        validation=ExtractionValidation(valid=True),
        usage=Usage(total_tokens=10),
        latency_ms=1,
    )


def _accept_valid(min_confidence: float | None = None) -> RoutingRule:
    return RoutingRule(
        when=RuleCondition(status="success", validation_valid=True, min_confidence=min_confidence),
        decision=RouteDecision.ACCEPT,
        reason="valid response accepted",
    )


# A cheap-first-then-strong policy: accept the cheap stage only if confident,
# else escalate to the strong one, which accepts anything valid.
ESCALATION_POLICY = RoutingPolicy(
    stages=[
        StagePolicy(name="cheap", rules=[_accept_valid(0.8)]),
        StagePolicy(name="strong", rules=[_accept_valid()]),
    ]
)


class _FakeService:
    """Stands in for ExtractionService: returns the canned response for its
    profile, recording the kwargs it was called with."""

    calls: dict[str, list[dict[str, Any]]] = {}
    responses: dict[str, ExtractionResponse] = {}

    def __init__(self, profile: ModelProfile, **_: Any) -> None:
        self.profile = profile

    async def extract_from_text(self, **kwargs: Any) -> ExtractionResponse:
        _FakeService.calls.setdefault(self.profile.name, []).append(kwargs)
        return _FakeService.responses[self.profile.name]

    async def extract_from_file(self, **kwargs: Any) -> ExtractionResponse:
        return await self.extract_from_text(**kwargs)


def _profile(name: str) -> ModelProfile:
    return ModelProfile(name=name, base_url="http://fake", model=name, api_key="x")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _FakeService.calls = {}
    _FakeService.responses = {
        "cheap": _response("cheap", confidence=0.4),  # below the 0.8 gate -> escalate
        "strong": _response("strong", confidence=0.95),
    }

    async def fake_resolve_profile(name: str | None) -> ModelProfile:
        return _profile(name or "studio_default")

    monkeypatch.setattr(api, "resolve_profile", fake_resolve_profile)
    monkeypatch.setattr(api, "ExtractionService", _FakeService)
    # build_extraction_router constructs ExtractionServiceStage(ExtractionService(...))
    # from its own module's import, so patch it there too.
    import docie_bench.benchmark.routing_config as routing_config

    monkeypatch.setattr(routing_config, "ExtractionService", _FakeService)
    monkeypatch.setattr(
        api,
        "get_routing_policy",
        lambda name: {"name": name, "policy": ESCALATION_POLICY.model_dump(mode="json")}
        if name == "cheap-then-strong"
        else None,
    )
    monkeypatch.setattr(api, "record_extraction", lambda *a, **k: None)
    monkeypatch.setattr(api.recency, "stamp_served_profile", lambda *a, **k: None)
    return TestClient(api.app)


def test_routing_policy_escalates_and_the_audit_rides_the_response(client: TestClient) -> None:
    resp = client.post(
        "/v1/extract/text",
        json={
            "text": "FACTURE ...",
            "schema_name": "invoice",
            "routing_policy": "cheap-then-strong",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The strong stage answered: cheap was tried first, fell below 0.8, escalated.
    assert body["model_profile"] == "strong"
    assert list(_FakeService.calls) == ["cheap", "strong"]
    audit = body["routing"]
    assert audit["policy"] == "cheap-then-strong"
    assert audit["selected_stage"] == "strong"
    assert audit["terminal_decision"] == "accept"
    assert audit["attempts"] == 2
    assert [s["stage"] for s in audit["stages"]] == ["cheap", "strong"]
    assert audit["stages"][0]["decision"] != "accept"  # the escalation is visible


def test_live_audit_strips_per_stage_output(client: TestClient) -> None:
    # The router stamps every stage's raw extraction (``output``) into the
    # audit -- fine for the benchmark, which scores every attempt. On the LIVE
    # surface that leaks the losing stages' extraction of a confidential
    # document the caller never asked for; the winner's is already ``result``.
    resp = client.post(
        "/v1/extract/text",
        json={
            "text": "FACTURE ...",
            "schema_name": "invoice",
            "routing_policy": "cheap-then-strong",
        },
    )
    assert resp.status_code == 200
    for stage in resp.json()["routing"]["stages"]:
        assert "output" not in stage
    # ...but everything a caller needs to understand the routing is still there.
    stage0 = resp.json()["routing"]["stages"][0]
    expected_keys = {"stage", "decision", "reason", "avg_confidence", "latency_ms", "total_tokens"}
    assert expected_keys <= set(stage0)


def test_no_routing_policy_is_the_unchanged_single_model_path(client: TestClient) -> None:
    resp = client.post(
        "/v1/extract/text",
        json={"text": "FACTURE ...", "schema_name": "invoice", "model_profile": "cheap"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_profile"] == "cheap"
    assert body["routing"] is None  # no router involved -> no audit
    assert list(_FakeService.calls) == ["cheap"]


def test_routing_policy_and_model_profile_are_mutually_exclusive(client: TestClient) -> None:
    resp = client.post(
        "/v1/extract/text",
        json={
            "text": "x",
            "schema_name": "invoice",
            "model_profile": "cheap",
            "routing_policy": "cheap-then-strong",
        },
    )
    assert resp.status_code == 400
    assert "mutually exclusive" in resp.json()["detail"]
    assert _FakeService.calls == {}  # rejected before any model call


def test_unknown_routing_policy_is_a_404_that_says_where_to_save_one(client: TestClient) -> None:
    resp = client.post(
        "/v1/extract/text",
        json={"text": "x", "schema_name": "invoice", "routing_policy": "does-not-exist"},
    )
    assert resp.status_code == 404
    assert "routing-policies" in resp.json()["detail"]


def test_every_stage_profile_is_resolved_up_front(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A policy whose escalation target can't be resolved must fail at request
    # time -- not on the one document that finally needed the strong stage.
    from fastapi import HTTPException

    async def resolve_only_cheap(name: str | None) -> ModelProfile:
        if name == "strong":
            raise HTTPException(status_code=400, detail="unknown model profile 'strong'")
        return _profile(name or "studio_default")

    monkeypatch.setattr(api, "resolve_profile", resolve_only_cheap)
    resp = client.post(
        "/v1/extract/text",
        json={"text": "x", "schema_name": "invoice", "routing_policy": "cheap-then-strong"},
    )
    assert resp.status_code == 400
    assert "strong" in resp.json()["detail"]
    assert _FakeService.calls == {}  # nothing was extracted


def test_router_with_no_answer_is_a_502_with_the_terminal_reason(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every stage errors -> RoutingResult.response is None. That's an honest
    # upstream failure with the router's own reason, not a 500.
    class _Boom(_FakeService):
        async def extract_from_text(self, **kwargs: Any) -> ExtractionResponse:
            raise RuntimeError("model exploded")

    import docie_bench.benchmark.routing_config as routing_config

    monkeypatch.setattr(routing_config, "ExtractionService", _Boom)
    resp = client.post(
        "/v1/extract/text",
        json={"text": "x", "schema_name": "invoice", "routing_policy": "cheap-then-strong"},
    )
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "cheap-then-strong" in detail
    assert "attempts=2" in detail


# ---------------------------------------------------------------------------
# The Studio worker path (POST /v1/studio/extract -> Inngest -> _run_extraction).
# This is the path the Playground actually uses, so a policy has to work HERE
# for the feature to exist in the Studio, not only via curl.
# ---------------------------------------------------------------------------

from docie_bench.inngest import functions  # noqa: E402


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    _FakeService.calls = {}
    _FakeService.responses = {
        "cheap": _response("cheap", confidence=0.4),
        "strong": _response("strong", confidence=0.95),
    }
    stamped: list[str] = []
    monkeypatch.setattr(functions, "ExtractionService", _FakeService)
    import docie_bench.benchmark.routing_config as routing_config

    monkeypatch.setattr(routing_config, "ExtractionService", _FakeService)
    monkeypatch.setattr(
        functions, "_resolve_profile", lambda model_profile=None, deployment=None: _profile(
            model_profile or deployment or "studio_default"
        )
    )
    monkeypatch.setattr(
        "docie_bench.studio.routing_policies.get_routing_policy",
        lambda name: {"name": name, "policy": ESCALATION_POLICY.model_dump(mode="json")}
        if name == "cheap-then-strong"
        else None,
    )
    monkeypatch.setattr(functions, "_record_observability", lambda *a, **k: None)
    monkeypatch.setattr(
        functions,
        "_stamp_deployment_recency",
        lambda *, explicit=None, profile_name=None: stamped.append(profile_name),
    )
    return {"stamped": stamped}


@pytest.mark.asyncio
async def test_worker_runs_a_policy_and_returns_the_same_shape_as_the_api(
    worker: dict[str, Any],
) -> None:
    result = await functions._run_extraction(
        {"text": "FACTURE ...", "schema_name": "invoice", "routing_policy": "cheap-then-strong"}
    )
    assert result["model_profile"] == "strong"
    assert list(_FakeService.calls) == ["cheap", "strong"]
    audit = result["routing"]
    assert audit["policy"] == "cheap-then-strong"
    assert audit["selected_stage"] == "strong"
    assert audit["attempts"] == 2
    for stage in audit["stages"]:
        assert "output" not in stage  # same live-audit shaping as the API path


@pytest.mark.asyncio
async def test_worker_stamps_recency_on_the_stage_that_answered(worker: dict[str, Any]) -> None:
    # The escalation target only ever runs on hard documents; if recency were
    # stamped on the first stage it would read idle and be the first eviction
    # victim, evicting exactly the model the policy needs for the hard cases.
    await functions._run_extraction(
        {"text": "x", "schema_name": "invoice", "routing_policy": "cheap-then-strong"}
    )
    assert worker["stamped"] == ["strong"]


@pytest.mark.asyncio
async def test_worker_single_model_event_is_unchanged(worker: dict[str, Any]) -> None:
    result = await functions._run_extraction(
        {"text": "x", "schema_name": "invoice", "model_profile": "cheap"}
    )
    assert result["model_profile"] == "cheap"
    assert result["routing"] is None
    assert worker["stamped"] == ["cheap"]


@pytest.mark.asyncio
async def test_worker_rejects_policy_plus_selector(worker: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        await functions._run_extraction(
            {"text": "x", "routing_policy": "cheap-then-strong", "deployment": "some-dep"}
        )
    assert _FakeService.calls == {}


def test_studio_extract_route_fails_fast_on_a_bad_policy_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bad selector must be a 4xx at the API edge, not a failed Inngest run
    # the caller discovers by polling. (The worker re-checks; this saves the
    # round-trip.)
    monkeypatch.setattr(
        "docie_bench.studio.routing_policies.get_routing_policy",
        lambda name: {"name": name, "policy": {}} if name == "exists" else None,
    )
    c = TestClient(api.app)
    both = c.post(
        "/v1/studio/extract",
        json={"text": "x", "routing_policy": "exists", "model_profile": "cheap"},
    )
    assert both.status_code == 400
    assert "mutually exclusive" in both.json()["detail"]
    missing = c.post("/v1/studio/extract", json={"text": "x", "routing_policy": "nope"})
    assert missing.status_code == 404
    assert "routing-policies" in missing.json()["detail"]
