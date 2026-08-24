"""A document-extraction agent's extractor as a saved routing policy.

``options.extractor = "policy:<name>"`` on an ``ocr``-kind agent in
``ocr_extract`` mode runs the named saved policy's confidence-gated cascade
(ExtractionRouter) as the extraction step instead of a single model. These
tests drive the REAL agents API (create agent → chat completion) the same way
``test_agents.py`` does; model calls are faked at the ExtractionService
boundary inside ``docie_bench.benchmark.routing_config`` (where
``build_extraction_router`` constructs each stage's service), the same seam
``test_live_routing_policies.py`` pins for the live extract routes — so the
router's own decision logic runs for real.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docie_bench.agents.api import configure_http_transport
from docie_bench.agents.api import router as agents_router
from docie_bench.extract.routing import (
    RouteDecision,
    RoutingPolicy,
    RoutingRule,
    RuleCondition,
    StagePolicy,
)
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation, Usage

UPSTREAM = ModelProfile(
    name="alpha", model="up-alpha", base_url="http://upstream/v1", api_key="k"
)


def _response(profile: str, *, confidence: float, invoice: str) -> ExtractionResponse:
    return ExtractionResponse(
        request_id=f"request-{profile}",
        schema_name="invoice",
        model_profile=profile,
        document_hash=None,
        result={
            "invoice_number": {
                "value": invoice,
                "confidence": confidence,
                "evidence_ids": ["b1"],
            }
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


# Cheap-first-then-strong: accept the cheap stage only above 0.8 confidence,
# else escalate to the strong one, which accepts anything valid.
ESCALATION_POLICY = RoutingPolicy(
    stages=[
        StagePolicy(name="cheap", rules=[_accept_valid(0.8)]),
        StagePolicy(name="strong", rules=[_accept_valid()]),
    ]
)


class _FakeService:
    """Stands in for ExtractionService inside routing_config: returns the
    canned response for its profile, recording the kwargs it was called with."""

    calls: dict[str, list[dict[str, Any]]] = {}
    responses: dict[str, ExtractionResponse] = {}

    def __init__(self, profile: ModelProfile, **_: Any) -> None:
        self.profile = profile

    async def extract_from_file(self, **kwargs: Any) -> ExtractionResponse:
        _FakeService.calls.setdefault(self.profile.name, []).append(kwargs)
        return _FakeService.responses[self.profile.name]


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    _FakeService.calls = {}
    _FakeService.responses = {
        # Below the policy's 0.8 accept gate -> the router must escalate.
        "cheap": _response("cheap", confidence=0.4, invoice="INV-CHEAP"),
        "strong": _response("strong", confidence=0.95, invoice="INV-STRONG"),
    }

    # _resolve_backing resolves stage names through the shared extraction
    # resolver; every stage resolves to a passthrough profile here so the
    # kind guard passes.
    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        return replace(UPSTREAM, name=str(model_profile), model=str(model_profile))

    monkeypatch.setattr(
        "docie_bench.agents.runtime.resolve_extraction_profile", fake_resolver
    )
    # build_extraction_router constructs ExtractionServiceStage(ExtractionService(...))
    # from its own module's import, so the fake goes THERE.
    import docie_bench.benchmark.routing_config as routing_config

    monkeypatch.setattr(routing_config, "ExtractionService", _FakeService)
    # runtime.py binds get_routing_policy at import time -> patch its binding.
    monkeypatch.setattr(
        "docie_bench.agents.runtime.get_routing_policy",
        lambda name: {"name": name, "policy": ESCALATION_POLICY.model_dump(mode="json")}
        if name == "cheap-then-strong"
        else None,
    )

    # No agent in these tests forwards a chat upstream; fail loudly if one does.
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected upstream call: {request.url}")

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(agents_router)
    client = TestClient(app)
    yield client
    configure_http_transport(None)


def _create_policy_agent(client: TestClient, **option_overrides: object) -> None:
    options: dict[str, object] = {
        "mode": "ocr_extract",
        "extractor": "policy:cheap-then-strong",
        "backend": "liteparse",
        "schema": "invoice",
    }
    options.update(option_overrides)
    created = client.post(
        "/v1/agents",
        json={"name": "doc-routed", "kind": "ocr", "options": options},
    )
    assert created.status_code == 201, created.text


def _chat(client: TestClient) -> httpx.Response:
    return client.post(
        "/v1/agents/doc-routed/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "extract structured data"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                }
            ]
        },
    )


def test_policy_extractor_escalates_and_the_audit_rides_docie_agent(
    client: TestClient,
) -> None:
    _create_policy_agent(client)
    resp = _chat(client)
    assert resp.status_code == 200, resp.text

    # The strong stage answered: cheap was tried first, fell below 0.8, escalated.
    assert list(_FakeService.calls) == ["cheap", "strong"]
    for stage_calls in _FakeService.calls.values():
        assert stage_calls[0]["schema_name"] == "invoice"
        assert stage_calls[0]["ocr_backend_name"] == "liteparse"

    payload = resp.json()
    assert payload["model"] == "policy:cheap-then-strong"
    content = json.loads(payload["choices"][0]["message"]["content"])
    assert content["invoice_number"] == "INV-STRONG"  # the winner's flat result

    routing = payload["docie_agent"]["routing"]
    assert routing["policy"] == "cheap-then-strong"
    assert routing["selected_stage"] == "strong"
    assert routing["attempts"] == 2
    assert [s["stage"] for s in routing["stages"]] == ["cheap", "strong"]
    # Live-audit shaping: no stage may carry its raw extraction ("output") —
    # that would leak the losing stage's read of a confidential document.
    for stage in routing["stages"]:
        assert "output" not in stage


def test_unknown_policy_is_a_4xx_naming_the_policy(client: TestClient) -> None:
    _create_policy_agent(client, extractor="policy:ghost")
    resp = _chat(client)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["type"] == "invalid_agent_config"
    assert "ghost" in error["message"]
    assert "routing-policies" in error["message"]  # says where to save one
    assert _FakeService.calls == {}  # nothing was extracted


def test_policy_extractor_requires_a_schema(client: TestClient) -> None:
    _create_policy_agent(client, schema=None)
    resp = _chat(client)
    assert resp.status_code == 500
    error = resp.json()["error"]
    assert error["type"] == "invalid_agent_config"
    assert "schema" in error["message"]
    assert _FakeService.calls == {}


def test_router_with_no_answer_is_a_502_with_the_terminal_reason(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every stage errors -> RoutingResult.response is None. That's an honest
    # upstream failure carrying the router's own terminal reason, not a 500.
    class _Boom(_FakeService):
        async def extract_from_file(self, **kwargs: Any) -> ExtractionResponse:
            raise RuntimeError("model exploded")

    import docie_bench.benchmark.routing_config as routing_config

    monkeypatch.setattr(routing_config, "ExtractionService", _Boom)
    _create_policy_agent(client)
    resp = _chat(client)
    assert resp.status_code == 502
    error = resp.json()["error"]
    assert error["type"] == "upstream_error"
    assert "cheap-then-strong" in error["message"]
    assert "routing policy exhausted" in error["message"]
    assert "attempts=2" in error["message"]
