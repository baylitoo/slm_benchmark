from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from docie_bench.extract.routing import (
    ExtractionRouter,
    ExtractionServiceStage,
    RouteDecision,
    RoutingBudget,
    RoutingPolicy,
    RoutingRequest,
    RoutingRule,
    RuleCondition,
    StageEvaluation,
    StagePolicy,
    StageResult,
    StageSelector,
)
from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation, Usage

_UNSET = object()


def _response(
    profile: str,
    *,
    valid: bool = True,
    confidence: float = 0.9,
    model_confidence: Any = _UNSET,
    warnings: list[str] | None = None,
    tokens: int = 10,
) -> ExtractionResponse:
    field: dict[str, Any] = {
        "value": "INV-1",
        "confidence": confidence,
        "evidence_ids": ["b1"],
    }
    if model_confidence is not _UNSET:
        field["model_confidence"] = model_confidence
    return ExtractionResponse(
        request_id=f"request-{profile}",
        schema_name="invoice",
        model_profile=profile,
        document_hash=None,
        result={"invoice_number": field},
        validation=ExtractionValidation(valid=valid, warnings=warnings or []),
        usage=Usage(total_tokens=tokens),
        latency_ms=1,
    )


class FakeStage:
    def __init__(
        self,
        name: str,
        *,
        response: ExtractionResponse | None = None,
        error: Exception | None = None,
        cost_units: float = 0.0,
        equivalence_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.equivalence_key = equivalence_key or name
        self.response = response
        self.error = error
        self.cost_units = cost_units
        self.metadata = metadata or {"fake": name}
        self.requests: list[RoutingRequest] = []

    async def execute(self, request: RoutingRequest) -> StageResult:
        self.requests.append(request)
        if self.error:
            raise self.error
        assert self.response is not None
        return StageResult(
            response=self.response,
            cost_units=self.cost_units,
            metadata=self.metadata,
        )


def _accept_valid(min_confidence: float | None = None) -> RoutingRule:
    return RoutingRule(
        when=RuleCondition(
            status="success",
            validation_valid=True,
            min_confidence=min_confidence,
        ),
        decision=RouteDecision.ACCEPT,
        reason="valid response accepted",
    )


@pytest.mark.asyncio
async def test_router_accepts_first_matching_stage_and_attaches_audit() -> None:
    primary = FakeStage("primary", response=_response("primary"))
    fallback = FakeStage("fallback", response=_response("fallback"))
    router = ExtractionRouter(
        stages=[primary, fallback],
        policy=RoutingPolicy(
            stages=[
                StagePolicy(name="primary", rules=[_accept_valid(0.8)]),
                StagePolicy(name="fallback", rules=[_accept_valid()]),
            ]
        ),
    )

    result = await router.extract_from_text(text="invoice", ocr_blocks=None, schema_name="invoice")

    assert result.response is not None
    assert result.response.model_profile == "primary"
    assert result.audit.terminal_decision is RouteDecision.ACCEPT
    assert result.audit.selected_stage == "primary"
    assert result.audit.policy_version == "1"
    assert result.audit.attempts == 1
    assert result.audit.stages[0].output == primary.response.result
    assert result.response.routing == result.audit.model_dump(mode="json")
    assert not fallback.requests


@pytest.mark.asyncio
async def test_low_confidence_response_falls_back_to_next_stage() -> None:
    primary = FakeStage("primary", response=_response("primary", confidence=0.4))
    fallback = FakeStage("fallback", response=_response("fallback", confidence=0.95))
    router = ExtractionRouter(
        stages=[primary, fallback],
        policy=RoutingPolicy(
            stages=[
                StagePolicy(
                    name="primary",
                    rules=[_accept_valid(0.8)],
                    default_reason="primary quality gate failed",
                ),
                StagePolicy(name="fallback", rules=[_accept_valid(0.8)]),
            ]
        ),
    )

    result = await router.extract_from_text(text="invoice", ocr_blocks=None, schema_name="invoice")

    assert result.response is not None
    assert result.response.model_profile == "fallback"
    assert result.audit.fallback_count == 1
    assert [stage.decision for stage in result.audit.stages] == [
        RouteDecision.FALLBACK,
        RouteDecision.ACCEPT,
    ]
    assert result.audit.stages[0].avg_confidence == 0.4


@pytest.mark.asyncio
async def test_stage_exception_is_audited_and_can_fallback() -> None:
    primary = FakeStage("primary", error=RuntimeError("model unavailable"))
    fallback = FakeStage("fallback", response=_response("fallback"))
    router = ExtractionRouter(
        stages=[primary, fallback],
        policy=RoutingPolicy(
            stages=[
                StagePolicy(
                    name="primary",
                    rules=[
                        RoutingRule(
                            when=RuleCondition(status="error"),
                            decision=RouteDecision.FALLBACK,
                            reason="recover from stage error",
                        )
                    ],
                ),
                StagePolicy(name="fallback", rules=[_accept_valid()]),
            ]
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.audit.terminal_decision is RouteDecision.ACCEPT
    assert result.audit.stages[0].status == "error"
    assert result.audit.stages[0].error == "RuntimeError('model unavailable')"


@pytest.mark.asyncio
async def test_stage_error_cannot_be_accepted_without_a_response() -> None:
    stage = FakeStage("primary", error=RuntimeError("failed"))
    router = ExtractionRouter(
        stages=[stage],
        policy=RoutingPolicy(
            stages=[
                StagePolicy(
                    name="primary",
                    rules=[
                        RoutingRule(
                            when=RuleCondition(status="error"),
                            decision=RouteDecision.ACCEPT,
                            reason="unsafe policy",
                        )
                    ],
                )
            ]
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.response is None
    assert result.audit.terminal_decision is RouteDecision.FAIL
    assert result.audit.terminal_reason == "stage cannot accept without a response"


@pytest.mark.asyncio
async def test_invalid_response_can_trigger_immediate_escalation() -> None:
    stage = FakeStage("primary", response=_response("primary", valid=False))
    router = ExtractionRouter(
        stages=[stage],
        policy=RoutingPolicy(
            stages=[
                StagePolicy(
                    name="primary",
                    rules=[
                        RoutingRule(
                            when=RuleCondition(status="success", validation_valid=False),
                            decision=RouteDecision.ESCALATE,
                            reason="invalid extraction requires review",
                        )
                    ],
                )
            ]
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.response is not None
    assert result.audit.terminal_decision is RouteDecision.ESCALATE
    assert result.audit.terminal_reason == "invalid extraction requires review"
    assert result.audit.selected_stage is None


@pytest.mark.asyncio
async def test_attempt_budget_stops_before_next_fallback_stage() -> None:
    primary = FakeStage("primary", response=_response("primary", confidence=0.2))
    fallback = FakeStage("fallback", response=_response("fallback"))
    router = ExtractionRouter(
        stages=[primary, fallback],
        policy=RoutingPolicy(
            stages=[StagePolicy(name="primary"), StagePolicy(name="fallback")],
            budget=RoutingBudget(max_stages=1),
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.audit.terminal_decision is RouteDecision.ESCALATE
    assert result.audit.budget_exhausted is True
    assert result.audit.terminal_reason == "max_stages budget exhausted"
    assert result.audit.attempts == 1
    assert not fallback.requests


@pytest.mark.asyncio
async def test_zero_latency_budget_stops_before_first_stage() -> None:
    stage = FakeStage("primary", response=_response("primary"))
    router = ExtractionRouter(
        stages=[stage],
        policy=RoutingPolicy(
            stages=[StagePolicy(name="primary", rules=[_accept_valid()])],
            budget=RoutingBudget(max_latency_ms=0),
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.response is None
    assert result.audit.budget_exhausted is True
    assert result.audit.terminal_reason == "max_latency_ms budget exhausted"
    assert not stage.requests


@pytest.mark.asyncio
async def test_request_context_selects_route_and_records_skipped_stages() -> None:
    text_stage = FakeStage("text", response=_response("text"))
    vision_stage = FakeStage("vision", response=_response("vision"))
    router = ExtractionRouter(
        stages=[text_stage, vision_stage],
        policy=RoutingPolicy(
            version="2026-06",
            stages=[
                StagePolicy(
                    name="text",
                    selector=StageSelector(context_min={"ocr_confidence": 0.8}),
                    rules=[_accept_valid()],
                ),
                StagePolicy(
                    name="vision",
                    selector=StageSelector(
                        context_equals={"file_suffix": ".pdf", "language": "fr"}
                    ),
                    rules=[_accept_valid()],
                ),
            ],
        ),
    )

    result = await router.extract_from_file(
        path=Path("scan.pdf"),
        ocr_backend_name="pdf_text",
        schema_name="invoice",
        language="fr",
    )

    assert result.response is not None
    assert result.response.model_profile == "vision"
    assert result.audit.policy_version == "2026-06"
    assert result.audit.skipped_stages == ["text"]
    assert not text_stage.requests


@pytest.mark.asyncio
async def test_prior_stage_result_can_select_escalation_stage() -> None:
    primary = FakeStage(
        "primary",
        response=_response("primary", valid=False),
        metadata={"ocr_confidence": 0.4},
    )
    review = FakeStage("review", response=_response("review"))
    router = ExtractionRouter(
        stages=[primary, review],
        policy=RoutingPolicy(
            stages=[
                StagePolicy(name="primary"),
                StagePolicy(
                    name="review",
                    selector=StageSelector(
                        requires_fallback=True,
                        prior_stage="primary",
                        prior_status="success",
                        prior_validation_valid=False,
                        prior_metadata_max={"ocr_confidence": 0.5},
                    ),
                    rules=[_accept_valid()],
                ),
            ]
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.response is not None
    assert result.response.model_profile == "review"
    assert result.audit.fallback_count == 1


@pytest.mark.asyncio
async def test_non_extraction_stage_can_feed_output_and_metadata_to_fallback() -> None:
    class FakeOcrStage:
        name = "ocr"
        equivalence_key = "ocr"

        async def execute(self, request: RoutingRequest) -> StageResult:
            return StageResult(
                output={"blocks": ["low-quality text"]},
                metadata={"ocr_confidence": 0.3},
            )

    extractor = FakeStage("extractor", response=_response("extractor"))
    router = ExtractionRouter(
        stages=[FakeOcrStage(), extractor],
        policy=RoutingPolicy(
            stages=[
                StagePolicy(name="ocr"),
                StagePolicy(
                    name="extractor",
                    selector=StageSelector(prior_metadata_max={"ocr_confidence": 0.5}),
                    rules=[_accept_valid()],
                ),
            ]
        ),
    )

    result = await router.route(RoutingRequest(operation="file", arguments={}))

    assert result.response is not None
    assert result.response.model_profile == "extractor"
    assert result.audit.stages[0].output == {"blocks": ["low-quality text"]}
    assert result.audit.stages[0].validation_valid is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget", "reason"),
    [
        (RoutingBudget(max_total_tokens=5), "max_total_tokens budget exhausted"),
        (RoutingBudget(max_cost_units=0.5), "max_cost_units budget exhausted"),
    ],
)
async def test_consumption_budget_overrides_stage_acceptance(
    budget: RoutingBudget,
    reason: str,
) -> None:
    stage = FakeStage("primary", response=_response("primary", tokens=10), cost_units=1.0)
    router = ExtractionRouter(
        stages=[stage],
        policy=RoutingPolicy(
            stages=[StagePolicy(name="primary", rules=[_accept_valid()])],
            budget=budget,
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.audit.terminal_decision is RouteDecision.ESCALATE
    assert result.audit.budget_exhausted is True
    assert result.audit.terminal_reason == reason
    assert result.audit.stages[0].decision is RouteDecision.ESCALATE


@pytest.mark.asyncio
async def test_request_budget_stops_retries_deterministically() -> None:
    primary = FakeStage("primary", response=_response("primary", valid=False))
    fallback = FakeStage("fallback", response=_response("fallback"))
    router = ExtractionRouter(
        stages=[primary, fallback],
        policy=RoutingPolicy(
            stages=[StagePolicy(name="primary"), StagePolicy(name="fallback")],
            budget=RoutingBudget(max_requests=1),
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.audit.terminal_reason == "max_requests budget exhausted"
    assert result.audit.attempts == 1
    assert not fallback.requests


@pytest.mark.asyncio
async def test_policy_exhaustion_returns_last_response_for_auditability() -> None:
    stage = FakeStage("primary", response=_response("primary", valid=False))
    router = ExtractionRouter(
        stages=[stage],
        policy=RoutingPolicy(
            stages=[StagePolicy(name="primary")],
            exhausted_decision=RouteDecision.FAIL,
            exhausted_reason="no usable extraction",
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.response is not None
    assert result.response.model_profile == "primary"
    assert result.audit.terminal_decision is RouteDecision.FAIL
    assert result.audit.terminal_reason == "no usable extraction"


@pytest.mark.asyncio
async def test_extraction_service_stage_dispatches_existing_service_contract() -> None:
    class FakeService:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []
            self.profile = SimpleNamespace(name="service")

        async def extract_from_text(self, **kwargs: Any) -> ExtractionResponse:
            self.calls.append(("text", kwargs))
            return _response("service")

        async def extract_from_file(self, **kwargs: Any) -> ExtractionResponse:
            self.calls.append(("file", kwargs))
            return _response("service")

    service = FakeService()
    stage = ExtractionServiceStage("service", service)  # type: ignore[arg-type]

    result = await stage.execute(
        RoutingRequest(operation="text", arguments={"text": "invoice", "schema_name": "invoice"})
    )

    assert result.response.model_profile == "service"
    assert service.calls == [("text", {"text": "invoice", "schema_name": "invoice"})]


def test_policy_rejects_empty_duplicate_and_unregistered_stages() -> None:
    with pytest.raises(ValidationError, match="at least one stage"):
        RoutingPolicy(stages=[])
    with pytest.raises(ValidationError, match="must be unique"):
        RoutingPolicy(stages=[StagePolicy(name="same"), StagePolicy(name="same")])
    with pytest.raises(ValidationError, match="equivalent stages"):
        RoutingPolicy(
            stages=[
                StagePolicy(name="first", equivalence_key="same"),
                StagePolicy(name="second", equivalence_key="same"),
            ]
        )
    with pytest.raises(ValueError, match="unregistered stages"):
        ExtractionRouter(
            stages=[],
            policy=RoutingPolicy(stages=[StagePolicy(name="missing")]),
        )
    with pytest.raises(ValueError, match="Registered routing stage names must be unique"):
        ExtractionRouter(
            stages=[FakeStage("same"), FakeStage("same")],
            policy=RoutingPolicy(stages=[StagePolicy(name="same")]),
        )
    with pytest.raises(ValueError, match="equivalent registered stages"):
        ExtractionRouter(
            stages=[
                FakeStage("first", equivalence_key="same"),
                FakeStage("second", equivalence_key="same"),
            ],
            policy=RoutingPolicy(
                stages=[StagePolicy(name="first"), StagePolicy(name="second")]
            ),
        )


def test_policy_can_be_loaded_from_declarative_data() -> None:
    policy = RoutingPolicy.model_validate(
        {
            "stages": [
                {
                    "name": "primary",
                    "rules": [
                        {
                            "when": {"status": "success", "min_confidence": 0.8},
                            "decision": "accept",
                            "reason": "quality threshold met",
                        }
                    ],
                }
            ],
            "budget": {"max_stages": 1, "max_total_tokens": 100},
        }
    )

    assert policy.stages[0].rules[0].decision is RouteDecision.ACCEPT
    assert policy.budget.max_total_tokens == 100


def test_rule_condition_matches_on_model_confidence_alone() -> None:
    condition = RuleCondition(min_model_confidence=-0.5)
    high = StageEvaluation(status="success", avg_model_confidence=-0.1)
    low = StageEvaluation(status="success", avg_model_confidence=-0.9)
    unknown = StageEvaluation(status="success", avg_model_confidence=None)

    assert condition.matches(high)
    assert not condition.matches(low)
    # Same conservative treatment `min_confidence` already gives an unknown
    # grounding confidence: an unset signal never satisfies a threshold.
    assert not condition.matches(unknown)


def test_rule_condition_with_both_thresholds_requires_both_to_pass() -> None:
    condition = RuleCondition(min_confidence=0.8, min_model_confidence=-0.5)

    # Grounding confidence is fine but the model's own confidence is low:
    # the AND-combination correctly fails, proving "escalate if EITHER
    # signal is too low" falls out of RuleCondition.matches()'s existing
    # every-set-predicate-must-pass semantics with no new combinator.
    assert not condition.matches(
        StageEvaluation(status="success", avg_confidence=0.95, avg_model_confidence=-0.9)
    )
    # And the reverse: model confidence is fine but grounding is low.
    assert not condition.matches(
        StageEvaluation(status="success", avg_confidence=0.2, avg_model_confidence=-0.1)
    )
    # Only when both clear their threshold does the rule match.
    assert condition.matches(
        StageEvaluation(status="success", avg_confidence=0.95, avg_model_confidence=-0.1)
    )


def test_min_model_confidence_rejects_positive_and_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        RuleCondition(min_model_confidence=0.5)
    # 0.0 itself is a valid (if degenerate) log-probability.
    RuleCondition(min_model_confidence=0.0)


@pytest.mark.asyncio
async def test_avg_model_confidence_is_none_when_result_carries_no_signal() -> None:
    """Always-computed, not rule-gated: cheap tree walk, same as `avg_confidence`.

    A response whose fields never got a `model_confidence` key at all (the
    opt-in logprob feature was never used for this request) resolves to
    `avg_model_confidence=None` regardless of whether any rule asked for it.
    """
    stage = FakeStage("primary", response=_response("primary"))
    router = ExtractionRouter(
        stages=[stage],
        policy=RoutingPolicy(stages=[StagePolicy(name="primary", rules=[_accept_valid()])]),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.audit.stages[0].avg_model_confidence is None


@pytest.mark.asyncio
async def test_low_model_confidence_escalates_even_when_grounded() -> None:
    # Well-grounded (confidence=0.99) but the model itself was unsure
    # (model_confidence=-3.5, below the -1.0 gate): the accept rule requires
    # BOTH signals, so this falls through to the stage's default decision.
    stage = FakeStage(
        "primary", response=_response("primary", confidence=0.99, model_confidence=-3.5)
    )
    router = ExtractionRouter(
        stages=[stage],
        policy=RoutingPolicy(
            stages=[
                StagePolicy(
                    name="primary",
                    rules=[
                        RoutingRule(
                            when=RuleCondition(
                                status="success",
                                validation_valid=True,
                                min_model_confidence=-1.0,
                            ),
                            decision=RouteDecision.ACCEPT,
                            reason="both signals clear",
                        )
                    ],
                    default_decision=RouteDecision.ESCALATE,
                    default_reason="model was unsure despite being grounded",
                )
            ]
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.audit.terminal_decision is RouteDecision.ESCALATE
    assert result.audit.stages[0].avg_confidence == 0.99
    assert result.audit.stages[0].avg_model_confidence == -3.5


@pytest.mark.asyncio
async def test_all_nested_schema_falls_through_to_escalation() -> None:
    """Documented-not-a-bug case (#345): a schema whose fields are all
    nested/list-valued always resolves `model_confidence: None` per #335's
    coverage (only top-level scalars are resolved). A rule requiring
    `min_model_confidence` predictably never matches such a schema, and the
    stage falls through to its default decision -- this is expected, not a
    regression, until #335's nested-field coverage expands.
    """
    response = ExtractionResponse(
        request_id="request-nested",
        schema_name="invoice",
        model_profile="primary",
        document_hash=None,
        result={
            "total": {
                "value": {"amount": "100.00", "currency": "EUR"},
                "confidence": 0.9,
                "evidence_ids": ["b1"],
                "model_confidence": None,
            },
            "line_items": [
                {
                    "value": "widget",
                    "confidence": 0.9,
                    "evidence_ids": ["b2"],
                    "model_confidence": None,
                }
            ],
        },
        validation=ExtractionValidation(valid=True, warnings=[]),
        usage=Usage(total_tokens=10),
        latency_ms=1,
    )
    stage = FakeStage("primary", response=response)
    router = ExtractionRouter(
        stages=[stage],
        policy=RoutingPolicy(
            stages=[
                StagePolicy(
                    name="primary",
                    rules=[
                        RoutingRule(
                            when=RuleCondition(
                                status="success",
                                validation_valid=True,
                                min_model_confidence=-2.0,
                            ),
                            decision=RouteDecision.ACCEPT,
                            reason="both signals clear",
                        )
                    ],
                    default_decision=RouteDecision.ESCALATE,
                    default_reason="model confidence unavailable for this schema",
                )
            ]
        ),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.audit.stages[0].avg_model_confidence is None
    assert result.audit.terminal_decision is RouteDecision.ESCALATE
    assert result.audit.terminal_reason == "model confidence unavailable for this schema"


@pytest.mark.asyncio
async def test_min_confidence_only_rules_are_unaffected_by_model_confidence_field() -> None:
    """Existing grounding-only policies see no behavior change: an
    unset `min_model_confidence` never gates a rule, whether or not the
    response happens to carry `model_confidence` values.
    """
    stage = FakeStage(
        "primary", response=_response("primary", confidence=0.9, model_confidence=-9.9)
    )
    router = ExtractionRouter(
        stages=[stage],
        policy=RoutingPolicy(stages=[StagePolicy(name="primary", rules=[_accept_valid(0.8)])]),
    )

    result = await router.route(RoutingRequest(operation="text", arguments={}))

    assert result.audit.terminal_decision is RouteDecision.ACCEPT
    assert result.audit.stages[0].avg_confidence == 0.9
