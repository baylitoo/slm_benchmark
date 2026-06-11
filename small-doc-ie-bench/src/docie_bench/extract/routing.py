from __future__ import annotations

import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from docie_bench.extract.service import ExtractionService
from docie_bench.schemas.common import ExtractionResponse


class RouteDecision(StrEnum):
    ACCEPT = "accept"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    FAIL = "fail"


class RoutingRequest(BaseModel):
    """Input contract shared by every stage in a route."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation: Literal["text", "file"]
    arguments: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    """Output contract shared by every stage in a route."""

    response: ExtractionResponse | None = None
    output: dict[str, Any] | None = None
    token_usage: int | None = Field(default=None, ge=0)
    cost_units: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionStage(Protocol):
    name: str

    async def execute(self, request: RoutingRequest) -> StageResult: ...


class ExtractionServiceStage:
    """Adapt an existing ExtractionService into a routing stage."""

    def __init__(
        self,
        name: str,
        service: ExtractionService,
        *,
        equivalence_key: str | None = None,
    ) -> None:
        self.name = name
        self.service = service
        self.equivalence_key = equivalence_key or f"extraction-service:{service.profile.name}"

    async def execute(self, request: RoutingRequest) -> StageResult:
        if request.operation == "text":
            response = await self.service.extract_from_text(**request.arguments)
        else:
            response = await self.service.extract_from_file(**request.arguments)
        return StageResult(
            response=response,
            metadata={"model_profile": response.model_profile},
        )


class RuleCondition(BaseModel):
    """All configured predicates must match for a rule to fire."""

    status: Literal["success", "error"] | None = None
    validation_valid: bool | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_warnings: int | None = Field(default=None, ge=0)

    def matches(self, evaluation: StageEvaluation) -> bool:
        if self.status is not None and self.status != evaluation.status:
            return False
        if (
            self.validation_valid is not None
            and self.validation_valid != evaluation.validation_valid
        ):
            return False
        if self.min_confidence is not None and (
            evaluation.avg_confidence is None or evaluation.avg_confidence < self.min_confidence
        ):
            return False
        return self.max_warnings is None or evaluation.warning_count <= self.max_warnings


class RoutingRule(BaseModel):
    when: RuleCondition
    decision: RouteDecision
    reason: str


class StageSelector(BaseModel):
    """Declaratively select a stage from request context and prior-stage results."""

    context_equals: dict[str, Any] = Field(default_factory=dict)
    context_min: dict[str, float] = Field(default_factory=dict)
    context_max: dict[str, float] = Field(default_factory=dict)
    requires_fallback: bool | None = None
    prior_stage: str | None = None
    prior_status: Literal["success", "error"] | None = None
    prior_validation_valid: bool | None = None
    prior_metadata_equals: dict[str, Any] = Field(default_factory=dict)
    prior_metadata_min: dict[str, float] = Field(default_factory=dict)
    prior_metadata_max: dict[str, float] = Field(default_factory=dict)

    def matches(
        self,
        *,
        request: RoutingRequest,
        audits: list[StageAudit],
        fallback_count: int,
    ) -> bool:
        context = _request_context(request)
        if any(context.get(key) != expected for key, expected in self.context_equals.items()):
            return False
        if any(
            not _numeric_at_least(context.get(key), minimum)
            for key, minimum in self.context_min.items()
        ):
            return False
        if any(
            not _numeric_at_most(context.get(key), maximum)
            for key, maximum in self.context_max.items()
        ):
            return False
        if self.requires_fallback is not None and self.requires_fallback != (fallback_count > 0):
            return False
        prior = audits[-1] if audits else None
        if self.prior_stage is not None and (prior is None or prior.stage != self.prior_stage):
            return False
        if self.prior_status is not None and (prior is None or prior.status != self.prior_status):
            return False
        if self.prior_validation_valid is not None and (
            prior is None or prior.validation_valid != self.prior_validation_valid
        ):
            return False
        if prior is None:
            return not (
                self.prior_metadata_equals or self.prior_metadata_min or self.prior_metadata_max
            )
        if any(
            prior.metadata.get(key) != expected
            for key, expected in self.prior_metadata_equals.items()
        ):
            return False
        if any(
            not _numeric_at_least(prior.metadata.get(key), minimum)
            for key, minimum in self.prior_metadata_min.items()
        ):
            return False
        return not any(
            not _numeric_at_most(prior.metadata.get(key), maximum)
            for key, maximum in self.prior_metadata_max.items()
        )


class StagePolicy(BaseModel):
    name: str
    selector: StageSelector = Field(default_factory=StageSelector)
    equivalence_key: str | None = None
    rules: list[RoutingRule] = Field(default_factory=list)
    default_decision: RouteDecision = RouteDecision.FALLBACK
    default_reason: str = "no routing rule matched"


class RoutingBudget(BaseModel):
    max_stages: int | None = Field(default=None, ge=1)
    max_requests: int | None = Field(default=None, ge=1)
    max_latency_ms: int | None = Field(default=None, ge=0)
    max_total_tokens: int | None = Field(default=None, ge=0)
    max_cost_units: float | None = Field(default=None, ge=0.0)


class RoutingPolicy(BaseModel):
    version: str = Field(default="1", min_length=1)
    stages: list[StagePolicy]
    budget: RoutingBudget = Field(default_factory=RoutingBudget)
    exhausted_decision: RouteDecision = RouteDecision.ESCALATE
    exhausted_reason: str = "routing policy exhausted"
    budget_exhausted_decision: RouteDecision = RouteDecision.ESCALATE

    @model_validator(mode="after")
    def validate_stages(self) -> RoutingPolicy:
        if not self.stages:
            raise ValueError("Routing policy must contain at least one stage")
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Routing policy stage names must be unique")
        equivalence_keys = [
            stage.equivalence_key for stage in self.stages if stage.equivalence_key is not None
        ]
        if len(equivalence_keys) != len(set(equivalence_keys)):
            raise ValueError("Routing policy cannot repeat equivalent stages")
        return self


class StageEvaluation(BaseModel):
    status: Literal["success", "error"]
    validation_valid: bool | None = None
    avg_confidence: float | None = None
    warning_count: int = 0


class StageAudit(BaseModel):
    stage: str
    status: Literal["success", "error"]
    decision: RouteDecision
    reason: str
    latency_ms: int
    validation_valid: bool | None = None
    avg_confidence: float | None = None
    warning_count: int = 0
    total_tokens: int = 0
    cost_units: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] | None = None


class RoutingAudit(BaseModel):
    policy_version: str
    terminal_decision: RouteDecision
    terminal_reason: str
    selected_stage: str | None
    attempts: int
    fallback_count: int
    latency_ms: int
    total_tokens: int
    cost_units: float
    budget_exhausted: bool
    skipped_stages: list[str]
    stages: list[StageAudit]


class RoutingResult(BaseModel):
    response: ExtractionResponse | None
    audit: RoutingAudit


class ExtractionRouter:
    def __init__(self, *, stages: list[ExtractionStage], policy: RoutingPolicy) -> None:
        self.policy = policy
        registered_names = [stage.name for stage in stages]
        if len(registered_names) != len(set(registered_names)):
            raise ValueError("Registered routing stage names must be unique")
        self.stages = {stage.name: stage for stage in stages}
        missing = [stage.name for stage in policy.stages if stage.name not in self.stages]
        if missing:
            raise ValueError(f"Routing policy references unregistered stages: {missing}")
        actual_keys = [
            getattr(self.stages[stage.name], "equivalence_key", stage.name)
            for stage in policy.stages
        ]
        if len(actual_keys) != len(set(actual_keys)):
            raise ValueError("Routing policy cannot repeat equivalent registered stages")

    async def extract_from_text(self, **kwargs: Any) -> RoutingResult:
        return await self.route(RoutingRequest(operation="text", arguments=kwargs))

    async def extract_from_file(self, **kwargs: Any) -> RoutingResult:
        return await self.route(RoutingRequest(operation="file", arguments=kwargs))

    async def route(self, request: RoutingRequest) -> RoutingResult:
        started = time.perf_counter()
        audits: list[StageAudit] = []
        last_response: ExtractionResponse | None = None
        total_tokens = 0
        total_cost = 0.0
        fallback_count = 0
        skipped_stages: list[str] = []

        for stage_policy in self.policy.stages:
            if not stage_policy.selector.matches(
                request=request,
                audits=audits,
                fallback_count=fallback_count,
            ):
                skipped_stages.append(stage_policy.name)
                continue
            budget_reason = self._budget_reason(
                attempts=len(audits),
                latency_ms=_elapsed_ms(started),
                total_tokens=total_tokens,
                total_cost=total_cost,
            )
            if budget_reason is not None:
                return self._finish(
                    response=last_response,
                    audits=audits,
                    started=started,
                    decision=self.policy.budget_exhausted_decision,
                    reason=budget_reason,
                    fallback_count=fallback_count,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    budget_exhausted=True,
                    skipped_stages=skipped_stages,
                )

            stage_started = time.perf_counter()
            result: StageResult | None = None
            error: str | None = None
            try:
                result = await self.stages[stage_policy.name].execute(request)
                if result.response is not None:
                    last_response = result.response
                evaluation = _evaluate_result(result)
                stage_tokens = (
                    result.token_usage
                    if result.token_usage is not None
                    else _usage_tokens(result.response)
                )
                total_tokens += stage_tokens
                total_cost += result.cost_units
            except Exception as exc:
                error = repr(exc)
                evaluation = StageEvaluation(status="error")
                stage_tokens = 0

            rule = next(
                (rule for rule in stage_policy.rules if rule.when.matches(evaluation)),
                None,
            )
            decision = rule.decision if rule else stage_policy.default_decision
            reason = rule.reason if rule else stage_policy.default_reason
            if decision is RouteDecision.ACCEPT and result is None:
                decision = RouteDecision.FAIL
                reason = "stage cannot accept without a response"
            audit = StageAudit(
                stage=stage_policy.name,
                status=evaluation.status,
                decision=decision,
                reason=reason,
                latency_ms=_elapsed_ms(stage_started),
                validation_valid=evaluation.validation_valid,
                avg_confidence=evaluation.avg_confidence,
                warning_count=evaluation.warning_count,
                total_tokens=stage_tokens,
                cost_units=result.cost_units if result else 0.0,
                error=error,
                metadata=result.metadata if result else {},
                output=(
                    result.output
                    if result and result.output is not None
                    else result.response.result
                    if result and result.response is not None
                    else None
                ),
            )
            audits.append(audit)

            budget_reason = self._budget_reason(
                attempts=len(audits),
                latency_ms=_elapsed_ms(started),
                total_tokens=total_tokens,
                total_cost=total_cost,
                include_stage_limit=False,
            )
            if budget_reason is not None:
                audit.decision = self.policy.budget_exhausted_decision
                audit.reason = budget_reason
                return self._finish(
                    response=last_response,
                    audits=audits,
                    started=started,
                    decision=self.policy.budget_exhausted_decision,
                    reason=budget_reason,
                    fallback_count=fallback_count,
                    total_tokens=total_tokens,
                    total_cost=total_cost,
                    budget_exhausted=True,
                    skipped_stages=skipped_stages,
                )
            if decision is RouteDecision.FALLBACK:
                fallback_count += 1
                continue
            return self._finish(
                response=last_response,
                audits=audits,
                started=started,
                decision=decision,
                reason=reason,
                fallback_count=fallback_count,
                total_tokens=total_tokens,
                total_cost=total_cost,
                selected_stage=stage_policy.name if decision is RouteDecision.ACCEPT else None,
                skipped_stages=skipped_stages,
            )

        return self._finish(
            response=last_response,
            audits=audits,
            started=started,
            decision=self.policy.exhausted_decision,
            reason=self.policy.exhausted_reason,
            fallback_count=fallback_count,
            total_tokens=total_tokens,
            total_cost=total_cost,
            skipped_stages=skipped_stages,
        )

    def _budget_reason(
        self,
        *,
        attempts: int,
        latency_ms: int,
        total_tokens: int,
        total_cost: float,
        include_stage_limit: bool = True,
    ) -> str | None:
        budget = self.policy.budget
        if include_stage_limit and budget.max_stages is not None and attempts >= budget.max_stages:
            return "max_stages budget exhausted"
        if (
            include_stage_limit
            and budget.max_requests is not None
            and attempts >= budget.max_requests
        ):
            return "max_requests budget exhausted"
        if budget.max_latency_ms is not None and latency_ms >= budget.max_latency_ms:
            return "max_latency_ms budget exhausted"
        if budget.max_total_tokens is not None and total_tokens > budget.max_total_tokens:
            return "max_total_tokens budget exhausted"
        if budget.max_cost_units is not None and total_cost > budget.max_cost_units:
            return "max_cost_units budget exhausted"
        return None

    def _finish(
        self,
        *,
        response: ExtractionResponse | None,
        audits: list[StageAudit],
        started: float,
        decision: RouteDecision,
        reason: str,
        fallback_count: int,
        total_tokens: int,
        total_cost: float,
        selected_stage: str | None = None,
        budget_exhausted: bool = False,
        skipped_stages: list[str],
    ) -> RoutingResult:
        audit = RoutingAudit(
            policy_version=self.policy.version,
            terminal_decision=decision,
            terminal_reason=reason,
            selected_stage=selected_stage,
            attempts=len(audits),
            fallback_count=fallback_count,
            latency_ms=_elapsed_ms(started),
            total_tokens=total_tokens,
            cost_units=total_cost,
            budget_exhausted=budget_exhausted,
            skipped_stages=skipped_stages,
            stages=audits,
        )
        if response is not None:
            response = response.model_copy(update={"routing": audit.model_dump(mode="json")})
        return RoutingResult(response=response, audit=audit)


def _evaluate_result(result: StageResult) -> StageEvaluation:
    response = result.response
    if response is None:
        confidence = result.metadata.get("confidence")
        return StageEvaluation(
            status="success",
            avg_confidence=(
                float(confidence)
                if isinstance(confidence, int | float) and not isinstance(confidence, bool)
                else None
            ),
        )
    confidences = _collect_confidences(response.result)
    return StageEvaluation(
        status="success",
        validation_valid=response.validation.valid,
        avg_confidence=round(sum(confidences) / len(confidences), 4) if confidences else None,
        warning_count=len(response.validation.warnings),
    )


def _collect_confidences(value: Any) -> list[float]:
    if isinstance(value, list):
        return [confidence for item in value for confidence in _collect_confidences(item)]
    if not isinstance(value, dict):
        return []
    confidences = []
    confidence = value.get("confidence")
    if isinstance(confidence, int | float) and not isinstance(confidence, bool):
        confidences.append(float(confidence))
    for child in value.values():
        confidences.extend(_collect_confidences(child))
    return confidences


def _usage_tokens(response: ExtractionResponse | None) -> int:
    if response is None or response.usage is None:
        return 0
    if response.usage.total_tokens is not None:
        return response.usage.total_tokens
    return (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)


def _request_context(request: RoutingRequest) -> dict[str, Any]:
    arguments = request.arguments
    context = {**request.metadata}
    argument_metadata = arguments.get("metadata")
    if isinstance(argument_metadata, dict):
        context.update(argument_metadata)
    context["operation"] = request.operation
    context["language"] = arguments.get("language")
    path = arguments.get("path")
    if isinstance(path, Path):
        context["file_suffix"] = path.suffix.lower()
    blocks = arguments.get("ocr_blocks")
    if isinstance(blocks, list):
        context["block_count"] = len(blocks)
        pages = {getattr(block, "page", None) for block in blocks}
        context["page_count"] = len(pages - {None})
        confidences = [
            float(block.confidence)
            for block in blocks
            if getattr(block, "confidence", None) is not None
        ]
        if confidences:
            context["ocr_confidence"] = sum(confidences) / len(confidences)
    text = arguments.get("text")
    if isinstance(text, str):
        context["text_length"] = len(text)
    return context


def _numeric_at_least(value: Any, minimum: float) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value >= minimum


def _numeric_at_most(value: Any, maximum: float) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value <= maximum


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
