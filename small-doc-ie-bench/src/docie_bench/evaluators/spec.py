"""An evaluator: which judge, which questions, which agents, and when a verdict
sends the run to human review."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from docie_bench.evaluators.judges import (
    DEFAULT_TRANSCRIPT_CHARS,
    DecisionJudge,
    Judge,
    JudgeQuestion,
    LlmJudge,
    ReadScope,
    Verdict,
)
from docie_bench.llm.model_profiles import ModelProfile

_SLUG = r"^[a-z0-9][a-z0-9_-]*$"


class QuestionSpec(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    type: Literal["choice", "score", "noul"]
    instructions: str = Field(min_length=1, max_length=2000)
    # choice: {option: description|null}; score: levels lowest first; noul: optional.
    criteria: dict[str, str | None] | list[str] | None = None

    @model_validator(mode="after")
    def _criteria_fit_type(self) -> QuestionSpec:
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise ValueError(f"choice question {self.id!r} needs at least two options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError(f"score question {self.id!r} needs 2 to 10 levels")
        elif self.criteria is not None and (
            not isinstance(self.criteria, dict) or set(self.criteria) - {"true", "false"}
        ):
            raise ValueError(f"noul question {self.id!r} criteria may only name true/false")
        return self

    def to_question(self) -> JudgeQuestion:
        return JudgeQuestion(self.id, self.type, self.instructions, self.criteria)


class ReviewRule(BaseModel):
    """Send the run to review when ``question`` answers this way.

    noul_above / noul_below compare P(true); score_above / score_below compare
    the (expected) level; choice_in matches the chosen option.
    """

    question: str
    when: Literal["noul_above", "noul_below", "score_above", "score_below", "choice_in"]
    value: float | list[str]

    def fires(self, verdict: Verdict) -> bool:
        if not verdict.ok or verdict.question_id != self.question:
            return False
        if self.when == "choice_in":
            return isinstance(self.value, list) and verdict.choice in self.value
        if isinstance(self.value, list):
            return False
        observed = verdict.noul if self.when.startswith("noul") else verdict.score
        if observed is None:
            return False
        return observed > self.value if self.when.endswith("above") else observed < self.value

    def describe(self, verdict: Verdict) -> str:
        if self.when == "choice_in":
            return f"{self.question}: answered {verdict.choice!r}"
        observed = verdict.noul if self.when.startswith("noul") else verdict.score
        side = ">" if self.when.endswith("above") else "<"
        return f"{self.question}: {observed:.2f} {side} {self.value}"


def _check_rule_value(rule: ReviewRule, question: QuestionSpec) -> None:
    """A rule whose value can never match would silently never fire."""
    if rule.when == "choice_in":
        if not isinstance(rule.value, list) or not rule.value:
            raise ValueError(f"review rule on {rule.question!r} needs a list of options")
        unknown = set(rule.value) - set(question.to_question().options())
        if unknown:
            raise ValueError(
                f"review rule on {rule.question!r} names unknown options {sorted(unknown)!r}"
            )
        return
    if isinstance(rule.value, list):
        raise ValueError(f"review rule {rule.when!r} on {rule.question!r} needs a number")
    if rule.when.startswith("noul") and not 0.0 <= rule.value <= 1.0:
        raise ValueError(
            f"review rule {rule.when!r} on {rule.question!r} compares a probability: "
            "use a value between 0 and 1"
        )
    if rule.when.startswith("score"):
        levels = len(question.to_question().options())
        if not 0.0 <= rule.value <= levels - 1:
            raise ValueError(
                f"review rule {rule.when!r} on {rule.question!r} compares a level: "
                f"use a value between 0 and {levels - 1}"
            )


class EvaluatorSpec(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=_SLUG)
    description: str = Field(default="", max_length=500)
    judge: Literal["decision", "llm_judge"]
    # A deployment / profile name, resolved at evaluation time.
    model: str = Field(min_length=1, max_length=200)
    questions: list[QuestionSpec] = Field(min_length=1, max_length=20)
    agents: list[str] = Field(default_factory=list)
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    review_when: list[ReviewRule] = Field(default_factory=list)
    # A failed judgement (model down, unparseable answer) also goes to review.
    review_on_error: bool = False
    max_transcript_chars: int = Field(default=DEFAULT_TRANSCRIPT_CHARS, ge=500, le=50_000)
    # run: the whole exchange; user: the user's messages only; answer: the
    # final answer only.
    reads: ReadScope = "run"
    enabled: bool = True

    @model_validator(mode="after")
    def _consistent(self) -> EvaluatorSpec:
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("question ids must be unique")
        types = {q.id: q.type for q in self.questions}
        for rule in self.review_when:
            if rule.question not in types:
                raise ValueError(f"review rule names unknown question {rule.question!r}")
            kind = types[rule.question]
            expected = "choice" if rule.when == "choice_in" else rule.when.split("_")[0]
            if kind != expected:
                raise ValueError(
                    f"review rule {rule.when!r} does not apply to {kind} question "
                    f"{rule.question!r}"
                )
            _check_rule_value(rule, next(q for q in self.questions if q.id == rule.question))
        return self

    def build_judge(self, profile: ModelProfile) -> Judge:
        questions = [q.to_question() for q in self.questions]
        if self.judge == "decision":
            return DecisionJudge(self.name, profile, questions)
        return LlmJudge(self.name, profile, questions)

    def review_reasons(self, verdicts: Sequence[Verdict]) -> list[str]:
        """Why this run should go to review; empty when it should not."""
        reasons = [
            rule.describe(v) for rule in self.review_when for v in verdicts if rule.fires(v)
        ]
        if self.review_on_error:
            reasons += [f"{v.question_id}: judge failed ({v.error})" for v in verdicts if not v.ok]
        return reasons
