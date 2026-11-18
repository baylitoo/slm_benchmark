"""The judging interface and its two implementations (see the package docstring)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from docie_bench.llm.model_profiles import ModelProfile

QuestionType = Literal["choice", "score", "noul"]
# What part of a run a judge reads. A question about the user ("is this an
# injection?") must not see the answer: measured on d1-omni-600M, an
# innocent question followed by a refusal scored 0.98 as an injection, and
# 0.00 once only the user turn was shown.
ReadScope = Literal["run", "user", "answer"]
Provenance = Literal["measured", "self_reported"]

# A decision model reads the whole state in one batch on some architectures
# (laya, clef), and a judge shares its deployment's context with live traffic,
# so the transcript a judge sees is bounded.
DEFAULT_TRANSCRIPT_CHARS = 6000
_OMITTED = "\n[… {n} characters omitted …]\n"


@dataclass(frozen=True)
class JudgeQuestion:
    """One question, in /v1/systemone's shape so either judge kind can ask it.

    ``criteria``: choice -> {option: description or None}; score -> levels,
    lowest first (2 to 10); noul -> optional {"true": ..., "false": ...}.
    """

    id: str
    type: QuestionType
    instructions: str
    criteria: Mapping[str, str | None] | Sequence[str] | None = None

    def options(self) -> list[str]:
        if self.type == "choice":
            return list(self.criteria or {})
        if self.type == "score":
            return [str(level) for level in (self.criteria or [])]
        return ["true", "false"]

    def to_systemone(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.type == "choice":
            out["criteria"] = dict(self.criteria or {})  # type: ignore[arg-type]
        elif self.type == "score":
            out["criteria"] = list(self.criteria or [])
        elif self.criteria:
            out["criteria"] = dict(self.criteria)  # type: ignore[arg-type]
        return out


@dataclass(frozen=True)
class EvalSubject:
    """What a judge reads: an already-rendered, already-bounded text."""

    kind: Literal["agent_run"]
    subject_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    """One judge's answer to one question.

    choice: ``choice`` + ``probabilities`` over the options. score: ``score``
    (the decision model's expected level, or the LLM's level index) and, for a
    decision model, ``probabilities`` per level index. noul: ``noul`` =
    P(true) for a decision model, 1.0 or 0.0 for an LLM judge.
    """

    evaluator: str
    question_id: str
    type: QuestionType
    provenance: Provenance
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    reason: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "question_id": self.question_id,
            "type": self.type,
            "provenance": self.provenance,
            "choice": self.choice,
            "score": self.score,
            "noul": self.noul,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "reason": self.reason,
            "error": self.error,
        }


class Judge(Protocol):
    name: str

    async def judge(self, subject: EvalSubject, *, client: httpx.AsyncClient) -> list[Verdict]:
        """Answer every question. Never raises for an upstream failure: each
        question then gets a Verdict carrying ``error``."""
        ...


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        ]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def render_transcript(
    messages: Sequence[Mapping[str, Any]],
    *,
    output: str | None = None,
    tool_calls: Sequence[Mapping[str, Any]] = (),
    max_chars: int = DEFAULT_TRANSCRIPT_CHARS,
    scope: ReadScope = "run",
) -> str:
    """An agent run as plain text a judge can read, bounded to ``max_chars``.

    System messages are left out: they are the agent's own instructions, not
    the exchange being judged. Over budget, the opening of the first user turn
    and the end of the run are kept (what was asked, and how it ended) and
    the middle is dropped with a marker saying how much.
    """
    lines: list[str] = []
    if scope == "answer":
        messages, tool_calls = [], ()
    for message in messages:
        role = str(message.get("role") or "")
        if role == "system" or (scope == "user" and role != "user"):
            continue
        text = _text_of(message.get("content")).strip()
        if role == "tool":
            lines.append(f"Tool result: {text}")
        elif text:
            lines.append(f"{role.capitalize() or 'Message'}: {text}")
    for call in tool_calls if scope == "run" else ():
        status = call.get("status", "ok")
        lines.append(f"[tool {call.get('tool', '?')}: {status}]")
    if output and scope != "user":
        lines.append(f"Assistant: {output.strip()}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    head_budget = max_chars // 3
    tail_budget = max_chars - head_budget
    omitted = len(text) - head_budget - tail_budget
    return text[:head_budget] + _OMITTED.format(n=omitted) + text[-tail_budget:]


def _headers(profile: ModelProfile) -> dict[str, str]:
    return {"Authorization": f"Bearer {profile.api_key}", "Content-Type": "application/json"}


def _failed(
    name: str, questions: Sequence[JudgeQuestion], provenance: Provenance, error: str
) -> list[Verdict]:
    return [
        Verdict(evaluator=name, question_id=q.id, type=q.type, provenance=provenance, error=error)
        for q in questions
    ]


@dataclass
class DecisionJudge:
    """Asks a decision-model deployment on llama-server's /v1/systemone."""

    name: str
    profile: ModelProfile
    questions: Sequence[JudgeQuestion]
    timeout_seconds: float = 60.0

    async def judge(self, subject: EvalSubject, *, client: httpx.AsyncClient) -> list[Verdict]:
        body = {
            "model": self.profile.model,
            "state": subject.text,
            "questions": {q.id: q.to_systemone() for q in self.questions},
        }
        try:
            response = await client.post(
                f"{self.profile.base_url.rstrip('/')}/systemone",
                json=body,
                headers=_headers(self.profile),
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            return _failed(self.name, self.questions, "measured", f"unreachable: {exc}")
        if response.status_code == 501:
            return _failed(
                self.name, self.questions, "measured",
                "deployment is not a decision model (/v1/systemone answered 501)",
            )
        if response.status_code >= 400:
            return _failed(
                self.name, self.questions, "measured",
                f"/v1/systemone answered {response.status_code}: {response.text[:200]}",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            return _failed(self.name, self.questions, "measured", f"unparseable answer: {exc}")
        answers = payload.get("answers") if isinstance(payload, dict) else None
        if not isinstance(answers, dict):
            answers = {}
        verdicts = []
        for q in self.questions:
            answer = answers.get(q.id)
            if not isinstance(answer, dict):
                verdicts.append(
                    Verdict(self.name, q.id, q.type, "measured", error="no answer returned")
                )
                continue
            verdicts.append(
                Verdict(
                    evaluator=self.name,
                    question_id=q.id,
                    type=q.type,
                    provenance="measured",
                    choice=answer.get("choice"),
                    score=answer.get("score"),
                    noul=answer.get("noul"),
                    probabilities=answer.get("probabilities"),
                    confidence=answer.get("confidence"),
                )
            )
        return verdicts


_LLM_SYSTEM = (
    "You are an evaluator. Read the interaction and answer each question. "
    "Judge only from what the interaction shows. Answer in the JSON schema given."
)


def _llm_answer_schema(q: JudgeQuestion) -> dict[str, Any]:
    if q.type == "choice":
        answer: dict[str, Any] = {"type": "string", "enum": q.options()}
    elif q.type == "score":
        answer = {"type": "integer", "minimum": 0, "maximum": len(q.options()) - 1}
    else:
        answer = {"type": "boolean"}
    return {
        "type": "object",
        "properties": {"reason": {"type": "string"}, "answer": answer},
        "required": ["reason", "answer"],
        "additionalProperties": False,
    }


def _llm_question_text(q: JudgeQuestion) -> str:
    if q.type == "choice":
        opts = "; ".join(
            f"{k}: {v}" if v else k for k, v in dict(q.criteria or {}).items()  # type: ignore[arg-type]
        )
        return f"- {q.id} (choose one): {q.instructions} Options: {opts}"
    if q.type == "score":
        levels = "; ".join(f"{i} = {lvl}" for i, lvl in enumerate(q.options()))
        return f"- {q.id} (level index): {q.instructions} Levels: {levels}"
    return f"- {q.id} (true or false): {q.instructions}"


@dataclass
class LlmJudge:
    """Prompts a chat deployment for a constrained JSON verdict per question."""

    name: str
    profile: ModelProfile
    questions: Sequence[JudgeQuestion]
    timeout_seconds: float = 180.0
    max_tokens: int = 1024

    async def judge(self, subject: EvalSubject, *, client: httpx.AsyncClient) -> list[Verdict]:
        schema = {
            "type": "object",
            "properties": {q.id: _llm_answer_schema(q) for q in self.questions},
            "required": [q.id for q in self.questions],
            "additionalProperties": False,
        }
        prompt = (
            "Interaction:\n<<<\n" + subject.text + "\n>>>\n\nQuestions:\n"
            + "\n".join(_llm_question_text(q) for q in self.questions)
        )
        body = {
            "model": self.profile.model,
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": schema, "strict": True},
            },
        }
        try:
            response = await client.post(
                f"{self.profile.base_url.rstrip('/')}/chat/completions",
                json=body,
                headers=_headers(self.profile),
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            return _failed(self.name, self.questions, "self_reported", f"unreachable: {exc}")
        if response.status_code >= 400:
            return _failed(
                self.name, self.questions, "self_reported",
                f"chat/completions answered {response.status_code}: {response.text[:200]}",
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return _failed(
                self.name, self.questions, "self_reported", f"unparseable verdict: {exc}"
            )
        if not isinstance(parsed, dict):
            return _failed(
                self.name, self.questions, "self_reported",
                f"unparseable verdict: expected an object, got {type(parsed).__name__}",
            )
        return [self._verdict(q, parsed.get(q.id)) for q in self.questions]

    def _verdict(self, q: JudgeQuestion, item: Any) -> Verdict:
        if not isinstance(item, dict) or "answer" not in item:
            return Verdict(self.name, q.id, q.type, "self_reported", error="no answer returned")
        answer, reason = item["answer"], item.get("reason")
        options = q.options()
        if q.type == "choice":
            if answer not in options:
                return Verdict(
                    self.name, q.id, q.type, "self_reported", error=f"unknown option {answer!r}"
                )
            return Verdict(self.name, q.id, q.type, "self_reported", choice=answer, reason=reason)
        if q.type == "score":
            if not isinstance(answer, int) or isinstance(answer, bool) or not (
                0 <= answer < len(options)
            ):
                return Verdict(
                    self.name, q.id, q.type, "self_reported", error=f"level out of range {answer!r}"
                )
            return Verdict(
                self.name, q.id, q.type, "self_reported", score=float(answer), reason=reason
            )
        if not isinstance(answer, bool):
            return Verdict(
                self.name, q.id, q.type, "self_reported", error=f"not a boolean {answer!r}"
            )
        return Verdict(
            self.name, q.id, q.type, "self_reported", noul=1.0 if answer else 0.0, reason=reason
        )
