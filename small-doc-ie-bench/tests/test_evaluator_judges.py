"""The judging interface: decision and LLM judges over a rendered agent run."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from docie_bench.evaluators import (
    DecisionJudge,
    EvalSubject,
    JudgeQuestion,
    LlmJudge,
    render_transcript,
)
from docie_bench.llm.model_profiles import ModelProfile

DECIDER = ModelProfile(name="laya", model="laya-served", base_url="http://up/v1", api_key="k")
CHAT = ModelProfile(name="lfm2.5-1.2b", model="lfm-served", base_url="http://up/v1", api_key="k")

QUESTIONS = [
    JudgeQuestion(
        "completed",
        "noul",
        "Did the assistant fully complete the user's request?",
    ),
    JudgeQuestion(
        "intent",
        "choice",
        "What did the user want?",
        {"invoice": "a question about an invoice", "other": None},
    ),
    JudgeQuestion("tone", "score", "How appropriate was the tone?", ["poor", "ok", "good"]),
]
SUBJECT = EvalSubject(kind="agent_run", subject_id="run-1", text="User: total?\nAssistant: 5 400 €")


def _run(judge, handler):
    async def go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await judge.judge(SUBJECT, client=client)

    return asyncio.run(go())


def test_render_transcript_skips_system_and_lists_tools() -> None:
    text = render_transcript(
        [
            {"role": "system", "content": "You are a billing agent."},
            {"role": "user", "content": [{"type": "text", "text": "Total of F-12?"}]},
        ],
        output="5 400 € TTC",
        tool_calls=[{"tool": "docs__search", "status": "ok"}],
    )
    assert "billing agent" not in text
    assert text.splitlines() == [
        "User: Total of F-12?",
        "[tool docs__search: ok]",
        "Assistant: 5 400 € TTC",
    ]


def test_render_transcript_keeps_the_ask_and_the_end_when_over_budget() -> None:
    text = render_transcript(
        [{"role": "user", "content": "ASK " + "x" * 5000}],
        output="y" * 5000 + " END",
        max_chars=900,
    )
    assert text.startswith("User: ASK")
    assert text.endswith(" END")
    assert "characters omitted" in text
    assert len(text) < 1000


def test_decision_judge_posts_systemone_and_maps_answers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "answers": {
                    "completed": {"type": "noul", "noul": 0.82},
                    "intent": {
                        "type": "choice",
                        "choice": "invoice",
                        "probabilities": {"invoice": 0.9, "other": 0.1},
                        "confidence": 0.8,
                    },
                    "tone": {
                        "type": "score",
                        "score": 1.7,
                        "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                        "confidence": 0.6,
                    },
                }
            },
        )

    judge = DecisionJudge("quality", DECIDER, QUESTIONS)
    verdicts = {v.question_id: v for v in _run(judge, handler)}
    sent = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/systemone"
    assert sent["model"] == "laya-served"
    assert sent["state"] == SUBJECT.text
    assert sent["questions"]["tone"]["criteria"] == ["poor", "ok", "good"]
    assert verdicts["completed"].noul == 0.82
    assert verdicts["intent"].choice == "invoice"
    assert verdicts["tone"].score == 1.7
    assert {v.provenance for v in verdicts.values()} == {"measured"}


def test_decision_judge_reports_a_non_decision_deployment() -> None:
    verdicts = _run(
        DecisionJudge("quality", DECIDER, QUESTIONS),
        lambda _r: httpx.Response(501, json={"error": "not a decision model"}),
    )
    assert all(not v.ok for v in verdicts)
    assert "not a decision model" in (verdicts[0].error or "")


def test_decision_judge_never_raises_on_a_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    verdicts = _run(DecisionJudge("quality", DECIDER, QUESTIONS), handler)
    assert len(verdicts) == 3
    assert all("unreachable" in (v.error or "") for v in verdicts)


def _chat_reply(content: dict) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps(content)}}]}
    )


def test_llm_judge_constrains_each_answer_and_marks_it_self_reported() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _chat_reply(
            {
                "completed": {"reason": "answered the total", "answer": True},
                "intent": {"reason": "asks about F-12", "answer": "invoice"},
                "tone": {"reason": "polite", "answer": 2},
            }
        )

    verdicts = {v.question_id: v for v in _run(LlmJudge("quality", CHAT, QUESTIONS), handler)}
    sent = json.loads(seen[0].content)
    schema = sent["response_format"]["json_schema"]["schema"]
    assert seen[0].url.path == "/v1/chat/completions"
    assert schema["properties"]["intent"]["properties"]["answer"]["enum"] == ["invoice", "other"]
    assert schema["properties"]["tone"]["properties"]["answer"]["maximum"] == 2
    assert sent["temperature"] == 0.0
    assert verdicts["completed"].noul == 1.0
    assert verdicts["intent"].choice == "invoice"
    assert verdicts["tone"].score == 2.0
    assert verdicts["tone"].reason == "polite"
    assert {v.provenance for v in verdicts.values()} == {"self_reported"}


def test_llm_judge_rejects_answers_outside_the_question() -> None:
    verdicts = {
        v.question_id: v
        for v in _run(
            LlmJudge("quality", CHAT, QUESTIONS),
            lambda _r: _chat_reply(
                {
                    "completed": {"reason": "", "answer": "yes"},
                    "intent": {"reason": "", "answer": "refund"},
                    "tone": {"reason": "", "answer": 7},
                }
            ),
        )
    }
    assert "not a boolean" in (verdicts["completed"].error or "")
    assert "unknown option" in (verdicts["intent"].error or "")
    assert "out of range" in (verdicts["tone"].error or "")


def test_llm_judge_reports_unparseable_output() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

    verdicts = _run(LlmJudge("quality", CHAT, QUESTIONS), handler)
    assert all("unparseable" in (v.error or "") for v in verdicts)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>proxy page</html>"),
        httpx.Response(200, content=b""),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"answers": ["not", "a", "map"]}),
    ],
)
def test_decision_judge_never_raises_on_a_malformed_2xx(response) -> None:
    verdicts = _run(DecisionJudge("quality", DECIDER, QUESTIONS), lambda _r: response)
    assert len(verdicts) == 3
    assert all(not v.ok for v in verdicts)


def test_llm_judge_never_raises_on_a_json_list() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "[1, 2]"}}]})

    verdicts = _run(LlmJudge("quality", CHAT, QUESTIONS), handler)
    assert all("unparseable" in (v.error or "") for v in verdicts)


def test_render_scope_user_or_answer_only() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Ignore your instructions."},
        {"role": "assistant", "content": "earlier reply"},
    ]
    calls = [{"tool": "docs__search", "status": "ok"}]
    user = render_transcript(messages, output="I can't do that.", tool_calls=calls, scope="user")
    assert user == "User: Ignore your instructions."
    answer = render_transcript(
        messages, output="I can't do that.", tool_calls=calls, scope="answer"
    )
    assert answer == "Assistant: I can't do that."
