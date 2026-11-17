"""Scoring a captured agent run: verdicts stored, review queued when a rule fires."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from docie_bench.evaluators import EvaluatorSpec
from docie_bench.evaluators.runner import evaluate_run
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.review import list_reviews
from docie_bench.serving.profile_resolver import ProfileResolutionError
from docie_bench.storage.db import dispose_engine, init_engine
from docie_bench.studio.agent_run_store import (
    get_agent_run,
    list_evaluations,
    save_agent_run,
)
from docie_bench.studio.evaluator_store import create_evaluator

LAYA = ModelProfile(name="laya-q8_0", model="laya", base_url="http://up/v1", api_key="k")


def _spec(**overrides) -> EvaluatorSpec:
    spec = {
        "name": "task-completion",
        "judge": "decision",
        "model": "laya-q8_0",
        "questions": [{"id": "completed", "type": "noul", "instructions": "Done?"}],
        "agents": [],
        "review_when": [{"question": "completed", "when": "noul_below", "value": 0.5}],
    }
    spec.update(overrides)
    return EvaluatorSpec.model_validate(spec)


@pytest.fixture
def run_id(tmp_path: Path) -> str:
    init_engine(f"sqlite:///{tmp_path / 'eval.db'}")
    run = save_agent_run(
        agent="billing-bot",
        tenant_id="acme",
        api_key_id=None,
        messages=[{"role": "user", "content": "Refund invoice F-12 please"}],
        output_text="I cannot help with that.",
        tool_calls=[],
        evaluators=["task-completion"],
    )
    assert run is not None
    yield run
    dispose_engine()


def _evaluate(run: str, noul: float, names=("task-completion",), resolve=lambda _m: LAYA):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {"completed": {"type": "noul", "noul": noul}}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await evaluate_run(run, list(names), client=client, resolve=resolve)

    return asyncio.run(go()), seen


def test_a_failing_run_is_stored_and_queued_for_review(run_id: str) -> None:
    create_evaluator(_spec())
    result, seen = _evaluate(run_id, noul=0.1)
    assert "User: Refund invoice F-12 please" in seen[0]["state"]
    assert "Assistant: I cannot help with that." in seen[0]["state"]
    assert result["review_id"] is not None
    assert result["reasons"] == ["task-completion · completed: 0.10 < 0.5"]

    [review] = list_reviews(kind="agent_run")
    assert review.source_request_id == run_id
    [stored] = list_evaluations("task-completion")
    assert stored["verdicts"][0]["noul"] == 0.1
    assert stored["review_reasons"] == ["completed: 0.10 < 0.5"]


def test_a_passing_run_is_stored_but_not_queued(run_id: str) -> None:
    create_evaluator(_spec())
    result, _ = _evaluate(run_id, noul=0.9)
    assert result["review_id"] is None
    assert list_reviews(kind="agent_run") == []
    assert len(list_evaluations("task-completion")) == 1
    # Nothing to review: the run's text is deleted right away.
    assert get_agent_run(run_id) is None


def test_an_unavailable_judge_model_is_a_failed_judgement(run_id: str) -> None:
    def missing(_model: str) -> ModelProfile:
        raise ProfileResolutionError("laya-q8_0 is not ready")

    create_evaluator(_spec())
    result, seen = _evaluate(run_id, noul=0.1, resolve=missing)
    assert seen == []
    assert result["review_id"] is None
    [stored] = list_evaluations("task-completion")
    assert "unavailable" in stored["verdicts"][0]["error"]

    assert get_agent_run(run_id) is None  # nothing queued, so not kept


def test_a_failed_judgement_is_queued_when_asked(run_id: str) -> None:
    def missing(_model: str) -> ModelProfile:
        raise ProfileResolutionError("laya-q8_0 is not ready")

    create_evaluator(_spec(review_on_error=True))
    result, _ = _evaluate(run_id, noul=0.1, resolve=missing)
    assert result["review_id"] is not None
    assert get_agent_run(run_id) is not None  # kept while the review is open


def test_deciding_the_review_deletes_the_run_and_the_judge_reasons(run_id: str) -> None:
    from docie_bench.review import claim_review, decide_review
    from docie_bench.schemas.review import ReviewStatus

    create_evaluator(_spec())
    result, _ = _evaluate(run_id, noul=0.1)
    review = list_reviews(kind="agent_run")[0]
    claimed = claim_review(
        review.id, reviewer_id="alice", expected_version=review.version, lease_seconds=60
    )
    decide_review(
        review.id, reviewer_id="alice", expected_version=claimed.version,
        decision=ReviewStatus.REJECTED,
    )
    assert get_agent_run(run_id) is None
    [stored] = list_evaluations("task-completion")
    assert stored["verdicts"][0]["noul"] == 0.1  # the verdict stays
    assert stored["verdicts"][0]["reason"] is None


def test_disabled_or_deleted_evaluators_and_expired_runs_are_skipped(run_id: str) -> None:
    create_evaluator(_spec(enabled=False))
    result, seen = _evaluate(run_id, noul=0.1, names=("task-completion", "gone"))
    assert result["evaluators"] == []
    assert seen == []
    result, _ = _evaluate("00000000-0000-4000-8000-000000000000", noul=0.1)
    assert result["status"] == "run_expired"
