"""Score one captured agent run with its evaluators and queue a review if a rule fires."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from docie_bench.evaluators.judges import EvalSubject, Verdict, render_transcript
from docie_bench.evaluators.spec import EvaluatorSpec
from docie_bench.llm.model_profiles import ModelProfile

logger = logging.getLogger("docie_bench.evaluators.runner")

ProfileResolver = Callable[[str], ModelProfile]


def _resolve(model: str) -> ModelProfile:
    from docie_bench.serving.profile_resolver import resolve_extraction_profile

    return resolve_extraction_profile(model_profile=model)


async def evaluate_run(
    run_id: str,
    evaluator_names: Sequence[str],
    *,
    client: httpx.AsyncClient,
    resolve: ProfileResolver = _resolve,
) -> dict[str, Any]:
    """Judge the run with each named evaluator, store every result, and queue
    one review carrying all the reasons when any rule fires.

    A run already purged, or an evaluator deleted or disabled since capture,
    is skipped rather than failed: there is nothing left to judge.
    """
    from docie_bench.review import enqueue_agent_run_review
    from docie_bench.studio.agent_run_store import get_agent_run, save_evaluation
    from docie_bench.studio.evaluator_store import get_evaluator

    run = get_agent_run(run_id)
    if run is None:
        return {"run_id": run_id, "status": "run_expired"}

    all_reasons: list[str] = []
    all_verdicts: list[dict[str, Any]] = []
    flagged_by: list[str] = []
    judged: list[str] = []
    for name in evaluator_names:
        spec = get_evaluator(name)
        if spec is None or not spec.enabled:
            continue
        verdicts = await _judge(spec, run, client=client, resolve=resolve)
        reasons = spec.review_reasons(verdicts)
        rows = [v.as_dict() for v in verdicts]
        save_evaluation(
            run_id=run_id,
            evaluator=spec.name,
            agent=run["agent"],
            tenant_id=run["tenant_id"],
            verdicts=rows,
            review_reasons=reasons,
        )
        judged.append(spec.name)
        all_verdicts.extend(rows)
        if reasons:
            flagged_by.append(spec.name)
            all_reasons.extend(f"{spec.name} · {reason}" for reason in reasons)

    review_id = None
    if all_reasons:
        review = enqueue_agent_run_review(
            run_id=run_id,
            agent=run["agent"],
            evaluators=flagged_by,
            reasons=all_reasons,
            verdicts=all_verdicts,
        )
        review_id = review.id
    return {
        "run_id": run_id,
        "status": "evaluated",
        "evaluators": judged,
        "review_id": review_id,
        "reasons": all_reasons,
    }


async def _judge(
    spec: EvaluatorSpec,
    run: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    resolve: ProfileResolver,
) -> list[Verdict]:
    try:
        profile = resolve(spec.model)
    except Exception as exc:
        # A judge model that is not live is a failed judgement, not a crash:
        # review_on_error decides whether a human looks at the run anyway.
        return [
            Verdict(
                spec.name,
                q.id,
                q.type,
                "measured" if spec.judge == "decision" else "self_reported",
                error=f"judge model {spec.model!r} unavailable: {exc}",
            )
            for q in spec.questions
        ]
    subject = EvalSubject(
        kind="agent_run",
        subject_id=run["id"],
        text=render_transcript(
            run["messages"],
            output=run["output"],
            tool_calls=run["tool_calls"],
            max_chars=spec.max_transcript_chars,
        ),
        metadata={"agent": run["agent"]},
    )
    verdicts = await spec.build_judge(profile).judge(subject, client=client)
    if any(v.ok for v in verdicts):
        # A judge call is traffic: without the stamp a deployment that only
        # judges looks idle to the autoloader and gets evicted.
        try:
            from docie_bench.serving import recency

            recency.stamp_served_profile(profile.name)
        except Exception:
            logger.warning("recency stamp failed for %s", profile.name, exc_info=True)
    return verdicts
