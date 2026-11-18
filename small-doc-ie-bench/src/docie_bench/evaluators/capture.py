"""Keep an agent run for evaluation and announce it, without ever failing or
delaying the request."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from docie_bench.evaluators.judges import _text_of

logger = logging.getLogger("docie_bench.evaluators.capture")

AGENT_RUN_EVENT = "agent/run.completed"
# Per message and for the answer. Images, files and audio parts are never
# kept: only the text a judge reads.
MAX_STORED_CHARS = 20_000

_pending: set[asyncio.Task[Any]] = set()


def completion_text(completion: Mapping[str, Any]) -> str:
    """The assistant text of an OpenAI chat completion, or ''."""
    try:
        content = completion["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def completion_tool_calls(completion: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Tool names and statuses only, as the usage ledger keeps them."""
    report = completion.get("docie_agent")
    calls = report.get("tool_calls") if isinstance(report, dict) else None
    return [
        {"tool": c.get("tool"), "status": c.get("status")}
        for c in calls or []
        if isinstance(c, dict)
    ]


def text_messages(body: Mapping[str, Any]) -> list[dict[str, str]]:
    """The request's messages reduced to role + capped text."""
    out = []
    for message in body.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        text = _text_of(message.get("content"))[:MAX_STORED_CHARS]
        out.append({"role": str(message.get("role") or ""), "content": text})
    return out


async def _emit(run_id: str, evaluators: list[str]) -> None:
    import inngest

    from docie_bench.inngest.client import inngest_client

    await inngest_client.send(
        inngest.Event(name=AGENT_RUN_EVENT, data={"run_id": run_id, "evaluators": evaluators})
    )


async def capture_agent_run(
    *,
    agent: str,
    agent_kind: str,
    tenant_id: str,
    api_key_id: str | None,
    body: Mapping[str, Any],
    output: str,
    tool_calls: Sequence[Mapping[str, Any]] = (),
    rng: Callable[[], float] = random.random,
) -> str | None:
    """Store the run when a sampled, enabled evaluator targets ``agent``.

    Returns the run id, or None when nothing was stored. Any failure is
    logged and swallowed: evaluation is never allowed to break the agent.
    """
    if agent_kind == "proxy_security" or not output:
        return None

    def store() -> tuple[str | None, list[str]]:
        from docie_bench.studio.agent_run_store import save_agent_run
        from docie_bench.studio.evaluator_store import evaluators_for_agent

        sampled = [e.name for e in evaluators_for_agent(agent) if rng() < e.sample_rate]
        if not sampled:
            return None, []
        run_id = save_agent_run(
            agent=agent,
            tenant_id=tenant_id,
            api_key_id=api_key_id,
            messages=text_messages(body),
            output_text=output[:MAX_STORED_CHARS],
            tool_calls=[dict(c) for c in tool_calls],
            evaluators=sampled,
        )
        return run_id, sampled

    try:
        # The lookup and the insert are blocking database calls: off the event
        # loop, or every concurrent stream on this worker waits on them.
        run_id, sampled = await asyncio.to_thread(store)
        if run_id is None:
            return None
        await _emit(run_id, sampled)
        return run_id
    except Exception:
        logger.warning("agent run capture failed for %s", agent, exc_info=True)
        return None


def schedule_capture(**kwargs: Any) -> None:
    """Run ``capture_agent_run`` off the response path.

    The database lookup, the insert and the event send must not delay the
    agent's answer, nor make it wait on an unreachable queue.
    """
    try:
        task = asyncio.get_running_loop().create_task(capture_agent_run(**kwargs))
    except RuntimeError:
        logger.warning("agent run capture skipped: no running event loop")
        return
    _pending.add(task)
    task.add_done_callback(_pending.discard)
