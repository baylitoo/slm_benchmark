"""Agent runs are kept for evaluation only when an evaluator wants them."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path

import pytest

from docie_bench.evaluators import EvaluatorSpec
from docie_bench.evaluators import capture as capture_mod
from docie_bench.evaluators.capture import capture_agent_run, completion_text
from docie_bench.storage.db import dispose_engine, init_engine
from docie_bench.studio.agent_run_store import get_agent_run, purge_expired_runs, save_agent_run
from docie_bench.studio.evaluator_store import create_evaluator

BODY = {
    "model": "billing-bot",
    "messages": [
        {"role": "system", "content": "You are a billing agent."},
        {"role": "user", "content": "Total of F-12?"},
    ],
}


def _evaluator(name: str = "completion", **overrides) -> EvaluatorSpec:
    spec = {
        "name": name,
        "judge": "decision",
        "model": "laya-q8_0",
        "questions": [{"id": "completed", "type": "noul", "instructions": "Done?"}],
        "agents": ["billing-bot"],
    }
    spec.update(overrides)
    return EvaluatorSpec.model_validate(spec)


@pytest.fixture
def emitted(tmp_path: Path, monkeypatch) -> list[tuple[str, list[str]]]:
    init_engine(f"sqlite:///{tmp_path / 'runs.db'}")
    sent: list[tuple[str, list[str]]] = []

    async def fake_emit(run_id: str, evaluators: list[str]) -> None:
        sent.append((run_id, evaluators))

    monkeypatch.setattr(capture_mod, "_emit", fake_emit)
    yield sent
    dispose_engine()


def _capture(**overrides) -> str | None:
    kwargs = {
        "agent": "billing-bot",
        "agent_kind": "custom",
        "tenant_id": "acme",
        "api_key_id": None,
        "body": BODY,
        "output": "5 400 € TTC",
        "tool_calls": [{"tool": "docs__search", "status": "ok"}],
        "rng": lambda: 0.0,
    }
    kwargs.update(overrides)
    return asyncio.run(capture_agent_run(**kwargs))


def test_a_targeted_run_is_stored_and_announced(emitted) -> None:
    create_evaluator(_evaluator())
    run_id = _capture()
    assert run_id is not None
    assert emitted == [(run_id, ["completion"])]
    run = get_agent_run(run_id)
    assert run is not None
    assert run["tenant_id"] == "acme"
    assert run["output"] == "5 400 € TTC"
    assert run["messages"][1]["content"] == "Total of F-12?"
    assert run["tool_calls"] == [{"tool": "docs__search", "status": "ok"}]


def test_nothing_is_stored_without_an_enabled_evaluator(emitted) -> None:
    assert _capture() is None
    create_evaluator(_evaluator(enabled=False))
    assert _capture() is None
    create_evaluator(_evaluator("other", agents=[]))
    assert _capture() is None
    assert emitted == []


def test_sampling_keeps_only_the_evaluators_that_drew_the_run(emitted) -> None:
    create_evaluator(_evaluator("always"))
    create_evaluator(_evaluator("rarely", sample_rate=0.1))
    run_id = _capture(rng=lambda: 0.5)
    assert emitted == [(run_id, ["always"])]


def test_proxy_security_and_empty_output_are_never_stored(emitted) -> None:
    create_evaluator(_evaluator())
    assert _capture(agent_kind="proxy_security") is None
    assert _capture(output="") is None
    assert emitted == []


def test_a_failure_never_reaches_the_request(emitted, monkeypatch) -> None:
    create_evaluator(_evaluator())

    async def broken_emit(run_id: str, evaluators: list[str]) -> None:
        raise RuntimeError("inngest down")

    monkeypatch.setattr(capture_mod, "_emit", broken_emit)
    assert _capture() is None


def _old_run(old: dt.datetime) -> str:
    run_id = save_agent_run(
        agent="billing-bot",
        tenant_id="acme",
        api_key_id=None,
        messages=[],
        output_text="x",
        tool_calls=[],
        evaluators=["completion"],
        now=old,
    )
    assert run_id is not None
    return run_id


def test_a_never_scored_run_is_purged_after_the_backstop(emitted) -> None:
    old = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    run_id = _old_run(old)
    assert purge_expired_runs(now=old + dt.timedelta(days=6)) == 0
    assert purge_expired_runs(now=old + dt.timedelta(days=8)) == 1
    assert get_agent_run(run_id) is None


def test_a_run_with_an_open_review_is_never_purged(emitted) -> None:
    from docie_bench.review import enqueue_agent_run_review

    old = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    run_id = _old_run(old)
    enqueue_agent_run_review(
        run_id=run_id, agent="billing-bot", evaluators=["completion"],
        reasons=["completed: 0.10 < 0.5"], verdicts=[],
    )
    assert purge_expired_runs(now=old + dt.timedelta(days=365)) == 0
    assert get_agent_run(run_id) is not None


def test_completion_text_reads_the_assistant_message() -> None:
    assert completion_text({"choices": [{"message": {"content": "ok"}}]}) == "ok"
    assert completion_text({"choices": []}) == ""


# ── Through the agent API: the non-stream and raw-stream paths both capture ──


@pytest.fixture
def agent_api(tmp_path: Path, monkeypatch):
    import json
    from collections.abc import AsyncIterator
    from dataclasses import replace

    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from docie_bench.agents.api import configure_http_transport
    from docie_bench.agents.api import router as agents_router
    from docie_bench.agents.registry import AgentRegistry
    from docie_bench.agents.spec import AgentSpec
    from docie_bench.llm.model_profiles import ModelProfile

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    monkeypatch.delenv("DOCIE_AGENTS_PATH", raising=False)
    upstream = ModelProfile(name="alpha", model="up", base_url="http://upstream/v1", api_key="k")
    monkeypatch.setattr(
        "docie_bench.agents.runtime.resolve_extraction_profile",
        lambda **_: replace(upstream),
    )
    AgentRegistry().create(AgentSpec(name="billing-bot", kind="custom", model_profile="alpha"))

    captured: list[dict] = []

    monkeypatch.setattr(
        "docie_bench.agents.api.schedule_capture", lambda **kw: captured.append(kw)
    )

    async def chunks(abort: bool) -> AsyncIterator[bytes]:
        for piece in ("5 400", " €"):
            yield f'data: {{"choices":[{{"delta":{{"content":"{piece}"}}}}]}}\n\n'.encode()
            if abort:
                raise httpx.ReadError("upstream dropped")
        yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            abort = "abort" in body["messages"][-1]["content"]
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=chunks(abort)
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "5 400 €"}}]}
        )

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(agents_router)
    yield TestClient(app), captured
    configure_http_transport(None)


@pytest.mark.parametrize("stream", [False, True])
def test_the_agent_api_hands_the_finished_run_to_capture(agent_api, stream) -> None:
    client, captured = agent_api
    response = client.post(
        "/v1/agents/chat/completions",
        json={"model": "billing-bot", "stream": stream, "messages": BODY["messages"][1:]},
    )
    assert response.status_code == 200
    _ = response.content  # drain the stream so the relay's finally runs
    assert len(captured) == 1
    assert captured[0]["agent"] == "billing-bot"
    assert captured[0]["output"] == "5 400 €"
    assert captured[0]["body"]["messages"][0]["content"] == "Total of F-12?"


def test_an_aborted_stream_is_not_captured(agent_api) -> None:
    import httpx

    client, captured = agent_api

    def post() -> bytes:
        return client.post(
            "/v1/agents/chat/completions",
            json={
                "model": "billing-bot",
                "stream": True,
                "messages": [{"role": "user", "content": "abort please"}],
            },
        ).content

    with pytest.raises(httpx.ReadError):
        post()
    assert captured == []


def test_only_capped_text_is_kept_from_messages() -> None:
    from docie_bench.evaluators.capture import text_messages

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this invoice"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {"role": "user", "content": "x" * 50_000},
        ]
    }
    kept = text_messages(body)
    assert kept[0] == {"role": "user", "content": "Read this invoice"}
    assert len(kept[1]["content"]) == 20_000
