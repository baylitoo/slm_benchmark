"""Evaluator specs, their review rules, and the /v1/studio/evaluators routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import docie_bench.api as api
from docie_bench.agents.registry import AgentRegistry
from docie_bench.agents.spec import AgentSpec
from docie_bench.evaluators import EvaluatorSpec, Verdict
from docie_bench.storage.db import dispose_engine, init_engine
from docie_bench.studio.evaluator_store import evaluators_for_agent


def _spec(**overrides) -> dict:
    spec = {
        "name": "task-completion",
        "judge": "decision",
        "model": "laya-q8_0",
        "questions": [
            {"id": "completed", "type": "noul", "instructions": "Was the request fully done?"},
            {
                "id": "intent",
                "type": "choice",
                "instructions": "What did the user want?",
                "criteria": {"invoice": None, "other": None},
            },
            {
                "id": "tone",
                "type": "score",
                "instructions": "Tone?",
                "criteria": ["poor", "ok", "good"],
            },
        ],
        "agents": ["billing-bot"],
        "review_when": [{"question": "completed", "when": "noul_below", "value": 0.5}],
    }
    spec.update(overrides)
    return spec


def _verdict(qid: str, qtype: str, **fields) -> Verdict:
    return Verdict("task-completion", qid, qtype, "measured", **fields)  # type: ignore[arg-type]


def test_review_rules_fire_on_their_question_only() -> None:
    spec = EvaluatorSpec.model_validate(
        _spec(
            review_when=[
                {"question": "completed", "when": "noul_below", "value": 0.5},
                {"question": "intent", "when": "choice_in", "value": ["other"]},
                {"question": "tone", "when": "score_below", "value": 1.0},
            ]
        )
    )
    verdicts = [
        _verdict("completed", "noul", noul=0.2),
        _verdict("intent", "choice", choice="invoice"),
        _verdict("tone", "score", score=0.4),
    ]
    assert spec.review_reasons(verdicts) == ["completed: 0.20 < 0.5", "tone: 0.40 < 1.0"]
    passing = [_verdict("completed", "noul", noul=0.9), _verdict("tone", "score", score=1.8)]
    assert spec.review_reasons(passing) == []


def test_failed_judgements_go_to_review_only_when_asked() -> None:
    failed = [_verdict("completed", "noul", error="unreachable")]
    assert EvaluatorSpec.model_validate(_spec()).review_reasons(failed) == []
    on_error = EvaluatorSpec.model_validate(_spec(review_on_error=True))
    assert on_error.review_reasons(failed) == ["completed: judge failed (unreachable)"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"review_when": [{"question": "ghost", "when": "noul_below", "value": 0.5}]}, "unknown"),
        ({"review_when": [{"question": "tone", "when": "noul_below", "value": 0.5}]}, "score"),
        (
            {
                "questions": [
                    {"id": "q", "type": "choice", "instructions": "x", "criteria": {"a": None}}
                ],
                "review_when": [],
            },
            "two options",
        ),
        (
            {"questions": [{"id": "q", "type": "score", "instructions": "x", "criteria": ["a"]}],
             "review_when": []},
            "2 to 10",
        ),
        ({"name": "Has Spaces"}, "pattern"),
        # Rules whose value can never match would silently never fire.
        ({"review_when": [{"question": "completed", "when": "noul_above", "value": 5}]},
         "between 0 and 1"),
        ({"review_when": [{"question": "completed", "when": "noul_above", "value": ["a"]}]},
         "needs a number"),
        ({"review_when": [{"question": "intent", "when": "choice_in", "value": ["Invoice"]}]},
         "unknown options"),
        ({"review_when": [{"question": "intent", "when": "choice_in", "value": []}]},
         "list of options"),
        ({"review_when": [{"question": "intent", "when": "choice_in", "value": 0.5}]},
         "list of options"),
        ({"review_when": [{"question": "tone", "when": "score_below", "value": 7}]},
         "between 0 and 2"),
    ],
)
def test_invalid_specs_are_rejected(overrides, message) -> None:
    with pytest.raises(ValidationError, match=message):
        EvaluatorSpec.model_validate(_spec(**overrides))


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    init_engine(f"sqlite:///{tmp_path / 'evaluators.db'}")
    agents = tmp_path / "agents.json"
    monkeypatch.setenv("DOCIE_AGENTS_PATH", str(agents))
    registry = AgentRegistry(agents)
    registry.create(AgentSpec(name="billing-bot", kind="custom", model_profile="lfm2.5-1.2b"))
    registry.create(
        AgentSpec.model_validate(
            {"name": "pii-proxy", "kind": "proxy_security", "model_profile": "lfm2.5-1.2b"}
        )
    )
    yield TestClient(api.app)
    dispose_engine()


def test_crud_round_trip_and_agent_lookup(client: TestClient) -> None:
    assert client.post("/v1/studio/evaluators", json=_spec()).status_code == 201
    assert client.post("/v1/studio/evaluators", json=_spec()).status_code == 409
    assert [e["name"] for e in client.get("/v1/studio/evaluators").json()] == ["task-completion"]
    assert [s.name for s in evaluators_for_agent("billing-bot")] == ["task-completion"]
    assert evaluators_for_agent("other-agent") == []

    disabled = client.put("/v1/studio/evaluators/task-completion", json=_spec(enabled=False))
    assert disabled.status_code == 200
    assert evaluators_for_agent("billing-bot") == []

    assert client.delete("/v1/studio/evaluators/task-completion").status_code == 200
    assert client.get("/v1/studio/evaluators/task-completion").status_code == 404


def test_targets_must_be_existing_non_proxy_agents(client: TestClient) -> None:
    unknown = client.post("/v1/studio/evaluators", json=_spec(agents=["ghost"]))
    assert unknown.status_code == 422
    proxy = client.post("/v1/studio/evaluators", json=_spec(agents=["pii-proxy"]))
    assert proxy.status_code == 422
    assert "never stored" in proxy.json()["detail"]


def test_an_evaluator_cannot_be_renamed(client: TestClient) -> None:
    client.post("/v1/studio/evaluators", json=_spec())
    renamed = client.put("/v1/studio/evaluators/task-completion", json=_spec(name="other"))
    assert renamed.status_code == 422


def test_every_template_becomes_a_valid_evaluator() -> None:
    from docie_bench.evaluators.templates import CATEGORIES, TEMPLATES

    category_ids = {c["id"] for c in CATEGORIES}
    for template in TEMPLATES:
        assert template["category"] in category_ids
        if not template["questions"]:
            continue  # the from-scratch templates start empty on purpose
        fields = {
            k: template[k]
            for k in ("judge", "questions", "review_when", "reads")
            if k in template
        }
        EvaluatorSpec.model_validate({**fields, "name": template["id"], "model": "m"})


def test_templates_results_and_runs_routes(client: TestClient) -> None:
    from docie_bench.studio.agent_run_store import save_agent_run, save_evaluation

    templates = client.get("/v1/studio/evaluators/templates").json()
    assert templates["templates"][0]["id"] == "task-completion"

    run_id = save_agent_run(
        agent="billing-bot", tenant_id="acme", api_key_id=None,
        messages=[{"role": "user", "content": "hi"}], output_text="hello",
        tool_calls=[], evaluators=["task-completion"],
    )
    save_evaluation(
        run_id=str(run_id), evaluator="task-completion", agent="billing-bot",
        tenant_id="acme", verdicts=[{"question_id": "completed", "noul": 0.9}],
        review_reasons=[],
    )
    results = client.get("/v1/studio/evaluators/task-completion/results").json()
    assert results[0]["run_id"] == run_id
    assert client.get(f"/v1/studio/agent-runs/{run_id}").json()["output"] == "hello"
    assert client.get("/v1/studio/agent-runs/unknown").status_code == 404


def _as_client_key(tenant_id: str):
    from docie_bench.security import TenantContext, tenant_guard

    api.app.dependency_overrides[tenant_guard] = lambda: TenantContext(
        tenant_id=tenant_id, authenticated=True
    )


def test_a_client_key_sees_only_its_tenant_and_cannot_manage(client: TestClient) -> None:
    from docie_bench.studio.agent_run_store import save_agent_run, save_evaluation

    client.post("/v1/studio/evaluators", json=_spec())
    run_id = save_agent_run(
        agent="billing-bot", tenant_id="acme", api_key_id=None,
        messages=[], output_text="hello", tool_calls=[], evaluators=["task-completion"],
    )
    for tenant in ("acme", "acme"):  # saved twice: a retried step must not duplicate
        save_evaluation(
            run_id=str(run_id), evaluator="task-completion", agent="billing-bot",
            tenant_id=tenant, verdicts=[], review_reasons=[],
        )
    try:
        _as_client_key("globex")
        assert client.get(f"/v1/studio/agent-runs/{run_id}").status_code == 404
        assert client.get("/v1/studio/evaluators/task-completion/results").json() == []
        assert client.post("/v1/studio/evaluators", json=_spec(name="other")).status_code == 403
        assert client.delete("/v1/studio/evaluators/task-completion").status_code == 403

        _as_client_key("acme")
        assert client.get(f"/v1/studio/agent-runs/{run_id}").status_code == 200
        assert len(client.get("/v1/studio/evaluators/task-completion/results").json()) == 1
    finally:
        api.app.dependency_overrides.clear()
