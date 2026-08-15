"""GET/POST/DELETE /v1/studio/routing-policies -- the missing "define once,
reuse by name" persistence layer for RoutingPolicy (extract/routing.py), which
was already fully validated and already drove benchmark runs, but only ever
from a server-side filesystem path until now.

Worker-side resolution (a saved policy name -> routing_policy_path at
benchmark-trigger time) is covered in test_studio_artifacts.py alongside the
rest of _run_benchmark_job's tests, not here."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import docie_bench.api as api
from docie_bench.storage.db import dispose_engine, init_engine


@pytest.fixture(autouse=True)
def policy_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'routing_policies.db'}")
    yield
    dispose_engine()


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def _payload(name: str = "cpu-cascade") -> dict:
    return {
        "name": name,
        "policy": {
            "version": "cpu-cascade-1",
            "stages": [
                {
                    "name": "ollama_qwen25_1b",
                    "rules": [
                        {
                            "when": {"status": "success", "validation_valid": True},
                            "decision": "accept",
                            "reason": "cleared the quality gate",
                        }
                    ],
                }
            ],
        },
    }


def test_creates_and_lists_a_routing_policy(client: TestClient) -> None:
    created = client.post("/v1/studio/routing-policies", json=_payload())
    assert created.status_code == 201
    assert created.json()["name"] == "cpu-cascade"

    listed = client.get("/v1/studio/routing-policies")
    assert listed.status_code == 200
    names = {p["name"] for p in listed.json()}
    assert names == {"cpu-cascade"}


def test_fetches_one_policy_by_name(client: TestClient) -> None:
    client.post("/v1/studio/routing-policies", json=_payload())

    resp = client.get("/v1/studio/routing-policies/cpu-cascade")

    assert resp.status_code == 200
    assert resp.json()["policy"]["version"] == "cpu-cascade-1"


def test_fetch_unknown_name_is_404(client: TestClient) -> None:
    resp = client.get("/v1/studio/routing-policies/does_not_exist")

    assert resp.status_code == 404
    assert resp.headers.get("X-Docie-Error") == "not_found"


def test_duplicate_name_is_409(client: TestClient) -> None:
    client.post("/v1/studio/routing-policies", json=_payload())

    resp = client.post("/v1/studio/routing-policies", json=_payload())

    assert resp.status_code == 409


def test_invalid_policy_is_422(client: TestClient) -> None:
    # Duplicate stage names -- RoutingPolicy.validate_stages rejects this.
    resp = client.post(
        "/v1/studio/routing-policies",
        json={
            "name": "bad-policy",
            "policy": {
                "stages": [
                    {"name": "dup", "rules": []},
                    {"name": "dup", "rules": []},
                ]
            },
        },
    )

    assert resp.status_code == 422


def test_over_length_name_is_422_not_a_db_error(client: TestClient) -> None:
    # Mirrors the routing_label overflow PR #194's review caught: reject an
    # over-length name at the API layer (String(64) column) rather than
    # letting it reach the DB as a DataError.
    resp = client.post("/v1/studio/routing-policies", json=_payload(name="x" * 65))

    assert resp.status_code == 422


def test_deletes_a_policy(client: TestClient) -> None:
    client.post("/v1/studio/routing-policies", json=_payload())

    resp = client.delete("/v1/studio/routing-policies/cpu-cascade")

    assert resp.status_code == 200
    assert client.get("/v1/studio/routing-policies/cpu-cascade").status_code == 404


def test_delete_unknown_name_is_404(client: TestClient) -> None:
    resp = client.delete("/v1/studio/routing-policies/does_not_exist")

    assert resp.status_code == 404
    assert resp.headers.get("X-Docie-Error") == "not_found"


def test_no_database_returns_empty_list_not_500(client: TestClient) -> None:
    dispose_engine()

    resp = client.get("/v1/studio/routing-policies")

    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Benchmark trigger: routing_policy / routing_policy_name / model_profile are
# mutually exclusive
# ---------------------------------------------------------------------------


def test_trigger_benchmark_rejects_routing_policy_name_with_model_profile(
    client: TestClient,
) -> None:
    resp = client.post(
        "/v1/studio/benchmark",
        json={"dataset": "ds", "model_profile": "p1", "routing_policy_name": "cpu-cascade"},
    )

    assert resp.status_code == 422


def test_trigger_benchmark_rejects_routing_policy_with_routing_policy_name(
    client: TestClient,
) -> None:
    resp = client.post(
        "/v1/studio/benchmark",
        json={
            "dataset": "ds",
            "routing_policy": "configs/routing-policy.example.yaml",
            "routing_policy_name": "cpu-cascade",
        },
    )

    assert resp.status_code == 422
