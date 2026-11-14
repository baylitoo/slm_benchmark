"""Decision-model families, their detection, and the /v1/systemone surface."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docie_bench.agents.api import configure_http_transport
from docie_bench.chat_api import router as chat_router
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.arch_registry import resolve_family
from docie_bench.serving.hf_hub import _runtime_for_family
from docie_bench.serving.model_store import FAMILIES, get_family
from docie_bench.serving.profile_resolver import ProfileResolutionError

# Hub metadata as ggml-org publishes it (checked against the live API).
LAYA_TAGS = ("gguf", "decision-model", "text-classification", "feature-extraction")
KEV_TAGS = ("gguf", "decision-model", "text-classification", "conversational")
OPENJEV_TAGS = ("gguf", "decision-model", "image-text-to-text", "conversational")


def test_decision_families_contract() -> None:
    text = get_family("decision")
    vision = get_family("decision_vl")
    assert text.decision
    assert vision.decision
    assert text.llama_server_args == ()  # GGUF metadata switches the server
    assert not text.needs_mmproj
    assert vision.needs_mmproj
    assert not text.embedding
    assert not text.reranker
    assert not text.snapshot
    assert sorted(n for n, f in FAMILIES.items() if f.decision) == ["decision", "decision_vl"]
    assert _runtime_for_family("decision", has_gguf=True) == "llama.cpp"


def test_tagged_encoder_gguf_routes_to_decision() -> None:
    verdict = resolve_family(
        "modern-bert",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="ggml-org/Laya-GGUF",
        tags=LAYA_TAGS,
        pipeline_tag="text-classification",
    )
    assert (verdict.verdict, verdict.family) == ("supported", "decision")
    assert "#29818" in (verdict.runtime_note or "")


def test_tagged_qwen35_is_decision_not_chat() -> None:
    verdict = resolve_family(
        "qwen35",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="ggml-org/Kev-4B-GGUF",
        tags=KEV_TAGS,
        pipeline_tag="text-classification",
    )
    assert (verdict.verdict, verdict.family) == ("supported", "decision")


def test_tagged_with_projector_routes_to_decision_vl() -> None:
    verdict = resolve_family(
        "qwen35",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=True,
        repo_id="ggml-org/OpenJev-GGUF",
        tags=OPENJEV_TAGS,
        pipeline_tag="image-text-to-text",
    )
    assert (verdict.verdict, verdict.family) == ("supported", "decision_vl")


def test_untagged_classification_encoder_defaults_to_decision_for_confirmation() -> None:
    verdict = resolve_family(
        "bert",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="someone/intent-classifier-GGUF",
        pipeline_tag="text-classification",
    )
    assert (verdict.verdict, verdict.family) == ("needs_family", "decision")


def test_safetensors_decision_model_points_to_a_gguf() -> None:
    verdict = resolve_family(
        "modernbert",
        has_gguf=False,
        has_safetensors=True,
        has_mmproj=False,
        repo_id="convaiinnovations/laya",
        tags=("decision-model",),
    )
    assert (verdict.verdict, verdict.family) == ("needs_family", "decision")
    assert "GGUF" in verdict.reason


@pytest.mark.parametrize(
    ("arch", "kwargs", "family"),
    [
        # An embedding encoder keeps its embedding route.
        ("bert", {"pipeline_tag": "sentence-similarity"}, "embedding"),
        # A cross-encoder tagged text-classification stays a reranker.
        (
            "bert",
            {"pipeline_tag": "text-classification", "tags": ("cross-encoder",)},
            "reranker",
        ),
    ],
)
def test_non_decision_encoders_keep_their_family(arch, kwargs, family) -> None:
    verdict = resolve_family(
        arch, has_gguf=True, has_safetensors=False, has_mmproj=False, repo_id="x/y", **kwargs
    )
    assert verdict.family == family


DECIDER = ModelProfile(
    name="laya", model="laya-served", base_url="http://upstream/v1", api_key="k"
)

QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": None, "shipping": None},
    },
    "urgent": {"type": "noul", "instructions": "Needs a human today?"},
}


@pytest.fixture
def api(monkeypatch) -> tuple[TestClient, list[httpx.Request]]:
    def fake_resolver(*, model_profile: str | None = None, **_: object) -> ModelProfile:
        if model_profile == "laya":
            return DECIDER
        raise ProfileResolutionError(f"model {model_profile!r} is not routable")

    monkeypatch.setattr("docie_bench.chat_api.resolve_extraction_profile", fake_resolver)
    recorded: list[dict] = []
    monkeypatch.setattr(
        "docie_bench.chat_api.usage_store.record_usage", lambda **kw: recorded.append(kw)
    )
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "laya-served",
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.97, "shipping": 0.03},
                        "confidence": 0.94,
                    },
                    "urgent": {"type": "noul", "noul": 0.2},
                },
                "usage": {"input_tokens": 120, "output_tokens": 0},
            },
        )

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)
    client.recorded = recorded  # type: ignore[attr-defined]
    yield client, captured
    configure_http_transport(None)


def test_systemone_forwards_with_served_id_and_records_tokens(api) -> None:
    client, captured = api
    response = client.post(
        "/v1/systemone",
        json={"model": "laya", "state": "Charged twice for order 4471", "questions": QUESTIONS},
    )
    assert response.status_code == 200, response.text
    assert response.json()["answers"]["route"]["choice"] == "billing"
    assert captured[-1].url.path == "/v1/systemone"
    sent = json.loads(captured[-1].content)
    assert sent["model"] == "laya-served"
    assert sent["questions"] == QUESTIONS
    row = client.recorded[-1]  # type: ignore[attr-defined]
    assert (row["surface"], row["prompt_tokens"], row["status"]) == ("decide", 120, "ok")


def test_systemone_accepts_image_only_request(api) -> None:
    client, _ = api
    response = client.post(
        "/v1/systemone",
        json={"model": "laya", "images": ["data:image/png;base64,AA=="], "questions": QUESTIONS},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {"state": "x", "questions": QUESTIONS},
        {"model": "laya", "questions": QUESTIONS},
        {"model": "laya", "state": "x"},
        {"model": "laya", "state": "x", "questions": {}},
        {"model": "laya", "state": "x", "questions": {"q": {"type": "rank"}}},
        {"model": "laya", "state": "x", "questions": {"q": "yes?"}},
    ],
)
def test_systemone_rejects_malformed_requests(api, body) -> None:
    client, captured = api
    assert client.post("/v1/systemone", json=body).status_code == 400
    assert not captured


def test_systemone_unknown_model_is_404(api) -> None:
    client, _ = api
    response = client.post(
        "/v1/systemone", json={"model": "ghost", "state": "x", "questions": QUESTIONS}
    )
    assert response.status_code == 404
