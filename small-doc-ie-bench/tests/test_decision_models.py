"""Decision-model families, their detection, and the /v1/systemone surface."""

from __future__ import annotations

import asyncio
import json
import struct

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docie_bench.agents.api import configure_http_transport
from docie_bench.chat_api import router as chat_router
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.arch_registry import resolve_family
from docie_bench.serving.hf_hub import (
    HfGgufFile,
    HfHubError,
    _decision_header_candidate,
    _runtime_for_family,
    read_gguf_decision_type,
)
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


def test_header_decision_type_routes_encoder_gguf_to_decision() -> None:
    verdict = resolve_family(
        "modern-bert",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="ggml-org/Laya-GGUF",
        tags=LAYA_TAGS,
        pipeline_tag="text-classification",
        gguf_decision_type="laya",
        gguf_header_read=True,
    )
    assert (verdict.verdict, verdict.family) == ("supported", "decision")
    assert "#29818" in (verdict.runtime_note or "")


def test_header_decision_type_beats_the_qwen35_chat_route() -> None:
    verdict = resolve_family(
        "qwen35",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="ggml-org/Kev-4B-GGUF",
        tags=KEV_TAGS,
        gguf_decision_type="kev",
        gguf_header_read=True,
    )
    assert (verdict.verdict, verdict.family) == ("supported", "decision")


def test_header_decision_type_with_projector_routes_to_decision_vl() -> None:
    verdict = resolve_family(
        "qwen35",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=True,
        repo_id="ggml-org/OpenJev-GGUF",
        tags=OPENJEV_TAGS,
        pipeline_tag="image-text-to-text",
        gguf_decision_type="openjev",
        gguf_header_read=True,
    )
    assert (verdict.verdict, verdict.family) == ("supported", "decision_vl")


def test_tagged_chat_finetune_without_decision_header_stays_chat() -> None:
    """Most repos tagged decision-model on the Hub are chat fine-tunes that
    answer in text; the header, not the tag, decides."""
    verdict = resolve_family(
        "qwen35",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="someone/Jev-Style-2B-Decision-GGUF",
        tags=("gguf", "decision-model", "text-classification"),
        pipeline_tag="text-classification",
        gguf_decision_type=None,
        gguf_header_read=True,
    )
    assert verdict.family == "openai_chat"


def test_tag_with_unread_header_needs_confirmation() -> None:
    verdict = resolve_family(
        "qwen35",
        has_gguf=True,
        has_safetensors=False,
        has_mmproj=False,
        repo_id="ggml-org/Kev-4B-GGUF",
        tags=KEV_TAGS,
    )
    assert (verdict.verdict, verdict.family) == ("needs_family", "decision")


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


def _gguf_header(kv: list[tuple[str, int, object]]) -> bytes:
    """A GGUF metadata block: string (8), u32 (4) and string-array (9) values."""

    def s(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
    for key, value_type, value in kv:
        out += s(key) + struct.pack("<I", value_type)
        if value_type == 8:
            out += s(str(value))
        elif value_type == 4:
            out += struct.pack("<I", int(value))  # type: ignore[call-overload]
        elif value_type == 9:
            items = list(value)  # type: ignore[call-overload]
            out += struct.pack("<I", 8) + struct.pack("<Q", len(items))
            out += b"".join(s(i) for i in items)
    return out + b"\0" * 64  # tensor info follows; never read


def _hub(body: bytes) -> httpx.AsyncClient:
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, content=body))
    return httpx.AsyncClient(transport=transport)


def test_header_reader_finds_decision_type_after_the_tokenizer() -> None:
    """Some conversions write decision.* after tokenizer.*, so the reader
    scans the whole metadata block."""
    body = _gguf_header(
        [
            ("general.architecture", 8, "qwen35"),
            ("qwen35.block_count", 4, 24),
            ("tokenizer.ggml.tokens", 9, ["a", "b", "c"]),
            ("qwen35.decision.type", 8, "kev"),
        ]
    )

    async def run() -> str | None:
        async with _hub(body) as client:
            return await read_gguf_decision_type("o/r", "m.gguf", client=client)

    assert asyncio.run(run()) == "kev"


def test_header_reader_returns_none_for_a_chat_model() -> None:
    body = _gguf_header(
        [("general.architecture", 8, "qwen35"), ("tokenizer.ggml.tokens", 9, ["a"])]
    )

    async def run() -> str | None:
        async with _hub(body) as client:
            return await read_gguf_decision_type("o/r", "m.gguf", client=client)

    assert asyncio.run(run()) is None


def test_header_reader_rejects_a_non_gguf_file() -> None:
    async def run() -> str | None:
        async with _hub(b"<html>not found</html>") as client:
            return await read_gguf_decision_type("o/r", "m.gguf", client=client)

    with pytest.raises(HfHubError):
        asyncio.run(run())


def test_header_candidate_skips_imatrix_and_prefers_the_decision_build() -> None:
    def f(name: str, size: int) -> HfGgufFile:
        return HfGgufFile(name, size, None, False, False)

    imatrix = [f("d3.imatrix.gguf", 27_000_000), f("d3.i1-IQ1_S.gguf", 7_000_000_000)]
    pick = _decision_header_candidate(imatrix, "qwen35", ["decision-model"], None)
    assert pick is not None
    assert pick.filename == "d3.i1-IQ1_S.gguf"

    mixed = [f("basal-Q5_K_M.gguf", 3_419_046_560), f("basal-Q5_K_M-decision.gguf", 3_419_049_632)]
    pick = _decision_header_candidate(mixed, "qwen35", ["decision-model"], None)
    assert pick is not None
    assert pick.filename == "basal-Q5_K_M-decision.gguf"


def test_header_is_not_read_without_a_hint() -> None:
    files = [HfGgufFile("chat-Q4_K_M.gguf", 1, "Q4_K_M", False, False)]
    assert _decision_header_candidate(files, "qwen35", ["gguf"], "text-generation") is None
