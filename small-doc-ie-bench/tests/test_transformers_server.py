"""Transformers (AutoModel) last-resort shim: server contract + image handling.

The real backend imports torch/transformers, so these tests inject a fake
backend and cover the server's request/response shaping and eager image
validation — the same seam ``test_encoders`` uses. Backend correctness is
verified empirically at deploy (like the encoder path).
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docie_bench.transformers_server.server import (
    _processor_is_multimodal,
    create_transformers_app,
    split_prompt,
    to_transformers_messages,
)

# A 1x1 PNG, base64 — enough to exercise the data: URI decode path.
_PNG_1x1 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000100000001080600000"
        "01f15c4890000000d49444154789c6360000002000155a2b4e00000000"
        "049454e44ae426082"
    )
).decode()
_DATA_URI = f"data:image/png;base64,{_PNG_1x1}"


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], int, float]] = []

    def generate(
        self, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float
    ) -> str:
        self.calls.append((messages, max_tokens, temperature))
        return "hello from the model"


@pytest.fixture
def tf_client() -> tuple[TestClient, FakeBackend]:
    backend = FakeBackend()
    app = create_transformers_app(model_id="fake-tf", backend=backend)
    return TestClient(app), backend


def test_healthz_and_models(tf_client) -> None:
    client, _ = tf_client
    assert client.get("/healthz").json()["kind"] == "transformers"
    models = client.get("/v1/models").json()
    assert [m["id"] for m in models["data"]] == ["fake-tf"]


def test_chat_returns_openai_completion(tf_client) -> None:
    client, backend = tf_client
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello from the model"
    assert body["choices"][0]["finish_reason"] == "stop"
    # max_tokens forwarded; default temperature applied.
    _, max_tokens, temperature = backend.calls[-1]
    assert max_tokens == 32
    assert temperature == 0.0


def test_max_completion_tokens_alias(tf_client) -> None:
    client, backend = tf_client
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 7},
    )
    assert backend.calls[-1][1] == 7


def test_multimodal_request_reaches_backend(tf_client) -> None:
    # SERVER contract only: the server passes the OpenAI messages to the backend
    # (which owns the transformers conversion — see to_transformers_messages,
    # tested separately). This does NOT assert vision generation works end to
    # end; that needs a real checkpoint and is validated at deploy.
    client, backend = tf_client
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": _DATA_URI}},
                    ],
                }
            ]
        },
    )
    assert response.status_code == 200
    messages = backend.calls[-1][0]
    assert messages[0]["content"][1]["type"] == "image_url"


def test_generate_failure_is_a_clean_openai_error_not_a_crash(tf_client) -> None:
    # A custom-code checkpoint (trust_remote_code) whose calling convention
    # diverges from the standard apply_chat_template contract this shim
    # assumes generically fails INSIDE generate() -- e.g. an AttributeError,
    # not a ValueError. That must still come back OpenAI-shaped, not an
    # unhandled 500 with a bare traceback.
    client, backend = tf_client

    def broken_generate(messages, *, max_tokens, temperature):
        raise AttributeError("'OvisModel' object has no attribute 'apply_chat_template'")

    backend.generate = broken_generate
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["type"] == "backend_error"
    assert "apply_chat_template" in body["error"]["message"]


def test_empty_messages_is_400(tf_client) -> None:
    client, _ = tf_client
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_remote_image_url_rejected_400(tf_client) -> None:
    client, backend = tf_client
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
                    ],
                }
            ]
        },
    )
    assert response.status_code == 400
    assert "data:" in response.json()["error"]["message"]
    # Rejected before ever reaching the backend.
    assert backend.calls == []


def test_bad_base64_image_is_400(tf_client) -> None:
    client, _ = tf_client
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,!!!notb64"},
                        }
                    ],
                }
            ]
        },
    )
    assert response.status_code == 400


def test_split_prompt_collects_image_uris() -> None:
    messages = [
        {"role": "system", "content": "be brief"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": _DATA_URI}},
            ],
        },
    ]
    _, image_urls = split_prompt(messages)
    assert image_urls == [_DATA_URI]


def test_split_prompt_rejects_imageurl_without_url() -> None:
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
    with pytest.raises(ValueError, match="missing its 'url'"):
        split_prompt(messages)


# ── OpenAI → transformers message conversion (the backend's contract) ────────


def test_to_transformers_rewrites_image_to_placeholder() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": _DATA_URI}},
            ],
        }
    ]
    tf_messages, image_urls = to_transformers_messages(messages)
    # The image_url part becomes a transformers {"type":"image"} PLACEHOLDER;
    # the pixels ride the returned image_urls (positionally aligned).
    parts = tf_messages[0]["content"]
    assert parts[0] == {"type": "text", "text": "describe"}
    assert parts[1] == {"type": "image"}
    assert image_urls == [_DATA_URI]


def test_to_transformers_passes_string_content_through() -> None:
    messages = [{"role": "system", "content": "be brief"}]
    tf_messages, image_urls = to_transformers_messages(messages)
    assert tf_messages == messages
    assert image_urls == []


# ── vision detection: processor SHAPE, not which AutoClass succeeded ────────


class _FakeVlmProcessor:
    """Shaped like a real transformers ProcessorMixin composition (e.g. a
    Qwen2-VL or a custom-code VLM's processor) -- carries image_processor."""

    def __init__(self) -> None:
        self.image_processor = object()
        self.tokenizer = object()


class _FakeTextTokenizer:
    """A bare tokenizer (AutoTokenizer fallback) -- no image_processor."""


def test_processor_is_multimodal_true_for_a_vlm_processor() -> None:
    assert _processor_is_multimodal(_FakeVlmProcessor()) is True


def test_processor_is_multimodal_false_for_a_bare_tokenizer() -> None:
    # This is the case that used to be misclassified: a custom-code VLM
    # registered only under AutoModelForCausalLM (Ovis's known pattern)
    # still loads a real multimodal processor, so this check must key off
    # the PROCESSOR's own shape, not which AutoModel constructor succeeded.
    assert _processor_is_multimodal(_FakeTextTokenizer()) is False


def test_to_transformers_preserves_multi_image_order() -> None:
    other = "data:image/png;base64,AAAA"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _DATA_URI}},
                {"type": "text", "text": "vs"},
                {"type": "image_url", "image_url": {"url": other}},
            ],
        }
    ]
    tf_messages, image_urls = to_transformers_messages(messages)
    types = [p["type"] for p in tf_messages[0]["content"]]
    assert types == ["image", "text", "image"]
    assert image_urls == [_DATA_URI, other]
