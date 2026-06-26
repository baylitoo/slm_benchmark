from __future__ import annotations

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from docie_bench.llm.model_profiles import ModelProfile, load_model_profiles
from docie_bench.serving.gateway import create_gateway_app
from docie_bench.serving.solutions import (
    SolutionError,
    _decode_data_uri,
    build_solution,
)


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeBackend:
    def extract(self, path):  # noqa: ANN001
        assert path.exists()  # the adapter must have written the document to disk
        return [_Block("HELLO"), _Block("WORLD")]


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "docie_bench.serving.solutions.get_ocr_backend",
        lambda name, *, language=None: _FakeBackend(),
    )


def _png_data_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(b"not-a-real-png").decode()


def _ocr_profile() -> ModelProfile:
    return ModelProfile(
        name="ocr_fake", model="", base_url="", api_key="", kind="ocr",
        options={"backend": "tesseract"},
    )


def _image_request(model: str) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": _png_data_uri()}}],
            }
        ],
    }


# ── adapter unit ─────────────────────────────────────────────────────────────


def test_decode_data_uri_picks_suffix_from_mime() -> None:
    raw, suffix = _decode_data_uri("data:image/jpeg;base64," + base64.b64encode(b"x").decode())
    assert raw == b"x"
    assert suffix == ".jpg"


def test_decode_data_uri_rejects_non_data_url() -> None:
    with pytest.raises(SolutionError):
        _decode_data_uri("https://example.com/a.png")


def test_pipeline_without_extractor_raises() -> None:
    # extractor is checked before the http client, so http_client=None is fine here.
    profile = ModelProfile(name="p", model="", base_url="", api_key="", kind="pipeline")
    with pytest.raises(SolutionError):
        build_solution(profile, profiles={}, http_client=None)


@pytest.mark.usefixtures("fake_ocr")
def test_ocr_solution_returns_text_completion() -> None:
    import asyncio

    completion = asyncio.run(build_solution(_ocr_profile()).complete(_image_request("ocr_fake")))
    assert completion["choices"][0]["message"]["content"] == "HELLO\nWORLD"
    assert completion["model"] == "ocr_fake"


# ── gateway dispatch ─────────────────────────────────────────────────────────


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        "docie_bench.serving.solutions.get_ocr_backend",
        lambda name, *, language=None: _FakeBackend(),
    )
    app = create_gateway_app(profiles={"ocr_fake": _ocr_profile()})
    return TestClient(app)


def test_gateway_dispatches_ocr_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch) as client:
        resp = client.post("/v1/chat/completions", json=_image_request("ocr_fake"))
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "HELLO\nWORLD"


def test_gateway_streams_ocr_solution_as_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    request = _image_request("ocr_fake") | {"stream": True}
    with _client(monkeypatch) as client:
        resp = client.post("/v1/chat/completions", json=request)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert b"HELLO" in resp.content
    assert b"[DONE]" in resp.content


def test_gateway_solution_error_without_image_is_400(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(monkeypatch) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "ocr_fake", "messages": [{"role": "user", "content": "no image"}]},
        )
    assert resp.status_code == 400


def test_gateway_pipeline_ocr_then_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "docie_bench.serving.solutions.get_ocr_backend",
        lambda name, *, language=None: _FakeBackend(),
    )
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "c",
                "model": "up-x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"ok":1}'},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    profiles = {
        "llm_x": ModelProfile(name="llm_x", model="up-x", base_url="http://up-x/v1", api_key="k"),
        "pipe": ModelProfile(
            name="pipe", model="", base_url="", api_key="", kind="pipeline",
            options={"ocr_backend": "tesseract", "extractor": "llm_x"},
        ),
    }
    app = create_gateway_app(profiles=profiles, transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=_image_request("pipe"))

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == '{"ok":1}'
    sent = json.loads(captured[0].content)
    assert sent["model"] == "up-x"  # forwarded to the extractor's upstream id
    # the image part was replaced by a text part carrying the OCR output
    text_parts = [
        part["text"]
        for message in sent["messages"]
        for part in (message["content"] if isinstance(message["content"], list) else [])
        if part.get("type") == "text"
    ]
    assert "HELLO\nWORLD" in text_parts


# ── backward compatibility ───────────────────────────────────────────────────


def test_existing_passthrough_profiles_still_load(tmp_path) -> None:  # noqa: ANN001
    # A profile with no `kind` must load as passthrough and keep requiring base_url.
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "profiles:\n"
        "  legacy:\n"
        "    model: qwen2.5:1.5b\n"
        "    base_url: http://localhost:11434/v1\n",
        encoding="utf-8",
    )
    profiles = load_model_profiles(cfg)
    assert profiles["legacy"].kind == "passthrough"
    assert profiles["legacy"].options == {}
