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


# ── VLM-as-OCR pipeline + liteparse alias ────────────────────────────────────


async def test_pipeline_vlm_ocr_then_llm_fixes_mojibake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two-stage with a VISION deployment as the OCR step: image -> text (VLM) ->
    # JSON (extractor). The VLM's mojibake is repaired before the extractor sees it.
    from docie_bench.vision import DocumentImage

    # The doc is rasterized to PNG page images before the VLM (llama-server
    # rejects a PDF URI); stub the rasterizer — this test is about routing.
    monkeypatch.setattr(
        "docie_bench.serving.solutions.load_document_images",
        lambda path, *, pdf_dpi=150: [
            DocumentImage(page=1, media_type="image/png", data=b"PNGBYTES")
        ],
    )
    seen: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "vlm-x":  # the OCR (vision) step
            seen["ocr"] = body
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "FACTURE universitÃ©"}}]}
            )
        seen["extractor"] = body  # the extractor step
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"vendor":"x"}'}}]}
        )

    profiles = {
        "vlm_x": ModelProfile(
            name="vlm_x", model="up-vlm", base_url="http://vlm-x/v1", api_key="k"
        ),
        "llm_x": ModelProfile(
            name="llm_x", model="up-llm", base_url="http://llm-x/v1", api_key="k"
        ),
    }
    pipe = ModelProfile(
        name="pipe", model="", base_url="", api_key="", kind="pipeline",
        options={"extractor": "llm_x", "ocr_model": "vlm_x"},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        solution = build_solution(pipe, profiles=profiles, http_client=http)
        result = await solution.complete(_image_request("pipe"))

    # OCR step went to the vision deployment carrying the RASTERIZED PNG page
    # image (never the original document URI — the whole point of the fix).
    assert seen["ocr"]["model"] == "up-vlm"
    assert seen["ocr"]["messages"][0]["role"] == "system"
    assert "OCR transcription engine" in seen["ocr"]["messages"][0]["content"]
    ocr_content = seen["ocr"]["messages"][1]["content"]
    assert ocr_content[0]["type"] == "text"  # the OCR instruction
    assert "Transcribe all visible text" in ocr_content[0]["text"]
    image_urls = [p["image_url"]["url"] for p in ocr_content if p.get("type") == "image_url"]
    assert image_urls == [
        DocumentImage(page=1, media_type="image/png", data=b"PNGBYTES").data_url()
    ]

    # Extractor received the MOJIBAKE-FIXED OCR text (image swapped out).
    extractor_texts = [
        part["text"]
        for message in seen["extractor"]["messages"]
        for part in (message["content"] if isinstance(message["content"], list) else [])
        if part.get("type") == "text"
    ]
    assert any("université" in t for t in extractor_texts)
    assert not any("universitÃ©" in t for t in extractor_texts)
    assert seen["extractor"]["model"] == "up-llm"
    assert result["choices"][0]["message"]["content"] == '{"vendor":"x"}'


def test_pipeline_unknown_ocr_model_raises() -> None:
    pipe = ModelProfile(
        name="pipe", model="", base_url="", api_key="", kind="pipeline",
        options={"extractor": "llm_x", "ocr_model": "nope"},
    )
    profiles = {"llm_x": ModelProfile(name="llm_x", model="x", base_url="http://x/v1", api_key="k")}
    with pytest.raises(SolutionError, match="ocr_model 'nope' is not configured"):
        build_solution(pipe, profiles=profiles, http_client=httpx.AsyncClient())


def test_factory_liteparse_is_pdf_text_backend() -> None:
    from docie_bench.ocr.factory import get_ocr_backend
    from docie_bench.ocr.pdf_text import PdfTextBackend

    assert isinstance(get_ocr_backend("liteparse"), PdfTextBackend)
    assert isinstance(get_ocr_backend("pdf_text"), PdfTextBackend)  # legacy alias
    with pytest.raises(ValueError, match="Unknown OCR backend"):
        get_ocr_backend("nonsense")


def test_apply_no_think_sets_flag_and_merges() -> None:
    from docie_bench.serving.solutions import apply_no_think

    body: dict = {"messages": []}
    apply_no_think(body)
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert body["reasoning_effort"] == "none"
    # merges with pre-existing chat_template_kwargs.
    body2 = {"chat_template_kwargs": {"foo": 1}}
    apply_no_think(body2)
    assert body2["chat_template_kwargs"] == {"foo": 1, "enable_thinking": False}
    assert body2["reasoning_effort"] == "none"


async def test_pipeline_no_think_rides_ocr_and_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A reasoning extractor otherwise burns the budget thinking and emits no JSON;
    # no_think must ride BOTH the VLM-OCR call and the extractor call.
    from docie_bench.vision import DocumentImage

    monkeypatch.setattr(
        "docie_bench.serving.solutions.load_document_images",
        lambda path, *, pdf_dpi=150: [
            DocumentImage(page=1, media_type="image/png", data=b"P")
        ],
    )
    seen: dict[str, dict] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["vlm" if request.url.host == "vlm-x" else "ext"] = body
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    profiles = {
        "vlm_x": ModelProfile(
            name="vlm_x", model="uv", base_url="http://vlm-x/v1", api_key="k"
        ),
        "llm_x": ModelProfile(
            name="llm_x", model="ul", base_url="http://llm-x/v1", api_key="k"
        ),
    }
    pipe = ModelProfile(
        name="pipe", model="", base_url="", api_key="", kind="pipeline",
        options={"extractor": "llm_x", "ocr_model": "vlm_x", "no_think": True},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await build_solution(pipe, profiles=profiles, http_client=http).complete(
            _image_request("pipe")
        )
    assert seen["vlm"]["chat_template_kwargs"]["enable_thinking"] is False
    assert seen["ext"]["chat_template_kwargs"]["enable_thinking"] is False
    assert seen["vlm"]["reasoning_effort"] == "none"
    assert seen["ext"]["reasoning_effort"] == "none"


def test_prefill_and_repair_helpers() -> None:
    from docie_bench.serving.solutions import (
        prefill_json_object,
        repair_prefilled_content,
        wants_json_schema,
    )

    body = {"messages": [{"role": "user", "content": "x"}]}
    prefill_json_object(body)
    assert body["messages"][-1] == {"role": "assistant", "content": "{"}
    assert wants_json_schema({"type": "json_schema"}) is True
    assert wants_json_schema({"type": "json_object"}) is False
    # continuation-only content (no leading brace) gets the brace back; a
    # complete object is left untouched.
    c1 = {"choices": [{"message": {"content": ' "a": 1}'}}]}
    assert repair_prefilled_content(c1)["choices"][0]["message"]["content"] == '{ "a": 1}'
    c2 = {"choices": [{"message": {"content": '{"a": 1}'}}]}
    assert repair_prefilled_content(c2)["choices"][0]["message"]["content"] == '{"a": 1}'


async def test_pipeline_prefills_json_and_repairs_extractor_content(fake_ocr: None) -> None:
    # A schema-constrained extraction prefills the assistant "{" (suppresses a
    # reasoning ramble) and repairs a continuation-only response to valid JSON.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = json.loads(request.content)
        return httpx.Response(  # server returns only the continuation
            200, json={"choices": [{"message": {"content": ' "total_ttc": 10}'}}]}
        )

    profiles = {
        "llm_x": ModelProfile(name="llm_x", model="ul", base_url="http://llm-x/v1", api_key="k")
    }
    pipe = ModelProfile(
        name="pipe", model="", base_url="", api_key="", kind="pipeline",
        options={"ocr_backend": "tesseract", "extractor": "llm_x"},
    )
    req = _image_request("pipe")
    req["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "invoice", "schema": {}},
    }

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await build_solution(pipe, profiles=profiles, http_client=http).complete(req)

    assert seen["req"]["messages"][-1] == {"role": "assistant", "content": "{"}
    assert result["choices"][0]["message"]["content"] == '{ "total_ttc": 10}'


async def test_pipeline_retries_grammar_without_prefill_on_sampler_400(fake_ocr: None) -> None:
    # Prefill can break a model's grammar-sampler init (400 "Failed to initialize
    # samplers"). The pipeline retries the SAME grammar without the prefill —
    # keeping schema enforcement — instead of surfacing the 400.
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        prefilled = (body.get("messages") or [])[-1:] == [{"role": "assistant", "content": "{"}]
        if prefilled:
            return httpx.Response(
                400, json={"error": {"message": "Failed to initialize samplers: std::exception"}}
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"total_ttc": 5}'}}]})

    profiles = {
        "llm_x": ModelProfile(name="llm_x", model="ul", base_url="http://llm-x/v1", api_key="k")
    }
    pipe = ModelProfile(
        name="pipe", model="", base_url="", api_key="", kind="pipeline",
        options={"ocr_backend": "tesseract", "extractor": "llm_x"},
    )
    req = _image_request("pipe")
    req["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "invoice", "schema": {}},
    }

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await build_solution(pipe, profiles=profiles, http_client=http).complete(req)

    assert seen[0]["messages"][-1] == {"role": "assistant", "content": "{"}  # 1st: prefill
    assert seen[1]["messages"][-1] != {"role": "assistant", "content": "{"}  # retry: no prefill
    assert seen[1]["response_format"]["type"] == "json_schema"  # grammar kept
    assert result["choices"][0]["message"]["content"] == '{"total_ttc": 5}'
    assert len(seen) == 2
