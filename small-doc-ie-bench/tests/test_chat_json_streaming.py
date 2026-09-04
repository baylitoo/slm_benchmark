"""``chat_json``'s additive streaming path (``on_delta``/``on_reset``, #397).

Mirrors ``chat_api.py``'s ``_post_upstream_streamed`` test style: exercise the
new streaming request maker directly against a fake multi-chunk SSE upstream,
and compare its reconstruction against the blocking path's own output for the
same logical content — the highest-risk part of any accumulate-then-replay
change. Also proves the two behaviors #397 explicitly calls out: an empty
stream must hit the SAME response-format downgrade the blocking path takes
for empty content (not a hard content-must-be-text raise), and a retry after
deltas already streamed must fire ``on_reset`` before the next attempt's own
deltas arrive.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from docie_bench.llm.model_gateway import reset_gateway_state
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.openai_client import OpenAICompatibleClient


@pytest.fixture(autouse=True)
def _reset_gateway() -> None:
    reset_gateway_state()


def _profile(**overrides: Any) -> ModelProfile:
    values: dict[str, Any] = {
        "name": "lfm2.5-350m",
        "model": "lfm2.5-350m-served",
        "base_url": "http://model.test/v1",
        "api_key": "test",
        "response_format_style": "openai_json_schema",
        "retry_max_attempts": 1,
        "retry_backoff_base_seconds": 0,
        "retry_backoff_max_seconds": 0,
        "queue_timeout_seconds": 1,
        "capability_discovery": "disabled",
    }
    values.update(overrides)
    return ModelProfile(**values)


async def _client(profile: ModelProfile, handler: Any) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(profile)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=profile.base_url, transport=httpx.MockTransport(handler)
    )
    return client


def _sse_body(*frames: dict[str, Any]) -> bytes:
    body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
    return (body + "data: [DONE]\n\n").encode()


async def test_streamed_and_blocking_reconstructions_are_byte_identical() -> None:
    frames = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": '{"vendor'}}]},
        {"choices": [{"index": 0, "delta": {"content": '_name": "'}}]},
        {"choices": [{"index": 0, "delta": {"content": 'Acme"}'}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
    ]

    def streaming_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body(*frames)
        )

    def blocking_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"vendor_name": "Acme"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    deltas: list[str] = []
    streamed_client = await _client(_profile(), streaming_handler)
    try:
        streamed_result, streamed_usage, streamed_raw = await streamed_client.chat_json(
            system_prompt="s",
            user_prompt="u",
            schema_name="invoice",
            schema={"type": "object"},
            on_delta=deltas.append,
        )
    finally:
        await streamed_client.aclose()

    blocking_client = await _client(_profile(), blocking_handler)
    try:
        blocking_result, blocking_usage, blocking_raw = await blocking_client.chat_json(
            system_prompt="s",
            user_prompt="u",
            schema_name="invoice",
            schema={"type": "object"},
        )
    finally:
        await blocking_client.aclose()

    assert deltas == ['{"vendor', '_name": "', 'Acme"}']
    assert streamed_result == blocking_result == {"vendor_name": "Acme"}
    assert streamed_usage == blocking_usage
    assert (
        streamed_raw["choices"][0]["message"]["content"]
        == blocking_raw["choices"][0]["message"]["content"]
    )


async def test_empty_stream_downgrades_to_the_next_rung_not_a_hard_raise() -> None:
    # json_schema streams zero content deltas (small-Ollama-style empty
    # defect, but over SSE); json_object streams a valid completion. Content
    # must reconstruct to "" (never None) for this to hit the SAME downgrade
    # path _test_response_negotiation.py already locks in for the blocking
    # case, not InvalidModelResponseError("content must be text").
    styles_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode("utf-8"))
        rf = body.get("response_format")
        style = rf.get("type") if rf else "none"
        styles_seen.append(style)
        if style == "json_schema":
            empty_frame = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse_body(empty_frame),
            )
        frames = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": '{"ok": true}'}}
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body(*frames)
        )

    deltas: list[str] = []
    client = await _client(_profile(), handler)
    try:
        result, _usage, _raw = await client.chat_json(
            system_prompt="s",
            user_prompt="u",
            schema_name="invoice",
            schema={"type": "object"},
            on_delta=deltas.append,
        )
    finally:
        await client.aclose()

    assert result == {"ok": True}
    assert styles_seen == ["json_schema", "json_object"]
    assert deltas == ['{"ok": true}']


async def test_on_reset_fires_before_the_next_attempts_deltas_after_a_downgrade() -> None:
    # First attempt (json_schema) streams a garbage/abandoned fragment that
    # never parses as JSON, so chat_json downgrades to the next rung
    # (json_object); that retry must see on_reset() BEFORE any of its own
    # deltas, or the abandoned fragment stays painted alongside the new one.
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode("utf-8"))
        rf = body.get("response_format")
        style = rf.get("type") if rf else "none"
        if style == "json_schema":
            frames = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"}}]}
            ]
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse_body(*frames)
            )
        # json_object: empty content -> also downgrades (no further rungs
        # accept a grammar 400 for "none", so end the ladder cleanly here).
        frames = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": '{"a": 1}'}}
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_body(*frames)
        )

    events: list[tuple[str, str | None]] = []
    client = await _client(_profile(), handler)
    try:
        result, _usage, _raw = await client.chat_json(
            system_prompt="s",
            user_prompt="u",
            schema_name="invoice",
            schema={"type": "object"},
            on_delta=lambda text: events.append(("delta", text)),
            on_reset=lambda: events.append(("reset", None)),
        )
    finally:
        await client.aclose()

    assert result == {"a": 1}
    # The first attempt's fragment streamed, THEN a reset cleared it before
    # the second attempt's own (complete, valid) delta arrived.
    assert events == [
        ("delta", "partial"),
        ("reset", None),
        ("delta", '{"a": 1}'),
    ]


async def test_on_delta_none_is_untouched_blocking_behavior() -> None:
    """Sanity check: omitting on_delta takes the exact prior code path."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"x": 1}'}, "finish_reason": "stop"}
                ],
                "usage": {},
            },
        )

    client = await _client(_profile(), handler)
    try:
        result, _usage, _raw = await client.chat_json(
            system_prompt="s",
            user_prompt="u",
            schema_name="invoice",
            schema={"type": "object"},
        )
    finally:
        await client.aclose()
    assert result == {"x": 1}
