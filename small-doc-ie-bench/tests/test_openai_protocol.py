"""The wire shapes shared by every OpenAI-compatible surface."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from fastapi.responses import StreamingResponse

from docie_bench.openai_protocol import (
    SSE_DONE,
    SSE_KEEPALIVE,
    error_payload,
    openai_error,
    queue_stream,
    single_chunk_stream,
    sse_event,
)


async def _collect(response: StreamingResponse) -> list[bytes]:
    generator = cast(AsyncGenerator[bytes, None], response.body_iterator)
    return [chunk async for chunk in generator]


def test_the_error_code_repeats_the_type_unless_one_is_given() -> None:
    assert error_payload("boom", "internal_error") == {
        "message": "boom",
        "type": "internal_error",
        "code": "internal_error",
    }
    assert error_payload("boom", "extraction_error", code="502")["code"] == "502"


def test_the_error_response_wraps_the_payload_under_error() -> None:
    response = openai_error("nope", status_code=400, error_type="invalid_request_error")
    assert response.status_code == 400
    assert json.loads(bytes(response.body)) == {
        "error": {
            "message": "nope",
            "type": "invalid_request_error",
            "code": "invalid_request_error",
        }
    }


def test_an_event_is_one_json_frame() -> None:
    assert sse_event({"a": 1}) == b'data: {"a": 1}\n\n'


@pytest.mark.asyncio
async def test_a_buffered_completion_streams_as_one_chunk_then_done() -> None:
    response = single_chunk_stream({"id": "chatcmpl-1"})
    assert response.media_type == "text/event-stream"
    assert await _collect(response) == [sse_event({"id": "chatcmpl-1"}), SSE_DONE]


@pytest.mark.asyncio
async def test_the_queue_is_drained_until_the_producer_signals_the_end() -> None:
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def drive() -> None:
        queue.put_nowait({"type": "delta", "n": 1})
        queue.put_nowait({"type": "delta", "n": 2})
        queue.put_nowait(None)

    chunks = await _collect(queue_stream(queue, drive))
    assert chunks == [
        sse_event({"type": "delta", "n": 1}),
        sse_event({"type": "delta", "n": 2}),
        SSE_DONE,
    ]


@pytest.mark.asyncio
async def test_a_silent_producer_is_covered_by_keepalive_comments() -> None:
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def drive() -> None:
        await asyncio.sleep(0.05)
        queue.put_nowait({"type": "result"})
        queue.put_nowait(None)

    chunks = await _collect(queue_stream(queue, drive, keepalive_seconds=0.01))
    assert SSE_KEEPALIVE in chunks
    assert chunks[-2:] == [sse_event({"type": "result"}), SSE_DONE]


@pytest.mark.asyncio
async def test_closing_the_stream_early_cancels_the_producer() -> None:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    started = asyncio.Event()
    finished = False

    async def drive() -> None:
        nonlocal finished
        started.set()
        await asyncio.sleep(10)
        finished = True

    response = queue_stream(queue, drive)
    generator = cast(AsyncGenerator[bytes, None], response.body_iterator)
    queue.put_nowait({"type": "delta"})
    assert await generator.__anext__() == sse_event({"type": "delta"})
    await started.wait()
    await generator.aclose()
    assert finished is False
