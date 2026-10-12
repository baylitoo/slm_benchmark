"""The wire shapes every OpenAI-compatible surface in this repo must agree on.

Five servers answer `/v1/chat/completions` here — the gateway, the agents
surface, the chat API, the encoders server, the transformers server and the
multi-vector server — and a consumer parses one error envelope and one stream
framing across all of them. Each had grown its own copy of both, and two modules
had resorted to importing another module's private helper, which is what a
missing shared module looks like.

Only the protocol lives here: the error envelope, the SSE framing, and the two
stream shapes (one buffered completion re-emitted as a single chunk, and a queue
of events drained until its producer signals the end). Nothing in here knows
what a model or a deployment is.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

SSE_MEDIA_TYPE = "text/event-stream"
SSE_DONE = b"data: [DONE]\n\n"
SSE_KEEPALIVE = b": keepalive\n\n"


def error_payload(message: str, error_type: str, *, code: str | None = None) -> dict[str, Any]:
    """The inner object of an OpenAI error response.

    ``code`` repeats ``error_type`` unless the caller has something more
    specific to say, such as the upstream HTTP status.
    """
    return {"message": message, "type": error_type, "code": code or error_type}


def openai_error(
    message: str, *, status_code: int, error_type: str, code: str | None = None
) -> JSONResponse:
    """An OpenAI-shaped error response: ``{"error": {message, type, code}}``."""
    return JSONResponse(
        status_code=status_code,
        content={"error": error_payload(message, error_type, code=code)},
    )


def sse_event(payload: dict[str, Any]) -> bytes:
    """One server-sent event carrying a JSON payload."""
    return f"data: {json.dumps(payload)}\n\n".encode()


def single_chunk_stream(chunk: dict[str, Any]) -> StreamingResponse:
    """A buffered completion re-emitted as a one-chunk event stream.

    For a surface that cannot stream token by token but was asked to stream:
    the client gets the streaming protocol it requested, in one frame.
    """

    async def body() -> AsyncIterator[bytes]:
        yield sse_event(chunk)
        yield SSE_DONE

    return StreamingResponse(body(), media_type=SSE_MEDIA_TYPE)


def queue_stream(
    queue: asyncio.Queue[Any],
    drive: Callable[[], Coroutine[Any, Any, None]],
    *,
    keepalive_seconds: float | None = None,
) -> StreamingResponse:
    """Stream the events ``drive`` puts on ``queue`` until it puts ``None``.

    The producer runs as a task for as long as the client reads. When the client
    disconnects, the generator is closed, the task is cancelled and awaited so
    it cannot outlive the request. Only a cancellation raised here is expected:
    ``drive`` is responsible for turning its own failures into an error event
    before its sentinel, so nothing else should surface at that await.

    ``keepalive_seconds`` emits an SSE comment whenever the producer has been
    silent that long. A run that streams no deltas -- a schema split across
    slots, say -- otherwise looks like a dead connection to a proxy, which
    drops it long before the work finishes.
    """

    async def body() -> AsyncIterator[bytes]:
        task: asyncio.Task[None] = asyncio.create_task(drive())
        try:
            while True:
                if keepalive_seconds is None:
                    item = await queue.get()
                else:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
                    except TimeoutError:
                        yield SSE_KEEPALIVE
                        continue
                if item is None:
                    break
                yield sse_event(item)
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        yield SSE_DONE

    return StreamingResponse(body(), media_type=SSE_MEDIA_TYPE)
