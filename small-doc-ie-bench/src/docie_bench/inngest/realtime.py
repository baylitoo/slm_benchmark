"""Best-effort realtime helpers.

The frontend subscribes to a per-request *channel* and receives *topics*
(``status``, ``progress``, ``result``, ``error``) as the job runs. Publishing
is best-effort: if the experimental realtime API is missing or shaped
differently in the installed SDK version, the job still completes and its
result remains retrievable via ``GET /v1/events/{event_id}/runs``.
"""

from __future__ import annotations

import logging
from typing import Any

from docie_bench.inngest.client import _realtime, inngest_client

logger = logging.getLogger("docie_bench.inngest.realtime")

# Topics published on a run channel.
TOPIC_STATUS = "status"
TOPIC_PROGRESS = "progress"
TOPIC_RESULT = "result"
TOPIC_ERROR = "error"


async def publish(channel: str, topic: str, data: Any) -> None:
    """Publish ``data`` to ``channel``/``topic``; never raises."""
    if _realtime is None:
        return
    try:
        await _realtime.publish(
            client=inngest_client,
            channel=channel,
            topic=topic,
            data=data,
        )
    except Exception:  # noqa: BLE001 - realtime is best-effort
        logger.debug("realtime publish failed (channel=%s topic=%s)", channel, topic, exc_info=True)


async def subscription_token(channel: str, topics: list[str]) -> Any:
    """Mint a short-lived token the frontend uses to subscribe to ``channel``."""
    if _realtime is None:
        raise RuntimeError("inngest.experimental.realtime is not available in this SDK build")
    return await _realtime.get_subscription_token(
        client=inngest_client,
        channel=channel,
        topics=topics,
    )


__all__ = [
    "publish",
    "subscription_token",
    "TOPIC_STATUS",
    "TOPIC_PROGRESS",
    "TOPIC_RESULT",
    "TOPIC_ERROR",
]
