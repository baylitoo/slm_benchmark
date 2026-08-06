"""The job queue being down must degrade honestly, never as raw 500s.

Every Studio mutation is an Inngest event; with the server unreachable the
reads all keep working, so the failure the user actually sees is "every
button errors". These tests pin the contract: a failed enqueue is an explicit
503 with an actionable detail, raised by ``send_or_503`` on every trigger
path.
"""

from __future__ import annotations

import asyncio
import typing

import pytest
from fastapi import HTTPException

from docie_bench.inngest import client as client_module
from docie_bench.inngest.client import send_or_503


class _DownClient:
    async def send(self, event: object) -> list[str]:
        raise ConnectionError("connection refused")


class _UpClient:
    def __init__(self) -> None:
        self.sent: list[object] = []

    async def send(self, event: object) -> list[str]:
        self.sent.append(event)
        return ["evt-1"]


def test_send_or_503_wraps_transport_failures() -> None:
    import inngest

    with pytest.raises(HTTPException) as exc:
        asyncio.run(send_or_503(typing.cast("inngest.Inngest", _DownClient()), inngest.Event(name="doc/extract.requested", data={})))

    assert exc.value.status_code == 503
    assert "job queue" in str(exc.value.detail)


def test_send_or_503_passes_through_when_up() -> None:
    import inngest

    up = _UpClient()
    ids = asyncio.run(send_or_503(typing.cast("inngest.Inngest", up), inngest.Event(name="doc/extract.requested", data={})))

    assert ids == ["evt-1"]
    assert len(up.sent) == 1


def test_every_trigger_path_routes_through_send_or_503() -> None:
    """No trigger route may call client.send directly — a new route bypassing
    the wrapper would regress the raw-500 behavior silently."""
    import inspect

    import docie_bench.inngest.serving_api as serving_api
    import docie_bench.inngest.studio_api as studio_api

    for module in (studio_api, serving_api):
        source = inspect.getsource(module)
        assert "inngest_client.send(" not in source, module.__name__
        assert "serving_client.send(" not in source, module.__name__


def test_realtime_token_requires_a_tenant() -> None:
    """The token route mints subscription credentials — it must be
    authenticated like every sibling trigger route (it was the one
    unauthenticated route on the studio router)."""
    import inspect

    from docie_bench.inngest.studio_api import realtime_token

    parameters = inspect.signature(realtime_token).parameters
    # `from __future__ import annotations` keeps these as strings.
    assert any(
        "TenantDependency" in str(parameter.annotation)
        for parameter in parameters.values()
    )


def test_module_client_is_untouched() -> None:
    # Sanity: the wrapper is additive; the shared clients still exist for the
    # worker's own use.
    assert client_module.inngest_client is not None
    assert client_module.serving_client is not None
