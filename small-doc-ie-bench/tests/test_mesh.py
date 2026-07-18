"""mesh-llm integration: mesh:<model> selector, status view, agent routing."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docie_bench.agents.api import configure_http_transport
from docie_bench.agents.api import router as agents_router
from docie_bench.serving.mesh import (
    MeshNotConfiguredError,
    mesh_view,
    resolve_mesh_profile,
)
from docie_bench.serving.profile_resolver import (
    ProfileResolutionError,
    resolve_extraction_profile,
)
from docie_bench.settings import get_settings

MESH_URL = "http://mesh-host:9337/v1"


@pytest.fixture()
def mesh_configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DOCIE_MESH_BASE_URL", MESH_URL)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def mesh_unconfigured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DOCIE_MESH_BASE_URL", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── selector resolution ──────────────────────────────────────────────────────


def test_resolve_mesh_profile_synthesizes_passthrough(mesh_configured) -> None:
    profile = resolve_mesh_profile("pooled-model")
    assert profile.name == "mesh:pooled-model"
    assert profile.model == "pooled-model"
    assert profile.base_url == MESH_URL
    assert profile.kind == "passthrough"


def test_resolve_mesh_profile_refuses_when_unconfigured(mesh_unconfigured) -> None:
    with pytest.raises(MeshNotConfiguredError, match="DOCIE_MESH_BASE_URL"):
        resolve_mesh_profile("pooled-model")


def test_resolver_routes_mesh_prefix(mesh_configured, tmp_path) -> None:
    # No models.yaml, no deployments — the explicit mesh: ref stands alone.
    profile = resolve_extraction_profile(
        model_profile="mesh:pooled-model",
        models_config_path=tmp_path / "missing.yaml",
        deployments=[],
    )
    assert profile.base_url == MESH_URL
    assert profile.model == "pooled-model"


def test_resolver_mesh_prefix_never_falls_through(mesh_unconfigured, tmp_path) -> None:
    with pytest.raises(ProfileResolutionError, match="DOCIE_MESH_BASE_URL"):
        resolve_extraction_profile(
            model_profile="mesh:pooled-model",
            models_config_path=tmp_path / "missing.yaml",
            deployments=[],
        )


def test_resolver_mesh_prefix_requires_model_id(mesh_configured, tmp_path) -> None:
    with pytest.raises(ProfileResolutionError, match="model id"):
        resolve_extraction_profile(
            model_profile="mesh:",
            models_config_path=tmp_path / "missing.yaml",
            deployments=[],
        )


# ── status view ──────────────────────────────────────────────────────────────


def _models_transport(status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        if status != 200:
            return httpx.Response(status, json={"error": "nope"})
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "llama-3.2-3b", "object": "model"},
                    {"id": "lfm2.5-1.2b", "object": "model"},
                ],
            },
        )

    return httpx.MockTransport(handler)


async def test_mesh_view_lists_models(mesh_configured) -> None:
    view = await mesh_view(transport=_models_transport())
    assert view["configured"] is True
    assert view["healthy"] is True
    assert view["models"] == ["lfm2.5-1.2b", "llama-3.2-3b"]


async def test_mesh_view_unconfigured(mesh_unconfigured) -> None:
    view = await mesh_view()
    assert view["configured"] is False
    assert view["healthy"] is False
    assert "not set" in view["detail"]


async def test_mesh_view_upstream_error_is_not_a_crash(mesh_configured) -> None:
    view = await mesh_view(transport=_models_transport(status=503))
    assert view["configured"] is True
    assert view["healthy"] is False
    assert "503" in view["detail"]


async def test_mesh_view_unreachable_is_not_a_crash(mesh_configured) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    view = await mesh_view(transport=httpx.MockTransport(boom))
    assert view["healthy"] is False
    assert "unreachable" in view["detail"]


# ── end to end: proxy agent backed by the mesh (real resolver, mock network) ─


@pytest.fixture()
def mesh_agent_api(tmp_path, monkeypatch, mesh_configured):
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        last = body["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"echo: {last}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    configure_http_transport(httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(agents_router)
    client = TestClient(app)
    yield client, captured
    configure_http_transport(None)


def test_proxy_agent_masks_then_routes_to_mesh(mesh_agent_api) -> None:
    """The pairing the integration exists for: local masking, pooled serving."""
    client, captured = mesh_agent_api
    created = client.post(
        "/v1/agents",
        json={
            "name": "mesh-flagger",
            "template": "proxy-security",
            "model_profile": "mesh:pooled-model",
            "options": {"mode": "placeholder"},
        },
    )
    assert created.status_code == 201, created.text
    response = client.post(
        "/v1/agents/mesh-flagger/chat/completions",
        json={"messages": [{"role": "user", "content": "mail jean@acme.fr please"}]},
    )
    assert response.status_code == 200, response.text
    sent = json.loads(captured[-1].content)
    # Only anonymized text left the node, addressed to the mesh endpoint.
    assert captured[-1].url.host == "mesh-host"
    assert sent["model"] == "pooled-model"
    assert sent["messages"][-1]["content"] == "mail [EMAIL_1] please"
    assert "jean@acme.fr" not in json.dumps(sent)
