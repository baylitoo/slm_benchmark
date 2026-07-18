"""mesh-llm connector — pooled multi-machine capacity behind the OpenAI seam.

`mesh-llm <https://github.com/Mesh-LLM/mesh-llm>`_ pools GPUs/RAM across
machines and exposes ONE OpenAI-compatible ``/v1`` that routes by the
``model`` field to whichever peer can serve it. That is exactly the routing
contract this framework already keys everything on, so the integration is a
selector, not a runtime: ``mesh:<model>`` resolves to a passthrough profile
pointed at the configured mesh endpoint — usable anywhere a ``model_profile``
is accepted (agents' backing model, the Playground, the benchmark, the
gateway) with no models.yaml editing.

Deliberately NOT a supervised runtime kind: the mesh manages its own nodes and
placement; this side only routes to it and reports its live model list
(:func:`mesh_view` backs ``GET /v1/serving/mesh``).

Security posture: point ``DOCIE_MESH_BASE_URL`` only at a PRIVATE
(invite-token) mesh you operate — never a publicly discovered one. Prompts
routed to the mesh leave this node; the intended pairing is a
``proxy_security`` agent whose LOCAL guard encoder masks PII first, so only
placeholder-anonymized text ever reaches pooled capacity.
"""

from __future__ import annotations

from typing import Any

import httpx

from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.settings import get_settings

# The selector prefix, sibling of the placement resolver's "store:".
MESH_PROFILE_PREFIX = "mesh:"


class MeshNotConfiguredError(ValueError):
    """A mesh selector was used but DOCIE_MESH_BASE_URL is not set."""


def mesh_base_url() -> str | None:
    """The configured mesh endpoint (normalized, no trailing slash) or None."""
    raw = get_settings().mesh_base_url.strip().rstrip("/")
    return raw or None


def resolve_mesh_profile(model: str) -> ModelProfile:
    """``mesh:<model>`` -> a passthrough profile at the configured mesh.

    No network round-trip: the mesh's own router owns model placement, so the
    model id is forwarded verbatim and an unknown id surfaces as the mesh's
    404 at request time (same trust model as any remote profile). Refuses
    loudly when no mesh is configured — an explicit selector must never fall
    through to another table.
    """
    base_url = mesh_base_url()
    if base_url is None:
        raise MeshNotConfiguredError(
            f"selector 'mesh:{model}' needs DOCIE_MESH_BASE_URL "
            "(the OpenAI endpoint of your PRIVATE mesh, e.g. http://mesh-host:9337/v1)"
        )
    if not model.strip():
        raise MeshNotConfiguredError("selector 'mesh:' needs a model id after the prefix")
    settings = get_settings()
    return ModelProfile(
        # The honest label surfaced in responses/reports: routing went to the mesh.
        name=f"mesh:{model}",
        model=model,
        base_url=base_url,
        api_key=settings.mesh_api_key.get_secret_value(),
        timeout_seconds=settings.mesh_timeout_seconds,
    )


async def mesh_view(*, transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    """Live mesh status for ``GET /v1/serving/mesh``: reachability + model list.

    Never raises: an unreachable/unconfigured mesh is a normal answer, not a
    500 — the UI renders the ``detail`` instead of a model list. ``transport``
    is the test seam (httpx.MockTransport).
    """
    base_url = mesh_base_url()
    if base_url is None:
        return {
            "configured": False,
            "endpoint": None,
            "healthy": False,
            "models": [],
            "detail": "DOCIE_MESH_BASE_URL is not set — mesh routing is disabled.",
        }
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.mesh_api_key.get_secret_value()}"}
    try:
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.get(
                f"{base_url}/models", headers=headers, timeout=10.0
            )
    except httpx.RequestError as exc:
        return {
            "configured": True,
            "endpoint": base_url,
            "healthy": False,
            "models": [],
            "detail": f"mesh endpoint is unreachable: {exc}",
        }
    if response.status_code >= 400:
        return {
            "configured": True,
            "endpoint": base_url,
            "healthy": False,
            "models": [],
            "detail": f"mesh /models returned HTTP {response.status_code}",
        }
    models: list[str] = []
    try:
        payload = response.json()
        for item in payload.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
    except ValueError:
        return {
            "configured": True,
            "endpoint": base_url,
            "healthy": False,
            "models": [],
            "detail": "mesh /models returned a non-JSON response",
        }
    return {
        "configured": True,
        "endpoint": base_url,
        "healthy": True,
        "models": sorted(models),
        "detail": None,
    }
