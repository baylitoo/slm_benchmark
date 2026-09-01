"""MCP catalog + registry management: browse, enable, disable, test."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from docie_bench import mcp_tools
from docie_bench.agents.api import agents_tenant_guard
from docie_bench.mcp_catalog import CATALOG, registry_entry_for
from docie_bench.settings import get_settings

router = APIRouter(tags=["mcp"], dependencies=[Depends(agents_tenant_guard)])


def _registry() -> dict[str, mcp_tools.MCPServerSpec]:
    try:
        return mcp_tools.load_mcp_registry()
    except mcp_tools.MCPConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/v1/mcp/catalog")
async def list_catalog() -> dict[str, Any]:
    enabled = set(_registry())
    return {
        "entries": [
            {
                "name": entry.name,
                "title": entry.title,
                "description": entry.description,
                "tools": list(entry.tools),
                "params": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "required": p.required,
                        "secret": p.secret,
                    }
                    for p in entry.params
                ],
                "enabled": entry.name in enabled,
            }
            for entry in CATALOG.values()
        ]
    }


class EnableRequest(BaseModel):
    catalog: str
    params: dict[str, str] = Field(default_factory=dict)


@router.post("/v1/mcp/servers", status_code=201)
async def enable_server(payload: EnableRequest) -> dict[str, Any]:
    entry = CATALOG.get(payload.catalog)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"catalog entry {payload.catalog!r} not found — see GET /v1/mcp/catalog",
        )
    known = {p.name for p in entry.params}
    unknown = sorted(set(payload.params) - known)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unknown param(s) for {entry.name!r}: {', '.join(unknown)}",
        )
    missing = [p.name for p in entry.params if p.required and not payload.params.get(p.name)]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"missing required param(s): {', '.join(missing)}"
        )
    record = registry_entry_for(entry, payload.params)
    mcp_tools.save_registry_entry(entry.name, record)
    return {
        "name": entry.name,
        "registered": True,
        "registry": str(get_settings().mcp_servers_config),
    }


@router.delete("/v1/mcp/servers/{name}")
async def disable_server(name: str) -> dict[str, Any]:
    if not mcp_tools.remove_registry_entry(name):
        raise HTTPException(status_code=404, detail=f"server {name!r} is not registered")
    return {"name": name, "registered": False}


@router.post("/v1/mcp/servers/{name}/test")
async def test_server(name: str) -> dict[str, Any]:
    """Connect to one registered server and list its tools — the preflight
    that catches a broken command/URL before the first chat request does."""
    registry = _registry()
    spec = registry.get(name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"server {name!r} is not registered")
    try:
        mcp_tools._require_mcp()
    except mcp_tools.MCPUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    try:
        async with AsyncExitStack() as stack:
            sessions = await mcp_tools.open_mcp_sessions(stack, [spec])
            listed = await sessions[name].list_tools()
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.input_schema,
                }
                for tool in listed.tools
            ]
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - any connect/handshake failure is the diagnostic
        raise HTTPException(status_code=502, detail=f"could not connect: {exc}") from exc
    return {"name": name, "ok": True, "tools": tools}


@router.get("/v1/mcp/servers/{name}/workers")
async def code_interpreter_workers(name: str) -> dict[str, Any]:
    """Judge0's own worker-pool/queue introspection (docker-compose's
    judge0-server, #264), surfaced as a status card nested in the Studio's
    MCP tab rather than a new page. Only meaningful for the code-interpreter
    server — 404s for anything else so the UI can render "not applicable"
    instead of a broken card."""
    if name != "code-interpreter":
        raise HTTPException(status_code=404, detail=f"server {name!r} has no worker-pool status")
    spec = _registry().get(name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"server {name!r} is not registered")
    from docie_bench.mcp_servers import code_interpreter

    url = spec.env.get(code_interpreter.URL_ENV) or code_interpreter._DEFAULT_URL
    token = spec.env.get(code_interpreter.TOKEN_ENV)
    if not token:
        raise HTTPException(
            status_code=422,
            detail=f"server {name!r} has no {code_interpreter.TOKEN_ENV} configured",
        )
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/workers", headers={"X-Auth-Token": token})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"could not reach judge0: {exc}") from exc
    return {"name": name, "queues": response.json()}
