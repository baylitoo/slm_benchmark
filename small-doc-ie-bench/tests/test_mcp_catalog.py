"""MCP catalog, first-party servers, and the management API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from docie_bench.mcp_api import router as mcp_router
from docie_bench.mcp_catalog import CATALOG, registry_entry_for
from docie_bench.mcp_servers import calculator, dates, web_fetch
from docie_bench.settings import get_settings

# ---------------------------------------------------------------- calculator


def test_calculate_arithmetic_and_functions() -> None:
    assert calculator.calculate("3 * 129.99 + 2 * 45.50") == pytest.approx(480.97)
    assert calculator.calculate("(1 + 2) ** 3 % 5") == pytest.approx(2.0)
    assert calculator.calculate("round(10 / 3, 2)") == pytest.approx(3.33)
    assert calculator.calculate("sum([1.1, 2.2, 3.3])") == pytest.approx(6.6)
    assert calculator.calculate("max(1, 2) + min(3, 4) + abs(-2) + sqrt(9)") == pytest.approx(10.0)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "().__class__",
        "x + 1",
        "'a' * 3",
        "True + 1",
        "9 ** 9 ** 9",
        "sum(x for x in [1])",
        "round(1.5, ndigits=0)",
        "1 " * 300 + "+ 1",
    ],
)
def test_calculate_rejects_non_arithmetic(expression: str) -> None:
    with pytest.raises(ValueError, match="allowed|too large|too long|not a valid|numbers"):
        calculator.calculate(expression)


def test_check_sum_match_and_mismatch() -> None:
    ok = calculator.check_sum([100.0, 20.5, 0.55], 121.05)
    assert ok["matches"] is True
    bad = calculator.check_sum([100.0, 20.5], 121.05, tolerance=0.01)
    assert bad["matches"] is False
    assert bad["difference"] == pytest.approx(-0.55)


# --------------------------------------------------------------------- dates


def test_parse_date_formats_and_dayfirst() -> None:
    assert dates.parse_date_text("March 4th, 2025") == "2025-03-04"
    assert dates.parse_date_text("03/04/2025") == "2025-03-04"
    assert dates.parse_date_text("03/04/2025", dayfirst=True) == "2025-04-03"
    with pytest.raises(ValueError, match="could not parse"):
        dates.parse_date_text("not a date at all zzz")


def test_diff_days() -> None:
    assert dates.diff_days("2025-03-04", "2025-04-03")["days"] == 30
    assert dates.diff_days("2025-04-03", "2025-03-04")["days"] == -30


# ----------------------------------------------------------------- web fetch


def test_check_url_allowlist() -> None:
    assert "scheme" in web_fetch.check_url("ftp://x.com/f", {"*"})
    assert "no hosts are allowlisted" in web_fetch.check_url("https://a.com/", set())
    assert web_fetch.check_url("https://a.com/x", {"a.com"}) is None
    assert web_fetch.check_url("https://b.com/x", {"a.com"}) is not None
    assert web_fetch.check_url("https://anything.io/", {"*"}) is None


async def test_fetch_url_redirects_reported_and_body_truncated(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "http://internal/secret"})
        return httpx.Response(200, text="x" * 300_000)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setenv(web_fetch.ALLOWED_HOSTS_ENV, "site.com")
    redirected = await web_fetch.fetch_url("https://site.com/redirect")
    assert redirected["ok"] is False
    assert "redirect" in redirected["error"]
    big = await web_fetch.fetch_url("https://site.com/big")
    assert big["ok"] is True
    assert big["truncated"] is True
    assert len(big["text"]) <= 200_000


# --------------------------------------------------- servers speak real MCP


@pytest.mark.parametrize(
    ("module", "expected_tools"),
    [
        (calculator, {"calc", "sum_check"}),
        (dates, {"parse_date", "date_diff", "today"}),
        (web_fetch, {"fetch"}),
    ],
)
async def test_build_server_exposes_expected_tools(module, expected_tools) -> None:
    server = module.build_server()
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        low = server._lowlevel_server
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: low.run(*server_streams, low.create_initialization_options())
            )
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                listed = await session.list_tools()
            tg.cancel_scope.cancel()
    assert {tool.name for tool in listed.tools} == expected_tools


# ------------------------------------------------------------ management API


@pytest.fixture
def registry_path(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "mcp-servers.json"
    monkeypatch.setenv("MCP_SERVERS_CONFIG", str(path))
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


@pytest.fixture
def client(registry_path: Path) -> TestClient:
    app = FastAPI()
    app.include_router(mcp_router)
    return TestClient(app)


def test_catalog_lists_entries_with_enabled_flag(client: TestClient, registry_path: Path) -> None:
    entries = {e["name"]: e for e in client.get("/v1/mcp/catalog").json()["entries"]}
    assert set(entries) == {"calculator", "dates", "web-fetch"}
    assert not entries["calculator"]["enabled"]
    assert entries["web-fetch"]["params"][0]["name"] == "allowed_hosts"

    client.post("/v1/mcp/servers", json={"catalog": "calculator"})
    entries = {e["name"]: e for e in client.get("/v1/mcp/catalog").json()["entries"]}
    assert entries["calculator"]["enabled"]


def test_enable_writes_registry_entry(client: TestClient, registry_path: Path) -> None:
    res = client.post(
        "/v1/mcp/servers",
        json={"catalog": "web-fetch", "params": {"allowed_hosts": "docs.example.com"}},
    )
    assert res.status_code == 201, res.text
    saved = json.loads(registry_path.read_text(encoding="utf-8"))["servers"]["web-fetch"]
    assert saved["transport"] == "stdio"
    assert saved["command"] == ["python", "-m", "docie_bench.mcp_servers.web_fetch"]
    assert saved["env"] == {"DOCIE_MCP_FETCH_ALLOWED_HOSTS": "docs.example.com"}
    assert saved["catalog"] == "web-fetch"


def test_enable_validates_catalog_and_params(client: TestClient) -> None:
    assert client.post("/v1/mcp/servers", json={"catalog": "nope"}).status_code == 404
    res = client.post(
        "/v1/mcp/servers", json={"catalog": "calculator", "params": {"bogus": "x"}}
    )
    assert res.status_code == 422


def test_enable_preserves_handwritten_entries(client: TestClient, registry_path: Path) -> None:
    registry_path.write_text(
        json.dumps(
            {"servers": {"remote": {"transport": "streamable-http", "url": "http://x/mcp"}}}
        ),
        encoding="utf-8",
    )
    client.post("/v1/mcp/servers", json={"catalog": "dates"})
    servers = json.loads(registry_path.read_text(encoding="utf-8"))["servers"]
    assert set(servers) == {"remote", "dates"}


def test_disable_removes_entry(client: TestClient, registry_path: Path) -> None:
    client.post("/v1/mcp/servers", json={"catalog": "calculator"})
    assert client.delete("/v1/mcp/servers/calculator").status_code == 200
    assert json.loads(registry_path.read_text(encoding="utf-8"))["servers"] == {}
    assert client.delete("/v1/mcp/servers/calculator").status_code == 404


def test_test_route_spawns_and_lists_tools(client: TestClient, registry_path: Path) -> None:
    # Real stdio spawn end-to-end; sys.executable so the venv's python is used.
    registry_path.write_text(
        json.dumps(
            {
                "servers": {
                    "calculator": {
                        "transport": "stdio",
                        "command": [sys.executable, "-m", "docie_bench.mcp_servers.calculator"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    res = client.post("/v1/mcp/servers/calculator/test")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ok"] is True
    assert {tool["name"] for tool in body["tools"]} == {"calc", "sum_check"}
    schema = next(t for t in body["tools"] if t["name"] == "calc")["input_schema"]
    assert schema["required"] == ["expression"]


def test_test_route_unknown_server_404(client: TestClient) -> None:
    assert client.post("/v1/mcp/servers/nope/test").status_code == 404


def test_registry_entry_for_omits_empty_env() -> None:
    entry = CATALOG["calculator"]
    assert "env" not in registry_entry_for(entry, {})
