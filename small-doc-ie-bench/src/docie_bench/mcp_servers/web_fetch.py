"""Web-fetch MCP server: allowlisted HTTP GET (deny by default, redirects
reported not followed — an allowlisted host redirecting elsewhere would
bypass the allowlist). HTML responses are converted to plain text before
truncation, so context budget goes to content instead of markup (see #349).
Run: ``python -m docie_bench.mcp_servers.web_fetch``.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any
from urllib.parse import urlsplit

ALLOWED_HOSTS_ENV = "DOCIE_MCP_FETCH_ALLOWED_HOSTS"
_MAX_RESPONSE_BYTES = 200_000
_TIMEOUT_SECONDS = 15.0
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


def is_html_content_type(content_type: str) -> bool:
    """Whether ``content_type`` (a raw ``Content-Type`` header value) is HTML."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type in _HTML_CONTENT_TYPES


def extract_text_from_html(html: str) -> str:
    """Convert HTML to plain text, dropping tags, scripts, and styles.

    ``liteparse`` (this codebase's PDF/document parser) does not accept HTML —
    it raises ``ParseError: unsupported file format: .html`` — so this uses
    ``html2text``, a small, dependency-free, actively maintained converter.
    """
    import html2text

    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.ignore_images = True
    return converter.handle(html)


def allowed_hosts() -> set[str]:
    raw = os.environ.get(ALLOWED_HOSTS_ENV, "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def check_url(url: str, hosts: set[str] | None = None) -> str | None:
    """Return a refusal reason, or None when ``url`` may be fetched."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return f"only http(s) URLs are allowed, got scheme {parts.scheme!r}"
    hostname = (parts.hostname or "").lower()
    if not hostname:
        return "URL has no hostname"
    hosts = allowed_hosts() if hosts is None else hosts
    if not hosts:
        return (
            "no hosts are allowlisted — the operator must set "
            f"{ALLOWED_HOSTS_ENV} (comma-separated hostnames, or '*') on this "
            "MCP server before fetch can be used"
        )
    if "*" in hosts or hostname in hosts:
        return None
    return f"host {hostname!r} is not in the allowlist"


async def fetch_url(url: str) -> dict[str, Any]:
    import httpx

    refusal = check_url(url)
    if refusal:
        return {"ok": False, "error": refusal}
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"request failed: {exc}"}
    if 300 <= response.status_code < 400:
        return {
            "ok": False,
            "error": f"HTTP {response.status_code} redirect to "
            f"{response.headers.get('location', '?')!r} — redirects are not followed; "
            "fetch the target directly if its host is allowlisted",
        }
    content_type = response.headers.get("content-type", "")
    body = response.text
    if is_html_content_type(content_type):
        # Fall back to the raw body rather than error out on pathological markup.
        with contextlib.suppress(Exception):
            body = extract_text_from_html(body)
    truncated = len(body.encode("utf-8", errors="ignore")) > _MAX_RESPONSE_BYTES
    if truncated:
        body = body.encode("utf-8", errors="ignore")[:_MAX_RESPONSE_BYTES].decode(
            "utf-8", errors="ignore"
        )
    return {
        "ok": response.status_code < 400,
        "status": response.status_code,
        "content_type": content_type,
        "truncated": truncated,
        "text": body,
    }


def build_server() -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("docie-web-fetch")

    @server.tool()
    async def fetch(url: str) -> dict[str, Any]:
        """HTTP GET an allowlisted URL and return its text (truncated if
        large). HTML responses are converted to plain text first, so the
        budget covers content rather than markup. Only hosts the operator
        allowlisted are reachable; redirects are reported, not followed."""
        return await fetch_url(url)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
