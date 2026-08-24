"""Dates MCP server: parse/normalize/diff dates deterministically with
dateutil. Run: ``python -m docie_bench.mcp_servers.dates``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from dateutil import parser as dateutil_parser


def parse_date_text(text: str, dayfirst: bool = False) -> str:
    try:
        parsed = dateutil_parser.parse(text, dayfirst=dayfirst, fuzzy=True)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"could not parse {text!r} as a date: {exc}") from exc
    parsed_date: dt.date = parsed.date()
    return parsed_date.isoformat()


def diff_days(start: str, end: str) -> dict[str, Any]:
    start_date = dt.date.fromisoformat(start)
    end_date = dt.date.fromisoformat(end)
    delta = (end_date - start_date).days
    return {"start": start_date.isoformat(), "end": end_date.isoformat(), "days": delta}


def current_date() -> str:
    return dt.date.today().isoformat()


def build_server() -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("docie-dates")

    @server.tool()
    def parse_date(text: str, dayfirst: bool = False) -> str:
        """Parse a date written in any common format ("4 mars 2025",
        "March 4th, 2025", "03/04/2025") into ISO 8601 (YYYY-MM-DD). Set
        dayfirst=true for European day/month/year order."""
        return parse_date_text(text, dayfirst)

    @server.tool()
    def date_diff(start: str, end: str) -> dict[str, Any]:
        """Exact number of days between two ISO dates (end - start) — e.g.
        invoice date to due date. Negative means end is before start."""
        return diff_days(start, end)

    @server.tool()
    def today() -> str:
        """Today's date as ISO 8601 (YYYY-MM-DD)."""
        return current_date()

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
