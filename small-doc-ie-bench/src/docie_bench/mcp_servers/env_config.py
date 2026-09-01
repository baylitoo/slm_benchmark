"""Shared env-var config reading for MCP servers, factored out once a second
server (sql_agent, alongside docs_search) needed the identical int-with-
fallback logic -- add here rather than re-copying if a third one does too."""

from __future__ import annotations

import os


def int_env(name: str, default: int) -> int:
    """``default`` unless ``name`` is set to a valid int -- an unset or
    malformed override falls back rather than crashing a tool call over an
    operator typo in an env var."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
