"""Session-scoped document uploads for docs-search (#296).

A file attached in the Playground's chat box never reaches docs-search's
operator-controlled directory (``DOCIE_MCP_DOCS_SEARCH_DIR``) -- the
attachment is only ever rendered to page images for vision, never written
anywhere as a real ``.pdf``/``.txt`` file. This module is the write side of
closing that gap: a per-conversation directory the model's docs-search
session gets pointed at INSTEAD of the shared corpus for that request (see
``chat_api._chat_with_mcp_tools``), isolated from every other session.

Security model, matching the MCP registry's own convention (a request picks
a server BY NAME from what the operator registered, never supplies its own
URL): a session id is a capability issued BY THIS MODULE, never accepted
from a client that invented one. ``save_document(None, ...)`` mints a fresh
id; passing back an id this module already returned adds another file to
the SAME session. Any other string is rejected outright -- a guessable or
replayed id would otherwise read another session's uploads. The stored
filename is likewise never the client's -- a random name sidesteps path-
traversal and collision handling entirely, and the model doesn't need a
human-readable name since it discovers the real one via docs-search's
eager ``list_files`` call (see ``mcp_tools._eager_list_context``).
"""

from __future__ import annotations

import re
import shutil
import time
import uuid
from pathlib import Path

from docie_bench.mcp_servers.docs_search import SUPPORTED_SUFFIXES
from docie_bench.settings import get_settings

_SESSION_ID_RE = re.compile(r"^[a-f0-9]{32}$")
# Anything outside this allowlist is dropped from the sanitized stem -- never
# built into a regex substitution the client's filename could influence.
_UNSAFE_STEM_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")
_MAX_STEM_LEN = 60


class SessionDocumentError(ValueError):
    """An upload was rejected: unknown session id, bad file type, too big,
    or the session already holds its maximum document count."""


def _root() -> Path:
    root = get_settings().mcp_session_documents_root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _sanitize_stem(filename: str) -> str:
    """The client's filename, minus its extension, made safe to live in a
    server-controlled path segment: ``Path(...).name`` first (drops any
    directory component -- the actual path-traversal guard), then every
    character outside ``[a-zA-Z0-9._-]`` dropped (no ``../`` survives this
    even if smuggled in via a name with no real path separators), collapsed
    to a bounded length so a pathological filename can't blow up the
    directory listing. An empty result (e.g. a filename that was ALL unsafe
    characters) falls back to "document" rather than an empty path segment.
    """
    stem = Path(filename).stem
    cleaned = _UNSAFE_STEM_CHARS.sub("_", stem).strip("._-")
    return (cleaned or "document")[:_MAX_STEM_LEN]


def _session_dir(session_id: str, *, create: bool) -> Path:
    if not _SESSION_ID_RE.match(session_id):
        raise SessionDocumentError("invalid session id")
    path = _root() / session_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise SessionDocumentError(f"unknown session id: {session_id!r}")
    return path


def save_document(session_id: str | None, filename: str, content: bytes) -> tuple[str, str]:
    """Write ``content`` into a session's directory.

    ``session_id=None`` starts a new session (a fresh id is minted and
    returned); passing an id this function already returned adds another
    file to that SAME session. Returns ``(session_id, stored_name)`` --
    ``stored_name`` is what docs-search's ``list_files`` will report:
    ``filename``'s own name, sanitized (see ``_sanitize_stem``), plus a
    short random suffix for collision-safety -- readable in a trace/log
    instead of a bare hex blob, while staying just as untrusted for path
    resolution as the fully-random name it replaces (``resolve_document``
    still resolves strictly inside the session directory regardless of what
    this string looks like).
    """
    settings = get_settings()
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise SessionDocumentError(
            f"unsupported file type {suffix!r} "
            f"(expected one of {', '.join(SUPPORTED_SUFFIXES)})"
        )
    if len(content) > settings.max_upload_bytes:
        raise SessionDocumentError(
            f"file too large: {len(content)} bytes (max {settings.max_upload_bytes})"
        )
    sid = _new_session_id() if session_id is None else session_id
    directory = _session_dir(sid, create=session_id is None)
    existing = sum(1 for _ in directory.glob("*"))
    if existing >= settings.mcp_session_documents_max_files:
        raise SessionDocumentError(
            f"session already has {existing} documents "
            f"(max {settings.mcp_session_documents_max_files})"
        )
    stored_name = f"{_sanitize_stem(filename)}-{uuid.uuid4().hex[:8]}{suffix}"
    (directory / stored_name).write_bytes(content)
    return sid, stored_name


def session_documents_dir(session_id: str) -> Path:
    """Resolve an already-issued session's directory (read side, e.g. for
    pointing docs-search's ``DOCS_DIR_ENV`` override at it)."""
    return _session_dir(session_id, create=False)


def gc_stale_sessions(max_age_hours: float | None = None) -> int:
    """Delete session directories whose newest file is older than the TTL.

    Newest mtime under the directory, not the directory's own mtime, since a
    multi-turn conversation adds files into an existing session directory
    over time. No reliance on the client ever calling a delete endpoint --
    a closed tab or crashed session leaves nothing else to reclaim it.
    """
    settings = get_settings()
    ttl_hours = (
        max_age_hours if max_age_hours is not None else settings.mcp_session_documents_max_age_hours
    )
    cutoff_age_s = ttl_hours * 3600.0
    root = _root()
    now = time.time()
    removed = 0
    try:
        children = list(root.iterdir())
    except OSError:
        children = []
    for child in children:
        if not child.is_dir():
            continue
        try:
            stamps = [child.stat().st_mtime, *(p.stat().st_mtime for p in child.rglob("*"))]
        except OSError:
            continue
        if now - max(stamps) > cutoff_age_s:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed
