"""Docs-search MCP server (#275): read-only agentic search over one shared
directory of documents. Run: ``python -m docie_bench.mcp_servers.docs_search``.

The retrieval strategy is a ``SearchBackend`` (#282), the same
factory-over-ABC shape ``docie_bench.ocr.factory`` already uses for OCR
backends -- ``substring`` is the only one shipped so far (case-insensitive,
over text already extracted via ``liteparse``), chosen so the point stays
legible: a small (350M-class) model can drive real tool-choice / tool-args /
answer-from-result agentic RAG instead of one-shot generation over a
stuffed prompt. The seam exists so bm25/vector (e.g. against
``multi_vector_server``)/hybrid backends can be added later without
changing ``search_text``'s signature -- backend choice is an operator-set
catalog param (``BACKEND_ENV``), never a per-call agent argument.

Extraction itself is memoized per (path, mtime, size) via
``_extracted_blocks`` -- re-parsing a PDF (OCR fallback especially) on
every ``search_text``/``read_document`` call against the same document
within one tool loop would be wasteful, and agentic search is expected to
hit the same document repeatedly. That in-memory cache alone only helps
WITHIN one tool loop though: this server is a fresh subprocess per chat
request (see agents/runtime.py's _complete_with_tools), so its memory is
gone before the next turn. A disk-backed second tier (see
``_load_disk_cache``/``_write_disk_cache``) survives that -- the next
turn's fresh subprocess, or a fresh subprocess for a different chat
request against the SAME document, skips OCR entirely instead of paying
it again.

The directory is read-only and operator-controlled via ``DOCS_DIR_ENV`` (see
``mcp_catalog.CATALOG["docs-search"]``) -- this process never writes to it,
and every path a tool receives is resolved strictly inside it (see
``resolve_document``). Note this subprocess only inherits a minimal, curated
environment from the MCP client SDK (PATH/HOME-like vars, not the parent
process's full env) -- ``DOCS_DIR_ENV``/``BACKEND_ENV`` reach it only
because the catalog registers them explicitly as per-server env vars, same
as web_fetch's allowed-hosts param.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from docie_bench.schemas.common import OCRBlock

DOCS_DIR_ENV = "DOCIE_MCP_DOCS_SEARCH_DIR"
BACKEND_ENV = "DOCIE_MCP_DOCS_SEARCH_BACKEND"
_DEFAULT_DOCS_DIR = "data/agent-docs"
_DEFAULT_BACKEND = "substring"
# Public: also the write-side allowlist for mcp_session_documents (#296) --
# an upload that isn't one of these suffixes would just sit in list_files
# forever with nothing able to read it back.
SUPPORTED_SUFFIXES = (".pdf", ".txt")
# Characters of surrounding context kept on each side of a match --
# 300 gave the model a near-useless few-word fragment; the full hit page
# (tried next) blew a 128k context after several searches on a long
# document (each snippet compounds across every round of one tool loop).
# 400 each side is enough to judge relevance without re-dumping the page.
_SNIPPET_WINDOW = 400
# Absolute safety cap on one match's snippet -- a page with many scattered
# hits (a common word) could otherwise join many windows into something
# still too large.
_SNIPPET_CHARS_MAX = 4000

# Keyed by (resolved path, mtime_ns, size) so a changed file misses rather
# than serving stale blocks. Agentic search is expected to hit the SAME
# document repeatedly within one tool loop (list, search it several times,
# then read it) -- this skips re-running OCR fallback on scanned pages
# after the first hit. No eviction: the server process is spawned fresh
# per request (see agents/runtime.py's _complete_with_tools), so the cache
# never outlives one tool loop and can't grow unbounded.
_EXTRACTION_CACHE: dict[tuple[str, int, int], list[OCRBlock]] = {}


def _cache_dir() -> Path:
    """A SIBLING of ``docs_dir()``, never inside it -- the docstring's "this
    process never writes to docs_dir()" invariant holds for the
    operator-controlled shared corpus (a read-only mount stays read-only)
    even though extraction results now persist to disk."""
    root = docs_dir()
    cache = root.parent / f"{root.name}.extraction-cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _disk_cache_path(path: Path) -> Path:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return _cache_dir() / f"{digest}.json"


def _load_disk_cache(path: Path, stat: os.stat_result) -> list[OCRBlock] | None:
    try:
        raw = json.loads(_disk_cache_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if raw.get("mtime_ns") != stat.st_mtime_ns or raw.get("size") != stat.st_size:
        return None  # the file changed since this was written -- a real miss
    try:
        return [OCRBlock.model_validate(b) for b in raw["blocks"]]
    except Exception:  # noqa: BLE001 - a malformed/partial cache file is just a miss
        return None


def _write_disk_cache(path: Path, stat: os.stat_result, blocks: list[OCRBlock]) -> None:
    payload = {
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "blocks": [b.model_dump(mode="json") for b in blocks],
    }
    cache_path = _disk_cache_path(path)
    tmp = cache_path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(cache_path)
    except OSError:
        pass  # best-effort -- this request's extraction still succeeded either way


def _extracted_blocks(path: Path) -> list[OCRBlock]:
    from docie_bench.ocr.factory import get_ocr_backend

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _EXTRACTION_CACHE.get(key)
    if cached is not None:
        return cached
    from_disk = _load_disk_cache(path, stat)
    if from_disk is not None:
        _EXTRACTION_CACHE[key] = from_disk
        return from_disk
    blocks = get_ocr_backend("liteparse").extract(path)
    _EXTRACTION_CACHE[key] = blocks
    _write_disk_cache(path, stat, blocks)
    return blocks


class SearchBackend(ABC):
    """A retrieval strategy over the documents directory. ``search_text``'s
    signature is stable across every backend -- only what happens inside
    ``search`` changes (substring today; bm25/vector/hybrid/sql/cypher are
    all just more backends behind this same call, see #282)."""

    @abstractmethod
    def search(self, query: str, targets: list[str]) -> list[dict[str, Any]]:
        """One match per hit page: ``{path, page, snippet}``."""
        raise NotImplementedError


def _windowed_snippet(text: str, needle: str) -> str:
    """Every occurrence of ``needle`` in ``text``, each with
    ``_SNIPPET_WINDOW`` characters of context on both sides -- overlapping
    windows merge into one span instead of duplicating the shared text."""
    lower = text.lower()
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = lower.find(needle, start)
        if idx == -1:
            break
        lo = max(0, idx - _SNIPPET_WINDOW)
        hi = min(len(text), idx + len(needle) + _SNIPPET_WINDOW)
        spans.append((lo, hi))
        start = idx + len(needle)
    merged: list[tuple[int, int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return "…".join(text[lo:hi] for lo, hi in merged)[:_SNIPPET_CHARS_MAX]


class SubstringSearchBackend(SearchBackend):
    """Case-insensitive substring match over liteparse-extracted text --
    today's default, and for now the only implemented backend."""

    def search(self, query: str, targets: list[str]) -> list[dict[str, Any]]:
        needle = query.lower()
        matches: list[dict[str, Any]] = []
        for relative in targets:
            blocks = _extracted_blocks(resolve_document(relative))
            page_lines: dict[int, list[str]] = {}
            hit_pages: list[int] = []
            for block in blocks:
                page_lines.setdefault(block.page, []).append(block.text)
                if needle in block.text.lower() and block.page not in hit_pages:
                    hit_pages.append(block.page)
            for page in hit_pages:
                page_text = "\n".join(page_lines[page])
                matches.append(
                    {
                        "path": relative,
                        "page": page,
                        "snippet": _windowed_snippet(page_text, needle),
                    }
                )
        return matches


def get_search_backend(name: str) -> SearchBackend:
    normalized = name.lower().strip()
    if normalized == "substring":
        return SubstringSearchBackend()
    raise ValueError(f"Unknown search backend {name!r}. Expected: substring.")


def docs_dir() -> Path:
    path = Path(os.environ.get(DOCS_DIR_ENV, _DEFAULT_DOCS_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_document(relative: str) -> Path:
    """Resolve ``relative`` strictly inside ``docs_dir()``.

    Raises ``ValueError`` (surfaced to the model as a tool-error string, see
    ``mcp_tools.execute_tool_call``) on an absolute path, a ``..`` escape, a
    missing file, or an unsupported extension. A missing file's error
    message includes the real listing -- a small model that invents a path
    instead of calling ``list_files`` first still gets corrected on its next
    attempt, rather than needing to have followed instructions perfectly the
    first time.
    """
    if Path(relative).is_absolute():
        raise ValueError(f"{relative!r} must be a path relative to the documents directory")
    root = docs_dir().resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{relative!r} is outside the documents directory")
    if not candidate.is_file():
        available = list_documents()
        hint = (
            f"available documents: {available}"
            if available
            else "no documents are available in the documents directory"
        )
        raise ValueError(
            f"no such document: {relative!r} -- call list_files and use one of its paths "
            f"exactly, don't invent one ({hint})"
        )
    if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported file type {candidate.suffix!r} "
            f"(expected one of {', '.join(SUPPORTED_SUFFIXES)})"
        )
    return candidate


def list_documents() -> list[str]:
    root = docs_dir()
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


# A "peek" (no page range requested, on a document long enough to matter)
# stops accumulating pages once their combined text would exceed this many
# characters -- character budget, not a page count, because page density
# varies wildly (a dense two-column legal page can hold 10x a sparse one).
# Sized to comfortably fit a small model's context alongside everything
# else in the conversation, while still being long enough to see a
# document's structure (title, table of contents, opening section).
PEEK_CHAR_BUDGET = 4000


def document_text(
    relative: str, start_page: int | None = None, end_page: int | None = None
) -> dict[str, Any]:
    """One document's text, grouped by page, in reading order.

    With no page range and the document is long, returns a PEEK (pages
    from the start, up to ``PEEK_CHAR_BUDGET`` characters) plus a
    ``notice`` naming the real page count and how to read more --
    never the whole body of a long document unconditionally, which would
    overflow a small model's context window on something like a
    150-page regulation. Pass ``start_page``/``end_page`` (1-indexed,
    inclusive) to read a specific section once ``search_text`` has
    located it -- an explicit range is trusted as-is, no budget applied.
    """
    path = resolve_document(relative)
    blocks = _extracted_blocks(path)
    pages: dict[int, list[str]] = {}
    for block in blocks:
        pages.setdefault(block.page, []).append(block.text)
    page_numbers = sorted(pages)
    page_texts = {page: "\n".join(lines) for page, lines in pages.items()}
    total_pages = len(page_numbers)

    notice: str | None = None
    if start_page is not None or end_page is not None:
        lo = start_page if start_page is not None else page_numbers[0]
        hi = end_page if end_page is not None else page_numbers[-1]
        selected = [p for p in page_numbers if lo <= p <= hi]
    elif total_pages <= 1:
        selected = page_numbers
    else:
        selected = []
        budget = PEEK_CHAR_BUDGET
        for p in page_numbers:
            if selected and len(page_texts[p]) > budget:
                break
            selected.append(p)
            budget -= len(page_texts[p])
            if budget <= 0:
                break
        if len(selected) < total_pages:
            notice = (
                f"This document has {total_pages} pages; showing pages "
                f"{selected[0]}-{selected[-1]} as a peek (a long document isn't "
                "returned in full, to avoid overflowing your context). Use "
                "search_text to find the pages that actually answer the "
                "question, then call read_document again with "
                "start_page/end_page to read that specific section."
            )

    result: dict[str, Any] = {
        "path": relative,
        "total_pages": total_pages,
        "pages": [{"page": p, "text": page_texts[p]} for p in selected],
    }
    if notice is not None:
        result["notice"] = notice
    return result


def search_documents(query: str, path: str | None = None) -> list[dict[str, Any]]:
    """Search for ``query``, across one document (``path`` given) or every
    document under ``docs_dir()`` -- one match per hit page, with a short
    snippet. The retrieval strategy is ``BACKEND_ENV`` (default
    ``substring``), an operator setting, not a tool argument."""
    if not query.strip():
        raise ValueError("query must not be empty")
    targets = [path] if path else list_documents()
    backend_name = os.environ.get(BACKEND_ENV, _DEFAULT_BACKEND)
    return get_search_backend(backend_name).search(query, targets)


def build_server() -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("docie-docs-search")

    @server.tool()
    def list_files() -> list[str]:
        """List every readable document (.pdf/.txt) under the shared
        documents directory, as paths relative to it. ALWAYS call this
        first, before read_document or search_text -- these tools only see
        this fixed directory, never a file attached elsewhere in the
        conversation. If the file you're looking for isn't in this list, it
        genuinely isn't available here; say so instead of guessing a path."""
        return list_documents()

    @server.tool()
    def read_document(
        path: str, start_page: int | None = None, end_page: int | None = None
    ) -> dict[str, Any]:
        """Read one document's text, grouped by page. `path` MUST be one of
        the exact strings returned by list_files -- never invent, guess, or
        construct a path from an id/number seen elsewhere. PDFs are parsed
        via liteparse (PDFium text + OCR fallback for scanned pages).

        For a LONG document (call this once with no page range first --
        the response's `total_pages` tells you how long it really is):
        1. Peek: call with no start_page/end_page. On a long document you
           get the opening pages plus a `notice`, not the whole body --
           never assume a short response means a short document.
        2. Search: call search_text with a specific query to find which
           page(s) actually answer the question. Each hit reports its page.
        3. Read: call read_document again with start_page/end_page set to
           the page(s) search_text pointed at, to read that section in full.
        Formulate the search query from what you actually need to answer,
        not a generic re-read of the whole document -- the question may
        need several distinct searches (several facts on different pages)
        before you have everything required to answer it."""
        return document_text(path, start_page, end_page)

    @server.tool()
    def search_text(query: str, path: str | None = None) -> list[dict[str, Any]]:
        """Search for `query` across one document (pass `path` -- one of the
        exact strings from list_files, never invented) or every document if
        `path` is omitted. Returns one match per hit page: {path, page, snippet}.
        Search before you answer -- don't guess at content you haven't read
        or found. On a long document, use a hit's `page` as read_document's
        start_page/end_page to read that section in full rather than
        answering from the snippet alone."""
        return search_documents(query, path)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
