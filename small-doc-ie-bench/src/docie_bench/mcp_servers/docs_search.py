"""Docs-search MCP server (#275): read-only agentic search over one shared
directory of documents. Run: ``python -m docie_bench.mcp_servers.docs_search``.

Deliberately simple -- substring search over text already extracted via
``liteparse`` (the same PDF backend the rest of the platform uses, see
``docie_bench.ocr.factory``), no vector index -- so the point stays legible:
a small (350M-class) model can drive real tool-choice / tool-args /
answer-from-result agentic RAG instead of one-shot generation over a
stuffed prompt.

The directory is read-only and operator-controlled via ``DOCS_DIR_ENV`` (see
``mcp_catalog.CATALOG["docs-search"]``) -- this process never writes to it,
and every path a tool receives is resolved strictly inside it (see
``resolve_document``). Note this subprocess only inherits a minimal, curated
environment from the MCP client SDK (PATH/HOME-like vars, not the parent
process's full env) -- ``DOCS_DIR_ENV`` reaches it only because the catalog
registers it explicitly as a per-server env var, same as web_fetch's
allowed-hosts param.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DOCS_DIR_ENV = "DOCIE_MCP_DOCS_SEARCH_DIR"
_DEFAULT_DOCS_DIR = "data/agent-docs"
_SUPPORTED_SUFFIXES = (".pdf", ".txt")
_SNIPPET_CHARS = 300


def docs_dir() -> Path:
    path = Path(os.environ.get(DOCS_DIR_ENV, _DEFAULT_DOCS_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_document(relative: str) -> Path:
    """Resolve ``relative`` strictly inside ``docs_dir()``.

    Raises ``ValueError`` (surfaced to the model as a tool-error string, see
    ``mcp_tools.execute_tool_call``) on an absolute path, a ``..`` escape, a
    missing file, or an unsupported extension -- never on a path that merely
    doesn't exist yet, since the model is expected to call this with a path
    it already saw from ``list_documents``.
    """
    if Path(relative).is_absolute():
        raise ValueError(f"{relative!r} must be a path relative to the documents directory")
    root = docs_dir().resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"{relative!r} is outside the documents directory")
    if not candidate.is_file():
        raise ValueError(f"no such document: {relative!r}")
    if candidate.suffix.lower() not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"unsupported file type {candidate.suffix!r} "
            f"(expected one of {', '.join(_SUPPORTED_SUFFIXES)})"
        )
    return candidate


def list_documents() -> list[str]:
    root = docs_dir()
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def document_text(relative: str) -> dict[str, Any]:
    """One document's full text, grouped by page, in reading order."""
    from docie_bench.ocr.factory import get_ocr_backend

    path = resolve_document(relative)
    blocks = get_ocr_backend("liteparse").extract(path)
    pages: dict[int, list[str]] = {}
    for block in blocks:
        pages.setdefault(block.page, []).append(block.text)
    return {
        "path": relative,
        "pages": [
            {"page": page, "text": "\n".join(lines)} for page, lines in sorted(pages.items())
        ],
    }


def search_documents(query: str, path: str | None = None) -> list[dict[str, Any]]:
    """Case-insensitive substring search for ``query``, across one document
    (``path`` given) or every document under ``docs_dir()`` -- one match per
    hit page, with a short snippet."""
    if not query.strip():
        raise ValueError("query must not be empty")
    needle = query.lower()
    targets = [path] if path else list_documents()
    from docie_bench.ocr.factory import get_ocr_backend

    backend = get_ocr_backend("liteparse")
    matches: list[dict[str, Any]] = []
    for relative in targets:
        for block in backend.extract(resolve_document(relative)):
            if needle in block.text.lower():
                matches.append(
                    {"path": relative, "page": block.page, "snippet": block.text[:_SNIPPET_CHARS]}
                )
    return matches


def build_server() -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("docie-docs-search")

    @server.tool()
    def list_files() -> list[str]:
        """List every readable document (.pdf/.txt) under the shared
        documents directory, as paths relative to it. Call this first."""
        return list_documents()

    @server.tool()
    def read_document(path: str) -> dict[str, Any]:
        """Read one document's full text (a path from list_files), grouped
        by page. PDFs are parsed via liteparse (PDFium text + OCR fallback
        for scanned pages)."""
        return document_text(path)

    @server.tool()
    def search_text(query: str, path: str | None = None) -> list[dict[str, Any]]:
        """Case-insensitive substring search for `query` across one document
        (pass `path` from list_files) or every document if `path` is
        omitted. Returns one match per hit page: {path, page, snippet}.
        Search before you answer -- don't guess at content you haven't read
        or found."""
        return search_documents(query, path)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
