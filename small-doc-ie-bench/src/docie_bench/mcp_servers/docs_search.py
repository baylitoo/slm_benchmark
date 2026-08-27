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

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

DOCS_DIR_ENV = "DOCIE_MCP_DOCS_SEARCH_DIR"
BACKEND_ENV = "DOCIE_MCP_DOCS_SEARCH_BACKEND"
_DEFAULT_DOCS_DIR = "data/agent-docs"
_DEFAULT_BACKEND = "substring"
_SUPPORTED_SUFFIXES = (".pdf", ".txt")
_SNIPPET_CHARS = 300


class SearchBackend(ABC):
    """A retrieval strategy over the documents directory. ``search_text``'s
    signature is stable across every backend -- only what happens inside
    ``search`` changes (substring today; bm25/vector/hybrid/sql/cypher are
    all just more backends behind this same call, see #282)."""

    @abstractmethod
    def search(self, query: str, targets: list[str]) -> list[dict[str, Any]]:
        """One match per hit page: ``{path, page, snippet}``."""
        raise NotImplementedError


class SubstringSearchBackend(SearchBackend):
    """Case-insensitive substring match over liteparse-extracted text --
    today's default, and for now the only implemented backend."""

    def search(self, query: str, targets: list[str]) -> list[dict[str, Any]]:
        from docie_bench.ocr.factory import get_ocr_backend

        needle = query.lower()
        ocr = get_ocr_backend("liteparse")
        matches: list[dict[str, Any]] = []
        for relative in targets:
            for block in ocr.extract(resolve_document(relative)):
                if needle in block.text.lower():
                    matches.append(
                        {
                            "path": relative,
                            "page": block.page,
                            "snippet": block.text[:_SNIPPET_CHARS],
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
        """Search for `query` across one document (pass `path` from
        list_files) or every document if `path` is omitted. Returns one
        match per hit page: {path, page, snippet}.
        Search before you answer -- don't guess at content you haven't read
        or found."""
        return search_documents(query, path)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
