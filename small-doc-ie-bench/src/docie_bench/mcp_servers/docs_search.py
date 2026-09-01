"""Docs-search MCP server (#275): read-only agentic search over one shared
directory of documents. Run: ``python -m docie_bench.mcp_servers.docs_search``.

The retrieval strategy is a ``SearchBackend`` (#282), the same
factory-over-ABC shape ``docie_bench.ocr.factory`` already uses for OCR
backends -- ``substring`` (case-insensitive, over text already extracted
via ``liteparse``) is the default, chosen so the point stays legible: a
small (350M-class) model can drive real tool-choice / tool-args /
answer-from-result agentic RAG instead of one-shot generation over a
stuffed prompt. ``hybrid`` (#339) is a second backend behind the same
call: it runs the substring match first as a cheap pre-filter, then reranks
the hit pages by semantic similarity via ``multi_vector_server``'s
``/v1/rerank`` -- useful when the query paraphrases the document rather
than sharing its literal wording. Backend choice is an operator-set
catalog param (``BACKEND_ENV``), never a per-call agent argument.

A PDF classified by ``pdf_inspector`` as fully text-based (no pages
needing OCR) skips liteparse entirely and extracts via ``pdf_inspector``'s
own positioned-text API instead (see ``_pdf_inspector_fast_path``) --
faster for the common case, since it never has to probe for or run OCR.
Anything else (scanned/image/mixed pages, or any classification/extraction
failure) falls through to the existing liteparse path unchanged; this is a
pure speed optimization, never a hard dependency, and never changes
``OCRBlock``'s shape -- only its ``source`` tag differs (``"pdf_inspector"``
vs ``"pdf_text"``) so provenance stays honest either way.

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

Every indexed document is ALSO exposed as an MCP resource (#332) under
``docs://<relative-path>``, in addition to the ``list_files``/
``read_document``/``search_text`` tool triad -- a capable client can
``list_resources()`` to browse the corpus and ``read_resource(uri)`` to
fetch a document directly, without going through a tool call. This is a
separate, simpler read path: it does NOT go through ``run_tool_loop`` /
``mcp_tools.py``, isn't traced through ``on_tool_call``, and a resource
read returns the FULL document text rather than applying
``read_document``'s peek/windowing discipline (see ``_full_document_text``)
-- those caps exist to protect a model's context budget inside the
agentic tool loop, which a resource read sits outside of.

``write_note``/``read_notes`` add a persistent, page-anchored memory layer
on top of all of the above: the ``_EXTRACTION_CACHE``/disk cache only ever
remembers extracted TEXT, never anything the model itself observed about
that text, and that memory is gone the moment the tool loop ends. A note
survives the same fresh-subprocess-per-request lifecycle the disk cache
does (see ``_notes_dir``), so a later conversation over the SAME document
can discover what an earlier one already found. This is deliberately the
safe subset of "agent memory": append-only (no edit/delete -- a model
correcting its own prior note risks silently erasing a right earlier
observation with a wrong later one), every note anchored to a required
``page`` (same evidence-grounding instinct as ``evidence_ids`` on an
extracted field), and length/count-capped per document. ``read_notes`` is
a plain callable tool, deliberately NOT the ``eager_list_tool`` the way
``list_files`` is -- notes can accumulate across many conversations, so
auto-injecting them into every request would be an unbounded context
cost; a model has to choose to ask.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from docie_bench.mcp_servers.env_config import int_env
from docie_bench.ocr.base import stable_block_id
from docie_bench.schemas.common import BoundingBox, OCRBlock

DOCS_DIR_ENV = "DOCIE_MCP_DOCS_SEARCH_DIR"
BACKEND_ENV = "DOCIE_MCP_DOCS_SEARCH_BACKEND"
# Base URL of a running multi_vector_server deployment -- only consulted by
# HybridSearchBackend (BACKEND_ENV=hybrid). No default: unlike docs_dir/
# backend, there's no sane fallback for "which reranker" the way there is a
# compose-network address for judge0-server (code_interpreter.py), since a
# multi_vector deployment's URL is assigned dynamically by the serving
# control plane -- an operator who picks "hybrid" must supply it explicitly.
RERANKER_URL_ENV = "DOCIE_MCP_DOCS_SEARCH_RERANKER_URL"
# Operator-tunable, same shape as DOCS_DIR_ENV/BACKEND_ENV -- these tune how
# much text one tool result feeds back into the model's context, not what
# the model can ask for, so they're catalog params (see
# mcp_catalog.CATALOG["docs-search"]), never per-call tool arguments.
SNIPPET_WINDOW_ENV = "DOCIE_MCP_DOCS_SEARCH_SNIPPET_WINDOW"
SNIPPET_MAX_CHARS_ENV = "DOCIE_MCP_DOCS_SEARCH_SNIPPET_MAX_CHARS"
PEEK_CHAR_BUDGET_ENV = "DOCIE_MCP_DOCS_SEARCH_PEEK_CHAR_BUDGET"
# Caps on write_note -- a note is a short observation, not another full-text
# dump, and it's never truncated or evicted once written (append-only, see
# module docstring), so both caps fail the call loudly instead.
NOTE_MAX_CHARS_ENV = "DOCIE_MCP_DOCS_SEARCH_NOTE_MAX_CHARS"
MAX_NOTES_PER_DOC_ENV = "DOCIE_MCP_DOCS_SEARCH_MAX_NOTES_PER_DOC"
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

# A note is a short observation ("the total on page 3 doesn't match the sum
# of page 2's line items"), not a summary of the whole page -- 2000 chars is
# generous for that while keeping read_notes cheap to call even after many
# conversations have each left one.
_NOTE_MAX_CHARS = 2000
# No eviction (append-only, see module docstring), so this is a hard ceiling
# on one document's total notes, not a rolling window -- 100 is far more
# than a real usage pattern should ever need per document.
_MAX_NOTES_PER_DOC = 100


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


def _notes_dir() -> Path:
    """A SIBLING of ``docs_dir()``, never inside it -- same invariant, same
    construction as ``_cache_dir()``, just a different suffix so notes and
    extraction-cache files never collide."""
    root = docs_dir()
    notes = root.parent / f"{root.name}.notes"
    notes.mkdir(parents=True, exist_ok=True)
    return notes


def _notes_path(path: Path) -> Path:
    """Same content-addressed digest scheme as ``_disk_cache_path`` -- a
    note file's name has no relation to the document's own name."""
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return _notes_dir() / f"{digest}.json"


def _load_notes(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(_notes_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return raw if isinstance(raw, list) else []


def _write_notes(path: Path, notes: list[dict[str, Any]]) -> None:
    """Atomic tmp-file-then-replace, same write-safety convention as
    ``_write_disk_cache`` -- unlike that cache (a regenerable optimization,
    so a failed write there is silently swallowed), a note is the only copy
    of an observation a model may believe it already persisted, so an
    ``OSError`` here is left to propagate rather than fail silently."""
    notes_path = _notes_path(path)
    tmp = notes_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(notes), encoding="utf-8")
    tmp.replace(notes_path)


def _pdf_inspector_fast_path(path: Path) -> list[OCRBlock] | None:
    """A fast, OCR-free extraction for a PDF ``pdf_inspector`` classifies as
    fully text-based -- ``None`` for anything else (scanned/image/mixed
    pages, or any classification/extraction failure), which sends the
    caller to the existing liteparse (PDFium + OCR fallback) path instead.

    This is a pure speed optimization for the common case, never a hard
    dependency: pdf_inspector erroring or misclassifying costs nothing
    beyond falling back to the path that ran unconditionally before this
    existed -- same fail-open convention this codebase uses for every other
    "nice when it works" signal.
    """
    import pdf_inspector

    try:
        classification = pdf_inspector.classify_pdf(str(path))
    except Exception:  # noqa: BLE001 - classification failure is just a cache miss for this path
        return None
    if classification.pdf_type != "text_based" or classification.pages_needing_ocr:
        return None
    try:
        items = pdf_inspector.extract_text_with_positions(str(path))
    except Exception:  # noqa: BLE001 - extraction failure falls back to liteparse
        return None
    blocks: list[OCRBlock] = []
    for idx, item in enumerate(items):
        text = (item.text or "").strip()
        if not text:
            continue
        bbox = BoundingBox(
            x0=float(item.x),
            y0=float(item.y),
            x1=float(item.x) + float(item.width),
            y1=float(item.y) + float(item.height),
        )
        blocks.append(
            OCRBlock(
                id=stable_block_id(item.page, idx, text),
                text=text,
                page=item.page,
                bbox=bbox,
                source="pdf_inspector",
            )
        )
    return blocks


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
    blocks = None
    if path.suffix.lower() == ".pdf":
        blocks = _pdf_inspector_fast_path(path)
    if blocks is None:
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
    ``SNIPPET_WINDOW_ENV`` characters of context on both sides -- overlapping
    windows merge into one span instead of duplicating the shared text."""
    window = int_env(SNIPPET_WINDOW_ENV, _SNIPPET_WINDOW)
    max_chars = int_env(SNIPPET_MAX_CHARS_ENV, _SNIPPET_CHARS_MAX)
    lower = text.lower()
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = lower.find(needle, start)
        if idx == -1:
            break
        lo = max(0, idx - window)
        hi = min(len(text), idx + len(needle) + window)
        spans.append((lo, hi))
        start = idx + len(needle)
    merged: list[tuple[int, int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return "…".join(text[lo:hi] for lo, hi in merged)[:max_chars]


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


class RerankerUnavailableError(RuntimeError):
    """``BACKEND_ENV=hybrid`` but ``multi_vector_server`` isn't reachable --
    either ``RERANKER_URL_ENV`` was never set, or it points at a URL that
    refused the connection / errored / timed out. Raised, never swallowed:
    the operator explicitly chose hybrid, so silently falling back to plain
    substring results would look like a correct answer that's quietly
    missing the semantic ranking they asked for -- a config error should
    read as a config error."""


class HybridSearchBackend(SearchBackend):
    """Substring pre-filter (:class:`SubstringSearchBackend`, cheap, exact-
    term) narrows the corpus to candidate hit pages, then
    ``multi_vector_server``'s ``/v1/rerank`` reorders those candidates by
    semantic similarity to ``query`` -- exact terms narrow the search space
    before anything gets embedded, so a corpus-wide query never embeds every
    page of every document (see #339's design comment).

    If the substring pre-filter finds NOTHING, this returns an empty list
    rather than falling back to embedding the whole corpus: a hybrid search
    with zero literal term overlap between query and corpus is the rare
    case (most paraphrases still share at least one distinctive word --
    a name, an amount, a term), and unconditionally embedding every page of
    every document on that miss would make the common case (a real
    zero-hit search) silently expensive every time, trading a fast/cheap
    "nothing matched" for a slow one on every miss. A future full "vector"
    backend (embed everything, no pre-filter) is the place for that
    fallback, not a special case bolted onto hybrid.
    """

    def search(self, query: str, targets: list[str]) -> list[dict[str, Any]]:
        # Checked BEFORE running the substring pass, not after: a misconfigured
        # RERANKER_URL_ENV must fail every hybrid search deterministically, not
        # only the ones whose query happens to substring-match something --
        # an operator smoke-testing with a miss query would otherwise see a
        # clean empty result and conclude hybrid works when it can't actually
        # rerank anything.
        url = os.environ.get(RERANKER_URL_ENV)
        if not url:
            raise RerankerUnavailableError(
                f"hybrid search backend needs {RERANKER_URL_ENV} set to a running "
                "multi_vector_server deployment's base URL -- set it on the "
                "docs-search MCP server's env before selecting BACKEND_ENV=hybrid."
            )
        candidates = SubstringSearchBackend().search(query, targets)
        if not candidates:
            return []
        # Rerank against each candidate PAGE's full extracted text (page-level
        # chunking, per #339's design comment) -- not the substring match's
        # windowed snippet, which is anchored to the literal needle and may
        # not represent the page's semantic content for a paraphrased query.
        # The snippet returned to the caller still comes from the substring
        # pass below, unchanged.
        documents: list[str] = []
        for match in candidates:
            page_texts = _page_texts(resolve_document(match["path"]))
            documents.append(page_texts.get(match["page"], match["snippet"]))
        results = _rerank(query, documents, url)
        return [candidates[result["index"]] for result in results]


def _rerank(query: str, documents: list[str], url: str) -> list[dict[str, Any]]:
    """POST to ``multi_vector_server``'s ``/v1/rerank`` (same wire contract
    ``chat_api.py``'s ``/v1/rerank`` proxy forwards to, see
    ``multi_vector_server/server.py``) and return its ``results`` --
    ``[{"index": int, "relevance_score": float}, ...]``, sorted descending,
    ``index`` into ``documents``. Any connection failure, timeout, or non-2xx
    response becomes :class:`RerankerUnavailableError` -- a distinct,
    recognizable error rather than a raw ``httpx`` exception leaking out of
    a search tool call."""
    import httpx

    try:
        response = httpx.post(
            f"{url.rstrip('/')}/v1/rerank",
            json={"query": query, "documents": documents},
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RerankerUnavailableError(
            f"hybrid search backend could not reach multi_vector_server at {url!r}: "
            f"{exc} -- is it deployed and healthy?"
        ) from exc
    results: list[dict[str, Any]] = response.json()["results"]
    return results


def get_search_backend(name: str) -> SearchBackend:
    normalized = name.lower().strip()
    if normalized == "substring":
        return SubstringSearchBackend()
    if normalized == "hybrid":
        return HybridSearchBackend()
    raise ValueError(f"Unknown search backend {name!r}. Expected: substring, hybrid.")


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


def append_note(path: str, page: int, note: str) -> dict[str, Any]:
    """Append one page-anchored note about ``path`` to its persistent note
    file, oldest-first (see ``_notes_dir``/``_notes_path`` for where and how
    that file lives). Append-only -- there is no edit/delete, by design (see
    module docstring): the only way to correct a wrong earlier note is to
    write a new one saying so.

    Raises ``ValueError`` -- same convention as ``resolve_document`` -- for
    a nonexistent/traversal ``path``, ``page < 1``, a ``note`` over
    ``NOTE_MAX_CHARS_ENV``, or the (N+1)th note once
    ``MAX_NOTES_PER_DOC_ENV`` is already hit. Caps fail the call loudly
    rather than silently truncating the note or evicting an older one.
    """
    resolved = resolve_document(path)
    if page < 1:
        raise ValueError(f"page must be >= 1, got {page}")
    max_chars = int_env(NOTE_MAX_CHARS_ENV, _NOTE_MAX_CHARS)
    if len(note) > max_chars:
        raise ValueError(
            f"note is {len(note)} characters, over the {max_chars}-character cap "
            f"({NOTE_MAX_CHARS_ENV}) -- write_note is for a short observation, not "
            "another full-text dump; shorten it."
        )
    existing = _load_notes(resolved)
    max_notes = int_env(MAX_NOTES_PER_DOC_ENV, _MAX_NOTES_PER_DOC)
    if len(existing) >= max_notes:
        raise ValueError(
            f"{path!r} already has {len(existing)} notes, at the {max_notes}-note "
            f"cap ({MAX_NOTES_PER_DOC_ENV}) -- notes are append-only and never "
            "evicted, so there's no room left for another one."
        )
    entry: dict[str, Any] = {
        "page": page,
        "note": note,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    existing.append(entry)
    _write_notes(resolved, existing)
    return entry


def list_notes(path: str) -> list[dict[str, Any]]:
    """Every note previously written about ``path`` (see ``append_note``),
    oldest first -- an empty list if none exist yet. Resolves ``path``
    through ``resolve_document`` like every other tool, so notes can only
    ever be attached to (or read from) a document that genuinely exists in
    the corpus."""
    resolved = resolve_document(path)
    return _load_notes(resolved)


# A "peek" (no page range requested, on a document long enough to matter)
# stops accumulating pages once their combined text would exceed this many
# characters -- character budget, not a page count, because page density
# varies wildly (a dense two-column legal page can hold 10x a sparse one).
# Sized to comfortably fit a small model's context alongside everything
# else in the conversation, while still being long enough to see a
# document's structure (title, table of contents, opening section).
PEEK_CHAR_BUDGET = 4000

# An explicit start_page/end_page range used to be trusted as-is, unbounded --
# a model asking for e.g. start_page=1, end_page=40 on a long document could
# still recreate the exact context-bloat problem the peek budget above
# exists to prevent. Page-count based, not character-based (unlike the peek
# budget), because the caller already chose these specific pages via
# search_text -- the cap is about how much of THAT choice comes back in one
# call, not about guessing page density.
MAX_EXPLICIT_RANGE_PAGES = 5


def _page_texts(path: Path) -> dict[int, str]:
    """Extracted text grouped by page number, in reading order. Shared by
    ``document_text`` (page-windowed) and ``_full_document_text`` (whole
    document, for a resource read) so both agree on how pages are
    assembled from the underlying OCR blocks."""
    blocks = _extracted_blocks(path)
    pages: dict[int, list[str]] = {}
    for block in blocks:
        pages.setdefault(block.page, []).append(block.text)
    return {page: "\n".join(lines) for page, lines in pages.items()}


def _full_document_text(relative: str) -> str:
    """The ENTIRE document's text, every page, no peek/window capping --
    unlike ``document_text`` (backing the ``read_document`` tool), a
    resource read has no per-call start_page/end_page the way a tool call
    does, and resource access sits outside the agentic tool loop (see
    module docstring): the result isn't stuffed into a model's context
    automatically the way a tool result is, so the PEEK_CHAR_BUDGET /
    MAX_EXPLICIT_RANGE_PAGES caps that protect that loop don't apply here
    -- a client browsing/reading a resource expects the resource, and owns
    how it uses a large result (page through it, summarize it, discard
    it)."""
    path = resolve_document(relative)
    page_texts = _page_texts(path)
    return "\n\n".join(page_texts[p] for p in sorted(page_texts))


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
    located it -- capped at ``MAX_EXPLICIT_RANGE_PAGES`` pages per call,
    with a ``notice`` naming where to continue if the requested range was
    wider than that.
    """
    path = resolve_document(relative)
    page_texts = _page_texts(path)
    page_numbers = sorted(page_texts)
    total_pages = len(page_numbers)

    notice: str | None = None
    if start_page is not None or end_page is not None:
        lo = start_page if start_page is not None else page_numbers[0]
        hi = end_page if end_page is not None else page_numbers[-1]
        selected = [p for p in page_numbers if lo <= p <= hi]
        if len(selected) > MAX_EXPLICIT_RANGE_PAGES:
            selected = selected[:MAX_EXPLICIT_RANGE_PAGES]
            notice = (
                f"start_page/end_page is capped at {MAX_EXPLICIT_RANGE_PAGES} pages per "
                f"call; showing pages {selected[0]}-{selected[-1]}. Call read_document "
                f"again with start_page={selected[-1] + 1} to continue reading past this "
                "section."
            )
    elif total_pages <= 1:
        selected = page_numbers
    else:
        selected = []
        budget = int_env(PEEK_CHAR_BUDGET_ENV, PEEK_CHAR_BUDGET)
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


# Every indexed document is also exposed as an MCP resource (#332) under
# this scheme, e.g. "docs://invoices/2024-01.pdf" for the file at
# "invoices/2024-01.pdf" under docs_dir() -- the identity URI, not a
# search/read verb, since a resource IS the document rather than an action
# on it.
RESOURCE_URI_SCHEME = "docs"


def resource_uri(relative: str) -> str:
    return f"{RESOURCE_URI_SCHEME}://{relative}"


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
        start_page/end_page returns at most 5 pages per call -- a wider
        range gets truncated with a `notice` telling you the start_page to
        continue from. Formulate the search query from what you actually
        need to answer, not a generic re-read of the whole document -- the
        question may need several distinct searches (several facts on
        different pages) before you have everything required to answer it."""
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

    @server.tool()
    def write_note(path: str, page: int, note: str) -> dict[str, Any]:
        """Leave a short, page-anchored note about a document -- persists to
        disk across separate conversations/subprocess restarts (this server
        is a fresh subprocess per chat request), so a later pass over the
        SAME document can discover what an earlier one already found: a
        summary, an observation, a discrepancy worth flagging ("the total on
        page 3 doesn't match the sum of page 2's line items"). `path` MUST be
        one of the exact strings from list_files. `page` is required --
        every note is anchored to the specific page it's about, never
        free-floating. Append-only: there is no edit/delete, so if an
        earlier note turns out wrong, write a new note saying so rather than
        trying to correct it in place. `note` is capped at roughly 2000
        characters (operator-tunable) -- this is for a short observation or
        discrepancy flag, NOT a summary or full-text dump of the page; a call
        over the cap fails outright rather than truncating, so keep it brief
        up front instead of finding out after the call fails. There is also
        a cap on how many notes one document can hold in total."""
        return append_note(path, page, note)

    @server.tool()
    def read_notes(path: str) -> list[dict[str, Any]]:
        """Every note previously left about a document via write_note,
        oldest first -- an empty list if none exist. Unlike list_files, this
        is NOT called automatically before every request: call it yourself,
        ideally before re-analyzing a document you may have already been
        told about, so you don't re-derive something a previous pass on this
        document may already have determined."""
        return list_notes(path)

    # One resource per indexed document (#332), same enumeration list_files
    # already uses -- resources and list_files can never disagree about
    # what's in the corpus. Static (not template) resources, registered up
    # front from that same fixed listing, so list_resources() enumerates
    # the corpus directly instead of a client having to guess URIs from a
    # template: this server is a fresh subprocess per chat request (see
    # module docstring), so the corpus can't drift mid-process the way a
    # long-lived server's would.
    def _make_reader(relative: str) -> Callable[[], str]:
        def _read() -> str:
            return _full_document_text(relative)

        return _read

    for relative in list_documents():
        server.resource(
            resource_uri(relative),
            name=relative,
            title=relative,
            description=f"Full text of {relative!r} from the shared documents directory.",
            mime_type="text/plain",
        )(_make_reader(relative))

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
