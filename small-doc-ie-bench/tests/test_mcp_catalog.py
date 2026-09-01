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
from docie_bench.mcp_servers import calculator, code_interpreter, dates, docs_search, web_fetch
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


def test_is_html_content_type() -> None:
    assert web_fetch.is_html_content_type("text/html; charset=utf-8") is True
    assert web_fetch.is_html_content_type("application/xhtml+xml") is True
    assert web_fetch.is_html_content_type("application/json") is False
    assert web_fetch.is_html_content_type("text/plain") is False
    assert web_fetch.is_html_content_type("") is False


async def test_fetch_url_extracts_text_from_html(monkeypatch: pytest.MonkeyPatch) -> None:
    page = (
        "<html><head><title>T</title>"
        "<style>body { color: red; }</style>"
        "<script>function evil() { alert('tracked'); }</script>"
        "</head><body>"
        "<h1>Hello World</h1>"
        "<p>This is the actual content.</p>"
        "<script>console.log('more tracking');</script>"
        "</body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=page, headers={"content-type": "text/html; charset=utf-8"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setenv(web_fetch.ALLOWED_HOSTS_ENV, "site.com")
    result = await web_fetch.fetch_url("https://site.com/page.html")
    assert result["ok"] is True
    assert result["content_type"] == "text/html; charset=utf-8"
    assert "Hello World" in result["text"]
    assert "actual content" in result["text"]
    assert "<" not in result["text"]
    assert "evil" not in result["text"]
    assert "alert" not in result["text"]
    assert "tracked" not in result["text"]
    assert "color: red" not in result["text"]


async def test_fetch_url_non_html_passes_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"a": 1, "b": [1, 2, 3]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=payload, headers={"content-type": "application/json"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setenv(web_fetch.ALLOWED_HOSTS_ENV, "site.com")
    result = await web_fetch.fetch_url("https://site.com/data.json")
    assert result["ok"] is True
    assert result["text"] == payload


async def test_fetch_url_truncates_extracted_html_text(monkeypatch: pytest.MonkeyPatch) -> None:
    page = "<html><body>" + "<p>word</p>" * 60_000 + "</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=page, headers={"content-type": "text/html"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setenv(web_fetch.ALLOWED_HOSTS_ENV, "site.com")
    result = await web_fetch.fetch_url("https://site.com/long.html")
    assert result["ok"] is True
    assert result["truncated"] is True
    assert len(result["text"].encode("utf-8")) <= 200_000
    assert "<" not in result["text"]


# ----------------------------------------------------------------- docs-search


@pytest.fixture
def docs(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv(docs_search.DOCS_DIR_ENV, str(tmp_path))
    docs_search._EXTRACTION_CACHE.clear()
    return tmp_path


def test_list_documents_only_lists_supported_files_recursively(docs: Path) -> None:
    (docs / "a.txt").write_text("hello")
    (docs / "ignored.docx").write_text("nope")
    sub = docs / "sub"
    sub.mkdir()
    (sub / "b.pdf").write_bytes(b"%PDF-fake")
    assert docs_search.list_documents() == ["a.txt", "sub/b.pdf"]


def test_resolve_document_rejects_escape_attempts(docs: Path) -> None:
    (docs / "a.txt").write_text("hello")
    # A POSIX-rooted path isn't `.is_absolute()` on a Windows test runner (no
    # drive letter), so it's caught by the parents-check instead of the
    # explicit absolute-path guard -- either way it must never resolve
    # inside docs_dir(), which is what actually matters.
    with pytest.raises(ValueError, match="relative|outside"):
        docs_search.resolve_document("/etc/passwd")
    with pytest.raises(ValueError, match="outside"):
        docs_search.resolve_document("../outside.txt")
    with pytest.raises(ValueError, match="no such document"):
        docs_search.resolve_document("missing.txt")
    (docs / "b.exe").write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported file type"):
        docs_search.resolve_document("b.exe")
    assert docs_search.resolve_document("a.txt") == (docs / "a.txt").resolve()


def test_missing_document_error_lists_the_real_files_so_a_retry_self_corrects(
    docs: Path,
) -> None:
    # A model that invents a path instead of calling list_files first should
    # still land on a real one after seeing the error -- the error message
    # IS the correction, not just an upfront instruction it may have ignored.
    (docs / "invoice.pdf").write_bytes(b"%PDF-fake")
    with pytest.raises(ValueError, match=r"invoice\.pdf"):
        docs_search.resolve_document("1/20")


def test_missing_document_error_says_nothing_is_available_when_the_directory_is_empty(
    docs: Path,
) -> None:
    with pytest.raises(ValueError, match="no documents are available"):
        docs_search.resolve_document("anything.pdf")


def test_document_text_groups_lines_by_page(docs: Path) -> None:
    (docs / "note.txt").write_text("first line\nsecond line\n")
    result = docs_search.document_text("note.txt")
    assert result == {
        "path": "note.txt",
        "total_pages": 1,
        "pages": [{"page": 1, "text": "first line\nsecond line"}],
    }
    assert "notice" not in result


def _multi_page_extractor(monkeypatch, page_count: int, chars_per_page: int):
    """Stub liteparse's extract() to return ``page_count`` pages of
    ``chars_per_page`` characters each -- .txt extraction always lands
    everything on page 1 (see ocr/base.py's text_to_blocks), so a real
    multi-page document needs a controlled fake extractor instead."""
    from docie_bench.ocr import factory as ocr_factory
    from docie_bench.schemas.common import OCRBlock

    def _page(n: int) -> OCRBlock:
        text = f"page {n} " + "x" * chars_per_page
        return OCRBlock(id=f"b{n}", text=text, page=n, source="manual")

    class _Stub:
        def extract(self, path: Path) -> list[OCRBlock]:
            return [_page(n) for n in range(1, page_count + 1)]

    monkeypatch.setattr(ocr_factory, "get_ocr_backend", lambda name: _Stub())


_BIG_PAGE_CHARS = 2000


def test_document_text_peeks_a_long_document_instead_of_returning_it_whole(
    docs: Path, monkeypatch
) -> None:
    # A 150-page regulation must not overflow a small model's context --
    # the default (no page range) call stops well short of the full body.
    (docs / "big.pdf").write_bytes(b"%PDF-fake")
    _multi_page_extractor(monkeypatch, page_count=4, chars_per_page=_BIG_PAGE_CHARS)

    result = docs_search.document_text("big.pdf")

    assert result["total_pages"] == 4
    assert len(result["pages"]) < 4
    assert "notice" in result
    assert "search_text" in result["notice"]
    assert "start_page" in result["notice"]


def test_document_text_honors_an_explicit_page_range_under_the_page_cap(
    docs: Path, monkeypatch
) -> None:
    (docs / "big.pdf").write_bytes(b"%PDF-fake")
    _multi_page_extractor(monkeypatch, page_count=4, chars_per_page=_BIG_PAGE_CHARS)

    result = docs_search.document_text("big.pdf", start_page=3, end_page=4)

    assert result["total_pages"] == 4
    assert [p["page"] for p in result["pages"]] == [3, 4]
    assert "notice" not in result  # 2 pages requested, under the 5-page cap


def test_document_text_caps_an_explicit_range_at_max_pages(docs: Path, monkeypatch) -> None:
    (docs / "big.pdf").write_bytes(b"%PDF-fake")
    _multi_page_extractor(monkeypatch, page_count=20, chars_per_page=10)

    result = docs_search.document_text("big.pdf", start_page=3, end_page=20)

    assert [p["page"] for p in result["pages"]] == [3, 4, 5, 6, 7]
    assert "notice" in result
    assert "start_page=8" in result["notice"]


def test_document_text_peek_budget_is_operator_tunable(docs: Path, monkeypatch) -> None:
    (docs / "big.pdf").write_bytes(b"%PDF-fake")
    _multi_page_extractor(monkeypatch, page_count=4, chars_per_page=_BIG_PAGE_CHARS)
    page_len = len("page 1 ") + _BIG_PAGE_CHARS  # matches _multi_page_extractor's format
    monkeypatch.setenv(docs_search.PEEK_CHAR_BUDGET_ENV, str(page_len * 3))

    result = docs_search.document_text("big.pdf")

    assert len(result["pages"]) == 3


def _stub_pdf_inspector(
    monkeypatch, *, pdf_type: str, pages_needing_ocr: list[int] | None = None
):
    """Stub pdf_inspector's classify_pdf/extract_text_with_positions --
    real classification needs a real, parseable PDF this test suite has no
    fixture for, so the library calls are mocked directly (same pattern
    _multi_page_extractor already uses for liteparse)."""
    import pdf_inspector

    class _Classification:
        def __init__(self) -> None:
            self.pdf_type = pdf_type
            self.pages_needing_ocr = pages_needing_ocr or []

    class _Item:
        def __init__(self, text: str, page: int) -> None:
            self.text = text
            self.page = page
            self.x = 0.0
            self.y = 0.0
            self.width = 10.0
            self.height = 10.0

    monkeypatch.setattr(pdf_inspector, "classify_pdf", lambda path: _Classification())
    monkeypatch.setattr(
        pdf_inspector,
        "extract_text_with_positions",
        lambda path: [_Item("hello from pdf_inspector", 1)],
    )


def test_extracted_blocks_uses_pdf_inspector_fast_path_for_text_based_pdf(
    docs: Path, monkeypatch
) -> None:
    from docie_bench.ocr import factory as ocr_factory

    (docs / "a.pdf").write_bytes(b"%PDF-fake")
    _stub_pdf_inspector(monkeypatch, pdf_type="text_based")

    def _fail_if_called(name: str, **kwargs: object) -> None:
        raise AssertionError("liteparse must not run when pdf_inspector's fast path applies")

    monkeypatch.setattr(ocr_factory, "get_ocr_backend", _fail_if_called)

    blocks = docs_search._extracted_blocks(docs / "a.pdf")

    assert len(blocks) == 1
    assert blocks[0].text == "hello from pdf_inspector"
    assert blocks[0].source == "pdf_inspector"


def test_extracted_blocks_falls_back_to_liteparse_for_a_scanned_pdf(
    docs: Path, monkeypatch
) -> None:
    from docie_bench.ocr import factory as ocr_factory
    from docie_bench.schemas.common import OCRBlock

    (docs / "scanned.pdf").write_bytes(b"%PDF-fake")
    _stub_pdf_inspector(monkeypatch, pdf_type="scanned", pages_needing_ocr=[0])

    class _Stub:
        def extract(self, path: Path) -> list[OCRBlock]:
            return [OCRBlock(id="b1", text="ocr'd text", page=1, source="pdf_text")]

    monkeypatch.setattr(ocr_factory, "get_ocr_backend", lambda name: _Stub())

    blocks = docs_search._extracted_blocks(docs / "scanned.pdf")

    assert len(blocks) == 1
    assert blocks[0].source == "pdf_text"


def test_extracted_blocks_falls_back_to_liteparse_when_pdf_inspector_errors(
    docs: Path, monkeypatch
) -> None:
    import pdf_inspector

    from docie_bench.ocr import factory as ocr_factory
    from docie_bench.schemas.common import OCRBlock

    (docs / "broken.pdf").write_bytes(b"%PDF-fake")
    monkeypatch.setattr(
        pdf_inspector,
        "classify_pdf",
        lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    class _Stub:
        def extract(self, path: Path) -> list[OCRBlock]:
            return [OCRBlock(id="b1", text="liteparse saved us", page=1, source="pdf_text")]

    monkeypatch.setattr(ocr_factory, "get_ocr_backend", lambda name: _Stub())

    blocks = docs_search._extracted_blocks(docs / "broken.pdf")

    assert blocks[0].text == "liteparse saved us"


def test_search_documents_across_all_or_one_file(docs: Path) -> None:
    (docs / "a.txt").write_text("the invoice total is 400 EUR")
    (docs / "b.txt").write_text("nothing relevant here")
    everywhere = docs_search.search_documents("invoice")
    assert [m["path"] for m in everywhere] == ["a.txt"]
    assert everywhere[0]["snippet"] == "the invoice total is 400 EUR"
    scoped = docs_search.search_documents("nothing", path="b.txt")
    assert len(scoped) == 1
    assert docs_search.search_documents("invoice", path="b.txt") == []


def test_search_documents_returns_one_match_per_page_with_a_windowed_snippet(
    docs: Path,
) -> None:
    # liteparse splits a page into many small blocks (one per line for a
    # .txt file) -- a page with the query on several lines must come back
    # as ONE match, its windows merged, not one near-duplicate per line.
    (docs / "a.txt").write_text(
        "Chapter I\nsome unrelated text\nChapter II mentions Chapter I\nabout Chapter III"
    )
    matches = docs_search.search_documents("chapter")
    assert len(matches) == 1
    assert matches[0]["page"] == 1
    assert "Chapter I" in matches[0]["snippet"]
    assert "Chapter III" in matches[0]["snippet"]


def test_search_documents_snippet_stays_windowed_on_a_large_page(docs: Path) -> None:
    # A single match on a page far larger than the window must not pull in
    # the whole page -- that's the exact bloat this window guards against.
    filler = "x" * 5000
    text = f"{filler} needle {filler}"
    (docs / "big.txt").write_text(text)
    matches = docs_search.search_documents("needle")
    assert len(matches) == 1
    snippet = matches[0]["snippet"]
    assert "needle" in snippet
    assert len(snippet) < len(text)
    assert len(snippet) <= 2 * docs_search._SNIPPET_WINDOW + len("needle")


def test_search_documents_snippet_window_is_operator_tunable(docs: Path, monkeypatch) -> None:
    monkeypatch.setenv(docs_search.SNIPPET_WINDOW_ENV, "10")
    filler = "x" * 5000
    text = f"{filler} needle {filler}"
    (docs / "big.txt").write_text(text)
    matches = docs_search.search_documents("needle")
    assert len(matches[0]["snippet"]) <= 2 * 10 + len("needle")


def test_search_documents_rejects_empty_query(docs: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        docs_search.search_documents("   ")


# ── pluggable search backends (#282): search_documents dispatches through
# get_search_backend, an operator-set env var picks which one -────────────


def test_get_search_backend_returns_substring_by_default() -> None:
    backend_cls = docs_search.SubstringSearchBackend
    assert isinstance(docs_search.get_search_backend("substring"), backend_cls)
    assert isinstance(docs_search.get_search_backend("SUBSTRING"), backend_cls)


def test_get_search_backend_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown search backend"):
        docs_search.get_search_backend("vector")


def test_get_search_backend_returns_hybrid(docs: Path) -> None:
    hybrid_cls = docs_search.HybridSearchBackend
    assert isinstance(docs_search.get_search_backend("hybrid"), hybrid_cls)
    assert isinstance(docs_search.get_search_backend("HYBRID"), hybrid_cls)
    # substring stays exactly what it was -- adding hybrid didn't touch it.
    substring_cls = docs_search.SubstringSearchBackend
    assert isinstance(docs_search.get_search_backend("substring"), substring_cls)


def _mock_reranker(monkeypatch, handler) -> list:
    """Patch ``httpx.post`` (used by ``docs_search._rerank``) to answer via
    ``handler``, same MockTransport pattern as code_interpreter's tests."""
    captured: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: real_client(transport=httpx.MockTransport(wrapped)).post(url, **kw),
    )
    return captured


def test_hybrid_backend_reranks_substring_candidates_by_semantic_score(
    docs: Path, monkeypatch
) -> None:
    # Both files literally contain "widget" (substring's pre-filter can't
    # tell "mentions it in passing" from "is actually about it" -- both hit,
    # in a.txt-then-b.txt document order) but only b.txt's page is actually
    # ABOUT a widget -- a semantic reranker should put it first, flipping
    # substring's document-order result.
    (docs / "a.txt").write_text("The gizmo costs 10 dollars and also needs a widget")
    (docs / "b.txt").write_text("A widget is a small mechanical part used in gizmos")
    monkeypatch.setenv(docs_search.RERANKER_URL_ENV, "http://reranker.local")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["query"] == "widget"
        scored = [
            (i, 0.9 if "small mechanical part" in doc else 0.1)
            for i, doc in enumerate(body["documents"])
        ]
        ranked = sorted(scored, key=lambda pair: -pair[1])
        return httpx.Response(
            200,
            json={"results": [{"index": i, "relevance_score": s} for i, s in ranked]},
        )

    captured = _mock_reranker(monkeypatch, handler)

    # substring's pre-filter matches both files (they both literally contain
    # "widget"), in a.txt-then-b.txt document order -- the reranker's scoring
    # above should flip that to b.txt first.
    result = docs_search.HybridSearchBackend().search("widget", ["a.txt", "b.txt"])

    assert [m["path"] for m in result] == ["b.txt", "a.txt"]
    assert len(captured) == 1


def test_hybrid_backend_returns_empty_when_substring_prefilter_finds_nothing(
    docs: Path, monkeypatch
) -> None:
    # A design choice (see #339): zero literal term overlap returns empty
    # rather than embedding the whole corpus -- fast/cheap for the common
    # "actually nothing matches" case, at the cost of missing a fully
    # paraphrased query with no shared words at all (left to a future
    # pure-vector backend). The reranker must never be called on this path.
    (docs / "a.txt").write_text("completely unrelated content")
    monkeypatch.setenv(docs_search.RERANKER_URL_ENV, "http://reranker.local")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("reranker must not be called when nothing pre-filtered")

    _mock_reranker(monkeypatch, handler)

    result = docs_search.HybridSearchBackend().search("nonexistent-term-xyz", ["a.txt"])

    assert result == []


def test_hybrid_backend_errors_clearly_when_reranker_url_is_unset(docs: Path, monkeypatch) -> None:
    (docs / "a.txt").write_text("the invoice total is 400 EUR")
    monkeypatch.delenv(docs_search.RERANKER_URL_ENV, raising=False)

    with pytest.raises(docs_search.RerankerUnavailableError, match=docs_search.RERANKER_URL_ENV):
        docs_search.HybridSearchBackend().search("invoice", ["a.txt"])


def test_hybrid_backend_errors_clearly_when_reranker_is_unreachable(
    docs: Path, monkeypatch
) -> None:
    (docs / "a.txt").write_text("the invoice total is 400 EUR")
    monkeypatch.setenv(docs_search.RERANKER_URL_ENV, "http://reranker.local")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _mock_reranker(monkeypatch, handler)

    with pytest.raises(docs_search.RerankerUnavailableError, match="reranker.local"):
        docs_search.HybridSearchBackend().search("invoice", ["a.txt"])


def test_search_documents_uses_the_backend_env_var(docs: Path, monkeypatch) -> None:
    (docs / "a.txt").write_text("the invoice total is 400 EUR")

    class _StubBackend(docs_search.SearchBackend):
        def search(self, query: str, targets: list[str]) -> list[dict]:
            return [{"path": "stub", "page": 0, "snippet": f"stub:{query}:{targets}"}]

    monkeypatch.setattr(docs_search, "get_search_backend", lambda name: _StubBackend())
    monkeypatch.setenv(docs_search.BACKEND_ENV, "whatever-the-stub-ignores")
    result = docs_search.search_documents("invoice")
    assert result == [{"path": "stub", "page": 0, "snippet": "stub:invoice:['a.txt']"}]


def test_search_documents_unknown_backend_is_a_clear_config_error(docs: Path, monkeypatch) -> None:
    (docs / "a.txt").write_text("hello")
    monkeypatch.setenv(docs_search.BACKEND_ENV, "vector")
    with pytest.raises(ValueError, match="Unknown search backend 'vector'"):
        docs_search.search_documents("hello")


# ── extraction memoization: repeated search/read against the SAME document
# skips re-running liteparse (expected agentic-search usage) ────────────────


def _counting_extractor(monkeypatch, calls: list[Path]):
    """Stub liteparse's extract() to log every call it receives -- for
    asserting the cache actually skips re-extraction, not just that search
    still returns the right answer."""
    from docie_bench.ocr import factory as ocr_factory
    from docie_bench.schemas.common import OCRBlock

    class _Stub:
        def extract(self, path: Path) -> list[OCRBlock]:
            calls.append(path)
            return [OCRBlock(id="b1", text=path.read_text(), page=1, source="manual")]

    monkeypatch.setattr(ocr_factory, "get_ocr_backend", lambda name: _Stub())


def test_repeated_search_against_the_same_document_extracts_once(docs: Path, monkeypatch) -> None:
    (docs / "a.txt").write_text("the invoice total is 400 EUR")
    calls: list[Path] = []
    _counting_extractor(monkeypatch, calls)

    docs_search.search_documents("invoice")
    docs_search.search_documents("invoice")
    docs_search.document_text("a.txt")

    assert len(calls) == 1


def test_extraction_cache_invalidates_when_the_file_changes(docs: Path, monkeypatch) -> None:
    path = docs / "a.txt"
    path.write_text("version one")
    calls: list[Path] = []
    _counting_extractor(monkeypatch, calls)

    docs_search.document_text("a.txt")
    assert len(calls) == 1

    # A changed mtime/size (however small) must miss, not serve stale text --
    # the cache key is (path, mtime_ns, size), not just the path.
    path.write_text("version two, now longer")
    docs_search.document_text("a.txt")
    assert len(calls) == 2


# ── disk-backed second tier: survives the fresh-subprocess-per-request
# lifecycle the in-memory cache alone can't (#298) ──────────────────────────


def test_extraction_survives_a_simulated_fresh_subprocess(docs: Path, monkeypatch) -> None:
    """docs-search is a new subprocess per chat request -- clearing
    _EXTRACTION_CACHE simulates that restart. The disk tier must still
    skip re-extraction."""
    path = docs / "a.txt"
    path.write_text("the invoice total is 400 EUR")
    calls: list[Path] = []
    _counting_extractor(monkeypatch, calls)

    docs_search.document_text("a.txt")
    assert len(calls) == 1

    docs_search._EXTRACTION_CACHE.clear()
    docs_search.document_text("a.txt")
    assert len(calls) == 1


def test_disk_cache_lives_beside_docs_dir_not_inside_it(docs: Path, monkeypatch) -> None:
    (docs / "a.txt").write_text("hello")
    _counting_extractor(monkeypatch, [])
    docs_search.document_text("a.txt")

    cache_dir = docs.parent / f"{docs.name}.extraction-cache"
    assert cache_dir.is_dir()
    assert list(cache_dir.glob("*.json"))
    # docs_dir() itself gets no new files -- "this process never writes to
    # docs_dir()" holds even for the operator's read-only shared corpus.
    assert list(docs.iterdir()) == [docs / "a.txt"]


def test_disk_cache_invalidates_when_the_file_changes_across_a_fresh_process(
    docs: Path, monkeypatch
) -> None:
    path = docs / "a.txt"
    path.write_text("version one")
    calls: list[Path] = []
    _counting_extractor(monkeypatch, calls)

    docs_search.document_text("a.txt")
    docs_search._EXTRACTION_CACHE.clear()

    path.write_text("version two, now longer")
    docs_search.document_text("a.txt")
    assert len(calls) == 2  # disk cache's own mtime/size check also misses correctly


def test_disk_cache_falls_back_gracefully_on_a_malformed_cache_file(
    docs: Path, monkeypatch
) -> None:
    path = docs / "a.txt"
    path.write_text("hello")
    calls: list[Path] = []
    _counting_extractor(monkeypatch, calls)

    cache_path = docs_search._disk_cache_path(docs_search.resolve_document("a.txt"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("not valid json{{{")

    result = docs_search.document_text("a.txt")
    assert len(calls) == 1  # gracefully re-extracted instead of crashing
    assert result["pages"][0]["text"] == "hello"


# ------------------------------------------------------------ code-interpreter


def test_submit_code_raises_when_token_missing(monkeypatch) -> None:
    monkeypatch.delenv(code_interpreter.TOKEN_ENV, raising=False)
    with pytest.raises(code_interpreter.CodeInterpreterUnavailableError, match="TOKEN"):
        code_interpreter.submit_code("print(1)")


def test_submit_code_posts_to_judge0_and_maps_the_response(monkeypatch) -> None:
    monkeypatch.setenv(code_interpreter.TOKEN_ENV, "secret-token")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "stdout": "6\n",
                "stderr": "",
                "exit_code": 0,
                "status": {"id": 3, "description": "Accepted"},
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: real_client(transport=httpx.MockTransport(handler)).post(url, **kw),
    )

    result = code_interpreter.submit_code("print(3 * 2)", url="http://judge0-server:2358")

    assert result == {
        "stdout": "6\n",
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
        "truncated": False,
    }
    (request,) = captured
    assert request.headers["x-auth-token"] == "secret-token"
    body = json.loads(request.content)
    assert body["language_id"] == 71
    assert body["source_code"] == "print(3 * 2)"
    assert request.url.params["wait"] == "true"


def test_submit_code_maps_time_limit_exceeded(monkeypatch) -> None:
    monkeypatch.setenv(code_interpreter.TOKEN_ENV, "secret-token")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "stdout": None,
                "stderr": None,
                "exit_code": None,
                "status": {"id": 5, "description": "Time Limit Exceeded"},
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: real_client(transport=httpx.MockTransport(handler)).post(url, **kw),
    )

    result = code_interpreter.submit_code("while True: pass", url="http://judge0-server:2358")
    assert result["timed_out"] is True
    assert result["stdout"] == ""


def test_submit_code_sends_stdin_when_given(monkeypatch) -> None:
    monkeypatch.setenv(code_interpreter.TOKEN_ENV, "secret-token")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "stdout": "hi\n",
                "stderr": "",
                "exit_code": 0,
                "status": {"id": 3, "description": "Accepted"},
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: real_client(transport=httpx.MockTransport(handler)).post(url, **kw),
    )

    code_interpreter.submit_code(
        "print(input())", url="http://judge0-server:2358", stdin="hi"
    )

    (request,) = captured
    body = json.loads(request.content)
    assert body["stdin"] == "hi"


def test_submit_code_omits_stdin_when_not_given(monkeypatch) -> None:
    monkeypatch.setenv(code_interpreter.TOKEN_ENV, "secret-token")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "stdout": "6\n",
                "stderr": "",
                "exit_code": 0,
                "status": {"id": 3, "description": "Accepted"},
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: real_client(transport=httpx.MockTransport(handler)).post(url, **kw),
    )

    code_interpreter.submit_code("print(3 * 2)", url="http://judge0-server:2358")

    (request,) = captured
    body = json.loads(request.content)
    assert "stdin" not in body


def test_run_python_threads_stdin_through_to_submit_code(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_submit_code(code, **kw):
        calls.append({"code": code, **kw})
        return {}

    monkeypatch.setattr(code_interpreter, "submit_code", fake_submit_code)

    server = code_interpreter.build_server()
    run_python = server._tool_manager._tools["run_python"].fn
    run_python("print(input())", stdin="hi")

    assert calls == [{"code": "print(input())", "stdin": "hi"}]


# --------------------------------------------------- servers speak real MCP


@pytest.mark.parametrize(
    ("module", "expected_tools"),
    [
        (calculator, {"calc", "sum_check"}),
        (dates, {"parse_date", "date_diff", "today"}),
        (web_fetch, {"fetch"}),
        (docs_search, {"list_files", "read_document", "search_text"}),
        (code_interpreter, {"run_python"}),
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
    assert set(entries) == {
        "calculator",
        "dates",
        "web-fetch",
        "docs-search",
        "code-interpreter",
        "call-llm",
        "sql-agent",
    }
    assert not entries["calculator"]["enabled"]
    assert entries["web-fetch"]["params"][0]["name"] == "allowed_hosts"
    docs_search_params = {p["name"] for p in entries["docs-search"]["params"]}
    assert docs_search_params == {
        "docs_dir",
        "backend",
        "reranker_url",
        "snippet_window",
        "snippet_max_chars",
        "peek_char_budget",
    }
    sql_agent_params = {p["name"]: p["secret"] for p in entries["sql-agent"]["params"]}
    assert sql_agent_params == {
        "host": False,
        "port": False,
        "user": False,
        "password": True,
        "dbname": False,
        "row_limit": False,
    }
    ci_params = {p["name"]: p["required"] for p in entries["code-interpreter"]["params"]}
    assert ci_params == {"url": False, "token": True}

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


def test_docs_search_spawned_end_to_end(
    client: TestClient, registry_path: Path, tmp_path: Path
) -> None:
    # Real stdio spawn with the env-var-scoped docs dir -- proves the
    # subprocess actually sees DOCIE_MCP_DOCS_SEARCH_DIR (this SDK only
    # inherits a minimal curated env by default, not the full parent env).
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "invoice.txt").write_text("total due: 42 EUR")
    registry_path.write_text(
        json.dumps(
            {
                "servers": {
                    "docs-search": {
                        "transport": "stdio",
                        "command": [
                            sys.executable, "-m", "docie_bench.mcp_servers.docs_search",
                        ],
                        "env": {docs_search.DOCS_DIR_ENV: str(docs)},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    res = client.post("/v1/mcp/servers/docs-search/test")
    assert res.status_code == 200, res.text
    assert {t["name"] for t in res.json()["tools"]} == {
        "list_files", "read_document", "search_text",
    }


def test_registry_entry_for_omits_empty_env() -> None:
    entry = CATALOG["calculator"]
    assert "env" not in registry_entry_for(entry, {})


def test_enable_docs_search_writes_its_dir_env_var(client: TestClient, registry_path: Path) -> None:
    res = client.post(
        "/v1/mcp/servers",
        json={"catalog": "docs-search", "params": {"docs_dir": "/data/agent-docs"}},
    )
    assert res.status_code == 201, res.text
    saved = json.loads(registry_path.read_text(encoding="utf-8"))["servers"]["docs-search"]
    assert saved["command"] == ["python", "-m", "docie_bench.mcp_servers.docs_search"]
    assert saved["env"] == {"DOCIE_MCP_DOCS_SEARCH_DIR": "/data/agent-docs"}


def test_enable_code_interpreter_requires_the_token_param(client: TestClient) -> None:
    res = client.post("/v1/mcp/servers", json={"catalog": "code-interpreter", "params": {}})
    assert res.status_code == 422
    assert "token" in res.json()["detail"]


def test_enable_code_interpreter_writes_url_and_token_env(
    client: TestClient, registry_path: Path
) -> None:
    res = client.post(
        "/v1/mcp/servers",
        json={
            "catalog": "code-interpreter",
            "params": {"token": "secret-token", "url": "http://judge0-server:2358"},
        },
    )
    assert res.status_code == 201, res.text
    saved = json.loads(registry_path.read_text(encoding="utf-8"))["servers"]["code-interpreter"]
    assert saved["command"] == ["python", "-m", "docie_bench.mcp_servers.code_interpreter"]
    assert saved["env"] == {
        "DOCIE_MCP_CODE_INTERPRETER_TOKEN": "secret-token",
        "DOCIE_MCP_CODE_INTERPRETER_URL": "http://judge0-server:2358",
    }


# ---------------------------------------------------- code-interpreter workers


def test_workers_route_404_for_a_non_code_interpreter_server(
    client: TestClient, registry_path: Path
) -> None:
    client.post("/v1/mcp/servers", json={"catalog": "calculator"})
    assert client.get("/v1/mcp/servers/calculator/workers").status_code == 404


def test_workers_route_404_when_unregistered(client: TestClient) -> None:
    assert client.get("/v1/mcp/servers/code-interpreter/workers").status_code == 404


def test_workers_route_422_without_a_token(client: TestClient, registry_path: Path) -> None:
    registry_path.write_text(
        json.dumps(
            {
                "servers": {
                    "code-interpreter": {
                        "transport": "stdio",
                        "command": ["python", "-m", "docie_bench.mcp_servers.code_interpreter"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    res = client.get("/v1/mcp/servers/code-interpreter/workers")
    assert res.status_code == 422
    assert "TOKEN" in res.json()["detail"]


def test_workers_route_proxies_judge0(
    client: TestClient, registry_path: Path, monkeypatch
) -> None:
    registry_path.write_text(
        json.dumps(
            {
                "servers": {
                    "code-interpreter": {
                        "transport": "stdio",
                        "command": ["python", "-m", "docie_bench.mcp_servers.code_interpreter"],
                        "env": {
                            "DOCIE_MCP_CODE_INTERPRETER_TOKEN": "secret-token",
                            "DOCIE_MCP_CODE_INTERPRETER_URL": "http://judge0-server:2358",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-auth-token"] == "secret-token"
        assert request.url.path == "/workers"
        return httpx.Response(200, json=[{"queue": "default", "size": 0, "available": 2,
                                           "idle": 2, "working": 0, "paused": 0, "failed": 0}])

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )

    res = client.get("/v1/mcp/servers/code-interpreter/workers")
    assert res.status_code == 200, res.text
    assert res.json()["queues"][0]["available"] == 2


def test_workers_route_502_when_judge0_unreachable(
    client: TestClient, registry_path: Path, monkeypatch
) -> None:
    registry_path.write_text(
        json.dumps(
            {
                "servers": {
                    "code-interpreter": {
                        "transport": "stdio",
                        "command": ["python", "-m", "docie_bench.mcp_servers.code_interpreter"],
                        "env": {
                            "DOCIE_MCP_CODE_INTERPRETER_TOKEN": "secret-token",
                            "DOCIE_MCP_CODE_INTERPRETER_URL": "http://judge0-server:2358",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )

    assert client.get("/v1/mcp/servers/code-interpreter/workers").status_code == 502
