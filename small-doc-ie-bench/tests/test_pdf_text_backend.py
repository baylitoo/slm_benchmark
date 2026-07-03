from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from docie_bench.ocr.pdf_text import PdfTextBackend


class _FakeParser:
    """Stand-in for liteparse.LiteParse that records init kwargs and returns
    two text items (one blank, to prove empty items are skipped)."""

    last_init: dict[str, object] = {}

    def __init__(self, **kwargs) -> None:
        type(self).last_init = kwargs

    def parse(self, path):  # noqa: ARG002 - path unused in the stub
        page = SimpleNamespace(
            page_num=1,
            width=612,
            height=792,
            text="Total 42.00",
            text_items=[
                SimpleNamespace(
                    text="Total 42.00", x=10.0, y=20.0, width=90.0, height=12.0, confidence=1.0
                ),
                SimpleNamespace(text="   ", x=0.0, y=0.0, width=1.0, height=1.0, confidence=0.5),
            ],
        )
        return SimpleNamespace(pages=[page], text="Total 42.00")


def test_pdf_text_maps_liteparse_items_to_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("docie_bench.ocr.pdf_text.LiteParse", _FakeParser)
    backend = PdfTextBackend(dpi=200, ocr_server_url="http://vlm/ocr", language="fr")

    blocks = backend.extract(Path("invoice.pdf"))

    # Blank text item is skipped.
    assert len(blocks) == 1
    block = blocks[0]
    assert block.text == "Total 42.00"
    assert block.page == 1
    assert block.source == "pdf_text"
    assert block.confidence == 1.0
    # bbox is (x, y, x+width, y+height).
    assert (block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1) == (10.0, 20.0, 100.0, 32.0)
    # OCR route/dpi/language flow into the parser and the cache configuration.
    assert _FakeParser.last_init["ocr_server_url"] == "http://vlm/ocr"
    assert _FakeParser.last_init["ocr_language"] == "fr"
    assert _FakeParser.last_init["dpi"] == 200.0
    cfg = backend.configuration()
    assert cfg == {
        "engine": "liteparse",
        "dpi": 200,
        "ocr_server_url": "http://vlm/ocr",
        "language": "fr",
    }
    assert backend.version().startswith("1:liteparse-")


def test_pdf_text_reads_plain_text_without_liteparse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # .txt short-circuits before touching liteparse.
    def _boom(**kwargs):  # pragma: no cover - must not be called
        raise AssertionError("liteparse should not be constructed for .txt")

    monkeypatch.setattr("docie_bench.ocr.pdf_text.LiteParse", _boom)
    doc = tmp_path / "doc.txt"
    doc.write_text("hello\nworld", encoding="utf-8")

    blocks = PdfTextBackend().extract(doc)

    assert [b.text for b in blocks] == ["hello", "world"]


def test_pdf_text_rejects_unsupported_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("docie_bench.ocr.pdf_text.LiteParse", _FakeParser)
    with pytest.raises(ValueError, match="supports .pdf and .txt only"):
        PdfTextBackend().extract(Path("scan.png"))
