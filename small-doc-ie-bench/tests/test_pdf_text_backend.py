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
        "layout": "xy-cut-v1",
        "dpi": 200,
        "ocr_server_url": "http://vlm/ocr",
        "language": "fr",
        "ocr_engine": "tesseract",
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


class _FakeParser221(_FakeParser):
    """LiteParse 2.21+: the engine is a named argument."""

    def __init__(self, *, ocr_backend: str | None = None, **kwargs) -> None:
        super().__init__(ocr_backend=ocr_backend, **kwargs)


def test_liteparse_names_tesseract_so_a_pulled_ocria_cannot_take_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("docie_bench.ocr.pdf_text.LiteParse", _FakeParser221)

    PdfTextBackend(ocr_server_url="").extract(Path("scan.pdf"))

    assert _FakeParser221.last_init["ocr_backend"] == "tesseract"


def test_ocria_nano_engine_pulls_its_models_once_and_keys_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from docie_bench.ocr.factory import get_ocr_backend

    pulled: list[str] = []
    installed: list[object] = []

    def pull(spec: str, **kwargs: object) -> str:
        pulled.append(spec)
        installed.append(SimpleNamespace(name="ocria-nano"))
        return "/models/ocria-nano"

    fake_models = SimpleNamespace(pull=pull, list=lambda: list(installed))
    monkeypatch.setitem(sys.modules, "liteparse.models", fake_models)
    monkeypatch.setattr("liteparse.models", fake_models, raising=False)
    monkeypatch.setattr("docie_bench.ocr.pdf_text.LiteParse", _FakeParser221)
    monkeypatch.setattr("docie_bench.ocr.pdf_text.get_settings", lambda: SimpleNamespace(
        ocr_language="fra", ocr_server_url=None, ocr_dpi=200
    ))

    backend = get_ocr_backend("ocria-nano")
    blocks = backend.extract(Path("scan.pdf"))
    backend.extract(Path("scan.pdf"))

    assert _FakeParser221.last_init["ocr_backend"] == "ocria-nano"
    assert pulled == ["ocria-nano"]
    assert blocks[0].bbox is not None  # evidence ids keep their boxes
    assert backend.configuration()["ocr_engine"] == "ocria-nano"
    assert get_ocr_backend("liteparse").configuration() != backend.configuration()


def test_ocria_nano_on_an_old_liteparse_says_what_to_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("docie_bench.ocr.pdf_text.LiteParse", _FakeParser)
    backend = PdfTextBackend(ocr_engine="ocria-nano", ocr_server_url="")

    with pytest.raises(RuntimeError, match="liteparse>=2.21.0"):
        backend.extract(Path("scan.pdf"))
