from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from docie_bench.ocr.factory import get_ocr_backend
from docie_bench.ocr.indra_backend import IndraOCRBackend


@dataclass
class _Line:
    box: list[float]
    text: str
    conf: float


@dataclass
class _Page:
    lines: list[_Line]
    source: str = "ocr"
    note: str = ""
    timings: dict = field(default_factory=dict)


class _Converter:
    def __init__(self, pages: list[_Page]) -> None:
        self.pages = pages

    def convert(self, path):
        yield from enumerate(self.pages)


def _install_fake(monkeypatch: pytest.MonkeyPatch, pages: list[_Page]) -> dict:
    calls: dict = {}
    module = types.ModuleType("indra_ocr")

    def load(**kwargs):
        calls.update(kwargs)
        return _Converter(pages)

    module.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "indra_ocr", module)
    return calls


def test_lines_become_blocks_with_page_bbox_and_confidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fake(
        monkeypatch,
        [
            _Page([_Line([1, 2, 30, 12], "Invoice #42", 0.97), _Line([0, 0, 1, 1], "  ", 0.1)]),
            _Page([_Line([5, 6, 50, 16], "Total 10.00", 1.0000001)]),
        ],
    )
    doc = tmp_path / "invoice.png"
    doc.write_bytes(b"\x89PNG")

    backend = get_ocr_backend("indra")
    assert isinstance(backend, IndraOCRBackend)
    blocks = backend.extract(doc)

    assert calls == {"threads": 4, "mode": "auto", "dpi": 200.0}
    assert [(b.page, b.text, b.source) for b in blocks] == [
        (1, "Invoice #42", "indra"),
        (2, "Total 10.00", "indra"),
    ]
    assert blocks[0].bbox is not None
    assert blocks[0].bbox.x1 == 30.0
    assert blocks[1].confidence == 1.0


def test_failed_page_raises_instead_of_returning_empty_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake(monkeypatch, [_Page([], source="error", note="ValueError: bad image")])
    doc = tmp_path / "scan.jpg"
    doc.write_bytes(b"\xff\xd8")

    with pytest.raises(RuntimeError, match="bad image"):
        IndraOCRBackend().extract(doc)
