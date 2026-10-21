from __future__ import annotations

import pytest

from docie_bench.chunking import FixedSizeChunker, HierarchicalChunker, parse_sections

DOC = """Preamble line.

# Invoice

Seller: ACME

### Line items

```python
# not a heading
```

| item | qty |
|------|-----|
| bolt | 2   |

## Totals

Total: 10.00

# Annex

Terms.
"""


def test_tree_keeps_levels_as_written_and_ignores_hashes_in_code() -> None:
    root = parse_sections(DOC)

    assert root.level == 0
    assert root.content == "Preamble line."
    shape = [(s.level, s.title) for s in root.children]
    assert shape == [(1, "Invoice"), (1, "Annex")]
    invoice = root.children[0]
    assert invoice.content == "Seller: ACME"
    assert [(s.level, s.title) for s in invoice.children] == [(3, "Line items"), (2, "Totals")]
    assert "# not a heading" in invoice.children[0].content
    assert DOC[invoice.start : invoice.end].startswith("# Invoice")
    assert DOC[invoice.start : invoice.end].rstrip().endswith("Total: 10.00")


def test_hierarchical_opens_only_oversized_sections() -> None:
    chunks = HierarchicalChunker(max_chars=100, overlap=10).chunk(DOC)

    paths = [c.section_path for c in chunks]
    assert paths == [
        [],
        ["Invoice"],
        ["Invoice", "Line items"],
        ["Invoice", "Totals"],
        ["Annex"],
    ]
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert DOC[c.start : c.end] == c.text
        assert len(c.text) <= 100


def test_hierarchical_falls_back_to_fixed_windows_for_a_long_section() -> None:
    doc = "# Notes\n\n" + "x" * 250
    chunks = HierarchicalChunker(max_chars=100, overlap=20).chunk(doc)

    assert len(chunks) > 1
    assert all(c.section_path == ["Notes"] and len(c.text) <= 100 for c in chunks)
    assert chunks[1].start == chunks[0].end - 20


def test_fixed_windows_overlap_and_cover_the_text() -> None:
    chunks = FixedSizeChunker(size=10, overlap=3).chunk("abcdefghijklmnopqrstu")

    assert [c.text for c in chunks] == ["abcdefghij", "hijklmnopq", "opqrstu"]
    assert FixedSizeChunker(size=10, overlap=3).chunk("") == []
    with pytest.raises(ValueError, match="overlap"):
        FixedSizeChunker(size=10, overlap=10)
