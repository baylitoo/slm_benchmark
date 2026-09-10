from docie_bench.ocr.layout import merge_lines, reading_order
from docie_bench.schemas.common import BoundingBox, OCRBlock


def _block(text: str, x0: float, y0: float, x1: float, y1: float, page: int = 1) -> OCRBlock:
    return OCRBlock(
        id=text, text=text, page=page, bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
    )


def test_two_columns_under_a_header_read_column_by_column() -> None:
    blocks = [_block("Header", 100, 20, 400, 32)]
    for row in range(6):
        y = 60 + 12 * row
        blocks.append(_block(f"L{row}", 30, y, 280, y + 10))
        blocks.append(_block(f"R{row}", 320, y, 560, y + 10))
    expected = ["Header"] + [f"L{i}" for i in range(6)] + [f"R{i}" for i in range(6)]
    assert [b.text for b in reading_order(blocks)] == expected


def test_sparse_label_column_stays_in_row_order() -> None:
    blocks: list[OCRBlock] = []
    expected: list[str] = []
    for index, label in enumerate(["Skills", "Tools", "Cloud", "Data", "Web", "Ops"]):
        top = 20 + 50 * index
        blocks.append(_block(label, 30, top, 120, top + 10))
        expected.append(label)
        for line in range(3):
            text = f"{label} values {line}"
            blocks.append(_block(text, 200, top + 12 * line, 560, top + 12 * line + 10))
            expected.append(text)
    assert [b.text for b in reading_order(blocks)] == expected


def test_words_of_a_justified_line_merge_but_columns_do_not() -> None:
    blocks = [
        _block("Contributions", 30, 60, 90, 70),
        _block("diverses", 96, 60, 130, 70),
        _block("allant", 136, 60, 160, 70),
        _block("R1", 320, 60, 560, 70),
    ]
    merged = merge_lines(blocks)
    assert [b.text for b in merged] == ["Contributions diverses allant", "R1"]
    assert merged[0].bbox is not None
    assert merged[0].bbox.x1 == 160


def test_label_and_value_on_a_tab_stop_stay_separate_blocks() -> None:
    blocks: list[OCRBlock] = []
    for row in range(4):
        y = 20 + 14 * row
        blocks.append(_block(f"Label {row}", 30, y, 150, y + 8))
        blocks.append(_block(f"value {row} a", 162, y, 300, y + 8))
    merged = merge_lines(blocks)
    assert [b.text for b in merged] == [
        t for row in range(4) for t in (f"Label {row}", f"value {row} a")
    ]
