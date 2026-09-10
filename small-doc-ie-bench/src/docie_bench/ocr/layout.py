"""Reading order for spatial text blocks: recursive XY-cut on bounding boxes.

Multi-column pages come out of PDF text extraction interleaved line by line
across columns. Cutting the page along its widest empty band (a column
gutter or a horizontal gap), recursively, then sorting each leaf region into
lines, yields column-by-column reading order for any layout with real
whitespace between regions. A vertical cut is only taken between balanced
regions: a sparse label column next to a dense value column (a skills table,
a dates gutter) is a row layout and stays in line order. Regions without a
usable gap fall back to top-to-bottom, left-to-right lines.
"""

from __future__ import annotations

from dataclasses import dataclass

from docie_bench.schemas.common import BoundingBox, OCRBlock

MIN_GAP_POINTS = 6.0
MIN_COLUMN_SHARE = 0.25
MIN_COLUMN_BLOCKS = 6
MAX_COLUMN_WIDTH_RATIO = 2.5
MERGE_GAP_FACTOR = 3.0
COLUMN_START_MIN_ITEMS = 3


@dataclass(frozen=True)
class _Gap:
    size: float
    position: float


def reading_order(blocks: list[OCRBlock], *, min_gap: float = MIN_GAP_POINTS) -> list[OCRBlock]:
    pages: dict[int, list[OCRBlock]] = {}
    for block in blocks:
        pages.setdefault(block.page, []).append(block)
    ordered: list[OCRBlock] = []
    for page in sorted(pages):
        items = pages[page]
        if any(block.bbox is None for block in items):
            ordered.extend(items)
        else:
            ordered.extend(_xy_cut(items, min_gap))
    return ordered


def merge_lines(blocks: list[OCRBlock]) -> list[OCRBlock]:
    """Join consecutive blocks that sit on the same line with a word-sized gap
    (justified text split into words by the extractor) into one block. A block
    that starts on one of the page's column starts (an x where several items
    begin: the value column of a label/value table) is never merged into the
    block on its left unless the two are practically touching."""
    starts_by_page = _column_starts(blocks)
    merged: list[OCRBlock] = []
    for block in blocks:
        previous = merged[-1] if merged else None
        if previous is not None and _same_line_neighbours(
            previous, block, starts_by_page.get(block.page, set())
        ):
            assert previous.bbox is not None
            assert block.bbox is not None
            merged[-1] = previous.model_copy(
                update={
                    "text": f"{previous.text} {block.text}",
                    "bbox": BoundingBox(
                        x0=min(previous.bbox.x0, block.bbox.x0),
                        y0=min(previous.bbox.y0, block.bbox.y0),
                        x1=max(previous.bbox.x1, block.bbox.x1),
                        y1=max(previous.bbox.y1, block.bbox.y1),
                    ),
                }
            )
        else:
            merged.append(block)
    return merged


def _column_starts(blocks: list[OCRBlock]) -> dict[int, set[int]]:
    counts: dict[int, dict[int, int]] = {}
    for block in blocks:
        if block.bbox is None:
            continue
        page_counts = counts.setdefault(block.page, {})
        x = round(block.bbox.x0)
        page_counts[x] = page_counts.get(x, 0) + 1
    return {
        page: {x for x, n in page_counts.items() if n >= COLUMN_START_MIN_ITEMS}
        for page, page_counts in counts.items()
    }


def _same_line_neighbours(left: OCRBlock, right: OCRBlock, column_starts: set[int]) -> bool:
    if left.page != right.page or left.bbox is None or right.bbox is None:
        return False
    height = max(left.bbox.y1 - left.bbox.y0, right.bbox.y1 - right.bbox.y0, 1.0)
    same_line = abs(left.bbox.y0 - right.bbox.y0) <= 0.5 * height
    gap = right.bbox.x0 - left.bbox.x1
    limit = height if round(right.bbox.x0) in column_starts else MERGE_GAP_FACTOR * height
    return same_line and -2.0 <= gap <= limit


def _xy_cut(items: list[OCRBlock], min_gap: float) -> list[OCRBlock]:
    if len(items) <= 2:
        return _sort_lines(items)
    # Columns first: a full-height gutter between two balanced, similarly
    # wide, multi-line regions is the primary structure of the page. A
    # header spanning the columns hides the gutter, so it is split off by
    # the horizontal cut below and the columns are found one level down.
    gap_x = _largest_gap(items, axis="x", min_gap=min_gap, min_share=MIN_COLUMN_SHARE)
    if gap_x is not None:
        left = [b for b in items if b.bbox.x1 <= gap_x.position]  # type: ignore[union-attr]
        right = [b for b in items if b.bbox.x1 > gap_x.position]  # type: ignore[union-attr]
        if _column_like(left, right):
            return _xy_cut(left, min_gap) + _xy_cut(right, min_gap)
    gap_y = _largest_gap(items, axis="y", min_gap=min_gap)
    if gap_y is None:
        return _sort_lines(items)
    first = [b for b in items if b.bbox.y1 <= gap_y.position]  # type: ignore[union-attr]
    second = [b for b in items if b.bbox.y1 > gap_y.position]  # type: ignore[union-attr]
    if not first or not second:
        return _sort_lines(items)
    return _xy_cut(first, min_gap) + _xy_cut(second, min_gap)


def _column_like(left: list[OCRBlock], right: list[OCRBlock]) -> bool:
    """Two text columns, not a key/value or line-item table: both sides are
    several lines tall and of comparable width."""
    if min(len(left), len(right)) < MIN_COLUMN_BLOCKS:
        return False
    width_left = max(b.bbox.x1 for b in left) - min(b.bbox.x0 for b in left)  # type: ignore[union-attr]
    width_right = max(b.bbox.x1 for b in right) - min(b.bbox.x0 for b in right)  # type: ignore[union-attr]
    narrow, wide = sorted((max(width_left, 1.0), max(width_right, 1.0)))
    return wide / narrow <= MAX_COLUMN_WIDTH_RATIO


def _largest_gap(
    items: list[OCRBlock], *, axis: str, min_gap: float, min_share: float = 0.0
) -> _Gap | None:
    """Widest empty band across ``axis``. With ``min_share``, only bands that
    leave at least that share of the blocks on each side qualify, so a page
    margin or a sparse label column never wins over a real gutter."""
    if axis == "x":
        spans = sorted((b.bbox.x0, b.bbox.x1) for b in items)  # type: ignore[union-attr]
    else:
        spans = sorted((b.bbox.y0, b.bbox.y1) for b in items)  # type: ignore[union-attr]
    best: _Gap | None = None
    reach = spans[0][1]
    for index, (lo, hi) in enumerate(spans[1:], start=1):
        gap = lo - reach
        if gap >= min_gap and (best is None or gap > best.size):
            share = min(index, len(spans) - index) / len(spans)
            if share >= min_share:
                best = _Gap(size=gap, position=(reach + lo) / 2)
        reach = max(reach, hi)
    return best


def _sort_lines(items: list[OCRBlock]) -> list[OCRBlock]:
    if not items:
        return items
    by_top = sorted(items, key=lambda b: (b.bbox.y0, b.bbox.x0))  # type: ignore[union-attr]
    heights = sorted(b.bbox.y1 - b.bbox.y0 for b in by_top)  # type: ignore[union-attr]
    tolerance = max(2.0, 0.5 * heights[len(heights) // 2])
    lines: list[list[OCRBlock]] = []
    for block in by_top:
        if lines and abs(block.bbox.y0 - lines[-1][0].bbox.y0) <= tolerance:  # type: ignore[union-attr]
            lines[-1].append(block)
        else:
            lines.append([block])
    ordered: list[OCRBlock] = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda b: b.bbox.x0))  # type: ignore[union-attr]
    return ordered
