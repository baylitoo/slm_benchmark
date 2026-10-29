from __future__ import annotations

from collections.abc import Iterator
from itertools import accumulate

from markdown_it import MarkdownIt
from pydantic import BaseModel, Field

from docie_bench.chunking.base import Chunk, Chunker, normalize_newlines
from docie_bench.chunking.fixed import FixedSizeChunker

_MD = MarkdownIt("commonmark").enable("table")


class Section(BaseModel):
    """A markdown heading and everything under it, up to the next heading at its level or above.

    ``level`` is the heading's number of ``#`` (0 for the root) and is kept as
    written: a ``###`` directly under a ``#`` stays level 3, without a synthetic
    level-2 parent. ``start``/``end`` are character offsets of the whole subtree,
    heading included; ``body_start`` is where the text after the heading begins.
    """

    level: int
    title: str | None
    content: str
    start: int
    body_start: int
    end: int
    children: list[Section] = Field(default_factory=list)

    @property
    def own_end(self) -> int:
        """End of this section's own text, before its first child."""
        return self.children[0].start if self.children else self.end

    def walk(self) -> Iterator[Section]:
        yield self
        for child in self.children:
            yield from child.walk()


def parse_sections(markdown: str) -> Section:
    """Build the heading tree of ``markdown``; text before the first heading belongs to the root.

    Only top-level ATX/setext headings open a section: a ``#`` inside a code
    fence, an HTML block, a list or a blockquote is content.
    """
    text = normalize_newlines(markdown)
    lines = text.splitlines(keepends=True)
    offsets = [0, *accumulate(len(line) for line in lines)]
    tokens = _MD.parse(text)
    headings = [
        (tok.map[0], tok.map[1], int(tok.tag[1:]), tokens[i + 1].content.strip())
        for i, tok in enumerate(tokens)
        if tok.type == "heading_open" and tok.level == 0 and tok.map is not None
    ]

    def span(first: int, last: int) -> str:
        return text[offsets[first] : offsets[last]].strip()

    first_heading = headings[0][0] if headings else len(lines)
    root = Section(
        level=0,
        title=None,
        content=span(0, first_heading),
        start=0,
        body_start=0,
        end=len(text),
    )
    stack = [root]
    for k, (head_start, head_end, level, title) in enumerate(headings):
        next_start = headings[k + 1][0] if k + 1 < len(headings) else len(lines)
        while stack[-1].level >= level:
            stack.pop().end = offsets[head_start]
        section = Section(
            level=level,
            title=title,
            content=span(head_end, next_start),
            start=offsets[head_start],
            body_start=offsets[head_end],
            end=len(text),
        )
        stack[-1].children.append(section)
        stack.append(section)
    return root


class HierarchicalChunker(Chunker):
    """One chunk per section subtree that fits in ``max_chars``.

    A subtree too large is opened: its own text becomes a chunk and each child
    is chunked the same way. Own text still over ``max_chars`` (a long section
    with no subheadings) falls back to fixed windows with ``overlap``.
    """

    name = "hierarchical"

    def __init__(self, max_chars: int = 4000, overlap: int = 200) -> None:
        self.max_chars = max_chars
        self.fallback = FixedSizeChunker(size=max_chars, overlap=overlap)

    def chunk(self, text: str) -> list[Chunk]:
        text = normalize_newlines(text)
        chunks: list[Chunk] = []
        self._emit(parse_sections(text), text, [], chunks)
        for index, chunk in enumerate(chunks):
            chunk.index = index
        return chunks

    def _emit(self, section: Section, text: str, path: list[str], out: list[Chunk]) -> None:
        if section.level > 0:
            path = [*path, section.title or ""]
        if section.end - section.start <= self.max_chars:
            self._add(text, section.start, section.end, path, section.level, out)
            return
        own = text[section.start : section.own_end]
        if len(own) <= self.max_chars:
            self._add(text, section.start, section.own_end, path, section.level, out)
        else:
            out.extend(self.fallback.chunk_span(own, section.start, path, section.level))
        for child in section.children:
            self._emit(child, text, path, out)

    @staticmethod
    def _add(
        text: str, start: int, end: int, path: list[str], level: int, out: list[Chunk]
    ) -> None:
        if text[start:end].strip():
            out.append(
                Chunk(
                    text=text[start:end],
                    index=len(out),
                    start=start,
                    end=end,
                    section_path=path,
                    level=level,
                )
            )
