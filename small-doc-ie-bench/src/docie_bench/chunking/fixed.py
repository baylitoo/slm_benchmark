from __future__ import annotations

from docie_bench.chunking.base import Chunk, Chunker, normalize_newlines


class FixedSizeChunker(Chunker):
    """Windows of ``size`` characters, each sharing ``overlap`` characters with the previous."""

    name = "fixed"

    def __init__(self, size: int = 2000, overlap: int = 200) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= overlap < size:
            raise ValueError("overlap must be in [0, size)")
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        return self.chunk_span(normalize_newlines(text), 0)

    def chunk_span(
        self,
        text: str,
        offset: int,
        section_path: list[str] | None = None,
        level: int | None = None,
    ) -> list[Chunk]:
        """Chunk ``text`` that starts at ``offset`` in a larger document."""
        chunks: list[Chunk] = []
        step = self.size - self.overlap
        start = 0
        while start < len(text):
            end = min(start + self.size, len(text))
            chunks.append(
                Chunk(
                    text=text[start:end],
                    index=len(chunks),
                    start=offset + start,
                    end=offset + end,
                    section_path=list(section_path or []),
                    level=level,
                )
            )
            if end == len(text):
                break
            start += step
        return chunks
