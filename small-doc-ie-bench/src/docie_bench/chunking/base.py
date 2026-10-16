from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One piece of a document, located by character offsets into the chunked text."""

    text: str
    index: int
    start: int
    end: int
    section_path: list[str] = Field(default_factory=list)
    level: int | None = None


class Chunker(ABC):
    name: str

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        raise NotImplementedError


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
