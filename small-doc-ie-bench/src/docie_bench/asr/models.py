from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TranscriptionOptions:
    """Backend-neutral controls shared by the HTTP API and CLI."""

    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0


@dataclass(frozen=True)
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str
    seek: int = 0
    tokens: tuple[int, ...] = ()
    temperature: float = 0.0
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration: float
    segments: tuple[TranscriptionSegment, ...] = field(default_factory=tuple)
    processing_seconds: float = 0.0
    model: str = ""
    backend: str = ""

    @property
    def real_time_factor(self) -> float | None:
        if self.duration <= 0:
            return None
        return self.processing_seconds / self.duration


class ASRBackend(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def backend_name(self) -> str: ...

    def transcribe(
        self,
        audio_path: Path,
        options: TranscriptionOptions,
    ) -> TranscriptionResult: ...
