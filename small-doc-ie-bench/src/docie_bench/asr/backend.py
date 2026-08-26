from __future__ import annotations

import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from docie_bench.asr.models import (
    TranscriptionOptions,
    TranscriptionResult,
    TranscriptionSegment,
)


class ASRError(RuntimeError):
    """Base class for stable ASR failures surfaced by API and CLI adapters."""


class ASRDependencyUnavailableError(ASRError):
    pass


class ASRModelLoadError(ASRError):
    pass


class ASRTranscriptionError(ASRError):
    pass


class FasterWhisperBackend:
    """Lazy ``faster-whisper`` adapter.

    Importing docie_bench never imports CTranslate2 or downloads weights. The
    first real transcription loads the configured model, and the process-wide
    backend cache then reuses it for subsequent calls.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        num_workers: int = 1,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        self._model_name = model
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def backend_name(self) -> str:
        return "faster-whisper"

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ASRDependencyUnavailableError(
                    "The faster-whisper backend is unavailable. Install it with "
                    "`pip install -e '.[asr]'`."
                ) from exc
            try:
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    num_workers=self.num_workers,
                )
            except Exception as exc:
                raise ASRModelLoadError(
                    f"Could not load ASR model {self.model_name!r}: {exc}"
                ) from exc
            return self._model

    def load(self) -> None:
        """Load model weights now so runtime readiness is truthful."""

        self._load_model()

    def transcribe(
        self,
        audio_path: Path,
        options: TranscriptionOptions,
    ) -> TranscriptionResult:
        model = self._load_model()
        started = time.perf_counter()
        try:
            raw_segments, info = model.transcribe(
                str(audio_path),
                language=options.language,
                initial_prompt=options.prompt,
                temperature=options.temperature,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
            )
            segments = tuple(
                TranscriptionSegment(
                    id=int(segment.id),
                    seek=int(segment.seek),
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text),
                    tokens=tuple(int(token) for token in segment.tokens),
                    temperature=float(segment.temperature),
                    avg_logprob=_optional_float(segment.avg_logprob),
                    compression_ratio=_optional_float(segment.compression_ratio),
                    no_speech_prob=_optional_float(segment.no_speech_prob),
                )
                for segment in raw_segments
            )
        except ASRError:
            raise
        except Exception as exc:
            raise ASRTranscriptionError(f"Audio transcription failed: {exc}") from exc
        elapsed = time.perf_counter() - started
        text = "".join(segment.text for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=str(info.language) if getattr(info, "language", None) else None,
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            segments=segments,
            processing_seconds=elapsed,
            model=self.model_name,
            backend=self.backend_name,
        )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


@lru_cache(maxsize=8)
def get_backend(
    *,
    model: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
    num_workers: int,
    beam_size: int,
    vad_filter: bool,
) -> FasterWhisperBackend:
    """Return a reusable configured backend without loading model weights yet."""

    return FasterWhisperBackend(
        model=model,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
