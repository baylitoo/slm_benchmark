from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile

AUDIO_MIME_BY_SUFFIX = {
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}

_MIME_ALIASES = {
    "application/ogg": "audio/ogg",
    "audio/mp4a-latm": "audio/mp4",
    "audio/vnd.wave": "audio/wav",
    "audio/x-flac": "audio/flac",
    "audio/x-m4a": "audio/mp4",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "video/mp4": "audio/mp4",
    "video/webm": "audio/webm",
}

_GENERIC_MIME_TYPES = {"", "application/octet-stream", "binary/octet-stream"}


@dataclass(frozen=True)
class StoredAudioUpload:
    path: Path
    suffix: str
    mime_type: str
    size_bytes: int


def detect_audio_mime_type(header: bytes) -> str | None:
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio/wav"
    if header.startswith(b"fLaC"):
        return "audio/flac"
    if header.startswith(b"OggS"):
        return "audio/ogg"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return "audio/webm"
    if header.startswith(b"ID3") or (
        len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
    ):
        return "audio/mpeg"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return "audio/mp4"
    return None


async def store_validated_audio_upload(
    file: UploadFile,
    *,
    max_bytes: int,
    allowed_mime_types: set[str],
) -> StoredAudioUpload:
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    expected_mime = AUDIO_MIME_BY_SUFFIX.get(suffix)
    if expected_mime is None:
        raise HTTPException(status_code=415, detail=f"Unsupported audio suffix: {suffix}")
    if expected_mime not in allowed_mime_types:
        raise HTTPException(status_code=415, detail=f"Audio type is disabled: {expected_mime}")

    claimed_mime = (file.content_type or "").lower().partition(";")[0].strip()
    claimed_mime = _MIME_ALIASES.get(claimed_mime, claimed_mime)
    if claimed_mime not in _GENERIC_MIME_TYPES and claimed_mime != expected_mime:
        raise HTTPException(
            status_code=415,
            detail="Declared content type does not match the audio suffix",
        )

    path: Path | None = None
    size = 0
    header = b""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            path = Path(handle.name)
            while chunk := await file.read(min(1024 * 1024, max_bytes + 1)):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio file too large. Max {max_bytes} bytes",
                    )
                if len(header) < 64:
                    header += chunk[: 64 - len(header)]
                handle.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        detected_mime = detect_audio_mime_type(header)
        if detected_mime != expected_mime:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Audio content does not match its suffix; detected "
                    f"{detected_mime or 'unknown'}"
                ),
            )
        return StoredAudioUpload(
            path=path,
            suffix=suffix,
            mime_type=detected_mime,
            size_bytes=size,
        )
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        raise
