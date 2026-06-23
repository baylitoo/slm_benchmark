"""Solution adapters — make non-LLM solutions answer `/v1/chat/completions`.

The gateway proxies `kind: passthrough` profiles to an upstream runtime. Other
kinds are served *here* by a local adapter that consumes the same OpenAI chat
request and returns an OpenAI chat-completion dict — so the benchmark can score
an OCR engine (or, later, an OCR→LLM pipeline) exactly like any model, through
the one unified endpoint.

Today: the `ocr` kind, reusing `docie_bench.ocr` backends (tesseract / paddleocr
/ pdf_text). The document arrives as an inline `image_url` data URI in the
request messages (the same shape the vision path already sends). `pipeline` is
reserved for the next adapter.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.ocr.factory import get_ocr_backend

_DATA_URI_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}


class SolutionError(Exception):
    """An adapter could not produce a completion (mapped to an OpenAI error)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


class Solution(Protocol):
    async def complete(self, request: dict[str, Any]) -> dict[str, Any]: ...


def build_solution(profile: ModelProfile) -> Solution:
    """Construct the adapter for a non-passthrough profile."""
    if profile.kind == "ocr":
        return OcrSolution(profile)
    if profile.kind == "pipeline":
        raise SolutionError(
            "the 'pipeline' adapter is not implemented yet",
            status_code=501,
            error_type="not_implemented",
        )
    raise SolutionError(
        f"no solution adapter for kind {profile.kind!r}",
        status_code=500,
        error_type="unsupported_kind",
    )


class OcrSolution:
    """Run a document image through an OCR backend; return its text as a completion.

    `options`: ``backend`` (tesseract | paddleocr | pdf_text, default tesseract)
    and ``language`` (backend-specific, e.g. 'en').
    """

    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile
        self.backend_name = str(profile.options.get("backend", "tesseract"))
        self.language = profile.options.get("language")
        # Fail fast on an unknown backend name at construction, not mid-request.
        get_ocr_backend(self.backend_name, language=self.language)

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        raw, suffix = _extract_document(request)
        text = await asyncio.to_thread(self._run_ocr, raw, suffix)
        return _chat_completion(self.profile.name, text)

    def _run_ocr(self, raw: bytes, suffix: str) -> str:
        backend = get_ocr_backend(self.backend_name, language=self.language)
        with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(raw)
            path = Path(handle.name)
        try:
            blocks = backend.extract(path)
        finally:
            path.unlink(missing_ok=True)
        return "\n".join(block.text for block in blocks)


def _extract_document(request: dict[str, Any]) -> tuple[bytes, str]:
    """Pull the inline document (image_url data URI) from the chat request."""
    for message in reversed(request.get("messages") or []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                return _decode_data_uri(url)
    raise SolutionError(
        "an OCR solution needs an inline image_url (data URI) in the request messages"
    )


def _decode_data_uri(url: str) -> tuple[bytes, str]:
    if not url.startswith("data:"):
        raise SolutionError("the OCR adapter only accepts inline 'data:' image_url payloads")
    header, _, encoded = url.partition(",")
    if ";base64" not in header:
        raise SolutionError("image_url data URI must be base64-encoded")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SolutionError(f"invalid base64 image data: {exc}") from exc
    mime = header[len("data:") :].split(";", 1)[0].strip().lower()
    return raw, _DATA_URI_SUFFIX.get(mime, ".png")


def _chat_completion(model: str, content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-solution",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
