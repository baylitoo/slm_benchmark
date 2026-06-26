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
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

import httpx

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


def build_solution(
    profile: ModelProfile,
    *,
    profiles: Mapping[str, ModelProfile] | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Solution:
    """Construct the adapter for a non-passthrough profile.

    `profiles`/`http_client` are supplied by the gateway and used by adapters that
    delegate to another profile (e.g. the OCR→LLM pipeline).
    """
    if profile.kind == "ocr":
        return OcrSolution(profile)
    if profile.kind == "pipeline":
        return PipelineSolution(profile, profiles=profiles or {}, http_client=http_client)
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
        text = await asyncio.to_thread(_ocr_to_text, self.backend_name, self.language, raw, suffix)
        return _chat_completion(self.profile.name, text)


class PipelineSolution:
    """OCR→LLM: OCR the document image, then extract with a passthrough LLM profile.

    `options`: ``ocr_backend`` (default tesseract), ``language``, and ``extractor``
    (the name of a passthrough LLM profile that performs the structured extraction).
    The document image in the request is replaced by its OCR text before the
    extractor is called, so the LLM does the field extraction over real text.
    """

    def __init__(
        self,
        profile: ModelProfile,
        *,
        profiles: Mapping[str, ModelProfile],
        http_client: httpx.AsyncClient | None,
    ) -> None:
        self.profile = profile
        self.backend_name = str(profile.options.get("ocr_backend", "tesseract"))
        self.language = profile.options.get("language")
        get_ocr_backend(self.backend_name, language=self.language)  # fail fast

        extractor_name = profile.options.get("extractor")
        if not extractor_name:
            raise SolutionError(
                f"pipeline profile {profile.name!r} requires options.extractor "
                "(the name of a passthrough LLM profile)",
                status_code=500,
                error_type="invalid_profile",
            )
        extractor = profiles.get(str(extractor_name))
        if extractor is None:
            raise SolutionError(
                f"pipeline extractor profile {extractor_name!r} is not configured",
                status_code=500,
                error_type="invalid_profile",
            )
        if extractor.kind != "passthrough":
            raise SolutionError(
                f"pipeline extractor {extractor_name!r} must be a passthrough LLM profile",
                status_code=500,
                error_type="invalid_profile",
            )
        if http_client is None:
            raise SolutionError(
                "pipeline adapter requires an HTTP client",
                status_code=500,
                error_type="invalid_profile",
            )
        self.extractor = extractor
        self.http = http_client

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        raw, suffix = _extract_document(request)
        text = await asyncio.to_thread(_ocr_to_text, self.backend_name, self.language, raw, suffix)
        # Hand the extractor the original prompt with the image swapped for OCR text.
        llm_request: dict[str, Any] = {
            **request,
            "model": self.extractor.model,
            "messages": _inject_ocr_text(request.get("messages") or [], text),
        }
        llm_request.pop("stream", None)  # the gateway re-streams the final completion
        url = f"{self.extractor.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.extractor.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = await self.http.post(
                url, json=llm_request, headers=headers, timeout=self.extractor.timeout_seconds
            )
        except httpx.RequestError as exc:
            raise SolutionError(
                f"pipeline extractor upstream is unreachable: {exc}",
                status_code=502,
                error_type="upstream_unavailable",
            ) from exc
        if resp.status_code >= 400:
            raise SolutionError(
                f"pipeline extractor returned {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
                error_type="upstream_error",
            )
        return resp.json()


def _ocr_to_text(backend_name: str, language: object, raw: bytes, suffix: str) -> str:
    backend = get_ocr_backend(backend_name, language=language)  # type: ignore[arg-type]
    with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(raw)
        path = Path(handle.name)
    try:
        blocks = backend.extract(path)
    finally:
        path.unlink(missing_ok=True)
    return "\n".join(block.text for block in blocks)


def _inject_ocr_text(messages: list[dict[str, Any]], ocr_text: str) -> list[dict[str, Any]]:
    """Return messages with every image_url part replaced by the OCR text part."""
    rewritten: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rewritten.append(message)
            continue
        parts = [
            {"type": "text", "text": ocr_text}
            if isinstance(part, dict) and part.get("type") == "image_url"
            else part
            for part in content
        ]
        rewritten.append({**message, "content": parts})
    return rewritten


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
