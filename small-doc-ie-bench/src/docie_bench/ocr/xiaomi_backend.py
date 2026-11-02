"""OCR through a served Xiaomi-OCR-0 deployment.

The first OCR backend that is a **model this fleet serves** rather than a
library this process imports. That difference is the whole module:

* Availability is not an import check. ``pytesseract`` is either installed or
  not; a deployment is reachable, unreachable, or not deployed at all, and only
  a resolution attempt can tell which.
* The work happens on whichever node holds a replica, so it inherits the
  routing, load balancing and failover the placement resolver already does —
  and the document crosses the network to get there.
* The request is unconstrained. The same deployment answers schema-driven
  extraction through the ``xiaomi_ocr`` family, which sends a grammar; this
  path must not, because a JSON grammar would corrupt a transcript. One
  deployment, two callers, two request shapes.

Multi-page documents are rasterized and transcribed page by page, so block
pages are real page numbers rather than everything collapsed onto page 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from docie_bench.ocr.base import OCRBackend, stable_block_id
from docie_bench.schemas.common import OCRBlock

#: The documented free-text instruction. Transcription and key extraction are
#: the same weights differing only in the prompt, so this wording is as much a
#: part of the contract as the schema skeleton is on the extraction path.
TRANSCRIBE_INSTRUCTION = "Convert the image to markdown"


class XiaomiOCRBackend(OCRBackend):
    """Transcribe a document with a served Xiaomi-OCR-0 deployment."""

    name = "xiaomi"

    def __init__(self, *, model: str | None = None, dpi: int | None = None) -> None:
        self._model = model
        self._dpi = dpi

    def _resolved_model(self) -> str:
        if self._model:
            return self._model
        from docie_bench.settings import get_settings

        return get_settings().ocr_xiaomi_model

    def version(self) -> str:
        return f"1:{self._resolved_model()}"

    def configuration(self) -> dict[str, Any]:
        """Cache-relevant configuration.

        The model name belongs here because two deployments of different
        builds produce different transcripts; the endpoint does not, since the
        resolver may legitimately route the same model to another replica
        between runs without the output changing.
        """
        return {"model": self._resolved_model(), "instruction": TRANSCRIBE_INSTRUCTION}

    def _profile(self) -> Any:
        from docie_bench.serving.profile_resolver import (
            ProfileResolutionError,
            resolve_extraction_profile,
        )

        model = self._resolved_model()
        try:
            return resolve_extraction_profile(model_profile=model)
        except ProfileResolutionError as exc:
            raise RuntimeError(
                f"the 'xiaomi' OCR backend needs a live deployment of {model!r}; "
                f"deploy it or set DOCIE_OCR_XIAOMI_MODEL to one that is live ({exc})"
            ) from exc

    def extract(self, path: Path) -> list[OCRBlock]:
        from docie_bench.vision import load_document_images

        profile = self._profile()
        # The profile's own vision knobs, not the generic ocr_dpi: this backend
        # IS a vision model, so it should rasterize the way that deployment's
        # other callers do.
        dpi = self._dpi if self._dpi is not None else profile.vision_pdf_dpi
        images = load_document_images(path, pdf_dpi=dpi)
        if not images:
            return []
        # Each page is its own sequential request to a CPU-bound VLM, so an
        # uncapped document is an unbounded transcription. Same ceiling the
        # extraction path applies to the same kind of model.
        images = images[: max(profile.vision_max_pages, 1)]

        blocks: list[OCRBlock] = []
        with httpx.Client(timeout=profile.timeout_seconds) as client:
            for image in images:
                text = self._transcribe(client, profile, image.data_url())
                blocks.extend(_text_to_page_blocks(text, page=image.page))
        return blocks

    def _transcribe(self, client: httpx.Client, profile: Any, data_url: str) -> str:
        response = client.post(
            f"{profile.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {profile.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": profile.model,
                "temperature": 0.0,
                "max_tokens": profile.max_tokens,
                # No response_format: a grammar would corrupt a transcript.
                # Thinking is never wanted for verbatim transcription, and is
                # sent unconditionally for the same reason the extraction path
                # does -- a template that ignores the flag is harmless, one
                # that honours it saves the whole budget.
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": TRANSCRIBE_INSTRUCTION},
                        ],
                    }
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, list):  # multimodal parts -> concatenate text
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content or "")


def _text_to_page_blocks(text: str, *, page: int) -> list[OCRBlock]:
    """One block per non-empty line, stamped with its real page.

    ``base.text_to_blocks`` hardcodes page 1, which is right for the
    single-image backends and wrong here: a transcript that loses its page
    numbers cannot be grounded back to the document.
    """
    blocks: list[OCRBlock] = []
    lines = (raw.strip() for raw in text.splitlines())
    for index, line in enumerate(lines):
        if not line:
            continue
        blocks.append(
            OCRBlock(
                id=stable_block_id(page, index, line),
                text=line,
                page=page,
                source="xiaomi",
            )
        )
    if not blocks and text.strip():
        blocks.append(
            OCRBlock(
                id=stable_block_id(page, 0, text.strip()),
                text=text.strip(),
                page=page,
                source="xiaomi",
            )
        )
    return blocks
