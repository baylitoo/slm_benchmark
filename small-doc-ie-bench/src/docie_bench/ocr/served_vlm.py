"""OCR backends that are a model this fleet serves, rather than a library.

Every other backend imports a package and runs in this process. These resolve a
deployment and send it an image, which changes three things the base class
exists to hold in one place:

* **Availability is not an import check.** A library is installed or not; a
  deployment is reachable, unreachable, or absent, and only a resolution
  attempt tells which.
* **Routing comes for free.** Resolution goes through the placement resolver,
  so the work lands on whichever node holds a live replica.
* **Pages are real pages.** ``base.text_to_blocks`` stamps everything page 1,
  which is right for a single image and wrong for a rasterized PDF.

What a subclass supplies is the instruction and which deployment to send it to.
That is the whole difference between transcribing with one of these models and
transcribing with another: the request shape is identical, and neither sends a
grammar, because a JSON grammar corrupts a transcript.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from docie_bench.ocr.base import OCRBackend, stable_block_id
from docie_bench.schemas.common import OCRBlock


class ServedVLMBackend(OCRBackend):
    """Transcribe a document through a deployed vision model."""

    #: The value that identifies this backend's block provenance. Must be in
    #: ``OCRBlock.source``'s vocabulary.
    source: str = "unknown"

    #: The environment variable that names this backend's deployment. Carried
    #: so an unresolvable model can say which setting to change -- the generic
    #: "point its model setting somewhere else" sends an operator looking.
    model_env_var: str = ""

    def __init__(self, *, model: str | None = None, dpi: int | None = None) -> None:
        self._model = model
        self._dpi = dpi

    # --- what a subclass supplies -----------------------------------------

    def configured_model(self) -> str:
        """The deployment name to transcribe through."""
        raise NotImplementedError

    def instruction(self) -> str:
        """The documented free-text instruction for this model."""
        raise NotImplementedError

    # --- shared ------------------------------------------------------------

    def resolved_model(self) -> str:
        return self._model or self.configured_model()

    def version(self) -> str:
        return f"1:{self.resolved_model()}"

    def configuration(self) -> dict[str, Any]:
        """Cache-relevant configuration.

        The model and the instruction belong here because either changing
        changes the transcript. The endpoint does not: the resolver may
        legitimately route the same model to another replica between runs
        without the output changing.
        """
        return {"model": self.resolved_model(), "instruction": self.instruction()}

    def _profile(self) -> Any:
        from docie_bench.serving.profile_resolver import (
            ProfileResolutionError,
            resolve_extraction_profile,
        )

        model = self.resolved_model()
        try:
            return resolve_extraction_profile(model_profile=model)
        except ProfileResolutionError as exc:
            setting = self.model_env_var or "its model setting"
            raise RuntimeError(
                f"the {self.name!r} OCR backend needs a live deployment of {model!r}; "
                f"deploy it or set {setting} to one that is live ({exc})"
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
        instruction = self.instruction()
        with httpx.Client(timeout=profile.timeout_seconds) as client:
            for image in images:
                text = self._transcribe(client, profile, image.data_url(), instruction)
                blocks.extend(self._page_blocks(text, page=image.page))
        return blocks

    def _transcribe(
        self, client: httpx.Client, profile: Any, data_url: str, instruction: str
    ) -> str:
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
                            {"type": "text", "text": instruction},
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

    def _page_blocks(self, text: str, *, page: int) -> list[OCRBlock]:
        """One block per non-empty line, stamped with its real page.

        Line-per-block is right for a transcript and wrong for the structured
        modes, which emit one HTML table or one LaTeX expression — those stay
        whole, because splitting markup on newlines produces fragments that are
        not valid anything. Subclasses that emit markup say so by overriding
        :meth:`emits_markup`.
        """
        if self.emits_markup():
            body = text.strip()
            if not body:
                return []
            return [
                OCRBlock(
                    id=stable_block_id(page, 0, body),
                    text=body,
                    page=page,
                    source=self.source,  # type: ignore[arg-type]
                )
            ]
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
                    source=self.source,  # type: ignore[arg-type]
                )
            )
        if not blocks and text.strip():
            blocks.append(
                OCRBlock(
                    id=stable_block_id(page, 0, text.strip()),
                    text=text.strip(),
                    page=page,
                    source=self.source,  # type: ignore[arg-type]
                )
            )
        return blocks

    def emits_markup(self) -> bool:
        """Whether this backend's current mode returns one markup document."""
        return False
