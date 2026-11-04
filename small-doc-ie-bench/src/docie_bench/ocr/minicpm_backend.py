"""OCR through a served MiniCPM-OCR-1.0 deployment.

Unlike the single-instruction models, this one documents FIVE recognition
tasks, each selected purely by the prompt — plain text, HTML tables, LaTeX
formulas, seals, and charts rendered as tables. One deployment answers all of
them, so the mode is configuration rather than a separate backend.

Two of those modes return one markup document rather than lines of text, which
is why the base class asks whether a mode emits markup: splitting an HTML table
on newlines yields fragments that are not valid HTML and cannot be grounded.
"""

from __future__ import annotations

from docie_bench.ocr.served_vlm import ServedVLMBackend

#: The documented prompt for each task, verbatim from the model card. The
#: trailing colon is part of the convention, not formatting.
MODE_INSTRUCTIONS: dict[str, str] = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "seal": "Seal Recognition:",
    "chart": "Chart to Table:",
}

#: Modes whose answer is one markup document, not a transcript.
MARKUP_MODES = frozenset({"table", "formula", "chart"})

DEFAULT_MODE = "text"


class MiniCPMOCRBackend(ServedVLMBackend):
    """Transcribe a document with a served MiniCPM-OCR-1.0 deployment."""

    name = "minicpm"
    source = "minicpm"
    model_env_var = "DOCIE_OCR_MINICPM_MODEL"

    def __init__(self, *, mode: str | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._mode = mode

    def resolved_mode(self) -> str:
        if self._mode:
            return _valid(self._mode)
        from docie_bench.settings import get_settings

        return _valid(get_settings().ocr_minicpm_mode)

    def configured_model(self) -> str:
        from docie_bench.settings import get_settings

        return get_settings().ocr_minicpm_model

    def instruction(self) -> str:
        return MODE_INSTRUCTIONS[self.resolved_mode()]

    def emits_markup(self) -> bool:
        return self.resolved_mode() in MARKUP_MODES

    def version(self) -> str:
        # The mode changes the output as much as the model does, so it belongs
        # in the cache key rather than only in the configuration.
        return f"1:{self.resolved_model()}:{self.resolved_mode()}"

    def configuration(self) -> dict[str, object]:
        return {**super().configuration(), "mode": self.resolved_mode()}


def _valid(mode: str) -> str:
    """The mode, or a refusal naming the ones that exist.

    An unknown mode would otherwise reach the model as a missing key, and a
    KeyError at transcription time says nothing useful about a setting typo.
    """
    normalized = (mode or "").strip().lower() or DEFAULT_MODE
    if normalized not in MODE_INSTRUCTIONS:
        raise ValueError(
            f"unknown MiniCPM OCR mode {mode!r}; expected one of "
            f"{', '.join(sorted(MODE_INSTRUCTIONS))}"
        )
    return normalized
