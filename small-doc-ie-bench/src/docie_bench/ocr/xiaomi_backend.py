"""OCR through a served Xiaomi-OCR-0 deployment.

One instruction, one mode. The shared mechanics -- resolving a deployment,
rasterizing, capping pages, sending an ungrammared request, stamping real page
numbers -- live in :mod:`docie_bench.ocr.served_vlm`; what is specific to this
model is the wording below and which deployment it goes to.

The same deployment also answers schema-driven extraction through the
``xiaomi_ocr`` family. That path sends a grammar and this one must not, which
is why the two are separate callers rather than one contract that switches.
"""

from __future__ import annotations

from docie_bench.ocr.served_vlm import ServedVLMBackend

#: The documented free-text instruction. Transcription and key extraction are
#: the same weights differing only in the prompt, so this wording is as much a
#: part of the contract as the schema skeleton is on the extraction path.
TRANSCRIBE_INSTRUCTION = "Convert the image to markdown"


class XiaomiOCRBackend(ServedVLMBackend):
    """Transcribe a document with a served Xiaomi-OCR-0 deployment."""

    name = "xiaomi"
    source = "xiaomi"
    model_env_var = "DOCIE_OCR_XIAOMI_MODEL"

    def configured_model(self) -> str:
        from docie_bench.settings import get_settings

        return get_settings().ocr_xiaomi_model

    def instruction(self) -> str:
        return TRANSCRIBE_INSTRUCTION
