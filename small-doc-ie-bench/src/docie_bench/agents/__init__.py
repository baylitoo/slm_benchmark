"""Preconfigured agents served over OpenAI-compatible endpoints.

An *agent* is a named, persisted configuration that fronts one served SLM (or
an OCR solution) with agent-specific pre/post processing, and is exposed as an
OpenAI-compatible model so any agents platform can consume it by pointing an
OpenAI client at ``/v1/agents`` (or ``/v1/agents/{name}``).

Shipped kinds:

* ``proxy_security`` — a security proxy: detects PII (emails, phones, IBANs,
  cards, national ids, IPs) in the incoming messages and anonymizes it with
  stable placeholders before the backing model ever sees it. A future encoder
  family (``options.guard_model``) can replace the regex analyzer.
* ``ocr`` — instantiates an OCR solution (tesseract / paddleocr / pdf_text),
  optionally piped into a structured extractor profile (e.g. NuExtract).
* ``custom`` — bring-your-own: a system prompt over any served model.
"""

from docie_bench.agents.spec import AgentSpec
from docie_bench.agents.registry import (
    AgentConflictError,
    AgentNotFoundError,
    AgentRegistry,
)

__all__ = ["AgentConflictError", "AgentNotFoundError", "AgentRegistry", "AgentSpec"]
