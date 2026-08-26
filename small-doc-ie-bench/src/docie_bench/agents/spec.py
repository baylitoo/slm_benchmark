"""The persisted agent configuration record."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

AgentKind = Literal["proxy_security", "ocr", "custom"]

_NAME_RE = r"^[a-z0-9][a-z0-9._-]{0,62}$"


def utcnow_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


class AgentSpec(BaseModel):
    """One configured agent, addressable as an OpenAI model id.

    ``model_profile`` is the backing SLM selector and accepts everything the
    extraction resolver does: a ``models.yaml`` profile name, a live deployment
    name, or ``store:<name>`` — resolved fresh at request time so the agent
    follows the deployment across restarts/ports. ``None`` uses the studio
    default profile. ``options`` is kind-specific:

    * ``proxy_security`` — ``entities`` (subset of :data:`~docie_bench.agents.pii.PII_TYPES`),
      ``mode`` (``placeholder`` | ``block`` | ``detect``), ``restore_pii`` (bool),
      ``guard_model`` (encoder-family analyzer endpoint selector; replaces the
      regex analyzer), ``guard_labels`` (zero-shot labels), ``guard_threshold``
      (min confidence), ``guard_fallback`` (``"regex"`` = degrade instead of
      failing closed when the guard is down).
    * ``ocr`` — a document-extraction agent with a staged ``mode``:
        - ``mode="ocr"`` (plain OCR): ``backend`` (tesseract | paddleocr |
          pdf_text), ``language`` → returns the document's OCR text.
        - ``mode="ocr_extract"`` (OCR → LLM): the above plus ``extractor`` (a
          passthrough LLM profile) and an optional ``schema`` (a name in the
          extraction SCHEMA_REGISTRY) → OCR the image, then the LLM extracts
          structured JSON (grammar-constrained when ``schema`` is set).
          ``extractor`` also accepts ``policy:<name>``, naming a saved routing
          policy (POST /v1/studio/routing-policies): the extraction step then
          runs the policy's confidence-gated cascade across model profiles
          instead of a single model, the sanitized routing audit rides the
          completion's ``docie_agent.routing``, and ``schema`` becomes
          required (the policy runs the structured extraction path).
        - ``mode="vision"`` (vision → structured): ``vision_model`` (a vision
          deployment selector) and ``schema`` → the image goes straight to the
          vision model, which grammar-generates JSON via ``response_format``
          (llama.cpp GBNF; no OCR step). NOTE: a NuExtract deployment here runs
          via generic GBNF, not its bespoke chat_template_kwargs path.
      Back-compat: an agent saved before ``mode`` existed derives it — an
      ``extractor`` present means ``ocr_extract``, otherwise ``ocr``.
    * ``custom`` — a passthrough chat agent (system prompt + backing model).
      Optional ``mcp_servers`` (list of registered MCP server names, see
      ``GET /v1/mcp/servers``) turns on tool use: the agent runs the same
      bounded model<->tools loop the generic chat surface uses, advertising
      each server's tools and executing any tool_calls the model returns.
      Absent/empty means the original bare-forward behavior, unchanged.
      Optional ``mcp_tools`` (``{server_name: [tool_name, ...]}``) restricts
      a server to a named subset of its tools instead of exposing all of
      them — validated against the server's own live tool list, so a typo'd
      or stale name is a clear config error rather than a silent no-op.
    """

    name: str = Field(pattern=_NAME_RE, max_length=63)
    kind: AgentKind
    display_name: str = ""
    description: str = ""
    model_profile: str | None = None
    system_prompt: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)

    @field_validator("options")
    @classmethod
    def _options_json_object(cls, value: dict[str, Any]) -> dict[str, Any]:
        # Persisted as JSON; reject non-string keys early instead of at save time.
        if any(not isinstance(key, str) for key in value):
            raise ValueError("options keys must be strings")
        return value
