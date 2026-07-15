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
      ``guard_model`` (reserved: encoder-family analyzer profile).
    * ``ocr`` — ``backend`` (tesseract | paddleocr | pdf_text), ``language``,
      ``extractor`` (optional passthrough profile name -> OCR→SLM pipeline).
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
