"""Preconfigured agent templates the Studio catalog offers.

A template is a starting point: ``defaults`` prefill the create form / request
body and remain fully overridable. Templates are code, not data — adding one
is a PR, so the catalog stays reviewed and reproducible.
"""

from __future__ import annotations

from typing import Any

from docie_bench.agents.pii import PII_TYPES

AGENT_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "proxy-security",
        "kind": "proxy_security",
        "display_name": "Security Proxy Agent",
        "description": (
            "Privacy firewall in front of any served SLM: detects PII "
            "(emails, phones, IBANs, cards, national ids, IPs) and anonymizes "
            "it with stable placeholders before the model sees the prompt. "
            "Modes: placeholder (mask), block (refuse), detect (annotate only). "
            "Also the seam for IP/confidentiality screening."
        ),
        "defaults": {
            "system_prompt": None,
            "options": {
                "entities": list(PII_TYPES),
                "mode": "placeholder",
                "restore_pii": False,
                # Reserved: a specialized encoder (PII/NER) analyzer profile.
                # When set, it replaces the built-in regex analyzer.
                "guard_model": None,
            },
        },
    },
    {
        "id": "ocr-agent",
        "kind": "ocr",
        "display_name": "OCR Agent",
        "description": (
            "Instantiates an OCR solution behind an OpenAI endpoint: send a "
            "document as an inline image_url data URI, get its text back. Set "
            "an extractor profile (e.g. a NuExtract deployment) to turn it "
            "into an OCR→SLM structured-extraction pipeline."
        ),
        "defaults": {
            "system_prompt": None,
            "options": {"backend": "tesseract", "language": None, "extractor": None},
        },
    },
    {
        "id": "custom",
        "kind": "custom",
        "display_name": "Custom Agent",
        "description": (
            "Bring your own: a system prompt over any served model. The "
            "starting point for building new agents on the platform."
        ),
        "defaults": {"system_prompt": "", "options": {}},
    },
]


def template_by_id(template_id: str) -> dict[str, Any] | None:
    for template in AGENT_TEMPLATES:
        if template["id"] == template_id:
            return template
    return None
