from __future__ import annotations

import json
from typing import Any

# Response-format styles whose contract is "emit a JSON object"; these can be
# gracefully downgraded to a weaker rung when a runtime returns empty content for
# the stronger one (the small-Ollama + json_schema empty-content defect). Every
# other style is purpose-built (nuextract3* delivers a template out-of-band,
# vllm_* drives guided decoding) and MUST NOT be silently downgraded, or the
# request is corrupted.
GENERIC_STYLES: frozenset[str] = frozenset(
    {"openai_json_schema", "json_object", "llamacpp_schema", "none"}
)

# Ordered downgrade ladders, strongest rung first. The terminal "none" rung
# relies on the client's parse-and-repair path (see openai_client._clean_content).
_STYLE_LADDERS: dict[str, tuple[str, ...]] = {
    "openai_json_schema": ("openai_json_schema", "json_object", "none"),
    "llamacpp_schema": ("llamacpp_schema", "json_object", "none"),
    "json_object": ("json_object", "none"),
    "none": ("none",),
}


def is_generic_style(style: str) -> bool:
    return style.lower().strip() in GENERIC_STYLES


def style_ladder(declared_style: str) -> tuple[str, ...]:
    """Return the ordered response-format styles to try, strongest first.

    Only the generic JSON-object family downgrades; purpose-built styles
    (``nuextract3``/``nuextract3_think``/``vllm_*``) return a singleton ladder so
    the negotiation path never rewrites their out-of-band contract.
    """
    normalized = declared_style.lower().strip()
    return _STYLE_LADDERS.get(normalized, (declared_style,))


def build_response_format(
    style: str, schema_name: str, schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return (response_format, extra_body) for common OpenAI-compatible servers.

    Runtimes evolve quickly. This adapter keeps the rest of the application stable.
    """
    normalized = style.lower().strip()
    if normalized == "openai_json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"{schema_name}_extraction",
                "strict": True,
                "schema": schema,
            },
        }, {}
    if normalized == "llamacpp_schema":
        # llama.cpp server variants commonly accept schema under response_format.
        return {"type": "json_object", "schema": schema}, {}
    if normalized == "json_object":
        return {"type": "json_object"}, {}
    if normalized == "vllm_guided_json":
        return None, {"guided_json": schema}
    if normalized == "vllm_structured_outputs":
        return None, {"structured_outputs": {"json": schema}}
    if normalized in {"nuextract3", "nuextract3_think"}:
        # NuExtract3 takes the extraction template out-of-band via
        # chat_template_kwargs (honoured by `llama-server --jinja`); the document
        # rides in the messages instead of the schema. `_think` enables the
        # model's reasoning mode. Verified working 2026-06-17.
        from docie_bench.llm.prompts import nuextract_template_for

        template = nuextract_template_for(schema_name)
        return None, {
            "chat_template_kwargs": {
                "template": json.dumps(template, ensure_ascii=False),
                "enable_thinking": normalized == "nuextract3_think",
            }
        }
    if normalized == "none":
        return None, {}
    raise ValueError(f"Unsupported response_format_style={style!r}")
