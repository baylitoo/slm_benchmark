from __future__ import annotations

import json
from typing import Any


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
