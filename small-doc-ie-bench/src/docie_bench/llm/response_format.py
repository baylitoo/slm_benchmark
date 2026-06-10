from __future__ import annotations

from typing import Any


def build_response_format(style: str, schema_name: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
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
    if normalized == "none":
        return None, {}
    raise ValueError(f"Unsupported response_format_style={style!r}")
