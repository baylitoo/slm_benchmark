from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from docie_bench.llm.model_gateway import (
    InvalidModelResponseError,
    ModelCapabilities,
    ModelGateway,
    ModelGatewayError,
    classify_response_error,
)
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.response_format import build_response_format
from docie_bench.settings import get_settings

logger = logging.getLogger(__name__)


def _clean_content(text: str) -> str:
    """Normalise raw LLM output to a single JSON object string.

    Handles:
    - NuExtract3's <|end-output|> continuation (take only what comes before it)
    - Markdown code fences (```json ... ```)
    - Run-on text after a complete JSON object (bracket-balance extraction)
    """
    # Strip NuExtract3 end token and anything after it
    if "<|end-output|>" in text:
        text = text[: text.index("<|end-output|>")]

    text = text.strip()

    # Unwrap markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    # Extract the first complete JSON object by bracket counting.
    # This handles hallucinated text appended after the closing brace.
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _fix_bare_keys(text[start : i + 1])

    return _fix_bare_keys(text[start:])  # incomplete — return what we have; json.loads will raise


_BARE_KEY_RE = re.compile(r'(?<=[{,\n])(\s*)(?!")([A-Za-z_]\w*)(\s*:)')


def _fix_bare_keys(text: str) -> str:
    """Quote bare (unquoted) JSON keys emitted as hallucinations.

    e.g. NuExtract mixing document text into output: `zaknur: {` → `"zaknur": {`
    Only matches after structural positions ({, comma, newline) to avoid
    touching string values that happen to contain word:colon patterns.
    """
    return _BARE_KEY_RE.sub(r'\1"\2"\3', text)


class LLMClientError(ModelGatewayError):
    pass


class OpenAICompatibleClient:
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile
        self._client = httpx.AsyncClient(
            base_url=profile.base_url,
            timeout=httpx.Timeout(profile.timeout_seconds),
            headers={"Authorization": f"Bearer {profile.api_key}"},
        )
        self._gateway = ModelGateway(profile, self._client)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def discover_capabilities(self, *, force: bool = False) -> ModelCapabilities:
        self._gateway.client = self._client
        return await self._gateway.discover_capabilities(force=force)

    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        image_urls: list[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
        import time as _time

        self._gateway.client = self._client
        await self._gateway.validate_request(needs_vision=bool(image_urls))
        response_format, extra_body = build_response_format(
            self.profile.response_format_style,
            schema_name,
            schema,
        )
        user_content: str | list[dict[str, Any]] = user_prompt
        if image_urls:
            user_content = [{"type": "text", "text": user_prompt}]
            user_content.extend(
                {"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls
            )
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.profile.temperature,
            "top_p": self.profile.top_p,
            "max_tokens": self.profile.max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if self.profile.stop_sequences:
            payload["stop"] = list(self.profile.stop_sequences)
        payload.update(extra_body)

        logger.debug(
            "llm_request",
            extra={
                "docie_step": "llm_request",
                "docie_base_url": self.profile.base_url,
                "docie_model": self.profile.model,
                "docie_schema_name": schema_name,
                "docie_response_format_style": self.profile.response_format_style,
                **(
                    {
                        "docie_system_prompt": system_prompt,
                        "docie_user_prompt": user_prompt,
                    }
                    if get_settings().log_document_content
                    else {}
                ),
                "docie_image_count": len(image_urls or []),
            },
        )

        async def operation() -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
            t0 = _time.perf_counter()
            resp = await self._client.post("/chat/completions", json=payload)
            llm_latency_ms = int((_time.perf_counter() - t0) * 1000)

            if resp.status_code >= 400:
                logger.error(
                    "LLM server error",
                    extra={"docie_status_code": resp.status_code, "docie_body": resp.text[:2000]},
                )
                raise classify_response_error(resp)
            try:
                data = resp.json()
            except ValueError as exc:
                raise InvalidModelResponseError("Model endpoint returned invalid JSON") from exc
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise InvalidModelResponseError(
                    f"Unexpected LLM response shape: {data}"
                ) from exc
            if isinstance(content, list):
                # Some multimodal-compatible gateways return content blocks.
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            if not isinstance(content, str):
                raise InvalidModelResponseError("Model response content must be text")

            logger.debug(
                "llm_response",
                extra={
                    "docie_step": "llm_response",
                    "docie_model": self.profile.model,
                    "docie_schema_name": schema_name,
                    **({"docie_raw_content": content} if get_settings().log_document_content else {}),
                    "docie_finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                    "docie_usage": data.get("usage"),
                    "docie_llm_latency_ms": llm_latency_ms,
                },
            )

            content = _clean_content(content)
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise InvalidModelResponseError(
                    f"Model returned non-JSON content: {content[:1000]}"
                ) from exc
            if not isinstance(parsed, dict):
                raise InvalidModelResponseError("Model returned JSON that is not an object")
            return parsed, data.get("usage"), data

        return await self._gateway.execute(operation)
