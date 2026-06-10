from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.response_format import build_response_format

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
                return text[start : i + 1]

    return text[start:]  # incomplete — return what we have; json.loads will raise


class LLMClientError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile
        self._client = httpx.AsyncClient(
            base_url=profile.base_url,
            timeout=httpx.Timeout(profile.timeout_seconds),
            headers={"Authorization": f"Bearer {profile.api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, LLMClientError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
        import time as _time

        response_format, extra_body = build_response_format(
            self.profile.response_format_style,
            schema_name,
            schema,
        )
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
                "docie_system_prompt": system_prompt,
                "docie_user_prompt": user_prompt,
            },
        )

        t0 = _time.perf_counter()
        resp = await self._client.post("/chat/completions", json=payload)
        llm_latency_ms = int((_time.perf_counter() - t0) * 1000)

        if resp.status_code >= 400:
            logger.error(
                "LLM server error",
                extra={"docie_status_code": resp.status_code, "docie_body": resp.text[:2000]},
            )
            raise LLMClientError(f"LLM server returned {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Unexpected LLM response shape: {data}") from exc
        if isinstance(content, list):
            # Some multimodal-compatible gateways return content blocks.
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

        logger.debug(
            "llm_response",
            extra={
                "docie_step": "llm_response",
                "docie_model": self.profile.model,
                "docie_schema_name": schema_name,
                "docie_raw_content": content,
                "docie_finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                "docie_usage": data.get("usage"),
                "docie_llm_latency_ms": llm_latency_ms,
            },
        )

        content = _clean_content(content)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Model returned non-JSON content: {content[:1000]}") from exc
        return parsed, data.get("usage"), data
