from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from docie_bench.llm.capability_probe import cached_probe_for_endpoint
from docie_bench.llm.model_gateway import (
    InvalidModelResponseError,
    ModelCapabilities,
    ModelGateway,
    ModelGatewayError,
    classify_response_error,
)
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.response_format import build_response_format, style_ladder
from docie_bench.settings import get_settings

logger = logging.getLogger(__name__)


def _clean_content(text: str) -> str:
    """Normalise raw LLM output to a single JSON object string.

    Handles:
    - NuExtract3 reasoning mode's <think>...</think> block (keep only the answer)
    - NuExtract v1's <|end-output|> continuation (take only what comes before it)
    - Markdown code fences (```json ... ```)
    - Run-on text after a complete JSON object (bracket-balance extraction)
    """
    # NuExtract3 reasoning mode prefixes the answer with a <think>...</think>
    # block; keep only what follows the final </think>.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    # Strip the NuExtract v1 end token and anything after it
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
        # The response-format style that actually produced a valid parse on the
        # most recent chat_json call. Recorded into predictions so constrained
        # (json_schema) vs unconstrained (none+repair) decoding is distinguishable.
        self.last_response_format_style: str | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def discover_capabilities(self, *, force: bool = False) -> ModelCapabilities:
        self._gateway.client = self._client
        return await self._gateway.discover_capabilities(force=force)

    def _negotiated_ladder(self) -> tuple[str, ...]:
        """Response-format styles to try, strongest confirmed rung first.

        The runtime downgrade is unconditional (it is the actual fix for the
        empty-content defect and must work even with capability discovery
        disabled). A cached probe only *prunes* rungs the endpoint already
        rejected, so real documents do not re-pay a downgrade round-trip.
        """
        ladder = style_ladder(self.profile.response_format_style)
        probe = cached_probe_for_endpoint(self.profile.base_url, self.profile.model)
        if probe is None or not probe.rejected_styles:
            return ladder
        pruned = tuple(style for style in ladder if style not in probe.rejected_styles)
        # Never collapse to nothing: keep the terminal rung as a safety net.
        return pruned or ladder[-1:]

    async def probe_style(self, style: str, *, schema: dict[str, Any]) -> bool:
        """Issue one minimal completion with a single style; report if honored.

        Bypasses the gateway retry/circuit machinery so a probe never trips the
        breaker. Returns ``False`` only for a *served* non-honor (a 4xx/5xx or an
        empty/invalid 200); transport errors PROPAGATE so the caller records the
        endpoint as unreachable rather than falsely marking every style rejected
        (which would prune the ladder and re-introduce the empty-content bug).
        """
        response_format, extra_body = build_response_format(style, "probe", schema)
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a JSON API. Reply with a single JSON object.",
                },
                {"role": "user", "content": 'Return exactly this JSON object: {"ok": "yes"}'},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        payload.update(extra_body)
        resp = await self._client.post("/chat/completions", json=payload)
        if resp.status_code >= 400:
            return False
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return False
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            return False
        try:
            return isinstance(json.loads(_clean_content(content)), dict)
        except json.JSONDecodeError:
            return False

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
        ladder = self._negotiated_ladder()
        declared_style = self.profile.response_format_style
        user_content: str | list[dict[str, Any]] = user_prompt
        if image_urls:
            user_content = [{"type": "text", "text": user_prompt}]
            user_content.extend(
                {"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        def build_payload(style: str) -> dict[str, Any]:
            response_format, extra_body = build_response_format(style, schema_name, schema)
            payload: dict[str, Any] = {
                "model": self.profile.model,
                "messages": messages,
                "temperature": self.profile.temperature,
                "top_p": self.profile.top_p,
                "max_tokens": self.profile.max_tokens,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            if self.profile.stop_sequences:
                payload["stop"] = list(self.profile.stop_sequences)
            payload.update(extra_body)
            return payload

        logger.debug(
            "llm_request",
            extra={
                "docie_step": "llm_request",
                "docie_base_url": self.profile.base_url,
                "docie_model": self.profile.model,
                "docie_schema_name": schema_name,
                "docie_response_format_style": declared_style,
                "docie_response_format_ladder": list(ladder),
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
            # Walk the negotiation ladder: HTTP/transport failures raise (so the
            # gateway's transient retry keeps working), but an empty or
            # unparseable 200 downgrades to the next weaker style instead of
            # retrying the same one — this is the fix for the empty-content bug.
            last_invalid: InvalidModelResponseError | None = None
            for position, style in enumerate(ladder):
                is_last_rung = position == len(ladder) - 1
                t0 = _time.perf_counter()
                resp = await self._client.post("/chat/completions", json=build_payload(style))
                llm_latency_ms = int((_time.perf_counter() - t0) * 1000)

                if resp.status_code >= 400:
                    logger.error(
                        "LLM server error",
                        extra={
                            "docie_status_code": resp.status_code,
                            "docie_body": resp.text[:2000],
                        },
                    )
                    raise classify_response_error(resp)
                try:
                    data = resp.json()
                except ValueError as exc:
                    raise InvalidModelResponseError(
                        "Model endpoint returned invalid JSON"
                    ) from exc
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise InvalidModelResponseError(
                        f"Unexpected LLM response shape: {data}"
                    ) from exc
                if isinstance(content, list):
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
                        "docie_response_format_style": style,
                        **(
                            {"docie_raw_content": content}
                            if getattr(get_settings(), "log_document_content", False)
                            else {}
                        ),
                        "docie_finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                        "docie_usage": data.get("usage"),
                        "docie_llm_latency_ms": llm_latency_ms,
                    },
                )
                cleaned = _clean_content(content)
                try:
                    parsed = json.loads(cleaned)
                    if not isinstance(parsed, dict):
                        raise InvalidModelResponseError("Model returned JSON that is not an object")
                except (json.JSONDecodeError, InvalidModelResponseError) as exc:
                    last_invalid = (
                        exc
                        if isinstance(exc, InvalidModelResponseError)
                        else InvalidModelResponseError(
                            f"Model returned non-JSON content: {cleaned[:1000]}"
                        )
                    )
                    if is_last_rung:
                        raise last_invalid from exc
                    logger.warning(
                        "structured-output downgrade",
                        extra={
                            "docie_step": "response_format_downgrade",
                            "docie_model_profile": self.profile.name,
                            "docie_model": self.profile.model,
                            "docie_schema_name": schema_name,
                            "docie_from_style": style,
                            "docie_to_style": ladder[position + 1],
                            "docie_reason": "empty_or_unparseable_content",
                        },
                    )
                    continue

                self.last_response_format_style = style
                if style != declared_style:
                    logger.info(
                        "structured-output negotiated",
                        extra={
                            "docie_step": "response_format_effective",
                            "docie_model_profile": self.profile.name,
                            "docie_model": self.profile.model,
                            "docie_declared_style": declared_style,
                            "docie_effective_style": style,
                        },
                    )
                return parsed, data.get("usage"), data

            raise last_invalid or InvalidModelResponseError(
                "No response-format style produced valid JSON"
            )

        return await self._gateway.execute(operation)
