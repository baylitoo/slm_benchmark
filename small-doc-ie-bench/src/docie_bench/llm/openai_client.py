from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
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
from docie_bench.llm.mojibake import fix_mojibake
from docie_bench.llm.response_format import (
    build_response_format,
    is_generic_style,
    style_ladder,
)
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
    # A <think> opened but never closed means generation was cut off (usually
    # max_tokens) WHILE the model was still reasoning -- there is no answer to
    # extract yet. Bail out here with the raw text (which starts with
    # "<think>", not "{", so json.loads fails cleanly) rather than falling
    # through to the bracket-balance extraction below: reasoning prose that
    # merely mentions a field in brace notation (e.g. musing about `{"vendor":
    # "string"}` while planning the answer) parses as syntactically valid,
    # semantically wrong, JSON otherwise -- a silent corruption, not a clean
    # failure. Confirmed reproducible, not hypothetical.
    if "<think>" in text and "</think>" not in text:
        return text

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


# Substrings (matched case-insensitively) that mark an HTTP 400 as a
# grammar/schema-compilation failure rather than a genuine bad request. Some
# backends (e.g. this project's Ollama) reject a strong response_format style
# with a hard 400 whose body reads like "Failed to initialize samplers: failed
# to parse grammar ..." instead of returning empty 200 content. For a generic
# style that is downgradable, this is the same signal as empty content: the rung
# is unsupported, so walk to the next weaker one instead of hard-failing.
# Narrow, specific compile-failure phrases only. Bare "grammar"/"json_schema"
# were too broad: a genuine bad-request 400 that merely echoes the style name
# (e.g. "response_format.type 'json_schema' is not supported") would spuriously
# downgrade on the SHARED sync /v1/extract + CLI paths, masking a real error.
_GRAMMAR_ERROR_MARKERS: tuple[str, ...] = (
    "failed to parse grammar",
    "initialize samplers",
)


def _is_grammar_compilation_error(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _GRAMMAR_ERROR_MARKERS)


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
        breaker. Returns ``False`` only for a *genuine* non-honor — a permanent
        4xx ("style unsupported") or an empty/invalid 200. Transient signals
        (429/5xx/timeouts/connection) are INCONCLUSIVE and PROPAGATE, classified
        exactly like the runtime path via ``classify_response_error``, so the
        caller records the endpoint as flaky rather than falsely marking the
        style rejected (which would prune the ladder and re-introduce the
        empty-content bug on a single transient blip during the canary).
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
            # Classify like the runtime: a retryable (transient/rate-limited)
            # status is inconclusive and must NOT be recorded as a rejection, so
            # propagate it. Only a permanent status is a genuine "style
            # unsupported" signal that legitimately rejects the style.
            error = classify_response_error(resp)
            if error.retryable:
                raise error
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

    async def _post_chat_streamed(
        self, payload: dict[str, Any], on_delta: Callable[[str], None]
    ) -> tuple[httpx.Response, dict[str, Any]]:
        """Streaming counterpart to the plain ``self._client.post(...)`` call
        below -- same request, ``stream: true`` + ``stream_options`` added
        (the convention ``chat_api._post_upstream_streamed`` already
        established), each ``delta.content`` fragment pushed to ``on_delta``
        the instant it arrives.

        Returns ``(resp, data)`` where ``resp`` is the real ``httpx.Response``
        (its body read via ``aread()`` before returning on a >=400 status, so
        ``resp.text``/``.status_code`` behave exactly like the non-streaming
        path's response for the caller's existing error handling) and
        ``data`` is a dict reconstructed to EXACTLY the shape
        ``resp.json()`` returns for an equivalent non-streaming request
        (``choices[0].message.{content, reasoning_content}``, ``usage`` from
        the trailing frame, ``choices[0].logprobs`` when the caller asked for
        per-token logprobs) -- so every line below this method that consumes
        ``data`` needs no branching of its own.

        ``content`` is always a string (``""`` when the stream produced zero
        content deltas), never ``None`` -- a zero-content stream must clean
        to empty and hit the SAME response-format downgrade the blocking
        path takes for an empty 200, not a hard "content must be text" raise
        that would bypass the negotiation ladder entirely.
        """
        body = dict(payload)
        body["stream"] = True
        stream_options = dict(body.get("stream_options") or {})
        stream_options.setdefault("include_usage", True)
        body["stream_options"] = stream_options

        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                return resp, {}

            role = "assistant"
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            logprobs_parts: list[dict[str, Any]] = []
            finish_reason: str | None = None
            usage: dict[str, Any] | None = None
            completion_id = ""
            model_name = str(payload.get("model") or "")

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:") :].strip()
                if chunk in ("", "[DONE]"):
                    continue
                try:
                    frame = json.loads(chunk)
                except ValueError:
                    continue
                if not isinstance(frame, dict):
                    continue
                frame_id = frame.get("id")
                if isinstance(frame_id, str) and frame_id:
                    completion_id = frame_id
                frame_model = frame.get("model")
                if isinstance(frame_model, str) and frame_model:
                    model_name = frame_model
                frame_usage = frame.get("usage")
                if isinstance(frame_usage, dict):
                    # Sent on the TRAILING frame (stream_options.include_usage,
                    # set above) -- a later frame's usage always wins, but
                    # there is normally at most one.
                    usage = frame_usage
                choices = frame.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                choice_finish = choice.get("finish_reason")
                if isinstance(choice_finish, str) and choice_finish:
                    finish_reason = choice_finish
                choice_logprobs = choice.get("logprobs")
                if isinstance(choice_logprobs, dict):
                    lp_content = choice_logprobs.get("content")
                    if isinstance(lp_content, list):
                        logprobs_parts.extend(
                            entry for entry in lp_content if isinstance(entry, dict)
                        )
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                delta_role = delta.get("role")
                if isinstance(delta_role, str) and delta_role:
                    role = delta_role
                content_piece = delta.get("content")
                if isinstance(content_piece, str) and content_piece:
                    content_parts.append(content_piece)
                    on_delta(content_piece)
                reasoning_piece = delta.get("reasoning_content")
                if isinstance(reasoning_piece, str) and reasoning_piece:
                    reasoning_parts.append(reasoning_piece)

            message: dict[str, Any] = {"role": role, "content": "".join(content_parts)}
            if reasoning_parts:
                message["reasoning_content"] = "".join(reasoning_parts)
            choice_data: dict[str, Any] = {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
            if logprobs_parts:
                choice_data["logprobs"] = {"content": logprobs_parts}
            data: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion",
                "model": model_name,
                "choices": [choice_data],
            }
            if usage is not None:
                data["usage"] = usage
            return resp, data

    async def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        image_urls: list[str] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        assistant_prefill: str | None = None,
        request_logprobs: bool = False,
        # Additive, opt-in live preview (#397): raw content-delta fragments
        # for the CURRENT attempt, pushed as they stream in. Never a
        # replacement for this method's return value -- chat_json still
        # post-processes the model's raw output (negotiation ladder,
        # mojibake fix, JSON cleanup) after the stream ends, and that
        # processed result is the only thing callers should treat as
        # authoritative. `on_delta=None` (the default) takes the exact same
        # blocking, non-streaming code path this method always has.
        on_delta: Callable[[str], None] | None = None,
        # Fires when a PREVIOUS attempt already streamed deltas and a new
        # attempt (a response-format downgrade, a reasoning-disable retry, or
        # a gateway-level transient retry) is about to start -- the caller
        # must clear its live buffer, or it keeps painting fragments from an
        # abandoned attempt alongside the new one. Never fires before the
        # very first delta of the whole call.
        on_reset: Callable[[], None] | None = None,
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
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        output_budget = max_tokens if max_tokens is not None else self.profile.max_tokens
        if output_budget < 1:
            raise ValueError("max_tokens must be positive")

        def build_payload(
            style: str,
            *,
            force_disable_reasoning: bool = False,
            use_prefill: bool = False,
        ) -> dict[str, Any]:
            response_format, extra_body = build_response_format(style, schema_name, schema)
            messages = list(base_messages)
            if use_prefill and assistant_prefill is not None:
                messages.append({"role": "assistant", "content": assistant_prefill})
            payload: dict[str, Any] = {
                "model": self.profile.model,
                "messages": messages,
                "temperature": self.profile.temperature,
                "top_p": self.profile.top_p,
                "max_tokens": output_budget,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            if self.profile.stop_sequences:
                payload["stop"] = list(self.profile.stop_sequences)
            if request_logprobs:
                # Opt-in, llama.cpp-only per-token confidence signal (#335,
                # see extract/logprob_confidence.py). Only the CHOSEN token's
                # own logprob is needed for MIN-aggregation, not runner-up
                # alternatives, so top_logprobs stays at the minimum (1).
                payload["logprobs"] = True
                payload["top_logprobs"] = 1
            payload.update(extra_body)
            if chat_template_kwargs:
                merged_template_kwargs = dict(payload.get("chat_template_kwargs") or {})
                merged_template_kwargs.update(chat_template_kwargs)
                payload["chat_template_kwargs"] = merged_template_kwargs
            if force_disable_reasoning:
                merged_template_kwargs = dict(payload.get("chat_template_kwargs") or {})
                merged_template_kwargs["enable_thinking"] = False
                payload["chat_template_kwargs"] = merged_template_kwargs
                payload["reasoning_effort"] = "none"
            elif (payload.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
                payload["reasoning_effort"] = "none"
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
                "docie_max_tokens": output_budget,
                "docie_assistant_prefill": assistant_prefill is not None,
                "docie_request_logprobs": request_logprobs,
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

        # Shared across every operation() invocation, INCLUDING one the
        # gateway's own transient-error retry re-enters from scratch (ladder
        # position 0 again) -- a single-element list so the closures below
        # can mutate it; a plain outer variable can't be reassigned from a
        # nested function without `nonlocal` scattered through two closures.
        streamed_any_delta = [False]

        def delta_cb(text: str) -> None:
            streamed_any_delta[0] = True
            on_delta(text)  # type: ignore[misc]  # only called when on_delta is not None

        async def operation() -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
            # Walk the negotiation ladder: HTTP/transport failures raise (so the
            # gateway's transient retry keeps working), while an ordinary empty
            # or unparseable 200 downgrades to the next weaker style. A response
            # that exhausted its output budget gets a distinct recovery below.
            last_invalid: InvalidModelResponseError | None = None
            # A length-truncated response with no visible content is not a
            # response-format failure: llama-server may have spent the entire
            # budget in reasoning_content. Insert one same-style, reasoning-off
            # recovery before considering any weaker schema style.
            attempts = [(style, False, assistant_prefill is not None) for style in ladder]
            for position, (style, force_disable_reasoning, use_prefill) in enumerate(attempts):
                is_last_rung = position == len(attempts) - 1
                t0 = _time.perf_counter()
                request_payload = build_payload(
                    style,
                    force_disable_reasoning=force_disable_reasoning,
                    use_prefill=use_prefill,
                )
                streamed_data: dict[str, Any] | None = None
                if on_delta is None:
                    resp = await self._client.post("/chat/completions", json=request_payload)
                else:
                    # A previous attempt (in THIS call, or a prior gateway-
                    # level retry that re-entered operation() from scratch)
                    # already streamed deltas the caller rendered -- this
                    # fresh attempt must not paint alongside them.
                    if streamed_any_delta[0]:
                        if on_reset is not None:
                            on_reset()
                        streamed_any_delta[0] = False
                    resp, streamed_data = await self._post_chat_streamed(
                        request_payload, delta_cb
                    )
                llm_latency_ms = int((_time.perf_counter() - t0) * 1000)

                if resp.status_code >= 400:
                    # llama.cpp can reject a JSON grammar when the final
                    # assistant turn is a continuation. Preserve the schema and
                    # retry the same style once without prefill; if that model
                    # then spends its budget reasoning, the next ladder rung gets
                    # another prefilled attempt.
                    if (
                        resp.status_code == 400
                        and use_prefill
                        and _is_grammar_compilation_error(resp.text)
                    ):
                        attempts.insert(position + 1, (style, force_disable_reasoning, False))
                        logger.warning(
                            "structured-output prefill downgrade",
                            extra={
                                "docie_step": "assistant_prefill_downgrade",
                                "docie_model_profile": self.profile.name,
                                "docie_model": self.profile.model,
                                "docie_schema_name": schema_name,
                                "docie_response_format_style": style,
                                "docie_reason": "grammar_compilation_error",
                                "docie_action": "retry_without_prefill",
                                "docie_upstream_error": resp.text[:1000],
                            },
                        )
                        continue
                    # Some backends reject a strong response_format style with a
                    # hard HTTP 400 (grammar/schema failed to compile) instead of
                    # returning empty 200 content. For a downgradable generic
                    # style this is the same "rung unsupported" signal, so walk to
                    # the next weaker style instead of hard-failing.
                    if (
                        resp.status_code == 400
                        and not is_last_rung
                        and is_generic_style(style)
                        and _is_grammar_compilation_error(resp.text)
                    ):
                        last_invalid = InvalidModelResponseError(
                            f"Style {style!r} rejected with grammar/schema "
                            f"compilation error: {resp.text[:1000]}"
                        )
                        logger.warning(
                            "structured-output downgrade",
                            extra={
                                "docie_step": "response_format_downgrade",
                                "docie_model_profile": self.profile.name,
                                "docie_model": self.profile.model,
                                "docie_schema_name": schema_name,
                                "docie_from_style": style,
                                "docie_to_style": attempts[position + 1][0],
                                "docie_reason": "grammar_compilation_error",
                                "docie_upstream_error": resp.text[:1000],
                            },
                        )
                        continue
                    logger.error(
                        "LLM server error",
                        extra={
                            "docie_status_code": resp.status_code,
                            "docie_body": resp.text[:2000],
                        },
                    )
                    raise classify_response_error(resp)
                if on_delta is None:
                    try:
                        data = resp.json()
                    except ValueError as exc:
                        raise InvalidModelResponseError(
                            "Model endpoint returned invalid JSON"
                        ) from exc
                else:
                    data = streamed_data or {}
                try:
                    message = data["choices"][0]["message"]
                    content = message["content"]
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

                # OpenAI-compatible servers differ on assistant continuation:
                # some return the assembled message, others return only the
                # generated suffix. Restore the prefix before JSON parsing.
                if use_prefill and assistant_prefill and content.strip():
                    if not content.lstrip().startswith(assistant_prefill.lstrip()):
                        content = assistant_prefill + content

                # Repair model-emitted UTF-8 mojibake (accented OCR/extraction on
                # small models) before parsing, so field values read correctly.
                if get_settings().fix_mojibake:
                    content = fix_mojibake(content)

                finish_reason = data.get("choices", [{}])[0].get("finish_reason")
                reasoning_content = message.get("reasoning_content")
                reasoning_chars = (
                    len(reasoning_content) if isinstance(reasoning_content, str) else 0
                )
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
                        "docie_finish_reason": finish_reason,
                        "docie_reasoning_chars": reasoning_chars,
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
                    unclosed_think = "<think>" in content and "</think>" not in content
                    output_budget_exhausted = finish_reason == "length" and (
                        not cleaned.strip() or unclosed_think
                    )
                    if output_budget_exhausted:
                        reasoning_already_disabled = (
                            force_disable_reasoning
                            or request_payload.get("reasoning_effort") == "none"
                        )
                        if not reasoning_already_disabled:
                            attempts.insert(position + 1, (style, True, use_prefill))
                            logger.warning(
                                "structured-output reasoning recovery",
                                extra={
                                    "docie_step": "reasoning_recovery",
                                    "docie_model_profile": self.profile.name,
                                    "docie_model": self.profile.model,
                                    "docie_schema_name": schema_name,
                                    "docie_response_format_style": style,
                                    "docie_reason": "output_budget_exhausted",
                                    "docie_action": "disable_reasoning",
                                    "docie_reasoning_chars": reasoning_chars,
                                    "docie_completion_tokens": (data.get("usage") or {}).get(
                                        "completion_tokens"
                                    ),
                                },
                            )
                            continue
                        logger.error(
                            "structured-output reasoning recovery exhausted",
                            extra={
                                "docie_step": "reasoning_recovery_exhausted",
                                "docie_model_profile": self.profile.name,
                                "docie_model": self.profile.model,
                                "docie_schema_name": schema_name,
                                "docie_response_format_style": style,
                                "docie_reason": "output_budget_exhausted",
                                "docie_reasoning_chars": reasoning_chars,
                                "docie_completion_tokens": (data.get("usage") or {}).get(
                                    "completion_tokens"
                                ),
                            },
                        )
                        if assistant_prefill is not None and not use_prefill and not is_last_rung:
                            logger.warning(
                                "structured-output prefill recovery",
                                extra={
                                    "docie_step": "assistant_prefill_recovery",
                                    "docie_model_profile": self.profile.name,
                                    "docie_model": self.profile.model,
                                    "docie_schema_name": schema_name,
                                    "docie_response_format_style": style,
                                    "docie_reason": "reasoning_exhausted_without_prefill",
                                    "docie_action": "try_next_style_with_prefill",
                                },
                            )
                            continue
                        raise InvalidModelResponseError(
                            f"Generation exhausted max_tokens ({output_budget}) "
                            "with no usable answer even after reasoning was disabled"
                        ) from exc
                    invalid_reason = (
                        "empty_content" if not cleaned.strip() else "unparseable_content"
                    )
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
                            "docie_to_style": attempts[position + 1][0],
                            "docie_reason": "empty_or_unparseable_content",
                            "docie_invalid_reason": invalid_reason,
                            "docie_finish_reason": finish_reason,
                            "docie_content_chars": len(content),
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
