"""Execute one agent request: pre-process → backing solution/SLM → post-process.

The OpenAI chat-completion dict is the contract on both sides, so anything an
agents platform can send to a model it can send to an agent. The backing model
selector is resolved FRESH per request through the shared extraction resolver
(profile name / live deployment / ``store:<name>``), so an agent survives its
deployment being unloaded and reloaded on another port.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from docie_bench.agents import pii
from docie_bench.agents.guard import (
    GuardAnalysisError,
    guard_analyze,
    labels_from_entities,
    moderation_flags,
)
from docie_bench.agents.spec import AgentSpec
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.profile_resolver import (
    ProfileResolutionError,
    resolve_extraction_profile,
)
from docie_bench.serving.solutions import SolutionError, build_solution

PROXY_MODES = ("placeholder", "block", "detect")


class AgentError(Exception):
    """Mapped to an OpenAI-style error payload by the API layer."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


async def complete_agent(
    spec: AgentSpec,
    body: dict[str, Any],
    *,
    http_client: httpx.AsyncClient,
) -> dict[str, Any]:
    """Serve one OpenAI chat request through ``spec``; returns a completion dict."""
    if not spec.enabled:
        raise AgentError(
            f"agent {spec.name!r} is disabled", status_code=403, error_type="agent_disabled"
        )
    if spec.kind == "ocr":
        return await _complete_ocr(spec, body, http_client=http_client)
    if spec.kind == "proxy_security":
        return await _complete_proxy(spec, body, http_client=http_client)
    return await _complete_custom(spec, body, http_client=http_client)


# ---------------------------------------------------------------------------
# OCR agent — reuse the gateway's solution adapters.
# ---------------------------------------------------------------------------


OCR_MODES = ("ocr", "ocr_extract", "vision")


def _resolve_ocr_mode(options: dict[str, Any]) -> str:
    """The staged mode, deriving it for agents saved before ``mode`` existed:
    an ``extractor`` present means the OCR→LLM pipeline, otherwise plain OCR."""
    mode = options.get("mode")
    if mode in OCR_MODES:
        return str(mode)
    return "ocr_extract" if options.get("extractor") else "ocr"


def _inject_response_format(body: dict[str, Any], schema_name: str) -> None:
    """Constrain the completion to the named extraction schema via OpenAI
    ``response_format`` (llama.cpp compiles it to a GBNF grammar). Shared by the
    OCR→LLM (over OCR text) and vision (over the image) paths.

    Uses the FLATTENED schema (flat_schema_json): the raw schema's per-field
    ``{value, evidence_ids, confidence}`` wrapper + a negative-lookahead regex on
    the money/number fields make llama.cpp fail to compile the grammar. The flat
    form (values only, no lookahead, additionalProperties:false) compiles, so the
    strict grammar actually applies — a clean ``{field: value | null}`` with no
    extra or duplicate keys, instead of falling back to shapeless json_object."""
    from docie_bench.schemas.extraction import flat_schema_json

    try:
        json_schema = flat_schema_json(schema_name)
    except ValueError as exc:
        raise AgentError(
            str(exc), status_code=400, error_type="invalid_request_error"
        ) from exc
    body["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
    }


def _schema_instruction(schema_name: str) -> str | None:
    """A prompt that names the fields to extract. The response_format grammar
    constrains the SHAPE but never tells the model WHICH fields to pull; when the
    grammar can't compile and we fall back to json_object, that's all the model
    has — without it a vision model just echoes the prompt (``{"OCR": true}``).
    So the target fields ride in the prompt too, for every attempt."""
    from docie_bench.schemas.extraction import schema_json

    try:
        properties = schema_json(schema_name).get("properties")
    except ValueError:
        return None
    if not isinstance(properties, dict) or not properties:
        return None
    fields = ", ".join(properties)
    return (
        "You extract structured data from the document in the image. Return ONLY "
        "a single JSON object with these fields (use null when a field is absent "
        f"from the document): {fields}. No prose, no explanation — JSON only."
    )


async def _post_chat_with_schema(
    upstream: ModelProfile,
    body: dict[str, Any],
    schema_name: str | None,
    *,
    http_client: httpx.AsyncClient,
) -> dict[str, Any]:
    """Forward a chat request, delivering ``schema_name`` as ``response_format``
    but resilient to a backend that cannot compile a deep schema's grammar.

    llama.cpp turns a strict json_schema into a GBNF grammar; a large nested
    schema (``$defs``/``$ref``/``anyOf`` — e.g. the invoice schema) makes it
    fail with 'failed to parse grammar' (HTTP 400). The extraction client
    survives this via its style ladder; the agent forward gets the same one
    here: strict json_schema -> json_object (valid-JSON, no schema grammar) ->
    no constraint. Only a grammar/response_format 400 downgrades; any other
    error propagates."""
    if not schema_name:
        return await _post_chat(upstream, dict(body), http_client=http_client)

    # Name the target fields in the prompt (every attempt), so the model knows
    # WHAT to extract even when the schema grammar is dropped — see
    # _schema_instruction.
    guided = dict(body)
    instruction = _schema_instruction(schema_name)
    if instruction:
        guided["messages"] = [
            {"role": "system", "content": instruction},
            *(guided.get("messages") or []),
        ]

    strict = dict(guided)
    _inject_response_format(strict, schema_name)  # unknown schema -> AgentError (propagates)
    json_object = {**guided, "response_format": {"type": "json_object"}}
    plain = dict(guided)
    attempts = (strict, json_object, plain)

    last_error: AgentError | None = None
    for index, attempt in enumerate(attempts):
        try:
            return await _post_chat(upstream, attempt, http_client=http_client)
        except AgentError as exc:
            downgradable = exc.status_code == 400 and any(
                token in str(exc).lower()
                for token in ("grammar", "response_format", "json_schema", "json schema")
            )
            if index < len(attempts) - 1 and downgradable:
                last_error = exc
                continue
            raise
    raise last_error  # pragma: no cover - the last attempt sends no response_format


async def _complete_ocr(
    spec: AgentSpec, body: dict[str, Any], *, http_client: httpx.AsyncClient
) -> dict[str, Any]:
    options = dict(spec.options)
    mode = _resolve_ocr_mode(options)

    # Vision → structured: the image goes straight to a vision deployment, which
    # grammar-generates JSON. No OCR, no solution adapter — just a schema-injected
    # forward (llama.cpp GBNF does the structuring).
    if mode == "vision":
        vision_selector = options.get("vision_model")
        if not vision_selector:
            raise AgentError(
                f"agent {spec.name!r} is in vision mode but has no options.vision_model",
                status_code=500,
                error_type="invalid_agent_config",
            )
        upstream = _resolve_backing(str(vision_selector))
        base = dict(body)
        if spec.system_prompt:
            base["messages"] = [
                {"role": "system", "content": spec.system_prompt},
                *(base.get("messages") or []),
            ]
        schema_name = options.get("schema")
        completion = await _post_chat_with_schema(
            upstream,
            base,
            str(schema_name) if schema_name else None,
            http_client=http_client,
        )
        completion["docie_agent"] = {"agent": spec.name, "kind": spec.kind, "mode": mode}
        return completion

    # OCR (A) or OCR→LLM (B): reuse the gateway's solution adapters.
    profiles: dict[str, ModelProfile] = {}
    if mode == "ocr_extract":
        extractor_name = options.get("extractor")
        if not extractor_name:
            raise AgentError(
                f"agent {spec.name!r} is in ocr_extract mode but has no options.extractor",
                status_code=500,
                error_type="invalid_agent_config",
            )
        extractor = _resolve_backing(str(extractor_name))
        profiles[extractor.name] = extractor
        kind = "pipeline"
        solution_options: dict[str, Any] = {
            "ocr_backend": options.get("backend", "tesseract"),
            "language": options.get("language"),
            "extractor": extractor.name,
        }
        # OCR step may be a deployed VISION model (VLM as OCR: image -> text)
        # instead of a built-in backend — resolved like the extractor and passed
        # to the pipeline adapter, which then transcribes with it.
        ocr_model_sel = options.get("ocr_model")
        if ocr_model_sel:
            ocr_vision = _resolve_backing(str(ocr_model_sel))
            profiles[ocr_vision.name] = ocr_vision
            solution_options["ocr_model"] = ocr_vision.name
        # An optional schema makes the OCR→LLM extraction reliably structured —
        # the same grammar constraint the vision path uses, over OCR text.
        schema_name = options.get("schema")
        if schema_name:
            _inject_response_format(body, str(schema_name))
    else:
        kind = "ocr"
        solution_options = {
            "backend": options.get("backend", "tesseract"),
            "language": options.get("language"),
        }
    profile = ModelProfile(
        name=spec.name,
        model=spec.name,
        base_url="",
        api_key="local-not-used",
        kind=kind,
        options=solution_options,
    )
    try:
        solution = build_solution(profile, profiles=profiles, http_client=http_client)
        completion = await solution.complete(body)
    except SolutionError as exc:
        raise AgentError(
            exc.message, status_code=exc.status_code, error_type=exc.error_type
        ) from exc
    completion["docie_agent"] = {"agent": spec.name, "kind": spec.kind, "mode": mode}
    return completion


# ---------------------------------------------------------------------------
# Security proxy + custom agents — forward to the backing SLM.
# ---------------------------------------------------------------------------


async def _complete_proxy(
    spec: AgentSpec, body: dict[str, Any], *, http_client: httpx.AsyncClient
) -> dict[str, Any]:
    options = dict(spec.options)
    mode = str(options.get("mode", "placeholder"))
    if mode not in PROXY_MODES:
        raise AgentError(
            f"agent {spec.name!r} has invalid options.mode {mode!r} "
            f"(expected one of {', '.join(PROXY_MODES)})",
            status_code=500,
            error_type="invalid_agent_config",
        )
    entities = options.get("entities")
    if entities is not None and not isinstance(entities, list):
        raise AgentError(
            f"agent {spec.name!r} has invalid options.entities (expected a list)",
            status_code=500,
            error_type="invalid_agent_config",
        )
    analyze_fn, analyzer_label, guard_state = _build_analyzer(
        spec, options, entities, http_client=http_client
    )

    messages = body.get("messages") or []
    placeholders: dict[str, str] = {}
    detected_types: dict[str, int] = {}
    masked_messages: list[dict[str, Any]] = []
    for message in messages:
        masked_messages.append(
            await _mask_message(message, analyze_fn, placeholders, detected_types)
        )

    # Moderation verdicts (GLiNER2 guardrail tasks) block BEFORE PII does: an
    # unsafe/jailbreak prompt must never reach the model even fully anonymized.
    unsafe = moderation_flags(guard_state.get("moderation") or {})
    if mode == "block" and unsafe:
        raise AgentError(
            f"request blocked by agent {spec.name!r}: flagged by moderation "
            f"({', '.join(unsafe)})",
            status_code=400,
            error_type="unsafe_blocked",
        )
    if mode == "block" and placeholders:
        summary = ", ".join(f"{t}×{n}" for t, n in sorted(detected_types.items()))
        raise AgentError(
            f"request blocked by agent {spec.name!r}: detected personal data ({summary})",
            status_code=400,
            error_type="pii_blocked",
        )

    forward = dict(body)
    if mode == "placeholder":
        forward["messages"] = masked_messages
    completion = await _forward_chat(spec, forward, http_client=http_client)

    if options.get("restore_pii") and placeholders:
        _restore_completion(completion, placeholders)

    pii_report: dict[str, Any] = {
        "mode": mode,
        "analyzer": analyzer_label,
        "detected": sum(detected_types.values()),
        # Types + placeholders only — raw values never leave the process.
        "entities": [
            {"type": t, "count": n} for t, n in sorted(detected_types.items())
        ],
        "placeholders": sorted(placeholders) if mode == "placeholder" else [],
    }
    if guard_state.get("degraded"):
        # The guard failed mid-request and options.guard_fallback kicked in —
        # callers must be able to see the analysis ran at regex recall.
        pii_report["degraded_to_regex"] = True
    report: dict[str, Any] = {
        "agent": spec.name,
        "kind": spec.kind,
        "pii": pii_report,
    }
    moderation = guard_state.get("moderation")
    if moderation:
        report["moderation"] = {"verdicts": moderation, "flags": unsafe}
    completion["docie_agent"] = report
    return completion


AnalyzeFn = Callable[[str], Awaitable[list[pii.PiiEntity]]]


def _merge_moderation(state: dict[str, Any], verdicts: dict[str, Any]) -> None:
    """Fold one message's verdicts into the request-level state (worst wins)."""
    merged: dict[str, Any] = state.setdefault("moderation", {})
    for task, verdict in verdicts.items():
        current = merged.get(task)
        if isinstance(verdict, list):
            existing = current if isinstance(current, list) else []
            merged[task] = sorted({*map(str, existing), *map(str, verdict)})
        elif isinstance(verdict, str):
            if str(current).strip().lower() == "unsafe":
                continue  # an earlier message already flagged this task
            merged[task] = verdict


def _build_analyzer(
    spec: AgentSpec,
    options: dict[str, Any],
    entities: list[str] | None,
    *,
    http_client: httpx.AsyncClient,
) -> tuple[AnalyzeFn, str, dict[str, Any]]:
    """The proxy's analyzer: the guard encoder when configured, else regex.

    Fail-closed by design: a configured guard that errors ABORTS the request
    (502 ``guard_unavailable``) — a security proxy must never silently forward
    unmasked text because its analyzer died. ``guard_fallback: "regex"`` opts
    into degraded regex analysis instead, flagged in the response report.
    ``guard_tasks`` (GLiNER2 guardrail checkpoints) adds moderation verdicts,
    accumulated across the request's messages into the returned state.
    """
    guard_state: dict[str, Any] = {"degraded": False}
    guard_selector = options.get("guard_model")
    if not guard_selector:

        async def regex_analyze(text: str) -> list[pii.PiiEntity]:
            return pii.analyze(text, entities)

        return regex_analyze, "regex", guard_state

    guard_profile = _resolve_backing(str(guard_selector))
    labels_raw = options.get("guard_labels")
    if labels_raw is not None and not isinstance(labels_raw, list):
        raise AgentError(
            f"agent {spec.name!r} has invalid options.guard_labels (expected a list)",
            status_code=500,
            error_type="invalid_agent_config",
        )
    labels = (
        [str(label) for label in labels_raw]
        if labels_raw
        else labels_from_entities(entities)
    )
    threshold_raw = options.get("guard_threshold")
    try:
        threshold = float(threshold_raw) if threshold_raw is not None else None
    except (TypeError, ValueError):
        raise AgentError(
            f"agent {spec.name!r} has invalid options.guard_threshold (expected a number)",
            status_code=500,
            error_type="invalid_agent_config",
        ) from None
    tasks_raw = options.get("guard_tasks")
    if tasks_raw is not None and not isinstance(tasks_raw, list):
        raise AgentError(
            f"agent {spec.name!r} has invalid options.guard_tasks (expected a list)",
            status_code=500,
            error_type="invalid_agent_config",
        )
    tasks = [str(task) for task in tasks_raw] if tasks_raw else None
    fallback_to_regex = options.get("guard_fallback") == "regex"

    async def analyze(text: str) -> list[pii.PiiEntity]:
        try:
            result = await guard_analyze(
                text,
                guard=guard_profile,
                http_client=http_client,
                labels=labels,
                threshold=threshold,
                tasks=tasks,
            )
        except GuardAnalysisError as exc:
            if fallback_to_regex:
                guard_state["degraded"] = True
                return pii.analyze(text, entities)
            raise AgentError(
                exc.message, status_code=502, error_type="guard_unavailable"
            ) from exc
        if result.moderation:
            _merge_moderation(guard_state, result.moderation)
        return result.entities

    return analyze, f"guard:{guard_profile.name}", guard_state


async def _complete_custom(
    spec: AgentSpec, body: dict[str, Any], *, http_client: httpx.AsyncClient
) -> dict[str, Any]:
    completion = await _forward_chat(spec, dict(body), http_client=http_client)
    completion["docie_agent"] = {"agent": spec.name, "kind": spec.kind}
    return completion


def _resolve_backing(selector: str | None) -> ModelProfile:
    try:
        profile = resolve_extraction_profile(model_profile=selector)
    except ProfileResolutionError as exc:
        raise AgentError(str(exc), status_code=400, error_type="model_not_found") from exc
    if profile.kind != "passthrough":
        raise AgentError(
            f"backing profile {profile.name!r} is a {profile.kind!r} solution; "
            "agents forward to passthrough (OpenAI-compatible) upstreams only",
            status_code=500,
            error_type="invalid_agent_config",
        )
    return profile


async def _forward_chat(
    spec: AgentSpec, body: dict[str, Any], *, http_client: httpx.AsyncClient
) -> dict[str, Any]:
    upstream = _resolve_backing(spec.model_profile)
    if spec.system_prompt:
        body["messages"] = [
            {"role": "system", "content": spec.system_prompt},
            *(body.get("messages") or []),
        ]
    return await _post_chat(upstream, body, http_client=http_client)


async def _post_chat(
    upstream: ModelProfile, body: dict[str, Any], *, http_client: httpx.AsyncClient
) -> dict[str, Any]:
    """POST an OpenAI chat request to a resolved passthrough upstream. The
    caller owns message/system-prompt/response_format shaping; this just sets
    the model id, forces a non-streaming call, posts, and normalizes errors."""
    body["model"] = upstream.model
    # The API layer re-emits the final completion as a single SSE chunk for
    # streaming clients; upstream is always asked for a plain completion.
    body.pop("stream", None)
    url = f"{upstream.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {upstream.api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = await http_client.post(
            url, json=body, headers=headers, timeout=upstream.timeout_seconds
        )
    except httpx.RequestError as exc:
        raise AgentError(
            f"backing model upstream {upstream.base_url} is unreachable: {exc}",
            status_code=502,
            error_type="upstream_unavailable",
        ) from exc
    if response.status_code >= 400:
        raise AgentError(
            f"backing model returned {response.status_code}: {response.text[:300]}",
            status_code=response.status_code,
            error_type="upstream_error",
        )
    try:
        completion = response.json()
    except ValueError as exc:
        raise AgentError(
            "backing model returned a non-JSON response",
            status_code=502,
            error_type="upstream_error",
        ) from exc
    if not isinstance(completion, dict):
        raise AgentError(
            "backing model returned an unexpected payload shape",
            status_code=502,
            error_type="upstream_error",
        )
    return completion


# ---------------------------------------------------------------------------
# Message masking / restoring
# ---------------------------------------------------------------------------


async def _mask_text(
    text: str,
    analyze_fn: AnalyzeFn,
    placeholders: dict[str, str],
    detected_types: dict[str, int],
) -> str:
    found = await analyze_fn(text)
    for entity in found:
        detected_types[entity.type] = detected_types.get(entity.type, 0) + 1
    masked, _ = pii.anonymize(text, found, placeholders=placeholders)
    return masked


async def _mask_message(
    message: dict[str, Any],
    analyze_fn: AnalyzeFn,
    placeholders: dict[str, str],
    detected_types: dict[str, int],
) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        masked = await _mask_text(content, analyze_fn, placeholders, detected_types)
        return {**message, "content": masked}
    if isinstance(content, list):
        parts: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                masked = await _mask_text(
                    str(part.get("text", "")), analyze_fn, placeholders, detected_types
                )
                parts.append({**part, "text": masked})
            else:
                parts.append(part)
        return {**message, "content": parts}
    return message


def _restore_completion(completion: dict[str, Any], placeholders: dict[str, str]) -> None:
    for choice in completion.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = pii.deanonymize(message["content"], placeholders)
