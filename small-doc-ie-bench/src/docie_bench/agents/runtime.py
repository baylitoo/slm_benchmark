"""Execute one agent request: pre-process → backing solution/SLM → post-process.

The OpenAI chat-completion dict is the contract on both sides, so anything an
agents platform can send to a model it can send to an agent. The backing model
selector is resolved FRESH per request through the shared extraction resolver
(profile name / live deployment / ``store:<name>``), so an agent survives its
deployment being unloaded and reloaded on another port.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
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
from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_gateway import ModelGatewayError
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.profile_resolver import (
    ProfileResolutionError,
    resolve_extraction_profile,
)
from docie_bench.serving.solutions import (
    SolutionError,
    _extract_document,
    apply_no_think,
    build_solution,
)

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


def _resolve_extraction_schema(schema_name: str) -> tuple[str, dict[str, Any] | None]:
    """Return ExtractionService's schema mode and optional saved specification."""
    from docie_bench.schemas.extraction import schema_json
    from docie_bench.studio.dynamic_schemas import get_dynamic_schema

    try:
        schema_json(schema_name)
        return "static", None
    except ValueError as exc:
        saved = get_dynamic_schema(schema_name)
        if saved is None:
            raise AgentError(
                f"Unknown schema_name={schema_name!r}",
                status_code=400,
                error_type="invalid_request_error",
            ) from exc
        return "dynamic", saved["spec"]


def _flatten_agent_result(value: Any, *, root: bool = True) -> Any:
    """Restore the Agent endpoint's flat value contract after shared validation.

    Playground keeps evidence/confidence wrappers for review and audit. Agent
    consumers historically receive direct field values, so unwrap those fields
    only at the final API boundary while retaining the richer internal result.
    """
    if isinstance(value, list):
        return [_flatten_agent_result(item, root=False) for item in value]
    if not isinstance(value, dict):
        return value
    if "value" in value and set(value) <= {"value", "evidence_ids", "confidence"}:
        return _flatten_agent_result(value.get("value"), root=False)
    omitted = {"evidence_ids", "confidence"}
    if root:
        omitted.update({"document_type", "extraction_notes"})
    return {
        key: _flatten_agent_result(item, root=False)
        for key, item in value.items()
        if key not in omitted
    }


async def _complete_structured_document(
    *,
    spec: AgentSpec,
    body: dict[str, Any],
    profile: ModelProfile,
    profiles: dict[str, ModelProfile],
    schema_name: str,
    ocr_backend_name: str,
    language: str | None,
    disable_thinking: bool,
    mode: str,
    output_model: str,
) -> dict[str, Any]:
    """Run an Agent document through the same extraction path as Playground.

    This preserves OCR blocks and their evidence ids, uses the shared prompt and
    response-format negotiation, and performs the same grounding/validation pass
    before adapting the result back to an OpenAI chat completion.
    """
    schema_mode, dynamic_schema = _resolve_extraction_schema(schema_name)
    raw, suffix = _extract_document(body)
    with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(raw)
        path = Path(handle.name)
    try:
        response = await ExtractionService(
            profile,
            profiles=profiles,
            disable_thinking=disable_thinking,
        ).extract_from_file(
            path=path,
            ocr_backend_name=ocr_backend_name,
            schema_name=schema_name,
            schema_mode=schema_mode,
            dynamic_schema=dynamic_schema,
            language=language,
            metadata={"agent": spec.name},
        )
    except ModelGatewayError as exc:
        raise AgentError(
            str(exc),
            status_code=exc.status_code or 502,
            error_type="upstream_error",
        ) from exc
    except ValueError as exc:
        raise AgentError(
            str(exc), status_code=400, error_type="invalid_request_error"
        ) from exc
    finally:
        path.unlink(missing_ok=True)

    usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
    usage.setdefault("prompt_tokens", 0)
    usage.setdefault("completion_tokens", 0)
    usage.setdefault("total_tokens", 0)
    agent_result = _flatten_agent_result(response.result)
    return {
        "id": f"chatcmpl-agent-{response.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": output_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(agent_result, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "docie_agent": {
            "agent": spec.name,
            "kind": spec.kind,
            "mode": mode,
            "validation": response.validation.model_dump(),
            "response_format_style": response.response_format_style,
        },
    }


async def _complete_ocr(
    spec: AgentSpec, body: dict[str, Any], *, http_client: httpx.AsyncClient
) -> dict[str, Any]:
    options = dict(spec.options)
    mode = _resolve_ocr_mode(options)

    # Schema-backed vision extraction uses the exact Playground pipeline:
    # document ingestion -> shared prompts/client -> grounding/validation.
    if mode == "vision":
        vision_selector = options.get("vision_model")
        if not vision_selector:
            raise AgentError(
                f"agent {spec.name!r} is in vision mode but has no options.vision_model",
                status_code=500,
                error_type="invalid_agent_config",
            )
        upstream = _resolve_backing(str(vision_selector))
        schema_name = options.get("schema")
        if schema_name:
            return await _complete_structured_document(
                spec=spec,
                body=body,
                profile=upstream,
                profiles={upstream.name: upstream},
                schema_name=str(schema_name),
                ocr_backend_name="vision",
                language=options.get("language"),
                disable_thinking=bool(options.get("no_think")),
                mode=mode,
                output_model=upstream.model,
            )
        base = dict(body)
        if spec.system_prompt:
            base["messages"] = [
                {"role": "system", "content": spec.system_prompt},
                *(base.get("messages") or []),
            ]
        if options.get("no_think"):
            apply_no_think(base)
        completion = await _post_chat(upstream, base, http_client=http_client)
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
            "no_think": bool(options.get("no_think")),
        }
        # OCR step may be a deployed VISION model (VLM as OCR: image -> text)
        # instead of a built-in backend — resolved like the extractor and passed
        # to the pipeline adapter, which then transcribes with it.
        ocr_model_sel = options.get("ocr_model")
        if ocr_model_sel:
            ocr_vision = _resolve_backing(str(ocr_model_sel))
            profiles[ocr_vision.name] = ocr_vision
            solution_options["ocr_model"] = ocr_vision.name
        # A schema selects the shared Playground extraction path. Schema-less
        # agents retain the generic OpenAI-compatible pipeline adapter below.
        schema_name = options.get("schema")
        if schema_name:
            pipeline_profile = ModelProfile(
                name=spec.name,
                model=spec.name,
                base_url="",
                api_key="local-not-used",
                kind=kind,
                options=solution_options,
            )
            return await _complete_structured_document(
                spec=spec,
                body=body,
                profile=pipeline_profile,
                profiles=profiles,
                schema_name=str(schema_name),
                ocr_backend_name=str(options.get("backend", "tesseract")),
                language=options.get("language"),
                disable_thinking=bool(options.get("no_think")),
                mode=mode,
                output_model=extractor.model,
            )
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
