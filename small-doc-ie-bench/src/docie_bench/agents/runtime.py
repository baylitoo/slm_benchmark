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
from docie_bench.benchmark.routing_config import build_extraction_router
from docie_bench.extract.routing import (
    ExtractionRouter,
    RoutingPolicy,
    RoutingResult,
    live_routing_audit,
)
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
from docie_bench.studio.routing_policies import (
    RoutingPolicyUnavailableError,
    get_routing_policy,
)

PROXY_MODES = ("placeholder", "block", "detect")

# A tool's arguments/result can be arbitrarily large (a document, an OCR
# dump) -- the "Try it" trace (#262) rides the live completion response, so
# cap each field rather than let one tool call balloon the payload.
_TRACE_TEXT_LIMIT = 4000


def _stringify_tool_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _truncate_trace_text(text: str) -> str:
    if len(text) <= _TRACE_TEXT_LIMIT:
        return text
    return text[:_TRACE_TEXT_LIMIT] + f"… ({len(text)} chars total)"


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
    if spec.kind == "workflow":
        return await _complete_workflow(spec, body, http_client=http_client)
    return await _complete_custom(spec, body, http_client=http_client)


# ---------------------------------------------------------------------------
# OCR agent — reuse the gateway's solution adapters.
# ---------------------------------------------------------------------------


OCR_MODES = ("ocr", "ocr_extract", "vision")

# ``options.extractor`` selector convention: ``policy:<name>`` runs a SAVED
# routing policy (POST /v1/studio/routing-policies) as the extraction step
# instead of a single model — the same registry the live extract routes and
# the Benchmark tab resolve by name.
_POLICY_PREFIX = "policy:"


def _resolve_ocr_mode(options: dict[str, Any]) -> str:
    """The staged mode, deriving it for agents saved before ``mode`` existed:
    an ``extractor`` present means the OCR→LLM pipeline, otherwise plain OCR."""
    mode = options.get("mode")
    if mode in OCR_MODES:
        return str(mode)
    return "ocr_extract" if options.get("extractor") else "ocr"


def _generation_max_tokens(body: dict[str, Any], options: dict[str, Any]) -> int | None:
    """Resolve request > agent > deployment/profile generation precedence."""
    raw = body.get("max_tokens", options.get("max_tokens"))
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or not 1 <= raw <= 131_072:
        raise AgentError(
            "max_tokens must be an integer between 1 and 131072",
            status_code=400,
            error_type="invalid_request_error",
        )
    return raw


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
    profile: ModelProfile | None = None,
    profiles: dict[str, ModelProfile] | None = None,
    schema_name: str,
    ocr_backend_name: str,
    language: str | None,
    disable_thinking: bool,
    mode: str,
    output_model: str,
    max_tokens: int | None,
    executor: ExtractionService | ExtractionRouter | None = None,
    policy_name: str | None = None,
) -> dict[str, Any]:
    """Run an Agent document through the same extraction path as Playground.

    This preserves OCR blocks and their evidence ids, uses the shared prompt and
    response-format negotiation, and performs the same grounding/validation pass
    before adapting the result back to an OpenAI chat completion.

    ``executor`` is anything exposing ``extract_from_file`` — when ``None``
    (every single-model caller), an ``ExtractionService`` is built from
    ``profile``/``profiles`` exactly as before. A ``policy:<name>`` extractor
    passes a prebuilt ``ExtractionRouter`` plus ``policy_name`` instead; its
    ``RoutingResult`` outcome is unwrapped here and the sanitized
    ``live_routing_audit`` (per-stage ``output`` stripped — a live surface must
    not leak the losing stages' extraction of a confidential document) rides
    the completion's ``docie_agent.routing``. Router caveats: each stage
    re-runs OCR through its own ``extract_from_file``, which the
    content-addressed OCR cache absorbs — no special handling needed; and
    ``disable_thinking``/``max_tokens`` do NOT reach router stages, because
    ``build_extraction_router`` constructs each stage's service without
    threading them (documented limitation, not hacked around here).
    """
    schema_mode, dynamic_schema = _resolve_extraction_schema(schema_name)
    raw, suffix = _extract_document(body)
    with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(raw)
        path = Path(handle.name)
    if executor is None:
        assert profile is not None, "either an executor or a profile is required"
        executor = ExtractionService(
            profile,
            profiles=profiles or {},
            disable_thinking=disable_thinking,
            max_tokens=max_tokens,
        )
    try:
        outcome = await executor.extract_from_file(
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

    routing_audit: dict[str, Any] | None = None
    if isinstance(outcome, RoutingResult):
        assert policy_name is not None, "a RoutingResult implies a policy_name"
        if outcome.response is None:
            # Every stage errored or the budget ran out before any stage
            # answered — an honest upstream failure with the router's own
            # terminal reason, not a 500.
            raise AgentError(
                f"routing policy {policy_name!r} produced no extraction: "
                f"{outcome.audit.terminal_reason} "
                f"(decision={outcome.audit.terminal_decision.value}, "
                f"attempts={outcome.audit.attempts})",
                status_code=502,
                error_type="upstream_error",
            )
        routing_audit = live_routing_audit(outcome, policy_name=policy_name)
        response = outcome.response
    else:
        response = outcome

    usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
    usage.setdefault("prompt_tokens", 0)
    usage.setdefault("completion_tokens", 0)
    usage.setdefault("total_tokens", 0)
    agent_result = _flatten_agent_result(response.result)
    docie_agent: dict[str, Any] = {
        "agent": spec.name,
        "kind": spec.kind,
        "mode": mode,
        "validation": response.validation.model_dump(),
        "response_format_style": response.response_format_style,
    }
    if routing_audit is not None:
        docie_agent["routing"] = routing_audit
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
        "docie_agent": docie_agent,
    }


async def _complete_ocr(
    spec: AgentSpec, body: dict[str, Any], *, http_client: httpx.AsyncClient
) -> dict[str, Any]:
    options = dict(spec.options)
    mode = _resolve_ocr_mode(options)
    max_tokens = _generation_max_tokens(body, options)

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
                max_tokens=max_tokens,
            )
        base = dict(body)
        if max_tokens is not None:
            base.setdefault("max_tokens", max_tokens)
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
        extractor_selector = str(extractor_name)
        # ``policy:<name>``: the extraction step is a SAVED routing policy —
        # a confidence-gated cascade across model profiles — instead of one
        # model. Runs through the shared structured path with a prebuilt
        # router as the executor.
        if extractor_selector.startswith(_POLICY_PREFIX):
            policy_name = extractor_selector[len(_POLICY_PREFIX):]
            schema_name = options.get("schema")
            if not schema_name:
                raise AgentError(
                    f"agent {spec.name!r} uses extractor {extractor_selector!r} but has "
                    "no options.schema — a routing-policy extractor runs the "
                    "structured extraction path, which needs an output schema",
                    status_code=500,
                    error_type="invalid_agent_config",
                )
            router = _build_policy_router(spec, policy_name)
            return await _complete_structured_document(
                spec=spec,
                body=body,
                schema_name=str(schema_name),
                ocr_backend_name=str(options.get("backend", "tesseract")),
                language=options.get("language"),
                disable_thinking=bool(options.get("no_think")),
                mode=mode,
                output_model=extractor_selector,
                max_tokens=max_tokens,
                executor=router,
                policy_name=policy_name,
            )
        extractor = _resolve_backing(extractor_selector)
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
                max_tokens=max_tokens,
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
    generation_body = dict(body)
    if max_tokens is not None:
        generation_body.setdefault("max_tokens", max_tokens)
    try:
        solution = build_solution(profile, profiles=profiles, http_client=http_client)
        completion = await solution.complete(generation_body)
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


def _resolve_mcp_servers(spec: AgentSpec) -> list[str]:
    """``options.mcp_servers`` — registry server names this agent may use as
    tool sources (registry-only, see mcp_tools.py: never a caller-supplied
    URL/command). Empty/absent means no tools — the existing bare-forward
    behavior, unchanged for every agent that doesn't opt in."""
    raw = spec.options.get("mcp_servers")
    if raw in (None, []):
        return []
    if not isinstance(raw, list) or not all(isinstance(n, str) and n for n in raw):
        raise AgentError(
            f"agent {spec.name!r}: options.mcp_servers must be a list of registered "
            "MCP server names",
            status_code=500,
            error_type="invalid_agent_config",
        )
    return raw


def _resolve_tool_allowlist(
    spec: AgentSpec,
    server_names: list[str],
    mapping: dict[str, tuple[str, str]],
    *,
    separator: str,
) -> set[str] | None:
    """``options.mcp_tools`` — an optional ``{server_name: [tool_name, ...]}``
    restricting which of a server's tools this agent may use, instead of
    every tool the server happens to expose. ``None`` means unrestricted
    (every listed tool allowed) — the default for an agent that only set
    ``mcp_servers``. Validated against the SERVER's own live tool list
    (``mapping``, from ``collect_openai_tools``) so a stale or typo'd tool
    name is a clear config error, not a silent no-op.
    """
    raw = spec.options.get("mcp_tools")
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise AgentError(
            f"agent {spec.name!r}: options.mcp_tools must be an object of "
            "{server_name: [tool_name, ...]}",
            status_code=500,
            error_type="invalid_agent_config",
        )
    allowed: set[str] = set()
    for server_name, tool_names in raw.items():
        if server_name not in server_names:
            raise AgentError(
                f"agent {spec.name!r}: options.mcp_tools names server {server_name!r}, "
                "which is not in options.mcp_servers",
                status_code=500,
                error_type="invalid_agent_config",
            )
        if not isinstance(tool_names, list) or not all(
            isinstance(t, str) and t for t in tool_names
        ):
            raise AgentError(
                f"agent {spec.name!r}: options.mcp_tools[{server_name!r}] must be a "
                "list of tool names",
                status_code=500,
                error_type="invalid_agent_config",
            )
        for tool_name in tool_names:
            qualified = f"{server_name}{separator}{tool_name}"
            if qualified not in mapping:
                raise AgentError(
                    f"agent {spec.name!r}: options.mcp_tools names tool {tool_name!r} "
                    f"on server {server_name!r}, which does not exist on that server",
                    status_code=500,
                    error_type="invalid_agent_config",
                )
            allowed.add(qualified)
    return allowed


async def _complete_custom(
    spec: AgentSpec, body: dict[str, Any], *, http_client: httpx.AsyncClient
) -> dict[str, Any]:
    server_names = _resolve_mcp_servers(spec)
    docie_agent: dict[str, Any] = {"agent": spec.name, "kind": spec.kind}
    if server_names:
        completion, tool_calls = await _complete_with_tools(
            spec, dict(body), server_names, http_client=http_client
        )
        if tool_calls:
            docie_agent["tool_calls"] = tool_calls
    else:
        completion = await _forward_chat(spec, dict(body), http_client=http_client)
    completion["docie_agent"] = docie_agent
    return completion


async def _complete_with_tools(
    spec: AgentSpec,
    body: dict[str, Any],
    server_names: list[str],
    *,
    http_client: httpx.AsyncClient,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A ``custom`` agent with ``options.mcp_servers`` set: connect, advertise
    the servers' tools, and run the same bounded model<->tools loop the
    generic chat surface uses (``chat_api._chat_with_mcp_tools``) — reused
    here via ``mcp_tools.run_tool_loop`` rather than duplicated, so both
    surfaces share one loop implementation and one error taxonomy.

    Returns ``(completion, tool_call_trace)`` — the trace (#261/#262) is
    every tool call this request actually executed, in order, each entry
    ``{"tool": qualified_name, "status": "ok"|"error", "latency_ms": int,
    "arguments": str, "result": str}`` (the latter two truncated at
    ``_TRACE_TEXT_LIMIT``). Empty when the model never called a tool.
    """
    from contextlib import AsyncExitStack

    from docie_bench import mcp_tools as mcp_mod
    from docie_bench.settings import get_settings

    upstream = _resolve_backing(spec.model_profile)
    try:
        registry = mcp_mod.load_mcp_registry()
    except mcp_mod.MCPConfigError as exc:
        raise AgentError(str(exc), status_code=500, error_type="mcp_config_error") from exc
    unknown = [name for name in server_names if name not in registry]
    if unknown:
        raise AgentError(
            f"agent {spec.name!r} references unregistered MCP server(s): "
            f"{', '.join(unknown)} — register them first (see GET /v1/mcp/servers)",
            status_code=400,
            error_type="mcp_server_not_registered",
        )
    try:
        mcp_mod._require_mcp()
    except mcp_mod.MCPUnavailableError as exc:
        raise AgentError(str(exc), status_code=501, error_type="mcp_unavailable") from exc
    specs = [registry[name] for name in server_names]

    forward = dict(body)
    if spec.system_prompt:
        forward["messages"] = [
            {"role": "system", "content": spec.system_prompt},
            *(forward.get("messages") or []),
        ]

    async def post(round_body: dict[str, Any]) -> dict[str, Any]:
        return await _post_chat(upstream, dict(round_body), http_client=http_client)

    tool_call_trace: list[dict[str, Any]] = []

    def record_tool_call(
        name: str, ok: bool, latency_ms: int, arguments: Any, result: str
    ) -> None:
        tool_call_trace.append(
            {
                "tool": name,
                "status": "ok" if ok else "error",
                "latency_ms": latency_ms,
                "arguments": _truncate_trace_text(_stringify_tool_value(arguments)),
                "result": _truncate_trace_text(result),
            }
        )

    # A config error (a bad allowlist) is captured rather than raised INSIDE
    # the AsyncExitStack body: raising there would propagate the exception
    # through the MCP session/task-group teardown machinery, which can
    # itself raise during unwind (anyio cancellation) and shadow the
    # original AgentError with a generic 502 -- confirmed empirically, not
    # theoretical. Deferring the raise to after the stack closes cleanly
    # sidesteps that race entirely.
    config_error: AgentError | None = None
    completion: dict[str, Any] | None = None
    try:
        async with AsyncExitStack() as stack:
            try:
                sessions = await mcp_mod.open_mcp_sessions(stack, specs)
                tools, mapping = await mcp_mod.collect_openai_tools(sessions)
            except Exception as exc:  # noqa: BLE001 - connect/handshake failure is a gateway error
                raise AgentError(
                    f"could not connect to MCP server(s): {exc}",
                    status_code=502,
                    error_type="mcp_server_unreachable",
                ) from exc
            try:
                allowed = _resolve_tool_allowlist(
                    spec, server_names, mapping, separator=mcp_mod.TOOL_SEPARATOR
                )
            except AgentError as exc:
                config_error = exc
            else:
                if allowed is not None:
                    # Filter BOTH the advertised list and the routing map: a
                    # disallowed tool must never be advertised to the model,
                    # and even a hallucinated call to a real-but-disallowed
                    # name must fall through run_tool_loop's "not all_mcp"
                    # branch (returned to the caller, never executed) rather
                    # than being honored.
                    tools = [t for t in tools if t["function"]["name"] in allowed]
                    mapping = {
                        name: target for name, target in mapping.items() if name in allowed
                    }
                completion = await mcp_mod.run_tool_loop(
                    post, forward, sessions, mapping, tools, on_tool_call=record_tool_call
                )
    except AgentError:
        raise
    except Exception as exc:  # noqa: BLE001 - transport teardown (ExitStack unwind) failure
        raise AgentError(
            f"MCP session error: {exc}", status_code=502, error_type="mcp_server_unreachable"
        ) from exc
    if config_error is not None:
        raise config_error
    if completion is None:
        raise AgentError(
            f"model kept calling tools for {get_settings().mcp_max_tool_iterations} rounds "
            "without a final answer",
            status_code=502,
            error_type="mcp_tool_loop_exhausted",
        )
    # run_tool_loop's only non-dict, non-None return is whatever `post` itself
    # returned as an error object -- but _post_chat never returns one, it
    # raises AgentError instead (caught above). A dict is the only remaining
    # possibility here.
    assert isinstance(completion, dict)
    return completion, tool_call_trace


def _resolve_workflow_steps(spec: AgentSpec) -> list[dict[str, Any]]:
    """``options.steps`` — a non-empty, ORDERED list of
    ``{model_profile, system_prompt?, mcp_servers?, mcp_tools?}`` steps
    (#265). Each step's shape mirrors a ``custom`` agent's own
    model_profile/options exactly, so the SAME per-step spec can drive
    ``_resolve_mcp_servers``/``_complete_with_tools``/``_forward_chat``
    unchanged (see ``_complete_workflow``)."""
    raw = spec.options.get("steps")
    if not isinstance(raw, list) or not raw:
        raise AgentError(
            f"agent {spec.name!r}: options.steps must be a non-empty list of "
            "{model_profile, system_prompt?, mcp_servers?, mcp_tools?} steps",
            status_code=500,
            error_type="invalid_agent_config",
        )
    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw):
        if not isinstance(raw_step, dict) or not raw_step.get("model_profile"):
            raise AgentError(
                f"agent {spec.name!r}: options.steps[{index}] must be an object "
                "with a non-empty model_profile",
                status_code=500,
                error_type="invalid_agent_config",
            )
        steps.append(raw_step)
    return steps


def _message_content(completion: dict[str, Any]) -> str | None:
    choices = completion.get("choices")
    message = (
        choices[0].get("message")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else None
    )
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else None


async def _complete_workflow(
    spec: AgentSpec, body: dict[str, Any], *, http_client: httpx.AsyncClient
) -> dict[str, Any]:
    """A ``workflow`` agent (#265): a fixed, ORDERED sequence of steps, each
    its own model_profile + prompt (+ optional MCP tools from #259), run
    server-side in one request — the "prompt chaining" pattern, deliberately
    narrow so a small (350M-class) model handles one well-scoped sub-task
    per step instead of the whole request in one shot.

    Step 1 receives the caller's own messages unchanged; each LATER step
    receives only the PREVIOUS step's answer as its single user message —
    not the accumulated history, not the original request. Usage sums
    across every step (same contract the MCP tool loop already uses); every
    step's own model/content and any tool calls it made ride
    ``docie_agent.steps``/``docie_agent.tool_calls`` for the "Try it" trace
    view (#262).
    """
    steps = _resolve_workflow_steps(spec)
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    step_trace: list[dict[str, Any]] = []
    tool_call_trace: list[dict[str, Any]] = []
    previous_content: str | None = None
    completion: dict[str, Any] | None = None
    for index, step in enumerate(steps):
        step_spec = spec.model_copy(
            update={
                "model_profile": step["model_profile"],
                "system_prompt": step.get("system_prompt"),
                "options": {
                    "mcp_servers": step.get("mcp_servers"),
                    "mcp_tools": step.get("mcp_tools"),
                },
            }
        )
        step_body = (
            dict(body)
            if index == 0
            else {"messages": [{"role": "user", "content": previous_content or ""}]}
        )
        server_names = _resolve_mcp_servers(step_spec)
        if server_names:
            completion, step_tool_calls = await _complete_with_tools(
                step_spec, step_body, server_names, http_client=http_client
            )
            tool_call_trace.extend(step_tool_calls)
        else:
            completion = await _forward_chat(step_spec, step_body, http_client=http_client)
        usage = completion.get("usage")
        if isinstance(usage, dict):
            for key in totals:
                value = usage.get(key)
                if isinstance(value, int):
                    totals[key] += value
        previous_content = _message_content(completion)
        step_trace.append(
            {"step": index, "model_profile": step["model_profile"], "content": previous_content}
        )
    assert completion is not None  # _resolve_workflow_steps guarantees >=1 step
    docie_agent: dict[str, Any] = {"agent": spec.name, "kind": spec.kind, "steps": step_trace}
    if tool_call_trace:
        docie_agent["tool_calls"] = tool_call_trace
    final = dict(completion)
    final["usage"] = totals
    final["docie_agent"] = docie_agent
    return final


def _build_policy_router(spec: AgentSpec, policy_name: str) -> ExtractionRouter:
    """A saved routing policy as this agent's extraction executor.

    Every stage profile is resolved UP FRONT through ``_resolve_backing`` —
    the same reasoning as api.py's ``resolve_extraction_executor``: the
    passthrough-kind guard applies to every stage, and a policy whose cold
    escalation target can't resolve fails at request time, not mid-route on
    the one document that finally needed it.
    """
    try:
        record = get_routing_policy(policy_name)
    except RoutingPolicyUnavailableError as exc:
        raise AgentError(
            str(exc), status_code=503, error_type="routing_policy_unavailable"
        ) from exc
    if record is None:
        raise AgentError(
            f"agent {spec.name!r} references unknown routing policy "
            f"{policy_name!r} — save one via POST /v1/studio/routing-policies "
            "(or pick it in the Studio)",
            status_code=400,
            error_type="invalid_agent_config",
        )
    policy = RoutingPolicy.model_validate(record["policy"])
    profiles = {stage.name: _resolve_backing(stage.name) for stage in policy.stages}
    return build_extraction_router(policy, profiles)


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
