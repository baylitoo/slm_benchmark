"""MCP servers as tool sources for served chat models.

A chat request to ``POST /v1/chat/completions`` can name registered MCP
servers (``"mcp_servers": ["calculator"]``); the serving side then runs the
whole agentic exchange that an OpenAI-tools caller would otherwise have to
drive by hand:

1. connect to each named server and ``list_tools`` — every MCP tool carries a
   JSON Schema (``input_schema``) that maps 1:1 onto an OpenAI function
   schema, so the conversion is mechanical;
2. advertise those functions to the model through the standard ``tools``
   field (llama-server renders them via ``--jinja`` — the ``tools`` family
   capability);
3. when the model answers with ``tool_calls``, execute each against its MCP
   server, append the results as ``role: "tool"`` messages, and re-ask;
4. loop (bounded) until the model produces a plain answer, which is returned
   as an ordinary chat completion with usage summed across every round.

Security model: only servers named in the registry file
(``settings.mcp_servers_config``) are reachable — a caller picks servers BY
NAME and can never supply its own URL or command line through the API. The
``mcp`` SDK is an optional dependency (``pip install docie-bench[mcp]``);
without it, requests that ask for MCP tools get a clear 501 instead of an
ImportError, and everything else is untouched.

Caller-owned tools compose: if the request also carries its own ``tools``,
both sets are advertised together, and the moment the model calls ANY
caller-owned tool the completion is returned as-is — executing a function the
caller (not this process) implements is the caller's job. Only rounds whose
tool calls are ALL MCP-owned are executed server-side.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docie_bench.settings import get_settings

if TYPE_CHECKING:
    from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)

# Qualifier between server and tool name in the advertised function name
# ("calculator__add"): two servers exporting the same tool name stay
# distinguishable, and a returned tool_call routes back to the right server
# without any extra bookkeeping in the completion itself.
TOOL_SEPARATOR = "__"

_TRANSPORTS = ("streamable-http", "stdio")


class MCPConfigError(ValueError):
    """The MCP server registry file is missing, malformed, or names an
    unknown transport."""


class MCPUnavailableError(RuntimeError):
    """The optional ``mcp`` SDK is not installed in this environment."""


class AskUserTimeoutError(RuntimeError):
    """A paused exchange (``ask_user``, or a user-initiated pause) got no
    answer within ``settings.mcp_ask_user_timeout_seconds`` — a closed tab
    or an abandoned session must not hold the request (or the in-memory
    registry entry below) open forever."""


@dataclass(frozen=True)
class MCPServerSpec:
    """One registry entry: how to reach a named MCP server.

    ``streamable-http`` talks to a remote server at ``url`` (optionally with
    auth ``headers``); ``stdio`` spawns ``command`` locally and speaks over
    its pipes. Both come exclusively from the operator-owned registry file —
    never from request bodies.
    """

    name: str
    transport: str
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    command: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


def _require_mcp() -> None:
    try:
        import mcp  # noqa: F401 - availability probe only
    except ImportError as exc:
        raise MCPUnavailableError(
            "MCP tool support needs the optional 'mcp' package — "
            "install with: pip install 'docie-bench[mcp]'"
        ) from exc


def load_mcp_registry(path: Path | None = None) -> dict[str, MCPServerSpec]:
    """Parse the registry file into named server specs.

    A missing file is an EMPTY registry, not an error — MCP support is
    opt-in, and the useful failure ("server 'x' is not registered") happens
    at request time with the names the caller actually asked for.
    """
    config_path = path if path is not None else get_settings().mcp_servers_config
    if not config_path.exists():
        return {}
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise MCPConfigError(f"{config_path} is not valid JSON: {exc}") from exc
    servers = raw.get("servers") if isinstance(raw, dict) else None
    if not isinstance(servers, dict):
        raise MCPConfigError(f"{config_path} must be an object with a 'servers' object")
    registry: dict[str, MCPServerSpec] = {}
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            raise MCPConfigError(f"server {name!r}: entry must be an object")
        transport = entry.get("transport")
        if transport not in _TRANSPORTS:
            raise MCPConfigError(
                f"server {name!r}: 'transport' must be one of {', '.join(_TRANSPORTS)}"
            )
        url = entry.get("url")
        command = entry.get("command")
        if transport == "streamable-http" and not (isinstance(url, str) and url):
            raise MCPConfigError(f"server {name!r}: streamable-http needs a 'url'")
        if transport == "stdio" and not (isinstance(command, list) and command):
            raise MCPConfigError(f"server {name!r}: stdio needs a non-empty 'command' list")
        registry[str(name)] = MCPServerSpec(
            name=str(name),
            transport=str(transport),
            url=str(url) if isinstance(url, str) else None,
            headers=dict(entry.get("headers") or {}),
            command=tuple(str(part) for part in (command or [])),
            env=dict(entry.get("env") or {}),
        )
    return registry


def save_registry_entry(name: str, record: dict[str, Any], path: Path | None = None) -> None:
    """Atomic read-modify-write of one server entry; preserves hand-written entries."""
    config_path = path if path is not None else get_settings().mcp_servers_config
    raw: dict[str, Any] = {"servers": {}}
    if config_path.exists():
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict) and isinstance(parsed.get("servers"), dict):
            raw = parsed
    raw.setdefault("servers", {})[name] = record
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = config_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(config_path)


def remove_registry_entry(name: str, path: Path | None = None) -> bool:
    config_path = path if path is not None else get_settings().mcp_servers_config
    if not config_path.exists():
        return False
    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    servers = parsed.get("servers") if isinstance(parsed, dict) else None
    if not isinstance(servers, dict) or name not in servers:
        return False
    del servers[name]
    tmp = config_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(config_path)
    return True


async def open_mcp_sessions(
    stack: AsyncExitStack, specs: list[MCPServerSpec]
) -> dict[str, ClientSession]:
    """Connect + initialize a ClientSession per spec, all owned by ``stack``.

    Sessions live for one chat request: connection setup rides the request's
    latency budget, and there is no pooled-session lifecycle to invalidate
    when an operator edits the registry file.
    """
    _require_mcp()
    from mcp.client.session import ClientSession

    settings = get_settings()
    sessions: dict[str, ClientSession] = {}
    for spec in specs:
        if spec.transport == "streamable-http":
            from mcp.client.streamable_http import (  # type: ignore[attr-defined]
                create_mcp_http_client,
                streamable_http_client,
            )

            http_client = (
                create_mcp_http_client(headers=dict(spec.headers)) if spec.headers else None
            )
            read, write = await stack.enter_async_context(
                streamable_http_client(str(spec.url), http_client=http_client)
            )
        else:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=spec.command[0],
                args=list(spec.command[1:]),
                env=dict(spec.env) or None,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(
            ClientSession(
                read,
                write,
                read_timeout_seconds=settings.mcp_tool_timeout_seconds,
            )
        )
        await session.initialize()
        sessions[spec.name] = session
    return sessions


async def collect_openai_tools(
    sessions: Mapping[str, ClientSession],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    """``list_tools`` every session and convert to OpenAI function schemas.

    Returns the advertisable ``tools`` array plus the routing map
    ``qualified name -> (server, tool)`` the executor needs to send a
    returned tool_call back to the right server.
    """
    tools: list[dict[str, Any]] = []
    mapping: dict[str, tuple[str, str]] = {}
    for server_name, session in sessions.items():
        listed = await session.list_tools()
        for tool in listed.tools:
            qualified = f"{server_name}{TOOL_SEPARATOR}{tool.name}"
            mapping[qualified] = (server_name, tool.name)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": qualified,
                        "description": tool.description or "",
                        # MCP input_schema IS JSON Schema — the exact contract
                        # OpenAI 'parameters' expects. Passed through verbatim.
                        "parameters": tool.input_schema
                        or {"type": "object", "properties": {}},
                    },
                }
            )
    return tools, mapping


async def execute_tool_call(
    sessions: Mapping[str, ClientSession],
    mapping: Mapping[str, tuple[str, str]],
    qualified_name: str,
    arguments: Any,
) -> str:
    """Run one model-emitted tool call against its MCP server.

    Always returns TEXT for the ``role: "tool"`` message — including on
    failure. A tool error is information the model should reason about
    ("error: ...") on the next round, not an exception that kills the chat
    request after the upstream already burned tokens on it.
    """
    server_name, tool_name = mapping[qualified_name]
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return f"error: tool arguments were not valid JSON: {arguments[:200]}"
    else:
        parsed = dict(arguments or {})
    if not isinstance(parsed, dict):
        return f"error: tool arguments must be a JSON object, got: {str(parsed)[:200]}"
    try:
        result = await sessions[server_name].call_tool(tool_name, parsed)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as text, never raised
        logger.warning("MCP tool %s failed: %s", qualified_name, exc)
        return f"error: MCP tool call failed: {exc}"
    parts: list[str] = []
    for content in result.content:
        text = getattr(content, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(json.dumps(content.model_dump(exclude_none=True), ensure_ascii=False))
    text_out = "\n".join(parts)
    if result.is_error:
        return f"error: {text_out or 'tool reported an error with no message'}"
    return text_out


def _accumulate_usage(totals: dict[str, int], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            totals[key] += value


# A tool's arguments/result can be arbitrarily large (a document, an OCR
# dump) -- the trace rides the live completion response, so cap each field
# rather than let one tool call balloon the payload. 4000 clipped routine
# docs-search results (a read_document on a multi-page PDF, a search_text
# with many hits) in practice (#284); 20000 covers those while still
# bounding a truly pathological result.
TRACE_TEXT_LIMIT = 20_000


def _stringify_tool_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _truncate_trace_text(text: str) -> str:
    if len(text) <= TRACE_TEXT_LIMIT:
        return text
    return text[:TRACE_TEXT_LIMIT] + f"… ({len(text)} chars total)"


# Generic across every MCP server, not docs-search-specific: any server that
# exposes a "list X" tool alongside tools that take one of X's identifiers as
# an argument has the same failure mode -- a small model skips the listing
# call, or invents a plausible-looking id instead of using one it actually
# saw. A per-tool error message that includes the real listing (see
# docs_search.resolve_document) helps a model self-correct AFTER a bad call,
# but doesn't stop the first one; this directive is the upfront half of that
# same fix.
TOOL_DISCIPLINE_DIRECTIVE = (
    "Tool discipline: when a tool takes an identifier (a path, id, or name) "
    "that another tool lists, call the listing tool first and use one of its "
    "results EXACTLY as returned. Never invent, guess, or construct an "
    "identifier from other context. If the listing has nothing relevant, say "
    "so directly instead of guessing."
)

# #391: the round that would otherwise exhaust mcp_max_tool_iterations gets
# its tools stripped (see run_tool_loop) instead of being allowed to call one
# more time and hit a hard 502 -- every prior round's tool results are
# already sitting in `messages`, discarding all of that for an error instead
# of a best-effort answer is a worse outcome than answering with whatever was
# actually found. This directive is folded in ONLY on that last round.
TOOL_BUDGET_EXHAUSTED_DIRECTIVE = (
    "You've reached the tool-call budget for this exchange -- no more tool "
    "calls are available. Answer the user's original question now, using "
    "only what you've already found in this conversation. If something is "
    "still genuinely missing, say so plainly rather than guessing or "
    "inventing it."
)

# Human-in-the-loop pause/resume (#383): a synthetic tool, never routed to an
# MCP server -- real MCP tools run as separate stdio subprocesses
# (open_mcp_sessions), and a tool call blocking on an event only a LATER,
# separate HTTP request can set doesn't cross that process boundary. ask_user
# is instead intercepted entirely inside run_tool_loop/_stream_chat_with_mcp_tools,
# in the same process, no subprocess involved. Advertised alongside real MCP
# tools (appended to `tools`, never to `mcp_tools`/`mapping` -- see
# run_tool_loop) only when the caller opted in (`exchange_id is not None`).
ASK_USER_TOOL_NAME = "ask_user"

ASK_USER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ASK_USER_TOOL_NAME,
        "description": (
            "Ask the human user a question and wait for their answer before "
            "continuing. Only for something genuinely ambiguous or a decision "
            "only the user can make -- not a substitute for a tool you already "
            "have, and not habitual: a question every round is a worse "
            "experience than doing your best with what you already know."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to show the user.",
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional multiple-choice options. Omit for a free-text answer."
                    ),
                },
            },
            "required": ["question"],
        },
    },
}

# Folded alongside TOOL_DISCIPLINE_DIRECTIVE (never replacing it -- existing
# callers that never enable ask_user keep exactly TOOL_DISCIPLINE_DIRECTIVE's
# text, unchanged) whenever a request opts into human-in-the-loop. Kept as
# its own line rather than appended into TOOL_DISCIPLINE_DIRECTIVE itself: the
# directive is a public constant several tests assert verbatim, and is folded
# unconditionally whenever mcp_tools is non-empty -- unrelated to whether
# ask_user is even offered this request.
ASK_USER_DISCIPLINE_LINE = (
    "You also have an ask_user tool. Use it ONLY when a request is genuinely "
    "ambiguous or needs a decision only the human user can make -- never "
    "habitually, and never in place of a tool you already have."
)


@dataclass
class _PendingInput:
    """One exchange's human-in-the-loop wait slot: the ``Event`` a paused
    ``run_tool_loop`` awaits, plus the answer text it's set with. Registered
    once per exchange (``open_pending_input``), re-armed in place for each
    successive pause within that same exchange (a conversation can pause
    more than once) rather than allocating a fresh registry entry per pause.
    """

    event: asyncio.Event = field(default_factory=asyncio.Event)
    answer: str | None = None
    # User-initiated pause (issue #383, as opposed to a model-issued
    # ask_user call): set by `request_pause`, consumed by run_tool_loop at
    # the top of its next round check -- the loop, not the request handler,
    # owns actually suspending between rounds.
    pause_requested: bool = False


# In-memory registry, keyed by the per-exchange id `_stream_chat_with_mcp_tools`
# mints (see its `exchange` SSE event) -- NOT a `channel` id reused from
# elsewhere in this codebase (Inngest's realtime channels are a distinct,
# unrelated concept): chat_api.py mints nothing per-request today, so this PR
# adds that minting as part of wiring HITL in. Single-process only, same as
# every other in-memory structure in this module (open_mcp_sessions'
# per-request ClientSessions, the MCP server registry cache) -- a paused
# exchange's `/respond` must land on the SAME api replica that is running its
# SSE stream. Bounded lifetime: `open_pending_input`/`close_pending_input`
# bracket exactly one exchange's `run_tool_loop` call, so an abandoned
# exchange never accumulates entries beyond its own timeout.
_pending_inputs: dict[str, _PendingInput] = {}


def open_pending_input(exchange_id: str) -> None:
    """Register a fresh wait slot for one exchange. Called once, before
    `run_tool_loop` starts, by the caller that minted `exchange_id`."""
    _pending_inputs[exchange_id] = _PendingInput()


def close_pending_input(exchange_id: str) -> None:
    """Drop the registry entry -- called from a `finally` once the exchange
    ends (answered, timed out, errored, or the client disconnected) so an
    abandoned exchange never leaks memory. A second call is a no-op."""
    _pending_inputs.pop(exchange_id, None)


def has_pending_input(exchange_id: str) -> bool:
    """Whether `exchange_id` still has a live registry entry. Introspection
    only (tests, and any future admin/debug surface) -- `run_tool_loop`
    itself never needs to ask this; it holds the entry directly."""
    return exchange_id in _pending_inputs


def request_pause(exchange_id: str) -> bool:
    """User-initiated pause: flags the exchange to suspend at its NEXT round
    boundary (run_tool_loop checks this at the top of every round). Returns
    False when no such exchange is registered -- already finished, wrong id,
    or the caller never opted into human-in-the-loop for it."""
    pending = _pending_inputs.get(exchange_id)
    if pending is None:
        return False
    pending.pause_requested = True
    return True


def submit_pending_answer(exchange_id: str, text: str) -> bool:
    """Resolve a paused exchange's wait with the user's answer -- for BOTH
    entry points (an ask_user question, or a user-initiated pause), since by
    the time an answer is being typed the exchange is in the exact same
    awaiting-input state either way. Safe to call before `run_tool_loop`
    actually reaches its `await`: `asyncio.Event.set()` is sticky, so the
    eventual `wait()` returns immediately with this answer already in place.
    Returns False when no such exchange is registered."""
    pending = _pending_inputs.get(exchange_id)
    if pending is None:
        return False
    pending.answer = text
    pending.event.set()
    return True


def _parse_ask_user_arguments(arguments: Any) -> tuple[str, list[str] | None]:
    """Same tolerant parsing as `execute_tool_call`'s arguments handling --
    a small model's JSON is not always well-formed, and a malformed
    `ask_user` call should degrade to a generic question, never raise."""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            parsed = {}
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        parsed = {}
    question = str(parsed.get("question") or "").strip() or "The model has a question for you."
    choices_raw = parsed.get("choices")
    choices = (
        [str(c) for c in choices_raw] if isinstance(choices_raw, list) and choices_raw else None
    )
    return question, choices


async def _await_pending_input(
    exchange_id: str,
    on_awaiting_input: Callable[[dict[str, Any]], None] | None,
    *,
    question: str | None = None,
    choices: list[str] | None = None,
) -> str:
    """Suspend until `exchange_id`'s registry entry is resolved (or time out).

    Fires `on_awaiting_input` once, with `{"question", "choices"}` when given
    (a model-issued `ask_user` call) or `{}` (a user-initiated pause, no
    question of its own -- the Playground renders a free-text "add context"
    prompt instead of a picker). Raises `AskUserTimeoutError` -- never hangs
    the request -- after `settings.mcp_ask_user_timeout_seconds` with no
    answer. Returns `""` if the registry entry vanished concurrently (the
    exchange was already torn down by its own `finally`), rather than waiting
    on nothing forever.
    """
    pending = _pending_inputs.get(exchange_id)
    if pending is None:
        return ""
    if on_awaiting_input is not None:
        payload: dict[str, Any] = {}
        if question is not None:
            payload["question"] = question
        if choices is not None:
            payload["choices"] = choices
        on_awaiting_input(payload)
    timeout = get_settings().mcp_ask_user_timeout_seconds
    try:
        await asyncio.wait_for(pending.event.wait(), timeout=timeout)
    except TimeoutError:
        raise AskUserTimeoutError(
            f"no answer arrived within {timeout:.0f}s -- the exchange was abandoned"
        ) from None
    answer = pending.answer or ""
    # Re-arm for a possible LATER pause in the same exchange (a conversation
    # can pause more than once) -- a fresh Event, since asyncio.Event has no
    # `clear()`-and-still-usable-by-a-concurrent-waiter guarantee once set.
    pending.event = asyncio.Event()
    pending.answer = None
    return answer

# Stable prefix marking the one line of the system message _with_context_budget_line
# owns -- lets it find-and-REPLACE its own line every round instead of
# appending a new stale snapshot each time (which would make the model's
# context problem worse while trying to warn it about one).
_CONTEXT_BUDGET_PREFIX = "Context budget:"


def _context_budget_line(used_tokens: int, ceiling: int) -> str:
    """The model-facing budget line for this round's CURRENT standing.

    The model is told about the budget from round 1 (0/ceiling) and this is
    re-folded every round after, always reflecting the latest cumulative
    total -- not a one-time warning near the wall, continuous awareness so
    the model can pace itself across the whole exchange (#344, revised per
    explicit feedback: a late warning alone isn't enough).
    """
    percent = round(100 * used_tokens / ceiling) if ceiling else 0
    return (
        f"{_CONTEXT_BUDGET_PREFIX} {used_tokens}/{ceiling} tokens used ({percent}%) in this "
        "exchange so far. Pace yourself -- as this climbs, wrap up and answer with what you "
        "have rather than continuing to call tools, unless one more targeted call is clearly "
        "necessary to answer at all."
    )


def _with_system_addendum(messages: list[dict[str, Any]], addendum: str) -> list[dict[str, Any]]:
    """Fold ``addendum`` into the request's system message — appended to the
    caller's own system prompt if one exists, otherwise inserted as a new
    one. Never adds a second ``system`` message: chat templates commonly
    assume at most one, and merging keeps that true regardless of what the
    caller supplied.
    """
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = f"{content}\n\n{addendum}"
                return messages
    return [{"role": "system", "content": addendum}, *messages]


def _with_context_budget_line(messages: list[dict[str, Any]], line: str) -> list[dict[str, Any]]:
    """Fold ``line`` into the system message, REPLACING any previous budget
    line (found via ``_CONTEXT_BUDGET_PREFIX``) rather than appending a new
    one each round. The model always sees its CURRENT standing as one line,
    never a growing history of stale snapshots eating its own budget.
    """
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                text_lines = content.split("\n")
                for i, existing in enumerate(text_lines):
                    if existing.startswith(_CONTEXT_BUDGET_PREFIX):
                        text_lines[i] = line
                        message["content"] = "\n".join(text_lines)
                        return messages
                message["content"] = f"{content}\n\n{line}"
                return messages
    return [{"role": "system", "content": line}, *messages]


def _catalog_usage_notes(mapping: Mapping[str, tuple[str, str]]) -> str | None:
    """Collect each server actually in play's own ``CatalogEntry.usage_notes``
    -- a multi-step protocol a tool's own JSON-schema description already
    documents but a small model reliably fails to extract (see
    ``CatalogEntry.usage_notes``'s own docstring for why). Promoted into the
    same clean, dedicated system-message channel ``TOOL_DISCIPLINE_DIRECTIVE``
    already uses. One entry per distinct server, first-seen order, dedup'd so
    a server with several tools doesn't repeat its note once per tool.
    ``None`` when no involved server declares any.
    """
    from docie_bench.mcp_catalog import CATALOG

    seen: set[str] = set()
    lines: list[str] = []
    for server_name, _tool_name in mapping.values():
        if server_name in seen:
            continue
        seen.add(server_name)
        entry = CATALOG.get(server_name)
        if entry is not None and entry.usage_notes:
            lines.append(entry.usage_notes)
    return "\n\n".join(lines) if lines else None


async def _eager_list_context(
    sessions: Mapping[str, ClientSession],
    mapping: Mapping[str, tuple[str, str]],
    on_tool_call: Callable[[str, bool, int, Any, str], None] | None,
) -> str | None:
    """Auto-call every advertised tool a catalog entry marks as its
    ``eager_list_tool`` (docs-search's ``list_files``) once, up front, and
    return their results as context text — or ``None`` if none apply.

    The model then starts the request already knowing the real identifiers
    for that server instead of discovering them (or inventing them) via a
    tool call it may skip or get wrong. Complements ``TOOL_DISCIPLINE_DIRECTIVE``:
    that's an instruction the model might ignore, this makes the listing
    true regardless of whether it does. Reported through ``on_tool_call``
    exactly like a model-issued call, same as a real one, so it shows up in
    the trace instead of happening invisibly.
    """
    from docie_bench.mcp_catalog import CATALOG

    lines: list[str] = []
    for qualified_name, (server_name, tool_name) in mapping.items():
        entry = CATALOG.get(server_name)
        if entry is None or entry.eager_list_tool != tool_name:
            continue
        call_started = time.monotonic()
        result = await execute_tool_call(sessions, mapping, qualified_name, {})
        if on_tool_call is not None:
            on_tool_call(
                qualified_name,
                not result.startswith("error:"),
                int((time.monotonic() - call_started) * 1000),
                {},
                result,
            )
        lines.append(
            f"{server_name}.{tool_name}() was already called for you. Its result -- "
            f"the ONLY valid identifiers for {server_name} right now -- is:\n{result}\n"
            f"Use one of these EXACTLY as returned whenever a {server_name} tool asks "
            "for an identifier. Never modify, translate, or construct a different one."
        )
    return "\n\n".join(lines) if lines else None


def make_trace_recorder(
    trace: list[dict[str, Any]],
) -> Callable[[str, bool, int, Any, str], None]:
    """Build an ``on_tool_call`` callback that appends each call's outcome to
    ``trace`` in the shared "Try it" trace shape (tool/status/latency_ms/
    arguments/result). One instance of this shape is used by both the Agents
    surface (#261/#262) and the generic ``mcp_servers`` chat surface, so a
    caller's trace list always renders identically regardless of which
    endpoint produced it.
    """

    def record(name: str, ok: bool, latency_ms: int, arguments: Any, result: str) -> None:
        trace.append(
            {
                "tool": name,
                "status": "ok" if ok else "error",
                "latency_ms": latency_ms,
                "arguments": _truncate_trace_text(_stringify_tool_value(arguments)),
                "result": _truncate_trace_text(result),
            }
        )

    return record


async def run_tool_loop(
    post: Callable[[dict[str, Any]], Awaitable[Any]],
    body: Mapping[str, Any],
    sessions: Mapping[str, ClientSession],
    mapping: Mapping[str, tuple[str, str]],
    mcp_tools: list[dict[str, Any]],
    max_iterations: int | None = None,
    on_tool_call: Callable[[str, bool, int, Any, str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    on_system_addendum: Callable[[str], None] | None = None,
    on_usage: Callable[[dict[str, Any]], None] | None = None,
    context_length_ceiling: int | None = None,
    on_context_budget: Callable[[dict[str, Any]], None] | None = None,
    exchange_id: str | None = None,
    on_awaiting_input: Callable[[dict[str, Any]], None] | None = None,
    on_tool_budget_forced: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """Drive the model↔tools exchange until a plain answer (or the bound).

    ``post`` is the caller's "one upstream completion" function; anything it
    returns that is not a dict (an error response) passes straight through.
    Returns the final completion dict with ``usage`` summed across rounds, or
    the error object from ``post``.

    ``max_iterations`` (or ``settings.mcp_max_tool_iterations``) is no longer
    a hard wall that discards the whole exchange (#391). Rounds 1..limit run
    exactly as before, full tools available every time. If round ``limit``
    ALSO ends in a tool call, ONE more round (``limit + 1``, never more) runs
    with tools stripped down to whatever the caller itself supplied (real MCP
    tools and ``ask_user`` both dropped) and ``TOOL_BUDGET_EXHAUSTED_DIRECTIVE``
    folded into the system message — forcing a plain answer built from
    everything already gathered in ``messages`` across every prior round,
    instead of a 502 that throws all of that work away. That forced round is
    a hard stop regardless of what comes back (even a broken ``post()`` that
    ignores the stripped tools and returns another tool-calls-shaped message
    anyway) — it is returned as the final answer either way, never looped
    back into. ``on_tool_budget_forced``, when given, is invoked once with
    ``{"rounds_used", "max_iterations"}`` right before that round's ``post()``
    call, so a caller can surface it transparently. A defensive ``return
    None`` still exists below (matching every other fail-open guard in this
    codebase) but is no longer reachable under normal operation — the forced
    round always returns a real completion, or an error from ``post()``,
    before the loop could ever reach it.

    ``on_tool_call``, when given, is invoked synchronously right after each
    executed tool call with ``(qualified_name, ok, latency_ms, arguments,
    result)`` — ``arguments`` is exactly what the model sent (string or
    object, unparsed), ``result`` the text handed back to the model. Pass
    ``make_trace_recorder``'s output to collect a "Try it"-shaped trace
    (#261/#262); both the Agents surface and the generic ``mcp_servers`` chat
    surface use it.

    Whenever ``mcp_tools`` is non-empty, ``TOOL_DISCIPLINE_DIRECTIVE`` is
    folded into the request's system message before the first round — a
    small model reliably skips a "call this first" tool docstring, so the
    same instruction is repeated at the request level for every caller
    automatically, not left to each caller's own system prompt. Any
    catalog-declared ``eager_list_tool`` (docs-search's ``list_files``) is
    also called once up front and its result folded in alongside it — see
    ``_eager_list_context``.

    ``on_reasoning``, when given, is invoked once per round with that
    round's ``message.reasoning_content`` whenever it's a non-empty string —
    a reasoning-capable model (LFM2.5's bundled chat template opens
    ``<think>`` unconditionally) emits its "why" for calling a tool (or for
    the final answer) in this SEPARATE field when ``--jinja`` is on;
    ``message.content``/``tool_calls`` stay clean either way. Fires for
    EVERY round, including the last one, so "is there a hidden thinking
    step before the tool call, or does the model just formulate it
    directly" has a real answer instead of that reasoning being silently
    discarded.

    ``on_system_addendum``, when given, is invoked exactly once — right
    after the addendum above is folded into ``messages``, before the first
    round ever runs — with the FULL addendum text actually used (directive
    alone, or directive plus eager-list context). Only fires when
    ``mcp_tools`` is non-empty, same condition that builds the addendum in
    the first place; a request with no MCP tools in play never calls this.
    This is real, load-bearing content injected on top of whatever system
    prompt the caller supplied, but until now it was never surfaced
    anywhere a caller could see it.

    ``on_usage``, when given, is invoked once per round, right after that
    round's ``completion.get("usage")`` is read, with a single dict:
    ``{"round": <that round's own usage dict, or {} when absent>,
    "cumulative": <running totals through this round>}``. Per-round shows
    the cost of the last action; cumulative shows total consumption so
    far — there is currently no other way to see how close a live agentic
    exchange is getting to the deployment's context window without reading
    the final completion's summed ``usage`` by hand after the whole
    exchange already finished. Fires for EVERY round, including the last
    one, same as ``on_reasoning``.

    ``context_length_ceiling``, when given, is the resolved deployment's own
    context window (its ``spec.launch.context_length`` deployment record) --
    the caller resolves this BEFORE the loop starts, from the same
    ``deployments.json`` this codebase already reads for capacity elsewhere.
    ``None`` means the ceiling could not be resolved (an unknown/unpriceable
    deployment, or a profile not backed by a live deployment record at all)
    and the check below is skipped entirely -- fail-open, the same convention
    every fit/pricing gate in this codebase uses for an unknowable value.

    Whenever ``context_length_ceiling`` is known, the model is told about its
    budget from the FIRST round on, not just warned once late: a
    ``_CONTEXT_BUDGET_PREFIX``-marked line is folded into the system message
    before round 1 (``0/ceiling``), then RE-FOLDED (replacing that same
    line, never appended anew) after every round with the latest cumulative
    total -- continuous awareness the model can pace itself against for the
    whole exchange, not a single late warning near the wall.

    Separately, ``on_context_budget``, when given, is invoked ONCE -- the
    first round cumulative ``total_tokens`` reaches
    ``settings.mcp_context_budget_warn_fraction`` (default 80%) of the
    ceiling -- with ``{"cumulative_tokens": <int>, "context_length": <int>,
    "threshold_fraction": <float>}``. This is the human/UI-facing mirror,
    for a caller that wants a one-time actionable warning of its own (the
    Playground renders it as a banner); it is independent of the continuous
    model-facing line above, which happens regardless of whether this
    callback is given.

    Neither truncates, summarizes, or otherwise forces the exchange to end --
    a cumulative total several real rounds deep can still hit a hard
    ``exceed_context_size_error`` from llama-server on some LATER round if
    the model doesn't act on its own budget awareness; this narrows that
    risk, it doesn't eliminate it.

    ``exchange_id``, when given, opts this call into human-in-the-loop
    pause/resume (#383) -- the caller must have already registered it with
    ``open_pending_input`` (and must ``close_pending_input`` it once this
    call returns or raises, typically from a ``finally``). Two things follow:

    1. The synthetic ``ASK_USER_TOOL_SCHEMA`` is advertised to the model
       alongside ``mcp_tools`` (never added to ``mcp_tools`` itself, so
       ``TOOL_DISCIPLINE_DIRECTIVE``'s ``mcp_tools``-gated folding, and the
       tests that assert its exact text, are unaffected). ``ASK_USER_DISCIPLINE_LINE``
       is folded into the system message alongside it.
    2. At the top of every round, a pending user-initiated pause
       (``request_pause``) suspends the loop -- awaiting an answer via the
       same registry entry a model-issued ``ask_user`` call would.

    Neither path consumes one of ``max_iterations`` rounds or touches the
    context-budget totals above -- only an actual ``post()`` call does
    either, and both pause paths return control to the loop BEFORE the next
    ``post()`` (or, for ``ask_user``, without an extra one at all -- the
    round that produced the tool call already counted). The user's answer
    folds in as a new ``role: "user"`` message (a pause) or as the
    ``ask_user`` tool call's own ``role: "tool"`` result -- either way, it
    only costs tokens on the NEXT real round, like any other message.

    A pause that gets no answer within ``settings.mcp_ask_user_timeout_seconds``
    raises ``AskUserTimeoutError`` -- propagated to the caller, never
    swallowed, so the exchange fails loudly (an "error" SSE event on the
    streaming surface) instead of hanging the request or leaking the
    registry entry forever.

    ``exchange_id=None`` (the default) disables all of the above: no
    ``ask_user`` tool is advertised, and the per-round pause check is
    skipped entirely -- today's behavior, byte-for-byte, for every existing
    caller (the non-streaming ``mcp_servers`` chat path, the standalone
    Agents surface). This mechanism ships Playground-only for now.
    """
    limit = max_iterations if max_iterations is not None else get_settings().mcp_max_tool_iterations
    forward = dict(body)
    messages = [dict(m) if isinstance(m, dict) else m for m in (forward.get("messages") or [])]
    if mcp_tools:
        addendum = TOOL_DISCIPLINE_DIRECTIVE
        usage_notes = _catalog_usage_notes(mapping)
        if usage_notes:
            addendum = f"{addendum}\n\n{usage_notes}"
        eager_context = await _eager_list_context(sessions, mapping, on_tool_call)
        if eager_context:
            addendum = f"{addendum}\n\n{eager_context}"
        if exchange_id is not None:
            addendum = f"{addendum}\n\n{ASK_USER_DISCIPLINE_LINE}"
        messages = _with_system_addendum(messages, addendum)
        if on_system_addendum is not None:
            on_system_addendum(addendum)
    elif exchange_id is not None:
        messages = _with_system_addendum(messages, ASK_USER_DISCIPLINE_LINE)
        if on_system_addendum is not None:
            on_system_addendum(ASK_USER_DISCIPLINE_LINE)
    if context_length_ceiling is not None:
        messages = _with_context_budget_line(
            messages, _context_budget_line(0, context_length_ceiling)
        )
    caller_tools = list(forward.get("tools") or [])
    caller_tool_names = {
        str(t.get("function", {}).get("name"))
        for t in caller_tools
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    forward["tools"] = (
        caller_tools + mcp_tools + ([ASK_USER_TOOL_SCHEMA] if exchange_id is not None else [])
    )
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    context_budget_warned = False
    rounds_used = 0

    def _is_known(name: str) -> bool:
        return name in mapping or (exchange_id is not None and name == ASK_USER_TOOL_NAME)

    while True:
        if exchange_id is not None:
            pending = _pending_inputs.get(exchange_id)
            if pending is not None and pending.pause_requested:
                # User-initiated pause: consume the flag and suspend right
                # here, BEFORE the next post() -- this round slot, and this
                # wait, cost nothing against max_iterations or the token
                # budget above.
                pending.pause_requested = False
                answer = await _await_pending_input(exchange_id, on_awaiting_input)
                messages.append({"role": "user", "content": answer})
                continue
        if rounds_used >= limit + 1:
            return None  # defensive fallback only -- see run_tool_loop's docstring
        rounds_used += 1
        # Rounds 1..limit run exactly as before (full tools, byte-identical
        # to pre-#391 behavior) -- only round limit+1 is forced, and only
        # reached at all when round `limit` ALSO ended in a tool call.
        is_forced_round = rounds_used == limit + 1
        if is_forced_round:
            # #391: every real round the budget allows has been used and the
            # last one still ended in a tool call. Grant exactly one more,
            # tools-stripped round to force a plain answer instead of letting
            # this end in a 502 -- strip mcp_tools/ask_user down to whatever
            # the caller itself supplied (untouched either way) and nudge the
            # model to answer NOW with what's already in `messages` from
            # every prior round, rather than discarding all of that.
            forward["tools"] = caller_tools
            messages = _with_system_addendum(messages, TOOL_BUDGET_EXHAUSTED_DIRECTIVE)
            if on_tool_budget_forced is not None:
                on_tool_budget_forced({"rounds_used": rounds_used, "max_iterations": limit})
        forward["messages"] = messages
        completion = await post(forward)
        if not isinstance(completion, dict):
            return completion
        round_usage = completion.get("usage")
        _accumulate_usage(totals, round_usage)
        if on_usage is not None:
            on_usage(
                {
                    "round": round_usage if isinstance(round_usage, dict) else {},
                    "cumulative": dict(totals),
                }
            )
        if context_length_ceiling is not None:
            # Model-facing: re-fold every round with the latest cumulative
            # total, independent of the one-time human-facing signal below.
            messages = _with_context_budget_line(
                messages, _context_budget_line(totals["total_tokens"], context_length_ceiling)
            )
            if on_context_budget is not None and not context_budget_warned:
                threshold_fraction = get_settings().mcp_context_budget_warn_fraction
                if totals["total_tokens"] >= context_length_ceiling * threshold_fraction:
                    context_budget_warned = True
                    on_context_budget(
                        {
                            "cumulative_tokens": totals["total_tokens"],
                            "context_length": context_length_ceiling,
                            "threshold_fraction": threshold_fraction,
                        }
                    )
        choices = completion.get("choices")
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else None
        )
        if on_reasoning is not None and isinstance(message, dict):
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                on_reasoning(reasoning)
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        names = [
            str(call.get("function", {}).get("name", ""))
            for call in (calls or [])
            if isinstance(call, dict)
        ]
        all_mcp = bool(names) and all(_is_known(name) for name in names)
        if not all_mcp or is_forced_round:
            # Plain answer, or at least one caller-owned/unknown tool call:
            # this completion belongs to the caller. Unknown names ride the
            # same path deliberately — hallucinated tool names are the
            # caller's signal, not something to swallow server-side. But a
            # hallucinated name is only "caller-owned" fiction if the caller
            # advertised tools at all; log the anomaly either way. ask_user
            # itself is never "unknown" when this exchange opted into it.
            #
            # `is_forced_round` short-circuits this unconditionally (#391):
            # mcp_tools were stripped from `forward` above, so a well-behaved
            # model structurally cannot emit one -- but nothing stops a
            # broken/adversarial `post()` (or a model ignoring what it was
            # actually offered) from returning a tool_calls-shaped message
            # anyway, whose name still happens to resolve via `mapping`. This
            # round is the hard stop either way: never loop back and execute
            # it, just return whatever came back as the final answer.
            for name in names:
                if name and not _is_known(name) and name not in caller_tool_names:
                    logger.warning("model called unknown tool %r — returning to caller", name)
            if any(totals.values()):
                completion = {**completion, "usage": dict(totals)}
            return completion
        assert message is not None  # all_mcp implies a message with tool_calls
        messages.append(message)
        for call in calls or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            call_name = str(function.get("name"))
            call_arguments = function.get("arguments")
            if call_name == ASK_USER_TOOL_NAME and exchange_id is not None:
                question, ask_choices = _parse_ask_user_arguments(call_arguments)
                call_started = time.monotonic()
                answer = await _await_pending_input(
                    exchange_id, on_awaiting_input, question=question, choices=ask_choices
                )
                if on_tool_call is not None:
                    on_tool_call(
                        ASK_USER_TOOL_NAME,
                        True,
                        int((time.monotonic() - call_started) * 1000),
                        call_arguments,
                        answer,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": answer,
                    }
                )
                continue
            call_started = time.monotonic()
            result_text = await execute_tool_call(sessions, mapping, call_name, call_arguments)
            if on_tool_call is not None:
                on_tool_call(
                    call_name,
                    not result_text.startswith("error:"),
                    int((time.monotonic() - call_started) * 1000),
                    call_arguments,
                    result_text,
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": result_text,
                }
            )
