// Agents (GET/POST /v1/agents — preconfigured agents over OpenAI endpoints)
// and the OpenAI-compatible chat completion calls (agent Try panel, generic
// serving-stack chat, and the streaming variant backing the Playground).

import { API_BASE } from "../env";
import { authHeader } from "../apiKey";
import {
  ApiError,
  ApiUnavailable,
  ModelLoading,
  detailOf,
  readBody,
  request,
  unauthorizedError,
} from "./core";

export type AgentKind = "proxy_security" | "ocr" | "custom" | "workflow";

/** A catalog template (GET /v1/agents/templates). */
export interface AgentTemplate {
  id: string;
  kind: AgentKind;
  display_name: string;
  description: string;
  /** Prefill for the create form: { system_prompt, options }. */
  defaults?: { system_prompt?: string | null; options?: Record<string, unknown> };
}

/** A configured agent (GET /v1/agents). */
export interface AgentView {
  name: string;
  kind: AgentKind;
  display_name?: string;
  description?: string;
  /** Backing SLM selector: profile name, live deployment name, or store:<name>. */
  model_profile?: string | null;
  system_prompt?: string | null;
  options?: Record<string, unknown>;
  enabled?: boolean;
  created_at?: string;
  updated_at?: string;
  /** API-relative OpenAI-compatible base path, e.g. "/v1/agents/pii-proxy". */
  endpoint?: string;
  [k: string]: unknown;
}

export interface CreateAgentRequest {
  name: string;
  template?: string;
  kind?: AgentKind;
  display_name?: string;
  description?: string;
  model_profile?: string | null;
  system_prompt?: string | null;
  options?: Record<string, unknown>;
  enabled?: boolean;
}

export interface UpdateAgentRequest {
  display_name?: string;
  description?: string;
  model_profile?: string | null;
  system_prompt?: string | null;
  options?: Record<string, unknown>;
  enabled?: boolean;
}

export function getAgents(): Promise<AgentView[]> {
  return request<AgentView[]>("/v1/agents");
}

export function getAgentTemplates(): Promise<AgentTemplate[]> {
  return request<AgentTemplate[]>("/v1/agents/templates");
}

export function createAgent(payload: CreateAgentRequest): Promise<AgentView> {
  return request<AgentView>("/v1/agents", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateAgent(name: string, patch: UpdateAgentRequest): Promise<AgentView> {
  return request<AgentView>(`/v1/agents/${encodeURIComponent(name)}`, {
    method: "PUT",
    body: JSON.stringify(patch),
  });
}

export function deleteAgent(name: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(`/v1/agents/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

/** Absolute OpenAI-compatible base_url for one agent (or the whole platform). */
export function agentBaseUrl(name?: string): string {
  return name
    ? `${API_BASE}/v1/agents/${encodeURIComponent(name)}`
    : `${API_BASE}/v1/agents`;
}

/** The proxy's per-request analysis report (docie_agent extension key). */
export interface AgentPiiReport {
  mode?: string;
  analyzer?: string;
  detected?: number;
  entities?: { type: string; count: number }[];
  placeholders?: string[];
  degraded_to_regex?: boolean;
}

/** One executed MCP tool call (#261/#262): status/latency for the usage
 * ledger, arguments/result for the "Try it" trace view. `step`/`step_name`
 * are set only for a workflow agent (#265) -- which step made this call,
 * since a multi-step workflow's calls would otherwise read as one
 * unattributed list. */
export interface AgentToolCallTrace {
  tool: string;
  status: "ok" | "error";
  latency_ms: number;
  arguments?: string;
  result?: string;
  step?: number;
  step_name?: string;
}

/** One round's token usage from the agentic tool loop (`on_usage`, #314):
 * `round` is that round's own usage, `cumulative` the running totals
 * through this round -- both raw counts, no context-window denominator. */
export interface AgentUsageTrace {
  round: Record<string, number>;
  cumulative: Record<string, number>;
}

/** Fires AT MOST ONCE per exchange (`on_context_budget`, #344): the first
 * round whose cumulative usage crosses `threshold_fraction` of the resolved
 * deployment's own `context_length`. A warning only -- a long agentic
 * exchange can otherwise run several real rounds before a LATER round's
 * cumulative usage exceeds the deployment's context window and the upstream
 * hard-400s, losing the whole in-progress exchange with no prior warning. */
export interface AgentContextBudgetTrace {
  cumulative_tokens: number;
  context_length: number;
  threshold_fraction: number;
}

/** Fires AT MOST ONCE per exchange (`tool_calls_unsupported`, #353), BEFORE
 * the tool loop runs a single round -- unlike `AgentContextBudgetTrace`,
 * this is known upfront from the resolved deployment's own health state
 * (llama-server's `chat_template_caps.supports_tool_calls`), not learned
 * mid-exchange. Means the model's chat template will never emit a real
 * `tool_calls` field for this deployment, no matter what tools are offered
 * -- it will describe using a tool in prose instead. */
export interface AgentToolCallsUnsupportedTrace {
  message: string;
}

/** Fires whenever a paused exchange (#383) is waiting for a human answer --
 * a model-issued `ask_user` tool call carries `question` (and `choices`
 * when the model gave any -- render a picker; free text otherwise), a
 * user-initiated pause (`pauseChatExchange`) carries neither -- render a
 * plain "add context" box instead. Can fire more than once per exchange. */
export interface AgentAwaitingInputTrace {
  question?: string;
  choices?: string[];
}

/** One workflow step's outcome (#265; `name`/`routed_to` added #266) -- the
 * "Try it" trace view's per-step detail, alongside any tool calls that step
 * made. `routed_to` is set only for a classifier (`route`) step -- the name
 * of the step it jumped to instead of falling through sequentially. */
export interface AgentWorkflowStepTrace {
  step: number;
  name?: string;
  model_profile: string;
  content: string | null;
  routed_to?: string | null;
}

export interface AgentChatResponse {
  model?: string;
  choices?: { message?: { role?: string; content?: string } }[];
  docie_agent?: {
    agent?: string;
    kind?: string;
    pii?: AgentPiiReport;
    tool_calls?: AgentToolCallTrace[];
    steps?: AgentWorkflowStepTrace[];
  };
  [k: string]: unknown;
}

/** POST an OpenAI-shaped chat body and surface OpenAI-shaped errors readably. */
async function openaiPost(url: string, payload: unknown): Promise<AgentChatResponse> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        ...authHeader(),
      },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    throw new ApiUnavailable(0, e instanceof Error ? e.message : "Network error");
  }
  const body = await readBody(res);
  if (res.status === 202 && body && typeof body === "object" && "status" in body) {
    const loading = body as { status?: string; deployment?: string; eta_seconds?: number };
    if (loading.status === "loading") {
      throw new ModelLoading(loading.deployment ?? "model", loading.eta_seconds ?? 0);
    }
  }
  if (res.ok) return body as AgentChatResponse;
  if (res.status === 401) throw unauthorizedError(body);
  const err =
    body && typeof body === "object" && "error" in body
      ? (body as { error?: { message?: string; type?: string } }).error
      : undefined;
  const detail = err?.message ?? detailOf(body, `Request failed (HTTP ${res.status})`);
  throw new ApiError(res.status, err?.type ? `${err.type}: ${detail}` : detail);
}

/**
 * One synchronous chat completion against an agent's OpenAI endpoint (the Try
 * panel). Errors arrive OpenAI-shaped — e.g. `guard_unavailable` when the
 * encoder deployment is unloaded.
 */
export function agentChat(
  name: string,
  messages: { role: string; content: unknown }[],
): Promise<AgentChatResponse> {
  return openaiPost(`${agentBaseUrl(name)}/chat/completions`, {
    model: name,
    messages,
  });
}

/**
 * Generic chat against the serving stack (POST /v1/chat/completions): `model`
 * is a live deployment name, a models.yaml profile, or store:<name>. Backs
 * the Playground's Chat mode.
 */
export function chatCompletion(
  model: string,
  messages: { role: string; content: unknown }[],
  mcpServers?: string[],
  sessionId?: string,
): Promise<AgentChatResponse> {
  return openaiPost(`${API_BASE}/v1/chat/completions`, {
    model,
    messages,
    // Named MCP servers: the backend advertises their tools, runs the tool
    // exchange, and returns the final completion (non-streaming only).
    ...(mcpServers && mcpServers.length > 0 ? { mcp_servers: mcpServers } : {}),
    // Points docs-search (if selected) at this session's uploaded documents
    // instead of the shared operator corpus (#296) — see uploadSessionDocument.
    ...(sessionId ? { session_id: sessionId } : {}),
  });
}

/**
 * Same call as `chatCompletion`, but with `stream: true` — the backend
 * relays llama-server's real token-by-token SSE frames (chat_api.py's
 * `_stream_chat_completions`), so `onToken` fires once per delta as the
 * model generates instead of once after the full completion lands.
 *
 * A cold `store:` model resolves BEFORE the stream ever opens (same
 * `_resolve_or_error` seam as the non-streaming path), so a 202 loading
 * response still arrives as one JSON body — `ModelLoading` throws exactly
 * like `chatCompletion`, before any `onToken` call.
 */
export async function chatCompletionStream(
  model: string,
  messages: { role: string; content: unknown }[],
  onToken: (text: string) => void,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...authHeader(),
      },
      body: JSON.stringify({ model, messages, stream: true }),
    });
  } catch (e) {
    throw new ApiUnavailable(0, e instanceof Error ? e.message : "Network error");
  }

  // 202 is a 2xx — res.ok is true for it too, so the loading check must run
  // BEFORE the ok/error branch, not inside it (a store: model resolves to a
  // plain JSON loading body before the stream ever opens; treating it as an
  // SSE body silently reads zero tokens instead of surfacing the load).
  if (res.status === 202) {
    const body = await readBody(res);
    if (body && typeof body === "object" && "status" in body) {
      const loading = body as { status?: string; deployment?: string; eta_seconds?: number };
      if (loading.status === "loading") {
        throw new ModelLoading(loading.deployment ?? "model", loading.eta_seconds ?? 0);
      }
    }
  }
  if (!res.ok) {
    const body = await readBody(res);
    if (res.status === 401) throw unauthorizedError(body);
    const err =
      body && typeof body === "object" && "error" in body
        ? (body as { error?: { message?: string; type?: string } }).error
        : undefined;
    const detail = err?.message ?? detailOf(body, `Request failed (HTTP ${res.status})`);
    throw new ApiError(res.status, err?.type ? `${err.type}: ${detail}` : detail);
  }
  if (!res.body) return;

  // SSE frames are separated by a blank line; a network chunk can split a
  // frame (or a multi-byte UTF-8 char) anywhere, so buffer and only consume
  // complete "\n\n"-terminated frames.
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let frameEnd: number;
    while ((frameEnd = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, frameEnd);
      buffer = buffer.slice(frameEnd + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        try {
          const delta = JSON.parse(data)?.choices?.[0]?.delta?.content;
          if (typeof delta === "string" && delta) onToken(delta);
        } catch {
          // Malformed/partial frame — skip it rather than kill the stream.
        }
      }
    }
  }
}

/** POST one of the human-in-the-loop control endpoints (#383) and surface
 * OpenAI-shaped errors readably -- same error handling as `openaiPost`, but
 * these return a small `{paused|accepted, exchange_id}` body, not a chat
 * completion. A 404 (`unknown_exchange`) means the exchange already
 * finished, timed out, or never opted into human-in-the-loop. */
async function exchangeControlPost(path: string, payload: unknown): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        ...authHeader(),
      },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    throw new ApiUnavailable(0, e instanceof Error ? e.message : "Network error");
  }
  if (res.ok) return;
  const body = await readBody(res);
  if (res.status === 401) throw unauthorizedError(body);
  const err =
    body && typeof body === "object" && "error" in body
      ? (body as { error?: { message?: string; type?: string } }).error
      : undefined;
  const detail = err?.message ?? detailOf(body, `Request failed (HTTP ${res.status})`);
  throw new ApiError(res.status, err?.type ? `${err.type}: ${detail}` : detail);
}

/**
 * User-initiated pause (#383): flags a running `chatCompletionMcpStream`
 * exchange (by the `exchangeId` its `onExchangeId` callback reported) to
 * suspend at its next round boundary -- usable any time the exchange is
 * running, including mid tool-search. This only REQUESTS the pause; the
 * exchange's own `onAwaitingInput` callback fires once it actually takes
 * effect. Throws (404 `unknown_exchange`) if the exchange already finished.
 */
export function pauseChatExchange(exchangeId: string): Promise<void> {
  return exchangeControlPost("/v1/chat/completions/pause", { exchange_id: exchangeId });
}

/**
 * Answers a paused exchange's `onAwaitingInput` callback (#383) -- a
 * model-issued `ask_user` question, or a user-initiated pause -- with
 * `text`. Safe to call even before the exchange has reached its pause
 * checkpoint. Throws (404 `unknown_exchange`) if the exchange already ended.
 */
export function respondToChatExchange(exchangeId: string, text: string): Promise<void> {
  return exchangeControlPost("/v1/chat/completions/respond", { exchange_id: exchangeId, text });
}

/**
 * Streaming variant of `chatCompletion` for the `mcp_servers` tool-loop
 * path: each executed tool call arrives as its own SSE event the moment it
 * finishes (`onToolCall`), instead of the whole exchange completing
 * silently before anything reaches the caller. NOT the OpenAI token-stream
 * format — there is no meaningful token stream for a tool-calling round —
 * so this parses `{"type": "content_delta"|"reasoning_delta"|
 * "system_addendum"|"tool_call"|"reasoning"|"usage"|"context_budget"|
 * "tool_calls_unsupported"|"content"|"error", ...}` frames, not
 * `choices[0].delta` (though `content_delta`/`reasoning_delta` themselves
 * now carry real per-round token deltas — see below).
 * `onReasoning`, when a reasoning-capable model's chat template emits one,
 * fires with that round's "why" (calling a tool, or the final answer)
 * BEFORE the tool call it precedes — answers "is there a hidden thinking
 * step" instead of leaving it invisible. `onSystemAddendum`, when given,
 * fires exactly once per request, before the first model round, with the
 * server-injected system-prompt text (`TOOL_DISCIPLINE_DIRECTIVE`, plus any
 * eager-list context) that `run_tool_loop` folds in on top of the caller's
 * own system prompt. For a docs-search request, its eager-list `tool_call`
 * event can arrive BEFORE this one -- that listing call happens while the
 * addendum text is still being assembled. `onUsage` fires once per round
 * with that round's own token usage plus the running cumulative totals
 * (raw counts, no context-window denominator — see `AgentUsageTrace`).
 * `onContextBudget` fires AT MOST ONCE per exchange, the first round
 * cumulative usage crosses the resolved deployment's context-budget warning
 * threshold (see `AgentContextBudgetTrace`) — a warning that the exchange
 * is at risk of a hard context-overflow error on some later round, not a
 * per-round log entry.
 * `onToolCallsUnsupported` fires AT MOST ONCE, BEFORE any round runs, when
 * the resolved deployment's chat template is known NOT to support real
 * tool-calling (see `AgentToolCallsUnsupportedTrace`) — known upfront from
 * the deployment's health state, unlike `onContextBudget`.
 * Human-in-the-loop pause/resume (#383) is always requested on this call
 * (`enable_ask_user: true` — this function is the Playground-only surface
 * the whole mechanism is scoped to). `onExchangeId`, when given, fires ONCE,
 * FIRST, before any other callback, with the fresh id this one exchange got
 * — pass it to `pauseChatExchange` at any later point while the exchange is
 * still running. `onAwaitingInput`, when given, fires every time the
 * exchange pauses for a human answer (see `AgentAwaitingInputTrace`) —
 * resolve it with `respondToChatExchange(exchangeId, text)`. A pause that
 * times out server-side (`settings.mcp_ask_user_timeout_seconds`) surfaces
 * as an ordinary thrown `ApiError` (`ask_user_timeout`), same as any other
 * `error` event.
 * `onContentDelta`/`onReasoningDelta` (#389), when given, fire as each
 * round's upstream call streams real token-by-token deltas — every
 * fragment for a round lands BEFORE that round's own `onUsage`/`onReasoning`
 * calls above and before the final `content` resolution, which are still
 * built from the round's fully-accumulated completion once its stream
 * ends. Purely additive, for live incremental rendering — never a
 * replacement for `onReasoning` or the resolved completion, which still
 * carry the complete text (and, for the resolved completion,
 * `docie_agent.tool_calls`).
 * Resolves with the final completion once a `content` event lands; throws
 * on an `error` event or a connection that ends without either.
 */
export async function chatCompletionMcpStream(
  model: string,
  messages: { role: string; content: unknown }[],
  mcpServers: string[],
  onToolCall: (call: AgentToolCallTrace) => void,
  sessionId?: string,
  onReasoning?: (text: string) => void,
  onSystemAddendum?: (text: string) => void,
  onUsage?: (usage: AgentUsageTrace) => void,
  onContextBudget?: (budget: AgentContextBudgetTrace) => void,
  onToolCallsUnsupported?: (warning: AgentToolCallsUnsupportedTrace) => void,
  onExchangeId?: (exchangeId: string) => void,
  onAwaitingInput?: (payload: AgentAwaitingInputTrace) => void,
  onContentDelta?: (text: string) => void,
  onReasoningDelta?: (text: string) => void,
): Promise<AgentChatResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...authHeader(),
      },
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        mcp_servers: mcpServers,
        enable_ask_user: true,
        ...(sessionId ? { session_id: sessionId } : {}),
      }),
    });
  } catch (e) {
    throw new ApiUnavailable(0, e instanceof Error ? e.message : "Network error");
  }

  if (res.status === 202) {
    const body = await readBody(res);
    if (body && typeof body === "object" && "status" in body) {
      const loading = body as { status?: string; deployment?: string; eta_seconds?: number };
      if (loading.status === "loading") {
        throw new ModelLoading(loading.deployment ?? "model", loading.eta_seconds ?? 0);
      }
    }
  }
  if (!res.ok) {
    const body = await readBody(res);
    if (res.status === 401) throw unauthorizedError(body);
    const err =
      body && typeof body === "object" && "error" in body
        ? (body as { error?: { message?: string; type?: string } }).error
        : undefined;
    const detail = err?.message ?? detailOf(body, `Request failed (HTTP ${res.status})`);
    throw new ApiError(res.status, err?.type ? `${err.type}: ${detail}` : detail);
  }
  if (!res.body) throw new ApiError(res.status, "empty response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completion: AgentChatResponse | null = null;
  let streamError: { message?: string; type?: string } | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let frameEnd: number;
    while ((frameEnd = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, frameEnd);
      buffer = buffer.slice(frameEnd + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") {
          if (streamError) {
            throw new ApiError(502, streamError.message ?? "MCP stream error");
          }
          if (completion) return completion;
          throw new ApiError(502, "MCP stream ended with no content");
        }
        try {
          const event = JSON.parse(data) as { type?: string; [k: string]: unknown };
          if (event.type === "tool_call") {
            const { type: _type, ...call } = event;
            onToolCall(call as unknown as AgentToolCallTrace);
          } else if (event.type === "content_delta") {
            if (onContentDelta && typeof event.text === "string") onContentDelta(event.text);
          } else if (event.type === "reasoning_delta") {
            if (onReasoningDelta && typeof event.text === "string") onReasoningDelta(event.text);
          } else if (event.type === "reasoning") {
            if (onReasoning && typeof event.text === "string") onReasoning(event.text);
          } else if (event.type === "system_addendum") {
            if (onSystemAddendum && typeof event.text === "string") onSystemAddendum(event.text);
          } else if (event.type === "usage") {
            if (onUsage) {
              onUsage({
                round: (event.round as Record<string, number>) ?? {},
                cumulative: (event.cumulative as Record<string, number>) ?? {},
              });
            }
          } else if (event.type === "context_budget") {
            if (onContextBudget) {
              onContextBudget({
                cumulative_tokens: Number(event.cumulative_tokens ?? 0),
                context_length: Number(event.context_length ?? 0),
                threshold_fraction: Number(event.threshold_fraction ?? 0),
              });
            }
          } else if (event.type === "tool_calls_unsupported") {
            if (onToolCallsUnsupported) {
              onToolCallsUnsupported({ message: String(event.message ?? "") });
            }
          } else if (event.type === "exchange") {
            if (onExchangeId && typeof event.exchange_id === "string") {
              onExchangeId(event.exchange_id);
            }
          } else if (event.type === "awaiting_input") {
            if (onAwaitingInput) {
              onAwaitingInput({
                ...(typeof event.question === "string" ? { question: event.question } : {}),
                ...(Array.isArray(event.choices)
                  ? { choices: event.choices as string[] }
                  : {}),
              });
            }
          } else if (event.type === "content") {
            completion = event.completion as AgentChatResponse;
          } else if (event.type === "error") {
            streamError = event.error as { message?: string; type?: string };
          }
        } catch {
          // Malformed/partial frame — skip it rather than kill the stream.
        }
      }
    }
  }
  if (streamError) throw new ApiError(502, streamError.message ?? "MCP stream error");
  if (completion) return completion;
  throw new ApiError(502, "MCP stream ended with no content");
}
