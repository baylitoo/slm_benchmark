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

export type AgentKind = "proxy_security" | "ocr" | "custom";

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

export interface AgentChatResponse {
  model?: string;
  choices?: { message?: { role?: string; content?: string } }[];
  docie_agent?: { agent?: string; kind?: string; pii?: AgentPiiReport };
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
): Promise<AgentChatResponse> {
  return openaiPost(`${API_BASE}/v1/chat/completions`, {
    model,
    messages,
    // Named MCP servers: the backend advertises their tools, runs the tool
    // exchange, and returns the final completion (non-streaming only).
    ...(mcpServers && mcpServers.length > 0 ? { mcp_servers: mcpServers } : {}),
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
