// Extraction trigger, realtime subscription, run-status polling fallback,
// document rasterization, and the Playground's own SSE streaming path.

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
  type TriggerResponse,
} from "./core";

export interface ExtractRequest {
  text?: string;
  content_b64?: string;
  filename?: string;
  schema_name?: string;
  /** Name of a schema saved via POST /v1/studio/schemas/dynamic. Wins over
   * schema_name when present -- the backend resolves the spec by name and
   * runs the extraction in schema_mode="dynamic". */
  dynamic_schema_name?: string;
  /** Free-text models.yaml/CLI profile name. Retained for back-compat. */
  model_profile?: string;
  /**
   * Explicit live-deployment selector = a DeploymentRecord `spec.name`. The
   * backend resolves it to that deployment's runtime endpoint (PR-a resolver);
   * it wins over `model_profile`. The Playground sends only this field.
   */
  deployment?: string;
  /**
   * A saved routing policy (POST /v1/studio/routing-policies) to run this
   * extraction through instead of a single model: first stage, escalate on
   * the policy's own confidence/validity rules and budgets. Mutually
   * exclusive with deployment/model_profile. The result's `routing` carries
   * the audit (which stage answered and why, per-attempt cost).
   */
  routing_policy?: string;
  ocr_backend?: string;
  language?: string;
}

export function triggerExtract(payload: ExtractRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/extract", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * SSE counterpart to `triggerExtract` for the Playground's extraction flow
 * ONLY (#397) — same `ExtractRequest` body, but posts directly to
 * `/v1/extract/stream` (a synchronous, non-Inngest route: no `TriggerResponse`,
 * no `PollingResult`/`RealtimeResult`, no run polling) and consumes the
 * response as an SSE stream instead of a fire-and-poll trigger.
 *
 * Frames: `{"type": "phase", "phase": ...}` fires once before generation
 * starts (`onPhase`). `{"type": "delta", "text": ...}` fires as the model's
 * raw output streams in (`onDelta`) — RAW, pre-normalization text for a live
 * preview only; never parse the accumulated deltas as the result.
 * `{"type": "reset"}` fires when the server abandons an in-progress attempt
 * and retries (`onReset`) — the caller must clear whatever it already
 * rendered from deltas, or it keeps painting fragments from that abandoned
 * attempt. Resolves with the parsed `result` object from the terminal
 * `{"type": "result", "result": ...}` frame — the SAME post-processed,
 * validated response `triggerExtract`'s worker path eventually delivers,
 * just synchronous. Throws on an `error` frame or a connection that ends
 * with neither `result` nor `error`. A cold/evicted deployment 202s before
 * any SSE framing starts (same `ModelLoading` contract as
 * `chatCompletionMcpStream`).
 */
export async function extractStream(
  payload: ExtractRequest,
  onDelta?: (text: string) => void,
  onReset?: () => void,
  onPhase?: (phase: string) => void,
): Promise<unknown> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/v1/extract/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...authHeader(),
      },
      body: JSON.stringify(payload),
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
    throw new ApiError(res.status, detailOf(body, `Request failed (HTTP ${res.status})`));
  }
  if (!res.body) throw new ApiError(res.status, "empty response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: unknown = undefined;
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
            throw new ApiError(502, streamError.message ?? "extraction stream error");
          }
          if (result !== undefined) return result;
          throw new ApiError(502, "extraction stream ended with no result");
        }
        try {
          const event = JSON.parse(data) as { type?: string; [k: string]: unknown };
          if (event.type === "delta") {
            if (onDelta && typeof event.text === "string") onDelta(event.text);
          } else if (event.type === "reset") {
            onReset?.();
          } else if (event.type === "phase") {
            if (onPhase && typeof event.phase === "string") onPhase(event.phase);
          } else if (event.type === "result") {
            result = event.result;
          } else if (event.type === "error") {
            streamError = event.error as { message?: string; type?: string };
          }
        } catch {
          // Malformed/partial frame — skip it rather than kill the stream.
        }
      }
    }
  }
  if (streamError) throw new ApiError(502, streamError.message ?? "extraction stream error");
  if (result !== undefined) return result;
  throw new ApiError(502, "extraction stream ended with no result");
}

// Inngest realtime subscription token. Shape is defined by the server SDK; we
// pass it through to the React hook untouched.
export type RealtimeToken = Record<string, unknown>;

export function getRealtimeToken(
  channel: string,
  topics: string[],
): Promise<RealtimeToken> {
  const params = new URLSearchParams();
  params.set("channel", channel);
  for (const t of topics) params.append("topics", t);
  return request<RealtimeToken>(`/v1/studio/realtime-token?${params.toString()}`);
}

// One run as proxied from Inngest's `/v1/events/{id}/runs`.
export interface InngestRun {
  run_id?: string;
  status?: string; // "Running" | "Completed" | "Failed" | "Cancelled" ...
  output?: unknown;
  [k: string]: unknown;
}

/**
 * Polling fallback. Inngest's run endpoint usually wraps the array as
 * `{ data: [...] }`; we accept both shapes and always return a plain array.
 */
export async function getRuns(eventId: string): Promise<InngestRun[]> {
  const raw = await request<unknown>(`/v1/studio/runs/${encodeURIComponent(eventId)}`);
  if (Array.isArray(raw)) return raw as InngestRun[];
  if (raw && typeof raw === "object" && Array.isArray((raw as { data?: unknown }).data)) {
    return (raw as { data: InngestRun[] }).data;
  }
  return [];
}

/** Rasterize an uploaded PDF (or image) to PNG page-image data URLs a vision
 * model can read (POST /v1/studio/render-document). One data URL per page.
 * `pages`, when given, is an explicit list of 1-indexed page numbers to
 * rasterize (e.g. `[1]` for a thumbnail preview) -- ONLY those pages render,
 * cheaply, regardless of the document's total length, and the server's
 * `max_pages` reject check does not apply. Omit it (the vision-send call
 * site) to keep the existing contract: rasterize everything, reject if the
 * document has more pages than the model will actually see. `total_pages` in
 * the response is the document's TRUE page count, which can exceed
 * `images.length` when a subset was requested. */
export function renderDocument(
  contentB64: string,
  filename: string,
  dpi?: number,
  pages?: number[],
): Promise<{ images: string[]; pages: number; total_pages: number }> {
  return request<{ images: string[]; pages: number; total_pages: number }>(
    "/v1/studio/render-document",
    {
      method: "POST",
      body: JSON.stringify({
        content_b64: contentB64,
        filename,
        ...(dpi ? { dpi } : {}),
        ...(pages ? { pages } : {}),
      }),
    },
  );
}

/**
 * Make an uploaded file searchable via the docs-search MCP server for THIS
 * conversation only (POST /v1/studio/session-documents). `sessionId` is
 * server-issued: omit it on the first upload of a conversation and pass the
 * returned `session_id` on every later upload/chat turn in that same
 * conversation — the id is a capability the backend must have issued, never
 * one the client invents.
 */
export function uploadSessionDocument(
  contentB64: string,
  filename: string,
  sessionId?: string,
): Promise<{ session_id: string; stored_name: string }> {
  return request<{ session_id: string; stored_name: string }>("/v1/studio/session-documents", {
    method: "POST",
    body: JSON.stringify({
      content_b64: contentB64,
      filename,
      ...(sessionId ? { session_id: sessionId } : {}),
    }),
  });
}
