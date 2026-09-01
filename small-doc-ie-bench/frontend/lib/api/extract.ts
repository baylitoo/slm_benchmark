// Extraction trigger, realtime subscription, run-status polling fallback,
// and document rasterization.

import { request, type TriggerResponse } from "./core";

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
