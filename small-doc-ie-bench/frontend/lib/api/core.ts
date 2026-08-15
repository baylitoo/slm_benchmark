// Shared HTTP plumbing every other module in this package needs: the
// error classes, the low-level fetch wrapper, and the handful of helpers
// (readBody/detailOf/unauthorizedError) that a call site bypassing
// `request()` for a non-JSON response (downloadArtifact, the streaming/
// OpenAI-shaped chat calls) still needs to format an error consistently.
// `export`ed here (not module-private) purely so sibling files in this
// package can import them -- nothing outside `lib/api/` should reach past
// `index.ts`'s barrel.

import { API_BASE } from "../env";
import { authHeader } from "../apiKey";

/** Raised when a backend endpoint is missing/unbuilt (404) or disabled (501). */
export class ApiUnavailable extends Error {
  constructor(
    public status: number,
    message?: string,
  ) {
    super(message || `Endpoint unavailable (HTTP ${status})`);
    this.name = "ApiUnavailable";
  }
}

/** Raised for other non-OK responses (validation, server errors, ...). */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Raised instead of returning a completion when the model is a cold `store:`
 * deployment the backend just triggered a load for (HTTP 202, load-on-demand
 * — see chat_api._resolve_or_error). Distinct from ApiError: this isn't a
 * failure, it's "keep waiting" — callers should show a status, not a red error.
 */
export class ModelLoading extends Error {
  constructor(
    public deployment: string,
    public etaSeconds: number,
  ) {
    super(`Model '${deployment}' is starting — retry in ~${Math.ceil(etaSeconds)}s.`);
    this.name = "ModelLoading";
  }
}

function isUnavailableStatus(status: number): boolean {
  return status === 404 || status === 501;
}

export async function readBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export function detailOf(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    return JSON.stringify(d);
  }
  if (typeof body === "string" && body) return body;
  return fallback;
}

/**
 * 401 from `TenantQuotaManager.authenticate` (see security.py) — the server
 * gives the same "A valid API key is required" detail whether no key was
 * sent or a wrong one was, so we don't try to distinguish; we just point the
 * operator at where to fix it (this Studio has no other UI for it).
 */
export function unauthorizedError(body: unknown): ApiError {
  const detail = detailOf(body, "A valid API key is required");
  return new ApiError(401, `${detail} — set one from the key icon in the top bar.`);
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...authHeader(),
        ...init?.headers,
      },
    });
  } catch (e) {
    // Network failure / CORS / server down. Treat as "unavailable" so callers
    // can degrade gracefully.
    throw new ApiUnavailable(0, e instanceof Error ? e.message : "Network error");
  }

  const body = await readBody(res);
  if (res.ok) return body as T;
  if (res.status === 401) throw unauthorizedError(body);

  // A BARE 404/501 means the route isn't built/enabled on this backend and the
  // UI degrades to its "not available" state. A 404 carrying X-Docie-Error is
  // a DOMAIN answer ("deployment 'x' not found") — surface its detail verbatim
  // instead of swallowing it as "endpoint unavailable".
  if (isUnavailableStatus(res.status) && !res.headers.get("x-docie-error")) {
    throw new ApiUnavailable(res.status, detailOf(body, "Endpoint not available"));
  }
  throw new ApiError(res.status, detailOf(body, `Request failed (HTTP ${res.status})`));
}

/** Trigger-shape response shared by extract/deploy/benchmark/seed. */
export interface TriggerResponse {
  event_ids: string[];
  channel: string;
  topics: string[];
}
