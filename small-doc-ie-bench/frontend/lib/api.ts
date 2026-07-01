// Typed client for the DocIE Studio backend.
//
// Every call is tolerant of endpoints that don't exist yet: a 404/501 is
// surfaced as a structured `ApiUnavailable` so the UI can render a friendly
// "coming soon" state instead of crashing.

import { API_BASE } from "./env";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ExtractRequest {
  text?: string;
  content_b64?: string;
  filename?: string;
  schema_name?: string;
  model_profile?: string;
  ocr_backend?: string;
  language?: string;
}

export interface TriggerResponse {
  event_ids: string[];
  channel: string;
  topics: string[];
}

// Inngest realtime subscription token. Shape is defined by the server SDK; we
// pass it through to the React hook untouched.
export type RealtimeToken = Record<string, unknown>;

// One run as proxied from Inngest's `/v1/events/{id}/runs`.
export interface InngestRun {
  run_id?: string;
  status?: string; // "Running" | "Completed" | "Failed" | "Cancelled" ...
  output?: unknown;
  [k: string]: unknown;
}

export interface DeployRequest {
  model: string;
  runtime?: string;
  name?: string;
  port?: number;
  context_length?: number;
  replicas?: number;
  [k: string]: unknown;
}

export interface BenchmarkRequest {
  dataset: string; // required server-side (POST /v1/studio/benchmark)
  split?: string;
  model_profile?: string;
  schema_name?: string;
  concurrency?: number;
  repeat?: number;
  language?: string;
  [k: string]: unknown;
}

// ---------------------------------------------------------------------------
// Serving domain shapes (from docie_bench.serving — see control_plane.py).
// All fields optional/defensive: the backend may grow/shrink the payload.
// ---------------------------------------------------------------------------

/** One runtime → compatibility record on a model manifest. */
export interface RuntimeCompatibility {
  compatible?: boolean;
  reason?: string;
  checked_version?: string | null;
}

/** A model manifest (GET /v1/serving/models). */
export interface ModelManifest {
  model_id?: string;
  source?: string;
  revision?: string;
  license?: string | null;
  state?: string; // "ready" | "serving" | "downloading" | "failed" | ...
  aliases?: string[];
  tags?: string[];
  supported_tasks?: string[];
  quantization?: string | null;
  precision?: string | null;
  context_length?: number | null;
  required_memory_gb?: number | null;
  required_disk_gb?: number | null;
  runtime_compatibility?: Record<string, RuntimeCompatibility>;
  [k: string]: unknown;
}

/** A runtime capability probe (GET /v1/serving/runtimes). */
export interface RuntimeCapability {
  runtime?: string;
  version?: string | null;
  installed?: boolean;
  compatible?: boolean;
  features?: string[];
  reasons?: string[];
  [k: string]: unknown;
}

/** A deployment record (GET /v1/serving/deployments). */
export interface DeploymentRecord {
  spec?: {
    name?: string;
    launch?: {
      runtime?: string;
      model?: string;
      alias?: string;
      host?: string;
      port?: number;
      context_length?: number;
      [k: string]: unknown;
    };
    desired_state?: string;
    [k: string]: unknown;
  };
  state?: string; // lifecycle: "running" | "stopped" | ...
  pid?: number | null;
  endpoint?: string | null;
  restart_count?: number;
  last_error?: string | null;
  updated_at?: number;
  [k: string]: unknown;
}

/** A GGUF model-store entry (GET /v1/serving/store). */
export interface StoreEntry {
  name: string;
  family?: string;
  vision?: boolean;
  /** Backends that can serve THIS model faithfully — the runtime picker source. */
  available_backends?: string[];
  has_mmproj?: boolean;
  source?: string | null;
  size_bytes?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
  [k: string]: unknown;
}

/** A model family contract (GET /v1/serving/families). */
export interface ModelFamily {
  name: string;
  vision?: boolean;
  needs_mmproj?: boolean;
  ollama_faithful?: boolean;
  template_delivery?: string;
  [k: string]: unknown;
}

/** Seed a store entry from a local Ollama/HF reference (POST /v1/studio/seed-ollama). */
export interface SeedOllamaRequest {
  reference: string; // e.g. "qwen2.5:1.5b"
  name: string; // store entry name
  family?: string; // defaults "openai_chat" server-side
}

/** A completed benchmark run (GET /v1/serving/benchmarks). */
export interface BenchmarkRun {
  run: string;
  path: string;
  metrics?: { summary?: Record<string, unknown>[] } | null;
  [k: string]: unknown;
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

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

function isUnavailableStatus(status: number): boolean {
  return status === 404 || status === 501;
}

async function readBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function detailOf(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    return JSON.stringify(d);
  }
  if (typeof body === "string" && body) return body;
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
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

  if (isUnavailableStatus(res.status)) {
    throw new ApiUnavailable(res.status, detailOf(body, "Endpoint not available yet"));
  }
  throw new ApiError(res.status, detailOf(body, `Request failed (HTTP ${res.status})`));
}

// ---------------------------------------------------------------------------
// Studio (live) endpoints
// ---------------------------------------------------------------------------

export function triggerExtract(payload: ExtractRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/extract", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getRealtimeToken(
  channel: string,
  topics: string[],
): Promise<RealtimeToken> {
  const params = new URLSearchParams();
  params.set("channel", channel);
  for (const t of topics) params.append("topics", t);
  return request<RealtimeToken>(`/v1/studio/realtime-token?${params.toString()}`);
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

// ---------------------------------------------------------------------------
// Serving / Deploy / Benchmark (not yet implemented server-side)
// ---------------------------------------------------------------------------

export function getModels(): Promise<ModelManifest[]> {
  return request<ModelManifest[]>("/v1/serving/models");
}

export function getRuntimes(): Promise<RuntimeCapability[]> {
  return request<RuntimeCapability[]>("/v1/serving/runtimes");
}

export function getDeployments(): Promise<DeploymentRecord[]> {
  return request<DeploymentRecord[]>("/v1/serving/deployments");
}

/** Deploy returns the same trigger shape as extract: { event_ids, channel, topics }. */
export function deployModel(payload: DeployRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/deploy", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Benchmark returns the same trigger shape as extract. */
export function triggerBenchmark(payload: BenchmarkRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/benchmark", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getBenchmarks(): Promise<BenchmarkRun[]> {
  return request<BenchmarkRun[]>("/v1/serving/benchmarks");
}

// ---------------------------------------------------------------------------
// Model store (Deploy tab source of truth)
// ---------------------------------------------------------------------------

/** Available models = the GGUF store catalog. 501 means the catalog isn't enabled. */
export function getStore(): Promise<StoreEntry[]> {
  return request<StoreEntry[]>("/v1/serving/store");
}

export function getFamilies(): Promise<ModelFamily[]> {
  return request<ModelFamily[]>("/v1/serving/families");
}

/** Seed the store from a local Ollama/HF reference. Returns a trigger to stream. */
export function seedOllama(payload: SeedOllamaRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/seed-ollama", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------------------------
// Derived helpers
// ---------------------------------------------------------------------------

/** Human-readable byte size, e.g. 1234567 -> "1.2 MB". */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

// ---------------------------------------------------------------------------
// Browser helpers
// ---------------------------------------------------------------------------

/** Read a File as base64 (without the `data:...;base64,` prefix). */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read file"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("Unexpected FileReader result"));
        return;
      }
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.readAsDataURL(file);
  });
}

export function statusIs(run: InngestRun, ...want: string[]): boolean {
  const s = (run.status ?? "").toString().toLowerCase();
  return want.some((w) => w.toLowerCase() === s);
}
