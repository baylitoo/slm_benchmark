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
  /** Free-text models.yaml/CLI profile name. Retained for back-compat. */
  model_profile?: string;
  /**
   * Explicit live-deployment selector = a DeploymentRecord `spec.name`. The
   * backend resolves it to that deployment's runtime endpoint (PR-a resolver);
   * it wins over `model_profile`. The Playground sends only this field.
   */
  deployment?: string;
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

// Record-derived view of the serving port window (GET /v1/serving/ports).
// `recommended_next` is a HINT: the worker re-derives + socket-probes at deploy
// time and may pick differently — never treat it as a reservation.
export interface PortsView {
  range: { start: number; end: number };
  deployments: { name: string | null; port: number; state: string | null }[];
  used: number[];
  free_sample: number[];
  recommended_next: number | null;
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

/** The reconciler's per-cycle observed overlay on a deployment (PR-1/PR-4). */
export interface ObservedPlacement {
  phase?: string | null; // hot | loading | cold | evicted | failed
  rss_bytes?: number | null;
  health_ok?: boolean | null;
  endpoint?: string | null;
  last_error?: string | null;
  last_probe_at?: string | null;
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
  /** Lifecycle-control metadata (PR-4): who stopped it — "manual" stays cold,
   * "managed" (evicted) auto-reloads on the next request. */
  activation?: string;
  /** Pinned deployments are never chosen for idle unload / eviction. */
  pinned?: boolean;
  last_served?: number | null;
  /** Reconciler-published observed state; null until first published,
   * observed_available=false when the database is unreachable. */
  observed?: ObservedPlacement | null;
  observed_available?: boolean;
  [k: string]: unknown;
}

/** Response of the lifecycle action endpoints (load/unload/pin/delete). */
export interface LifecycleActionResponse {
  event_ids: string[];
  channel: string;
  name: string;
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

// ---------------------------------------------------------------------------
// Sizing (GET /v1/serving/sizing, POST /v1/serving/sizing/whatif) — PR-3.
// ---------------------------------------------------------------------------

/** The reconciler-published node snapshot (also under /v1/serving/resources). */
export interface NodeSnapshot {
  total_bytes: number;
  free_bytes: number;
  /** "cgroup" (authoritative limit) | "vm" (soft — badge it). */
  source: string;
  sum_rss_bytes: number;
  reclaimable_bytes?: number;
  updated_at?: string | null;
  [k: string]: unknown;
}

/** One fit-table row: how a store model prices and how many more fit now. */
export interface SizingModelFit {
  name: string;
  family?: string | null;
  predicted_bytes?: number | null;
  calibrated_bytes?: number | null;
  /** True when a measured steady-state RSS backs the footprint. */
  calibrated?: boolean;
  /** max(calibrated, predicted); null = unpriceable (see detail). */
  footprint_bytes?: number | null;
  /** Live instances (display only — their RSS is already inside "used"). */
  running_instances?: number;
  /** null = unpriceable or no node snapshot. */
  fits_now?: number | null;
  detail?: string | null;
  [k: string]: unknown;
}

export interface SizingView {
  observed_available: boolean;
  detail?: string | null;
  total_bytes?: number | null;
  free_bytes?: number | null;
  source?: string | null;
  safety_margin_bytes?: number | null;
  /** RAM reserved for mid-load (mmap-ramp) deployments not yet fully resident. */
  loading_reserved_bytes?: number | null;
  /** free - margin - loading reserve; may be negative (honest red number). */
  free_effective_bytes?: number | null;
  assumptions?: {
    context_length?: number;
    n_parallel?: number;
    margin_fraction?: number;
  };
  per_model: SizingModelFit[];
  node?: NodeSnapshot | null;
  [k: string]: unknown;
}

/** One staged line of a hypothetical mix (POST body item). */
export interface WhatIfPlanEntry {
  model: string;
  instances: number;
  context_length?: number | null;
}

export interface WhatIfItemResult {
  model: string;
  instances: number;
  context_length: number;
  footprint_bytes: number;
  subtotal_bytes: number;
  calibrated: boolean;
  [k: string]: unknown;
}

export interface WhatIfView {
  observed_available: boolean;
  detail?: string | null;
  total_predicted_bytes: number;
  free_effective_bytes?: number | null;
  safety_margin_bytes?: number | null;
  /** RAM reserved for mid-load (mmap-ramp) deployments not yet fully resident. */
  loading_reserved_bytes?: number | null;
  remaining_bytes?: number | null;
  /** true fits · false deficit · null = no snapshot to judge against. */
  ok?: boolean | null;
  deficit_bytes?: number | null;
  margin_fraction?: number;
  per_item: WhatIfItemResult[];
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
  /** On-disk vision projector (GGUF) for needs_mmproj families whose pull ships none. */
  mmproj?: string;
}

/** A downloadable run artifact (report.html / predictions.jsonl / metrics.json). */
export interface RunArtifact {
  id: string;
  name: string;
  media_type?: string;
  size_bytes?: number;
  sha256?: string;
  /** Addressable, path-independent URI: `/v1/studio/artifacts/{id}`. */
  uri: string;
}

/**
 * A durable benchmark run (GET /v1/studio/runs). Keyed by the Inngest event id;
 * metrics come from the index, artifacts are fetched by id from the blob store.
 * Legacy `run`/`path` fields are kept optional for back-compat with older rows.
 */
export interface BenchmarkRun {
  event_id?: string;
  run?: string; // legacy (runs_dir scan)
  path?: string; // legacy
  status?: string;
  dataset?: string | null;
  model_profile?: string | null;
  metrics?: { summary?: Record<string, unknown>[]; [k: string]: unknown } | null;
  artifacts?: RunArtifact[];
  created_at?: string | null;
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

/** Live port-allocation view for the Deploy admin table (record-derived). */
export function getPorts(): Promise<PortsView> {
  return request<PortsView>("/v1/serving/ports");
}

/** Deploy returns the same trigger shape as extract: { event_ids, channel, topics }. */
export function deployModel(payload: DeployRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/deploy", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------------------------
// Deployment lifecycle actions (PR-4). Each fires a serving/* event at the
// single-replica serving service and returns the event ids to poll.
// ---------------------------------------------------------------------------

/** Cold-start a deployment (idempotent server-side; may evict LRU victims). */
export function loadDeployment(name: string): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}/load`,
    { method: "POST" },
  );
}

/** Evict a deployment: process killed, record + port + row kept (phase=evicted). */
export function unloadDeployment(name: string): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}/unload`,
    { method: "POST" },
  );
}

/** Set/clear the eviction shield. */
export function pinDeployment(
  name: string,
  pinned: boolean,
): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}/pin`,
    { method: "POST", body: JSON.stringify({ pinned }) },
  );
}

/** Real teardown: kills the process, frees the port, deletes the row. */
export function deleteDeployment(name: string): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
}

/** Benchmark returns the same trigger shape as extract. */
export function triggerBenchmark(payload: BenchmarkRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/benchmark", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Durable, tenant-scoped benchmark runs with addressable artifacts
 * (GET /v1/studio/runs). Reachable from any replica — resolved from the shared
 * blob store + Postgres index rather than the worker's local filesystem.
 */
export function getBenchmarks(): Promise<BenchmarkRun[]> {
  return request<BenchmarkRun[]>("/v1/studio/runs");
}

/** Absolute URL for a run artifact's addressable URI (prepends the API base). */
export function artifactUrl(uri: string): string {
  return `${API_BASE}${uri}`;
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

// ---------------------------------------------------------------------------
// Sizing tab (PR-3)
// ---------------------------------------------------------------------------

/** Per-model fit table + capacity numbers, from the observed surface. */
export function getSizing(): Promise<SizingView> {
  return request<SizingView>("/v1/serving/sizing");
}

/** Price a hypothetical deployment mix — fits or an explicit deficit. */
export function whatifSizing(plan: WhatIfPlanEntry[]): Promise<WhatIfView> {
  return request<WhatIfView>("/v1/serving/sizing/whatif", {
    method: "POST",
    body: JSON.stringify({ plan }),
  });
}

/** mesh-llm status (GET /v1/serving/mesh): private pooled capacity, if configured. */
export interface MeshView {
  configured: boolean;
  endpoint: string | null;
  healthy: boolean;
  /** Model ids the mesh currently serves — routable as `mesh:<id>`. */
  models: string[];
  detail?: string | null;
}

export function getMesh(): Promise<MeshView> {
  return request<MeshView>("/v1/serving/mesh");
}

/** Seed the store from a local Ollama/HF reference. Returns a trigger to stream. */
export function seedOllama(payload: SeedOllamaRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/seed-ollama", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------------------------
// Agents (GET/POST /v1/agents — preconfigured agents over OpenAI endpoints)
// ---------------------------------------------------------------------------

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

/**
 * One synchronous chat completion against an agent's OpenAI endpoint (the Try
 * panel). Unlike `request()`, errors here arrive OpenAI-shaped
 * (`{"error": {"message", "type"}}`), so surface that message directly —
 * e.g. `guard_unavailable` when the encoder deployment is unloaded.
 */
export async function agentChat(
  name: string,
  messages: { role: string; content: unknown }[],
): Promise<AgentChatResponse> {
  let res: Response;
  try {
    res = await fetch(`${agentBaseUrl(name)}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ model: name, messages }),
    });
  } catch (e) {
    throw new ApiUnavailable(0, e instanceof Error ? e.message : "Network error");
  }
  const body = await readBody(res);
  if (res.ok) return body as AgentChatResponse;
  const err =
    body && typeof body === "object" && "error" in body
      ? (body as { error?: { message?: string; type?: string } }).error
      : undefined;
  const detail = err?.message ?? detailOf(body, `Request failed (HTTP ${res.status})`);
  throw new ApiError(res.status, err?.type ? `${err.type}: ${detail}` : detail);
}

// ---------------------------------------------------------------------------
// Derived helpers
// ---------------------------------------------------------------------------

/**
 * A deployment that is live RIGHT NOW: lifecycle `ready` AND a concrete
 * `endpoint`. Mirrors the backend resolver's `_is_live` gate.
 */
export function isLiveDeployment(r: DeploymentRecord): boolean {
  return r.state === "ready" && !!r.endpoint;
}

/**
 * A deployment a request would AUTO-RELOAD (PR-4 cold-start-on-demand):
 * evicted by the autoloader (`activation === "managed"`) or with a load
 * already in flight (`desired_state === "running"`, still starting). Mirrors
 * the worker's `_autoload_target` gate. Manually stopped deployments stay
 * cold and are deliberately NOT selectable.
 */
export function isAutoReloadable(r: DeploymentRecord): boolean {
  if (isLiveDeployment(r)) return false;
  return r.activation === "managed" || r.spec?.desired_state === "running";
}

/**
 * Deployments the Playground may route an extraction to: live ones, PLUS
 * evicted/loading `managed` ones — sending a request to those triggers the
 * worker's autoload (TTFT = model load time, by design). Requires a
 * `spec.name` (the token the backend resolver keys on). Manually stopped /
 * terminally failed deployments are excluded.
 */
export function selectableDeployments(records: DeploymentRecord[]): DeploymentRecord[] {
  return records.filter(
    (r) => !!r.spec?.name && (isLiveDeployment(r) || isAutoReloadable(r)),
  );
}

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
