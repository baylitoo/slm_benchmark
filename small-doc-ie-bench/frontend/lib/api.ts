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
  /** Measured generation rate. Null when never measured — never a guess: this
   * number only ever comes from an actual request to the deployment. */
  tokens_per_second?: number | null;
  /** Time to the first token, in milliseconds. Null when the runtime did not
   * report it (a non-streamed total round-trip is not a TTFT). */
  ttft_ms?: number | null;
  /** When the measurement was taken (ISO). Drives the "measured N ago" label
   * and the stale marker. */
  throughput_measured_at?: string | null;
  /** Where the numbers came from: "timings" (the runtime's own server-side
   * measurement), "wall-clock" (timed by the caller), "not-applicable" (a
   * deployment that generates no tokens), "unmeasured". */
  throughput_source?: string | null;
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
  /** Coarse failure category derived server-side from state + last_error, for a
   * badge. One of: "oom" | "insufficient-memory" | "port-conflict" |
   * "spawn-error" | "crashed" | "unhealthy". Null when not in a failure. */
  failure_kind?: string | null;
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
  /** True for embedding families (served via /v1/embeddings). */
  embedding?: boolean;
  /** True for reranker families (served via /v1/rerank). */
  reranker?: boolean;
  /** True for analyzer (encoder) families (served by the encoder runtime). */
  analyzer?: boolean;
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
  /** True for embedding families — served with --embedding, used via /v1/embeddings. */
  embedding?: boolean;
  /** True for reranker families — served with --reranking --embedding --pooling
   * rank, used via /v1/rerank. */
  reranker?: boolean;
  /** True for analyzer (encoder) families — served by the encoder runtime. */
  analyzer?: boolean;
  /** The analyzer library for an analyzer family (e.g. "gliner" | "gliner2"). */
  encoder_backend?: string | null;
  /** True for the transformers/AutoModel LAST-RESORT families (no GGUF path). */
  transformers_runtime?: boolean;
  /** True when the family executes the repo's custom Python on the serving node. */
  trust_remote_code?: boolean;
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

  // A BARE 404/501 means the route isn't built/enabled on this backend and the
  // UI degrades to its "not available" state. A 404 carrying X-Docie-Error is
  // a DOMAIN answer ("deployment 'x' not found") — surface its detail verbatim
  // instead of swallowing it as "endpoint unavailable".
  if (isUnavailableStatus(res.status) && !res.headers.get("x-docie-error")) {
    throw new ApiUnavailable(res.status, detailOf(body, "Endpoint not available"));
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

/** Recover a stuck/failed deployment on a (re)allocated port (no delete+recreate).
 *  `port` omitted / null = auto-reallocate a free port (steps around an orphan
 *  still holding the old one); an explicit port is honored verbatim. */
export function repairDeployment(
  name: string,
  port?: number | null,
): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}/repair`,
    { method: "POST", body: JSON.stringify(port != null ? { port } : {}) },
  );
}

/** A deployment's runtime log tail (GET /v1/serving/deployments/{name}/logs). */
export interface DeploymentLogs {
  name: string;
  /** Reconciler one-line failure summary, if any. */
  last_error?: string | null;
  /** Raw stdout/stderr tail (most recent last). */
  lines: string[];
}

export function getDeploymentLogs(name: string, lines = 200): Promise<DeploymentLogs> {
  return request<DeploymentLogs>(
    `/v1/serving/deployments/${encodeURIComponent(name)}/logs?lines=${lines}`,
  );
}

/** A registered benchmark dataset (GET /v1/studio/datasets) — the values a
 * Benchmark run's `dataset` field can actually reference. */
export interface DatasetSummary {
  name: string;
  description: string | null;
  latest: string | null;
  versions: string[];
  documents: number | null;
}

/** Registered datasets, so the Benchmark form can offer a picker instead of a
 * free-text guess at a name that may not be registered. */
export function listDatasets(): Promise<DatasetSummary[]> {
  return request<DatasetSummary[]>("/v1/studio/datasets");
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
// Human review queue (Observability tab tile — GET /v1/reviews/metrics)
// ---------------------------------------------------------------------------

export interface ReviewMetricsView {
  queue_depth: Record<string, number>;
  correction_rate: number | null;
  reviewer_agreement: number | null;
  average_queue_latency_seconds: number | null;
  workload_by_reviewer: Record<string, Record<string, number>>;
}

/** Review-queue health: depth by status, correction/agreement rates. 422
 * ("Review workflow requires DATABASE_URL") when database persistence isn't
 * enabled — the whole review workflow needs it. */
export function getReviewMetrics(): Promise<ReviewMetricsView> {
  return request<ReviewMetricsView>("/v1/reviews/metrics");
}

// ---------------------------------------------------------------------------
// OCR cache utilization (Observability tab tile — GET /v1/serving/ocr-cache)
// ---------------------------------------------------------------------------

export interface OcrCacheStatsView {
  enabled: boolean;
  entry_count?: number;
  total_bytes?: number;
  max_bytes?: number;
  utilization_pct?: number | null;
  oldest_entry_age_seconds?: number | null;
  newest_entry_age_seconds?: number | null;
}

/** On-disk OCR cache size/entry count, scanned live from the shared cache
 * dir — accurate regardless of which api/worker replica served a request.
 * Hit-rate isn't included: that needs aggregation across replicas the same
 * way an autoscale load signal would, and isn't built yet. */
export function getOcrCacheStats(): Promise<OcrCacheStatsView> {
  return request<OcrCacheStatsView>("/v1/serving/ocr-cache");
}

// ---------------------------------------------------------------------------
// Store-model activity (Observability tab tile — GET /v1/serving/activity)
// ---------------------------------------------------------------------------

export interface ActivityEntry {
  model_name: string;
  window_count: number;
  window_started_at: string | null;
  last_request_at: string | null;
  updated_at: string | null;
  /** Placements for this model currently state=ready with a live endpoint. */
  live_replica_count: number;
  /** All non-removed placements for this model, live or not (0 for a
   * model that has request history but was never actually deployed, e.g.
   * a stale row from a since-cleared catalog). */
  total_replica_count: number;
}

export interface ActivityView {
  entries: ActivityEntry[];
  detail?: string;
}

/** Per-store-model request counts since the window last reset — purely
 * observational (see catalog.ModelActivity's docstring): nothing scales on
 * this yet, it's here so an operator can see load next to the Sizing tab's
 * fit numbers before anyone builds a decision on top of it. */
export function getActivity(): Promise<ActivityView> {
  return request<ActivityView>("/v1/serving/activity");
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

/**
 * Result of scaling a store model to a target replica count
 * (POST /v1/serving/store/{name}/scale). Idempotent: `adding` is empty when the
 * model is already at/above the requested target.
 */
export interface ScaleResult {
  model: string;
  /** The requested TARGET TOTAL number of deployments of this model. */
  target: number;
  /** Deployments of this model that already existed when scaling. */
  current: number;
  /** New deployment names spun up (empty when already at target). */
  adding: string[];
  event_ids: string[];
  channel: string | null;
  [k: string]: unknown;
}

/**
 * Scale a store model to a TARGET TOTAL replica count (idempotent): at or above
 * the target the server spawns nothing. `replicas` is the absolute target, not
 * a delta — callers add the desired count to the running instances.
 */
export function scaleStoreModel(name: string, replicas: number): Promise<ScaleResult> {
  return request<ScaleResult>(
    `/v1/serving/store/${encodeURIComponent(name)}/scale`,
    { method: "POST", body: JSON.stringify({ replicas }) },
  );
}

/** Extraction schema names available for structured output (GET /v1/schemas). */
export function listSchemas(): Promise<string[]> {
  return request<{ schemas: string[] }>("/v1/schemas").then((r) => r.schemas ?? []);
}

/** A seed download's latest progress — emitted on the realtime `progress` topic
 * AND persisted to a pollable sidecar so the polling fallback shows the same bar. */
export interface SeedProgress {
  percent?: number | null;
  received_bytes?: number;
  total_bytes?: number | null;
  stage?: string;
  file?: string;
}

/** Poll a seed run's latest download progress by channel (realtime-free). Returns
 * null progress until the first chunk lands (or once the download settles). */
export function getSeedProgress(channel: string): Promise<SeedProgress | null> {
  return request<{ channel: string; progress: SeedProgress | null }>(
    `/v1/serving/seed-progress?channel=${encodeURIComponent(channel)}`,
  ).then((r) => r.progress ?? null);
}

/** Rasterize an uploaded PDF (or image) to PNG page-image data URLs a vision
 * model can read (POST /v1/studio/render-document). One data URL per page. */
export function renderDocument(
  contentB64: string,
  filename: string,
  dpi?: number,
  maxPages?: number,
): Promise<{ images: string[]; pages: number }> {
  return request<{ images: string[]; pages: number }>("/v1/studio/render-document", {
    method: "POST",
    body: JSON.stringify({
      content_b64: contentB64,
      filename,
      ...(dpi ? { dpi } : {}),
      ...(maxPages ? { max_pages: maxPages } : {}),
    }),
  });
}

/** Seed the store from a local Ollama/HF reference. Returns a trigger to stream. */
export function seedOllama(payload: SeedOllamaRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/seed-ollama", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------------------------
// Hugging Face direct seeding (preferred path — no Ollama dependency)
// ---------------------------------------------------------------------------

/** One GGUF file of a Hub repo (GET /v1/studio/hf/repo). */
export interface HfGgufFileView {
  filename: string;
  size_bytes?: number | null;
  quant?: string | null;
  is_mmproj?: boolean;
  is_multipart?: boolean;
}

export interface HfRepoView {
  repo: string;
  suggested_name: string;
  ggufs: HfGgufFileView[];
}

/** A provider-curated collection (GET /v1/studio/hf/collection). */
export interface HfCollectionView {
  slug: string;
  title: string;
  models: string[];
}

export interface SeedHfRequest {
  repo: string;
  quant?: string | null;
  /** True (batch/collection): quant is a preference — fall back per repo instead of failing. */
  quant_prefer?: boolean;
  name?: string | null;
  family?: string;
}

/** A Hub search result card (GET /v1/studio/hf/search). */
export interface HfSearchCard {
  id: string;
  downloads?: number | null;
  likes?: number | null;
  gated?: boolean;
  tags?: string[];
}

/** Pre-flight support verdict for a repo (GET /v1/studio/hf/inspect). */
export interface HfInspect {
  repo: string;
  architecture?: string | null;
  /** "supported" | "needs_family" | "unsupported". */
  verdict: string;
  family?: string | null;
  reason?: string;
  /** A runtime-support caveat (e.g. "needs a recent llama-server"), if any. */
  runtime_note?: string | null;
  has_gguf?: boolean;
  has_safetensors?: boolean;
  has_mmproj?: boolean;
  /** Transformers last resort only: the checkpoint runs custom repo code on the
   * node (config auto_map) — deploy via the transformers_trust_remote_code family. */
  needs_trust_remote_code?: boolean;
  quants?: string[];
  suggested_name?: string;
}

/** Search the Hub for deployable models (server-side proxy). */
export function searchHf(query: string, gguf = true): Promise<HfSearchCard[]> {
  return request<HfSearchCard[]>(
    `/v1/studio/hf/search?query=${encodeURIComponent(query)}&gguf_only=${gguf}`,
  );
}

// ---------------------------------------------------------------------------
// Catalog browser (enriched /v1/studio/hf/search — the Azure-Foundry-like view)
// ---------------------------------------------------------------------------

/** How the catalog orders results. Empty query + a sort = "top trending/new". */
export type CatalogSort = "trending" | "downloads" | "likes" | "recent";

/**
 * An enriched Hub search result — the same `/v1/studio/hf/search` endpoint as
 * `searchHf`, extended with an in-list pre-verdict + serving-family resolution
 * so the catalog can segment models by support tier before any download.
 *
 * EVERY enriched field is defensively nullable: while the backend is still
 * emitting the plain `HfSearchCard` shape these keys arrive as `undefined`, so
 * the UI guards each with `== null` and renders a dash. `verdict`/`family`/
 * `prelim` are a FAST pre-check — `/v1/studio/hf/inspect` remains authoritative.
 */
export interface CatalogCard {
  id: string;
  downloads: number | null;
  downloads_all_time: number | null;
  likes: number | null;
  trending_score: number | null;
  gated: boolean;
  tags: string[];
  pipeline_tag: string | null;
  library_name: string | null;
  created_at: string | null;
  last_modified: string | null;
  license: string | null;
  architecture: string | null;
  params: number | null;
  param_label: string | null;
  size_est_bytes: number | null;
  verdict: "supported" | "needs_family" | "unsupported" | null;
  family: string | null;
  reason: string;
  runtime_note: string | null;
  prelim: boolean;
  /** Estimated size fits the node's deploy budget (free − safety margin — the
   *  exact quantity the deploy fit-check uses). false = too big; null = unknown
   *  (no node snapshot / DB down) — never render a false "fits" for null. */
  fits_node: boolean | null;
  /** That deploy budget in bytes, for a tooltip. */
  node_available_bytes: number | null;
}

/** Query for the enriched catalog search. All fields optional/defensive. */
export interface CatalogSearchParams {
  query?: string;
  sort?: CatalogSort;
  pipeline_tag?: string | null;
  author?: string | null;
  gguf_only?: boolean;
  limit?: number;
}

/**
 * Enriched Hub search for the Catalog browser. Hits the SAME endpoint as
 * `searchHf` (`/v1/studio/hf/search`, backward compatible) with the extra facet
 * + sort params. An empty `query` combined with a `sort` returns the top
 * trending/most-downloaded/newest models — the catalog's default first paint.
 */
export function searchCatalog(params: CatalogSearchParams = {}): Promise<CatalogCard[]> {
  const q = new URLSearchParams();
  q.set("query", params.query ?? "");
  if (params.sort) q.set("sort", params.sort);
  if (params.pipeline_tag) q.set("pipeline_tag", params.pipeline_tag);
  if (params.author) q.set("author", params.author);
  q.set("gguf_only", String(params.gguf_only ?? true));
  if (params.limit != null) q.set("limit", String(params.limit));
  return request<CatalogCard[]>(`/v1/studio/hf/search?${q.toString()}`);
}

/** Pre-flight support verdict + suggested family for a repo (no download). */
export function inspectHf(repo: string): Promise<HfInspect> {
  return request<HfInspect>(`/v1/studio/hf/inspect?repo=${encodeURIComponent(repo)}`);
}

/** Live GGUF/quant listing of a Hub repo (server-side proxy, HF_TOKEN aware). */
export function getHfRepo(repo: string): Promise<HfRepoView> {
  return request<HfRepoView>(`/v1/studio/hf/repo?repo=${encodeURIComponent(repo)}`);
}

/** The model repos of a HF collection (owner/slug-hash or its full URL). */
export function getHfCollection(slug: string): Promise<HfCollectionView> {
  return request<HfCollectionView>(
    `/v1/studio/hf/collection?slug=${encodeURIComponent(slug)}`,
  );
}

/** Download a GGUF straight from the Hub into the store; progress streams live. */
export function seedHf(payload: SeedHfRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/seed-hf", {
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

/** POST an OpenAI-shaped chat body and surface OpenAI-shaped errors readably. */
async function openaiPost(url: string, payload: unknown): Promise<AgentChatResponse> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
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
): Promise<AgentChatResponse> {
  return openaiPost(`${API_BASE}/v1/chat/completions`, { model, messages });
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
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
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
 * The semantic model type of a deployment — what it can be USED for.
 * "chat" (generative SLMs: extraction, agent backing, Playground chat),
 * "encoder" (analyzers: agent guard models), "embedding" (vectors:
 * /v1/embeddings, RAG), "reranker" (relevance scoring: /v1/rerank). Encoder is
 * derived from the launch runtime; embedding/reranker need the family, so pass
 * `embeddingNames`/`rerankerNames` (deployment names whose store family is
 * that family — see embeddingDeploymentNames / rerankerDeploymentNames).
 */
export type DeployedModelType = "chat" | "encoder" | "embedding" | "reranker";

export function deploymentModelType(
  r: DeploymentRecord,
  embeddingNames?: Set<string>,
  rerankerNames?: Set<string>,
): DeployedModelType {
  if (r.spec?.launch?.runtime === "encoder") return "encoder";
  const name = r.spec?.name;
  if (name && embeddingNames?.has(name)) return "embedding";
  if (name && rerankerNames?.has(name)) return "reranker";
  return "chat";
}

/**
 * Deployment names that are embedding models: a store entry whose family is
 * flagged `embedding`. Cross-references the store catalog and the family
 * contracts (the deployment record itself carries no family). A store entry's
 * name equals its store-deployed deployment name.
 */
export function embeddingDeploymentNames(
  store: StoreEntry[] | null | undefined,
  families: ModelFamily[] | null | undefined,
): Set<string> {
  const embeddingFamilies = new Set(
    (families ?? []).filter((f) => f.embedding).map((f) => f.name),
  );
  return new Set(
    (store ?? [])
      .filter((e) => e.family && embeddingFamilies.has(e.family))
      .map((e) => e.name),
  );
}

/**
 * Deployment names that are reranker models: a store entry whose family is
 * flagged `reranker`. Mirrors embeddingDeploymentNames exactly.
 */
export function rerankerDeploymentNames(
  store: StoreEntry[] | null | undefined,
  families: ModelFamily[] | null | undefined,
): Set<string> {
  const rerankerFamilies = new Set(
    (families ?? []).filter((f) => f.reranker).map((f) => f.name),
  );
  return new Set(
    (store ?? [])
      .filter((e) => e.family && rerankerFamilies.has(e.family))
      .map((e) => e.name),
  );
}

/**
 * Deployment names that are VISION models: a store entry with vision=true
 * (lfm2_vl / nuextract3 / vision_ocr families). The deployment record carries
 * no vision flag, so this cross-references the store catalog. A store entry's
 * name equals its store-deployed deployment name.
 */
export function visionDeploymentNames(
  store: StoreEntry[] | null | undefined,
): Set<string> {
  return new Set((store ?? []).filter((e) => e.vision).map((e) => e.name));
}

/** One embedding vector from GET-shaped /v1/embeddings response. */
export interface EmbeddingResponse {
  data?: { index?: number; embedding?: number[] }[];
  model?: string;
  [k: string]: unknown;
}

/** OpenAI embeddings against an embedding deployment (POST /v1/embeddings). */
export function embed(model: string, input: string | string[]): Promise<EmbeddingResponse> {
  return request<EmbeddingResponse>("/v1/embeddings", {
    method: "POST",
    body: JSON.stringify({ model, input }),
  });
}

export interface RerankResponse {
  results?: { index?: number; relevance_score?: number }[];
  model?: string;
  [k: string]: unknown;
}

/** Rerank documents against a query over a reranker deployment (POST /v1/rerank). */
export function rerank(
  model: string,
  query: string,
  documents: string[],
  topN?: number,
): Promise<RerankResponse> {
  return request<RerankResponse>("/v1/rerank", {
    method: "POST",
    body: JSON.stringify({
      model,
      query,
      documents,
      ...(topN != null ? { top_n: topN } : {}),
    }),
  });
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
