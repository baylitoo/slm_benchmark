// Serving domain: model manifests, runtimes, deployments, ports, deploy
// trigger, and the GGUF store catalog + family contracts.
//
// Shapes are from docie_bench.serving — see control_plane.py. All fields
// optional/defensive: the backend may grow/shrink the payload.

import { request, type TriggerResponse } from "./core";

export interface DeployRequest {
  model: string;
  runtime?: string;
  name?: string;
  port?: number;
  context_length?: number;
  max_tokens?: number;
  /** TARGET total instances of the store model (default 1). >1 fans out one
   * deploy per missing replica on auto-allocated ports — store-entry (Auto)
   * deploys only, and incompatible with an explicit name/port. */
  replicas?: number;
  /** llama-server request slots (default 1, meaningless for non-llamacpp
   * runtimes). context_length is divided across slots by the runtime, so the
   * caller must size context_length for a single slot's worth of context. */
  n_parallel?: number;
  /** llama-server --cache-reuse: min chunk size to reuse from KV cache. */
  cache_reuse?: number;
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
      max_tokens?: number | null;
      [k: string]: unknown;
    };
    desired_state?: string;
    [k: string]: unknown;
  };
  state?: string; // lifecycle: "stopped" | "starting" | "ready" | "degraded" | "failed"
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
  /** The shared launch alias this record's requests are load-balanced behind
   * (the base store name for a scaled model). */
  replica_group?: string | null;
  /** How many deployment records share `replica_group` right now (1 = not
   * scaled). Derived server-side from the records, never stored. */
  replicas?: number;
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

/** A model family contract (GET /v1/serving/families). */
export interface ModelFamily {
  name: string;
  vision?: boolean;
  needs_mmproj?: boolean;
  /** True when the family serves OpenAI tool calling (llama-server --jinja). */
  tools?: boolean;
  /** True for embedding families — served with --embedding, used via /v1/embeddings. */
  embedding?: boolean;
  /** True for reranker families — served with --reranking --embedding --pooling
   * rank, used via /v1/rerank. */
  reranker?: boolean;
  /** True for multi-vector (ColBERT / PyLate late-interaction) families — a
   * safetensors snapshot served by the multi-vector runtime (sentence-
   * transformers MultiVectorEncoder), also used via /v1/rerank. */
  multi_vector?: boolean;
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

export function getModels(): Promise<ModelManifest[]> {
  return request<ModelManifest[]>("/v1/serving/models");
}

export function getRuntimes(): Promise<RuntimeCapability[]> {
  return request<RuntimeCapability[]>("/v1/serving/runtimes");
}

export function getDeployments(): Promise<DeploymentRecord[]> {
  return request<DeploymentRecord[]>("/v1/serving/deployments");
}

/** One llama-server ``GET /slots`` entry, defensively normalized server-side
 * (#315) -- the schema varies by build, so every field but `id` is optional
 * and a field simply absent here means THIS build didn't report it, not an
 * error. `prompt` is truncated server-side. */
export interface LlamaCppSlot {
  id?: number;
  is_processing?: boolean;
  n_ctx?: number;
  prompt?: string;
  id_task?: number;
  n_past?: number;
  n_remain?: number;
  n_decoded?: number;
  cache_n?: number;
  prompt_n?: number;
  prompt_ms?: number;
  predicted_n?: number;
  predicted_ms?: number;
  tokens_per_second?: number;
  next_token?: Record<string, boolean | number | string>;
  [k: string]: unknown;
}

/** llama-server's own worker-pool-equivalent introspection (#315): per-slot
 * prompt state, KV cache reuse, and (on builds that report it) prefill/
 * decode timing, queried live and on demand -- same pattern as
 * getCodeInterpreterWorkers's live call to Judge0's worker pool. 404s for a
 * non-llama.cpp deployment or one with no live endpoint yet. */
export function getDeploymentSlots(
  name: string,
): Promise<{ name: string; slots: LlamaCppSlot[] }> {
  return request(`/v1/serving/deployments/${encodeURIComponent(name)}/slots`);
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

/** Available models = the GGUF store catalog. 501 means the catalog isn't enabled. */
export function getStore(): Promise<StoreEntry[]> {
  return request<StoreEntry[]>("/v1/serving/store");
}

export function getFamilies(): Promise<ModelFamily[]> {
  return request<ModelFamily[]>("/v1/serving/families");
}
