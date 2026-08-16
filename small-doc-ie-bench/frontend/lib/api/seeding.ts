// Model acquisition: seeding the store from a local Ollama/HF reference,
// Hugging Face Hub search/inspect/collection browsing (direct-from-Hub
// seeding, the preferred path), and the enriched catalog browser built on
// the same search endpoint.

import { request, type TriggerResponse } from "./core";

/** Seed a store entry from a local Ollama/HF reference (POST /v1/studio/seed-ollama). */
export interface SeedOllamaRequest {
  reference: string; // e.g. "qwen2.5:1.5b"
  name: string; // store entry name
  family?: string; // defaults "openai_chat" server-side
  /** On-disk vision projector (GGUF) for needs_mmproj families whose pull ships none. */
  mmproj?: string;
}

/** Seed the store from a local Ollama/HF reference. Returns a trigger to stream. */
export function seedOllama(payload: SeedOllamaRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/seed-ollama", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * A durable seed-download record (GET /v1/studio/seeds) -- the Downloads
 * tab's source of truth. `getSeedProgress` below covers a download's LIVE
 * percentage while it's in flight; this is what happened to it afterward
 * (or while it's running, alongside the live progress), including the error
 * text a failed seed's realtime publish would otherwise only ever reach a
 * subscriber watching at that exact moment.
 */
export interface SeedRunSummary {
  event_id: string;
  channel: string;
  kind: "ollama" | "hf";
  reference: string | null;
  name: string;
  status: "running" | "completed" | "failed";
  error: string | null;
  result: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

/** This tenant's recent seed-download jobs, newest first. */
export function listSeedRuns(): Promise<SeedRunSummary[]> {
  return request<SeedRunSummary[]>("/v1/studio/seeds");
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
  revision?: string | null;
  architecture?: string | null;
  /** "supported" | "needs_family" | "unsupported". */
  verdict: string;
  /** Deploy-time outcome after artifacts, RAM, and node capacity are considered. */
  readiness?: "ready" | "caution" | "blocked";
  family?: string | null;
  runtime?: "llama.cpp" | "transformers" | "encoder" | null;
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
  recommended_quant?: string | null;
  context_length?: number;
  node_available_bytes?: number | null;
  fits_node?: boolean | null;
  download_size_bytes?: number | null;
  estimated_ram_bytes?: number | null;
  required_files?: HfPreflightFile[];
  artifact_options?: HfArtifactOption[];
  blockers?: HfPreflightMessage[];
  warnings?: HfPreflightMessage[];
  recommendations?: string[];
  suggested_name?: string;
}

export interface HfPreflightFile {
  filename: string;
  role: "model" | "vision_projector" | "weights" | "support";
  size_bytes?: number | null;
}

export interface HfPreflightMessage {
  code: string;
  message: string;
}

export interface HfArtifactOption {
  kind: "gguf" | "snapshot";
  label: string;
  quant?: string | null;
  filename?: string | null;
  required_files: HfPreflightFile[];
  download_size_bytes?: number | null;
  estimated_ram_bytes?: number | null;
  node_available_bytes?: number | null;
  fits_node?: boolean | null;
  recommended: boolean;
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
export function inspectHf(repo: string, contextLength?: number): Promise<HfInspect> {
  const q = new URLSearchParams({ repo });
  if (contextLength != null) q.set("context_length", String(contextLength));
  return request<HfInspect>(`/v1/studio/hf/inspect?${q.toString()}`);
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
