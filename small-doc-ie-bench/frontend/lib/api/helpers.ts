// Derived helpers over DeploymentRecord/StoreEntry/ModelFamily (live/
// reloadable/selectable classification, model-type resolution), embedding +
// reranker calls, and small display formatters.

import { request } from "./core";
import type { DeploymentRecord, ModelFamily, StoreEntry } from "./serving";

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
 * flagged `reranker` OR `multi_vector`. Both answer on /v1/rerank — a GGUF
 * cross-encoder/ColBERT via llama-server --reranking, a safetensors ColBERT
 * via the multi-vector runtime — so for "what can this deployment be USED
 * for" (the Playground Rerank tab, the Deployments type filter) they are the
 * same thing. Mirrors embeddingDeploymentNames exactly.
 */
export function rerankerDeploymentNames(
  store: StoreEntry[] | null | undefined,
  families: ModelFamily[] | null | undefined,
): Set<string> {
  const rerankerFamilies = new Set(
    (families ?? []).filter((f) => f.reranker || f.multi_vector).map((f) => f.name),
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
