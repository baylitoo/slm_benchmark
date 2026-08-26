// Observability tab tiles: human-review queue health, OCR cache
// utilization, and store-model request activity.

import { request } from "./core";

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

export type UsageWindow = "24h" | "7d" | "30d";

/** One tool's call/error counts + average latency, folded from every agent
 * request in the window that ran the MCP tool loop (surface "agent" only). */
export interface UsageToolCall {
  tool: string;
  calls: number;
  errors: number;
  avg_latency_ms: number | null;
}

/** One deployment/profile's aggregates over the requested window, folded at
 * read time from the raw usage_records ledger. ``tool_calls`` is only ever
 * non-empty for an agent that ran MCP tools during the window. */
export interface UsageDeployment {
  deployment: string;
  requests: number;
  errors: number;
  prompt_tokens: number;
  completion_tokens: number;
  avg_latency_ms: number | null;
  p95_latency_ms: number | null;
  last_used_at: string | null;
  tool_calls: UsageToolCall[];
}

export interface UsageSummaryView {
  window: UsageWindow;
  deployments: UsageDeployment[];
}

/** Per-deployment usage over a bounded window (requests, errors, tokens
 * in/out, avg+p95 latency, last used). Rows are written by the serving
 * surfaces themselves (chat/embed/rerank/extract); no DATABASE_URL degrades
 * to an empty listing, same contract as the other Studio listings. */
export function getUsageSummary(window: UsageWindow): Promise<UsageSummaryView> {
  return request<UsageSummaryView>(`/v1/studio/usage?window=${encodeURIComponent(window)}`);
}
