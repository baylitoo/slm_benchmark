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
