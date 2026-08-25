// Sizing tab (PR-3): node RAM snapshot, per-model fit table, what-if
// planning, and scaling a store model to a target replica count.

import { request } from "./core";

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
 * (POST /v1/serving/store/{name}/scale). Idempotent both ways: `adding` and
 * `removing` are empty when the model is already at the requested target.
 */
export interface ScaleResult {
  model: string;
  /** The requested TARGET TOTAL number of deployments of this model. */
  target: number;
  /** Deployments of this model that already existed when scaling. */
  current: number;
  /** New deployment names spun up (empty when already at/above target). */
  adding: string[];
  /** Replica names being drained/deleted (scale-down; highest suffix first,
   * the bare base record always survives). */
  removing?: string[];
  event_ids: string[];
  channel: string | null;
  [k: string]: unknown;
}

/**
 * Scale a store model to a TARGET TOTAL replica count (idempotent). Above the
 * current count, the server RAM-checks N x the per-instance footprint and fans
 * out one deploy per missing replica (a provable deficit is a 422); below it,
 * the server drains surplus replicas via the real delete path. `replicas` is
 * the absolute target, not a delta.
 */
export function scaleStoreModel(name: string, replicas: number): Promise<ScaleResult> {
  return request<ScaleResult>(
    `/v1/serving/store/${encodeURIComponent(name)}/scale`,
    { method: "POST", body: JSON.stringify({ replicas }) },
  );
}
