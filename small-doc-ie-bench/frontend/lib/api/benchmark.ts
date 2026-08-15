// Benchmark trigger, durable run index + artifact download, and run
// comparison (deltas, CI, sign test, root causes, budget verdicts).

import { API_BASE } from "../env";
import { authHeader } from "../apiKey";
import {
  ApiError,
  ApiUnavailable,
  detailOf,
  readBody,
  request,
  unauthorizedError,
  type TriggerResponse,
} from "./core";

export interface BenchmarkRequest {
  dataset: string; // required server-side (POST /v1/studio/benchmark)
  split?: string;
  model_profile?: string;
  /** Server-side path to a routing-policy YAML (multi-stage fallback/escalation
   * across several profiles — see configs/routing-policy.example.yaml). Mutually
   * exclusive with model_profile and routing_policy_name; the backend 422s a
   * request carrying more than one. */
  routing_policy?: string;
  /** Name of a RoutingPolicy saved via createRoutingPolicy — the discoverable
   * alternative to routing_policy's raw filesystem path. */
  routing_policy_name?: string;
  schema_name?: string;
  concurrency?: number;
  repeat?: number;
  language?: string;
  [k: string]: unknown;
}

/** Benchmark returns the same trigger shape as extract. */
export function triggerBenchmark(payload: BenchmarkRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/benchmark", {
    method: "POST",
    body: JSON.stringify(payload),
  });
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

/**
 * Download a run artifact as `filename`, attaching the operator's X-API-Key.
 *
 * A plain `<a href={artifactUrl(...)}>` navigation can't carry a custom
 * header, so with `AUTH_REQUIRED=true` the browser hits the artifact route
 * unauthenticated and gets a 401 — the download silently fails instead of
 * saving the file. This fetches the artifact as an authenticated request,
 * then triggers the save via a short-lived object URL.
 */
export async function downloadArtifact(uri: string, filename: string): Promise<void> {
  let res: Response;
  try {
    res = await fetch(artifactUrl(uri), { headers: { ...authHeader() } });
  } catch (e) {
    throw new ApiUnavailable(0, e instanceof Error ? e.message : "Network error");
  }
  if (!res.ok) {
    if (res.status === 401) throw unauthorizedError(await readBody(res));
    throw new ApiError(
      res.status,
      detailOf(await readBody(res), `Download failed (HTTP ${res.status})`),
    );
  }
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

// One (dimension, group, metric) delta row from POST /v1/studio/comparisons.
// "aggregate" dimension rows (group: {}) are the whole-run summary; the other
// dimensions (model_profile/schema_name/language/document/field) are the same
// shape sliced narrower -- the UI only renders "aggregate" for now (see
// build_comparison_payload's own docstring on why root_causes, not a
// dimension drill-down, is the mechanism for "what got worse").
export interface ComparisonRow {
  dimension: string;
  group: Record<string, string>;
  metric: string;
  direction: "higher" | "lower";
  baseline: number;
  candidate: number;
  delta: number;
  signed_improvement: number;
  paired_samples: number;
  baseline_only: number;
  candidate_only: number;
  confidence_interval_95: [number, number];
  sign_test_p_value: number;
  warnings: string[];
}

// status is "warn" specifically for a judge-metric budget that would have
// failed but is downgraded to non-blocking because the judge isn't
// calibrated yet (reason: "judge_uncalibrated_non_blocking") -- distinct
// from "error", which means the budget itself is broken/unmatched
// (reason: "no_matching_comparison") and metric/dimension are absent.
export interface ComparisonBudgetCheck {
  name: string;
  status: "pass" | "fail" | "warn" | "error";
  metric?: string;
  dimension?: string;
  reason?: string;
  [k: string]: unknown;
}

interface ComparisonRunMeta {
  event_id?: string;
  dataset?: string | null;
  model_profile?: string | null;
  created_at?: string | null;
}

/** POST /v1/studio/comparisons response — the CLI's compare_runs payload, live. */
export interface ComparisonPayload {
  contract_version: number;
  generated_at: string;
  baseline: ComparisonRunMeta;
  candidate: ComparisonRunMeta;
  verdict: "pass" | "fail" | "error";
  comparisons: ComparisonRow[];
  budget_checks: ComparisonBudgetCheck[];
  compatibility_errors: string[];
  root_causes: { documents: ComparisonRow[]; fields: ComparisonRow[] };
}

/** Compare two completed runs (deltas, CI, sign test, root causes, budget verdicts). */
export function compareRuns(
  baselineEventId: string,
  candidateEventId: string,
): Promise<ComparisonPayload> {
  return request<ComparisonPayload>("/v1/studio/comparisons", {
    method: "POST",
    body: JSON.stringify({
      baseline_event_id: baselineEventId,
      candidate_event_id: candidateEventId,
    }),
  });
}
