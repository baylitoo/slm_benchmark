import type { BenchmarkRun } from "@/lib/api";

/** The per-profile summary row a run's headline numbers come from. */
export interface RunHeadline {
  profile: string | null;
  /** Summary rows beyond the first: a routed run scores each stage's profile. */
  extraProfiles: number;
  docs: number | null;
  fieldAccuracy: number | null;
  validRate: number | null;
  hallucinationRate: number | null;
  evidenceCoverage: number | null;
  rowF1: number | null;
  p50LatencyMs: number | null;
  p95LatencyMs: number | null;
  throughputPerMin: number | null;
}

function num(row: Record<string, unknown> | undefined, key: string): number | null {
  const value = row?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function runHeadline(run: BenchmarkRun): RunHeadline {
  const summary = run.metrics?.summary ?? [];
  const row = summary[0];
  const profile = typeof row?.model_profile === "string" ? row.model_profile : null;
  return {
    profile: run.model_profile ?? profile,
    extraProfiles: Math.max(0, summary.length - 1),
    docs: num(row, "docs"),
    fieldAccuracy: num(row, "field_accuracy"),
    validRate: num(row, "valid_rate"),
    hallucinationRate: num(row, "hallucination_rate"),
    evidenceCoverage: num(row, "evidence_coverage"),
    rowF1: num(row, "row_f1"),
    p50LatencyMs: num(row, "p50_latency_ms"),
    p95LatencyMs: num(row, "p95_latency_ms"),
    throughputPerMin: num(row, "throughput_docs_per_min"),
  };
}

export function formatPct(value: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

export function formatLatency(ms: number | null): string {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

export function formatAgo(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "—";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return iso;
  const mins = Math.max(0, Math.round((now - then) / 60_000));
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days} d ago`;
  return iso.slice(0, 10);
}

export const STATUS_TONE: Record<string, "ok" | "err" | "warn" | "neutral"> = {
  completed: "ok",
  failed: "err",
  running: "warn",
  queued: "neutral",
};
