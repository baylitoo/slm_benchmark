"use client";

import { Fragment, useCallback, useEffect, useState } from "react";
import {
  BarChart3,
  ExternalLink,
  Workflow,
  Gauge,
  ClipboardCheck,
  HardDrive,
  Activity,
  TrendingUp,
  ChevronRight,
  ChevronDown,
  Cpu,
  RefreshCw,
} from "lucide-react";
import { GRAFANA_URL, GRAFANA_DASHBOARD_URL, INNGEST_URL, METRICS_URL } from "@/lib/env";
import {
  getReviewMetrics,
  getOcrCacheStats,
  getActivity,
  getUsageSummary,
  getDeployments,
  getDeploymentSlots,
  ApiError,
  type ReviewMetricsView,
  type OcrCacheStatsView,
  type ActivityEntry,
  type ActivityView,
  type UsageDeployment,
  type UsageSummaryView,
  type UsageWindow,
  type DeploymentRecord,
  type LlamaCppSlot,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Card, Badge } from "./ui";
import { PageHeader } from "./patterns/PageHeader";
import { T } from "@/lib/i18n";

const REVIEW_POLL_MS = 10000;
const OCR_CACHE_POLL_MS = 15000;
const ACTIVITY_POLL_MS = 15000;
const USAGE_POLL_MS = 15000;
const SLOTS_DEPLOYMENTS_POLL_MS = 15000;

const USAGE_WINDOWS: UsageWindow[] = ["24h", "7d", "30d"];

/**
 * Observability = external tooling: quick-link tiles (Grafana / Inngest /
 * Prometheus) plus the docie Grafana dashboard embedded in an iframe, plus a
 * live review-queue summary tile (the actual claim/correct/approve/reject
 * workflow now lives in its own Review tab — this tile is a health signal +
 * deep-link, not the only way to reach it). One view — a prior "links"
 * sub-view was dropped: it rendered the same tiles already shown here, a
 * strict subset with nothing the combined page lacks.
 */
// Narrowed to this section's two deep-link targets, same trick Playground.tsx
// (NavigateToDeploy) and Agents.tsx (NavigateWithinAgents) use -- AppShell's
// onNavigate(id: SectionId, view?, query?) is a valid supertype.
type NavigateFromObservability = (
  id: "review" | "deploy",
  view?: string,
  query?: Record<string, string>,
) => void;

export function Observability({
  active = true,
  onNavigate,
}: {
  active?: boolean;
  onNavigate?: NavigateFromObservability;
}) {
  return (
    <div>
      <PageHeader
        title="Observability"
        subtitle="Dashboards, runs and metrics from the serving stack."
        actions={
          <a
            href={GRAFANA_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-card px-3 py-1.5 text-sm font-medium text-foreground transition hover:bg-muted"
          >
            Open Grafana <ExternalLink className="h-3.5 w-3.5" />
          </a>
        }
      />

      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-3">
          <LinkTile
            title="Grafana"
            href={GRAFANA_URL}
            desc="Dashboards & charts"
            icon={<BarChart3 className="h-5 w-5" />}
          />
          <LinkTile
            title="Inngest"
            href={INNGEST_URL}
            desc="Runs, events & functions"
            icon={<Workflow className="h-5 w-5" />}
          />
          <LinkTile
            title="Prometheus metrics"
            href={METRICS_URL}
            desc="Raw /metrics endpoint"
            icon={<Gauge className="h-5 w-5" />}
          />
        </div>
        <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
          <ReviewQueueCard active={active} onNavigate={onNavigate} />
          <OcrCacheCard active={active} />
          <ActivityCard active={active} onNavigate={onNavigate} />
          <SlotsCard active={active} />
        </div>
        <UsageCard active={active} />
        <Card
          title="Small Document IE Benchmark"
          subtitle="Agent requests, PII detections, latency, and gate blocks — live from Prometheus."
          actions={
            <a
              href={GRAFANA_DASHBOARD_URL.replace(/&kiosk$/, "")}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs font-medium text-accent hover:underline"
            >
              Open <ExternalLink className="h-3.5 w-3.5" />
            </a>
          }
          bodyClassName="p-3"
        >
          <div className="overflow-hidden rounded-md border border-border bg-background">
            <iframe
              src={GRAFANA_DASHBOARD_URL}
              title="Small Document IE Benchmark"
              className="h-[70vh] w-full"
              sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
            />
          </div>
          <p className="mt-2 px-1 text-xs text-muted-foreground">
            <T>Blank panel? Grafana needs anonymous Viewer access and embedding enabled — set on the grafana service in docker-compose.yml (GF_AUTH_ANONYMOUS_ENABLED, GF_SECURITY_ALLOW_EMBEDDING). Rebuild the grafana container after changing them.</T>
          </p>
        </Card>
      </div>
    </div>
  );
}

function ReviewQueueCard({
  active,
  onNavigate,
}: {
  active: boolean;
  onNavigate?: (id: "review") => void;
}) {
  const metrics = usePolling<ReviewMetricsView>(getReviewMetrics, REVIEW_POLL_MS, active);
  const notEnabled = metrics.error instanceof ApiError && metrics.error.status === 422;

  return (
    <Card
      icon={<ClipboardCheck className="h-5 w-5" />}
      title="Review queue"
      subtitle="Extractions admitted for human review — low confidence, weak evidence, arithmetic mismatches, or model disagreement."
      actions={
        onNavigate && (
          <button
            type="button"
            onClick={() => onNavigate("review")}
            className="text-xs font-medium text-accent hover:underline"
          >
            <T>Open queue →</T>
          </button>
        )
      }
    >
      {notEnabled ? (
        <p className="text-sm text-muted-foreground">
          Not enabled — the review workflow needs DATABASE_URL set (persistence
          is what the queue is stored in).
        </p>
      ) : metrics.error ? (
        <p className="text-sm text-muted-foreground">
          Couldn&apos;t load review metrics. Is the API reachable?
        </p>
      ) : metrics.loading ? (
        <p className="text-sm text-muted-foreground"><T>Loading…</T></p>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={(metrics.data?.queue_depth.pending ?? 0) > 0 ? "warn" : "ok"}>
              {metrics.data?.queue_depth.pending ?? 0} pending
            </Badge>
            <Badge tone="info">{metrics.data?.queue_depth.claimed ?? 0} claimed</Badge>
            <Badge tone="ok">{metrics.data?.queue_depth.approved ?? 0} approved</Badge>
            {/* err, not neutral -- matches Review.tsx's own STATUS_TONE for
                the same status, so the two surfaces read consistently
                instead of one treating "rejected" as a non-event. */}
            <Badge tone="err">{metrics.data?.queue_depth.rejected ?? 0} rejected</Badge>
          </div>
          <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
            <span>
              Correction rate:{" "}
              {metrics.data?.correction_rate != null
                ? `${(metrics.data.correction_rate * 100).toFixed(0)}%`
                : "n/a — no decided tasks yet"}
            </span>
            <span>
              Reviewer agreement:{" "}
              {metrics.data?.reviewer_agreement != null
                ? `${(metrics.data.reviewer_agreement * 100).toFixed(0)}%`
                : "n/a — needs 2+ reviewers on the same task"}
            </span>
          </div>
        </div>
      )}
    </Card>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}

function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
}

function OcrCacheCard({ active }: { active: boolean }) {
  const stats = usePolling<OcrCacheStatsView>(getOcrCacheStats, OCR_CACHE_POLL_MS, active);
  const data = stats.data;

  return (
    <Card
      icon={<HardDrive className="h-5 w-5" />}
      title="OCR cache"
      subtitle="Content-addressed OCR artifacts on disk — how full, how old. Hit rate isn't tracked yet (needs aggregating across replicas)."
    >
      {stats.error ? (
        <p className="text-sm text-muted-foreground">
          Couldn&apos;t load OCR cache stats. Is the API reachable?
        </p>
      ) : stats.loading ? (
        <p className="text-sm text-muted-foreground"><T>Loading…</T></p>
      ) : data && !data.enabled ? (
        <p className="text-sm text-muted-foreground">
          Disabled — set OCR_CACHE_ENABLED=true to cache OCR results across runs.
        </p>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="info">{data?.entry_count ?? 0} entries</Badge>
            <Badge
              tone={
                (data?.utilization_pct ?? 0) > 90
                  ? "err"
                  : (data?.utilization_pct ?? 0) > 70
                    ? "warn"
                    : "ok"
              }
            >
              {formatBytes(data?.total_bytes ?? 0)} / {formatBytes(data?.max_bytes ?? 0)} (
              {data?.utilization_pct ?? 0}%)
            </Badge>
          </div>
          <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
            <span>
              Oldest entry:{" "}
              {data?.oldest_entry_age_seconds != null
                ? `${formatAge(data.oldest_entry_age_seconds)} ago`
                : "n/a — cache is empty"}
            </span>
            <span>
              Newest entry:{" "}
              {data?.newest_entry_age_seconds != null
                ? `${formatAge(data.newest_entry_age_seconds)} ago`
                : "n/a — cache is empty"}
            </span>
          </div>
        </div>
      )}
    </Card>
  );
}

function secondsAgo(iso: string): number {
  return Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
}

function replicaBadge(e: ActivityEntry): { tone: "ok" | "warn" | "neutral"; label: string } {
  if (e.live_replica_count > 0) {
    return {
      tone: "ok",
      label: e.live_replica_count === 1 ? "1 replica" : `${e.live_replica_count} replicas`,
    };
  }
  if (e.total_replica_count > 0) {
    // Had a placement, none of them live now — idle-unloaded or stopped,
    // not gone. This is the signal the Deployments tab's scale stepper
    // (#92-95) needs: a model with request history but nothing serving it.
    return { tone: "warn", label: "evicted" };
  }
  return { tone: "neutral", label: "no placement" };
}

function ActivityCard({
  active,
  onNavigate,
}: {
  active: boolean;
  onNavigate?: NavigateFromObservability;
}) {
  const activity = usePolling<ActivityView>(getActivity, ACTIVITY_POLL_MS, active);
  const entries = activity.data?.entries ?? [];
  const detail = activity.data?.detail;
  const shown = entries.slice(0, 6);

  return (
    <Card
      icon={<Activity className="h-5 w-5" />}
      title="Model activity"
      subtitle="Requests per store: model since the window last reset, next to how many replicas are actually live — a load signal, not a rate. Nothing scales on it yet."
    >
      {activity.error ? (
        <p className="text-sm text-muted-foreground">
          Couldn&apos;t load activity. Is the API reachable?
        </p>
      ) : activity.loading ? (
        <p className="text-sm text-muted-foreground"><T>Loading…</T></p>
      ) : detail ? (
        <p className="text-sm text-muted-foreground">{detail}</p>
      ) : entries.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          <T>No store: model requests tracked yet — activity is recorded when a request routes through a store: reference (chat/embeddings/rerank/extract), not on every deployment.</T>
        </p>
      ) : (
        <div className="space-y-2">
          {shown.map((e) => {
            const replicas = replicaBadge(e);
            return (
              <div key={e.model_name} className="flex items-center justify-between gap-2 text-sm">
                {onNavigate ? (
                  <button
                    type="button"
                    onClick={() => onNavigate("deploy", "deployments", { q: e.model_name })}
                    title="Open in Deployments, filtered to this model"
                    className="truncate font-medium text-foreground hover:text-accent hover:underline"
                  >
                    {e.model_name}
                  </button>
                ) : (
                  <span className="truncate font-medium text-foreground">{e.model_name}</span>
                )}
                <div className="flex shrink-0 items-center gap-2">
                  <Badge tone="info">{e.window_count} req</Badge>
                  <Badge tone={replicas.tone}>{replicas.label}</Badge>
                  <span className="w-14 text-right text-xs text-muted-foreground">
                    {e.last_request_at ? `${formatAge(secondsAgo(e.last_request_at))} ago` : "n/a"}
                  </span>
                </div>
              </div>
            );
          })}
          {entries.length > shown.length && (
            <p className="text-xs text-muted-foreground">
              +{entries.length - shown.length} more
            </p>
          )}
        </div>
      )}
    </Card>
  );
}

/** Exported for direct unit testing, same pattern as {@link formatCount}. */
export function llamaCppRunningDeploymentNames(records: DeploymentRecord[]): string[] {
  // "ready" is the live-serving state every backend record actually uses
  // (catalog.py/dashboard.py/placement_resolver.py) -- "running" never
  // occurs, so this filter matched nothing against a real deployment.
  return records
    .filter((r) => r.spec?.launch?.runtime === "llamacpp" && r.state === "ready")
    .map((r) => r.spec?.name)
    .filter((name): name is string => Boolean(name));
}

/** One deployment's slots, fetched on demand (not eagerly for every
 * deployment on page load) -- mirrors McpView's CodeInterpreterWorkers:
 * collapsed by default, a click expands and triggers the live GET /slots
 * query, a refresh icon re-queries. Every slot field is optional (#315):
 * this only ever renders what the specific llama-server build reported. */
function SlotRow({ name }: { name: string }) {
  const [open, setOpen] = useState(false);
  const [slots, setSlots] = useState<LlamaCppSlot[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await getDeploymentSlots(name);
      setSlots(res.slots);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  function toggle() {
    const next = !open;
    setOpen(next);
    if (next && slots === null && !loading) void load();
  }

  return (
    <div className="rounded-md border border-border bg-muted/40 p-2 text-xs">
      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={toggle}
          className="flex min-w-0 items-center gap-1 font-medium text-foreground hover:text-accent"
        >
          {open ? (
            <ChevronDown className="h-3.5 w-3.5 shrink-0" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 shrink-0" />
          )}
          <span className="truncate" title={name}>
            {name}
          </span>
        </button>
        {open && (
          <button
            type="button"
            aria-label={`Refresh ${name} slots`}
            onClick={() => void load()}
            className="shrink-0 text-muted-foreground hover:text-foreground"
          >
            <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
          </button>
        )}
      </div>
      {open && (
        <div className="mt-1.5 space-y-1 pl-5">
          {error ? (
            <p className="text-destructive">
              <T>Couldn&apos;t reach this deployment.</T> {error}
            </p>
          ) : loading && slots === null ? (
            <p className="text-muted-foreground">
              <T>Loading…</T>
            </p>
          ) : !slots || slots.length === 0 ? (
            <p className="text-muted-foreground">
              <T>No slots reported.</T>
            </p>
          ) : (
            slots.map((slot, i) => (
              <div key={slot.id ?? i} className="flex flex-wrap items-center gap-2">
                <Badge tone={slot.is_processing ? "info" : "neutral"}>
                  slot {slot.id ?? i} · {slot.is_processing ? "busy" : "idle"}
                </Badge>
                {slot.n_ctx != null && (
                  <span className="text-muted-foreground">ctx {slot.n_ctx}</span>
                )}
                {slot.cache_n != null && (
                  <span className="text-muted-foreground">cache {slot.cache_n}</span>
                )}
                {slot.prompt_ms != null && (
                  <span className="text-muted-foreground">
                    prefill {formatLatency(slot.prompt_ms)}
                  </span>
                )}
                {slot.predicted_ms != null && (
                  <span className="text-muted-foreground">
                    decode {formatLatency(slot.predicted_ms)}
                  </span>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function SlotsCard({ active }: { active: boolean }) {
  const deployments = usePolling<DeploymentRecord[]>(
    getDeployments,
    SLOTS_DEPLOYMENTS_POLL_MS,
    active,
  );
  const names = llamaCppRunningDeploymentNames(deployments.data ?? []);

  return (
    <Card
      icon={<Cpu className="h-5 w-5" />}
      title="llama.cpp slots"
      subtitle="Per-slot prompt state and cache/timing straight from each deployment's own GET /slots — expand a deployment to query it live."
    >
      {deployments.error ? (
        <p className="text-sm text-muted-foreground">
          Couldn&apos;t load deployments. Is the API reachable?
        </p>
      ) : deployments.loading ? (
        <p className="text-sm text-muted-foreground">
          <T>Loading…</T>
        </p>
      ) : names.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          <T>No running llama.cpp deployments.</T>
        </p>
      ) : (
        <div className="space-y-1.5">
          {names.map((name) => (
            <SlotRow key={name} name={name} />
          ))}
        </div>
      )}
    </Card>
  );
}

/** Compact token/request count: 950 -> "950", 12_400 -> "12.4k", 3_200_000 -> "3.2M". */
export function formatCount(value: number): string {
  if (value < 1000) return `${value}`;
  if (value < 1_000_000) return `${(value / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
}

function formatLatency(ms: number | null): string {
  if (ms == null) return "n/a";
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

function totalTokens(entry: UsageDeployment): number {
  return entry.prompt_tokens + entry.completion_tokens;
}

// Exported for Observability.test.tsx -- rendered only by this page.
export function UsageCard({ active }: { active: boolean }) {
  const [usageWindow, setUsageWindow] = useState<UsageWindow>("24h");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const fetchUsage = useCallback(() => getUsageSummary(usageWindow), [usageWindow]);
  const usage = usePolling<UsageSummaryView>(fetchUsage, USAGE_POLL_MS, active);
  const { refresh } = usage;
  // usePolling keeps the latest fetch closure in a ref but only re-arms its
  // interval on interval/enabled changes -- switching the window needs an
  // immediate refetch, not a wait for the next tick.
  useEffect(() => {
    refresh();
  }, [usageWindow, refresh]);

  const entries = usage.data?.deployments ?? [];
  const maxTokens = Math.max(1, ...entries.map(totalTokens));
  const anyTokens = entries.some((entry) => totalTokens(entry) > 0);

  return (
    <Card
      icon={<TrendingUp className="h-5 w-5" />}
      title="Usage"
      subtitle="Per-deployment traffic from the durable usage ledger — every chat, embeddings, rerank and extract request served, aggregated over the selected window."
      actions={
        <div className="flex items-center gap-1 rounded-md border border-border bg-muted p-0.5">
          {USAGE_WINDOWS.map((candidate) => (
            <button
              key={candidate}
              type="button"
              onClick={() => setUsageWindow(candidate)}
              aria-pressed={usageWindow === candidate}
              className={`rounded px-2 py-0.5 text-xs font-medium transition ${
                usageWindow === candidate
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {candidate}
            </button>
          ))}
        </div>
      }
    >
      {usage.error ? (
        <p className="text-sm text-muted-foreground">
          Couldn&apos;t load usage. Is the API reachable?
        </p>
      ) : usage.loading ? (
        <p className="text-sm text-muted-foreground"><T>Loading…</T></p>
      ) : entries.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          <T>No usage recorded in this window yet — a row is written each time a request is served (chat, embeddings, rerank or extract). Recording needs DATABASE_URL set on the API.</T>
        </p>
      ) : (
        <div className="space-y-4">
          {anyTokens && (
            <div className="space-y-1.5">
              <p className="text-xs font-medium text-muted-foreground">
                <T>Token volume (in + out)</T>
              </p>
              {entries.map((entry) => (
                <div key={entry.deployment} className="flex items-center gap-2 text-xs">
                  <span
                    className="w-40 shrink-0 truncate font-medium text-foreground"
                    title={entry.deployment}
                  >
                    {entry.deployment}
                  </span>
                  <div className="h-3 flex-1 overflow-hidden rounded-sm bg-muted">
                    {/* Two-segment CSS bar: prompt (solid) + completion
                        (translucent), both scaled against the busiest
                        deployment so the rows compare visually. */}
                    <div
                      className="flex h-full"
                      style={{ width: `${Math.max(1, (totalTokens(entry) / maxTokens) * 100)}%` }}
                    >
                      <div
                        className="h-full bg-accent"
                        style={{
                          width: `${totalTokens(entry) > 0 ? (entry.prompt_tokens / totalTokens(entry)) * 100 : 0}%`,
                        }}
                      />
                      <div className="h-full flex-1 bg-accent/50" />
                    </div>
                  </div>
                  <span className="w-14 shrink-0 text-right tabular-nums text-muted-foreground">
                    {formatCount(totalTokens(entry))}
                  </span>
                </div>
              ))}
            </div>
          )}
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead>
                <tr className="border-b border-border text-xs text-muted-foreground">
                  <th className="py-1.5 pr-3 font-medium"><T>Deployment</T></th>
                  <th className="py-1.5 pr-3 text-right font-medium"><T>Requests</T></th>
                  <th className="py-1.5 pr-3 text-right font-medium"><T>Errors</T></th>
                  <th className="py-1.5 pr-3 text-right font-medium"><T>Tokens in</T></th>
                  <th className="py-1.5 pr-3 text-right font-medium"><T>Tokens out</T></th>
                  <th className="py-1.5 pr-3 text-right font-medium"><T>Avg latency</T></th>
                  <th className="py-1.5 pr-3 text-right font-medium"><T>p95</T></th>
                  <th className="py-1.5 text-right font-medium"><T>Last used</T></th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => {
                  const hasTools = entry.tool_calls.length > 0;
                  const isOpen = expanded.has(entry.deployment);
                  return (
                    <Fragment key={entry.deployment}>
                      <tr className="border-b border-border/60 last:border-0">
                        <td className="max-w-[220px] py-1.5 pr-3 font-medium text-foreground">
                          <div className="flex items-center gap-1">
                            {hasTools ? (
                              <button
                                type="button"
                                aria-label={
                                  isOpen
                                    ? `Collapse ${entry.deployment} tool calls`
                                    : `Expand ${entry.deployment} tool calls`
                                }
                                onClick={() =>
                                  setExpanded((prev) => {
                                    const next = new Set(prev);
                                    if (next.has(entry.deployment)) {
                                      next.delete(entry.deployment);
                                    } else {
                                      next.add(entry.deployment);
                                    }
                                    return next;
                                  })
                                }
                                className="shrink-0 text-muted-foreground hover:text-foreground"
                              >
                                {isOpen ? (
                                  <ChevronDown className="h-3.5 w-3.5" />
                                ) : (
                                  <ChevronRight className="h-3.5 w-3.5" />
                                )}
                              </button>
                            ) : (
                              <span className="w-3.5 shrink-0" />
                            )}
                            <span className="truncate" title={entry.deployment}>
                              {entry.deployment}
                            </span>
                          </div>
                        </td>
                        <td className="py-1.5 pr-3 text-right tabular-nums">
                          {formatCount(entry.requests)}
                        </td>
                        <td className="py-1.5 pr-3 text-right tabular-nums">
                          {entry.errors > 0 ? (
                            <Badge tone="err">{formatCount(entry.errors)}</Badge>
                          ) : (
                            <span className="text-muted-foreground">0</span>
                          )}
                        </td>
                        <td className="py-1.5 pr-3 text-right tabular-nums">
                          {formatCount(entry.prompt_tokens)}
                        </td>
                        <td className="py-1.5 pr-3 text-right tabular-nums">
                          {formatCount(entry.completion_tokens)}
                        </td>
                        <td className="py-1.5 pr-3 text-right tabular-nums">
                          {formatLatency(entry.avg_latency_ms)}
                        </td>
                        <td className="py-1.5 pr-3 text-right tabular-nums">
                          {formatLatency(entry.p95_latency_ms)}
                        </td>
                        <td className="py-1.5 text-right text-xs text-muted-foreground">
                          {entry.last_used_at
                            ? `${formatAge(secondsAgo(entry.last_used_at))} ago`
                            : "n/a"}
                        </td>
                      </tr>
                      {hasTools && isOpen && (
                        <tr className="border-b border-border/60 last:border-0">
                          <td colSpan={8} className="bg-muted/40 py-2 pl-8 pr-3">
                            <table className="w-full max-w-md text-left text-xs">
                              <thead>
                                <tr className="text-muted-foreground">
                                  <th className="pb-1 pr-3 font-medium"><T>Tool</T></th>
                                  <th className="pb-1 pr-3 text-right font-medium"><T>Calls</T></th>
                                  <th className="pb-1 pr-3 text-right font-medium"><T>Errors</T></th>
                                  <th className="pb-1 text-right font-medium"><T>Avg latency</T></th>
                                </tr>
                              </thead>
                              <tbody>
                                {entry.tool_calls.map((tool) => (
                                  <tr key={tool.tool}>
                                    <td className="max-w-[200px] truncate py-0.5 pr-3 font-medium text-foreground">
                                      {tool.tool}
                                    </td>
                                    <td className="py-0.5 pr-3 text-right tabular-nums">
                                      {formatCount(tool.calls)}
                                    </td>
                                    <td className="py-0.5 pr-3 text-right tabular-nums">
                                      {tool.errors > 0 ? (
                                        <Badge tone="err">{formatCount(tool.errors)}</Badge>
                                      ) : (
                                        <span className="text-muted-foreground">0</span>
                                      )}
                                    </td>
                                    <td className="py-0.5 text-right tabular-nums">
                                      {formatLatency(tool.avg_latency_ms)}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted-foreground">
            <T>Streamed chats count tokens when the upstream sends a final usage frame; older runtimes or a caller that opts out still count requests and latency only.</T>
          </p>
        </div>
      )}
    </Card>
  );
}

function LinkTile({
  title,
  href,
  desc,
  icon,
}: {
  title: string;
  href: string;
  desc: string;
  icon: React.ReactNode;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="group flex items-start gap-3 rounded-xl border border-border bg-background p-4 transition hover:border-accent hover:shadow-card"
    >
      <span className="grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-border bg-muted text-accent">
        {icon}
      </span>
      <div className="min-w-0">
        <p className="flex items-center gap-1 text-sm font-semibold text-foreground group-hover:text-accent">
          {title}
          <ExternalLink className="h-3.5 w-3.5 opacity-0 transition group-hover:opacity-100" />
        </p>
        <p className="mt-0.5 text-xs text-muted-foreground">{desc}</p>
        <p className="mt-1.5 truncate text-xs text-muted-foreground/70">{href}</p>
      </div>
    </a>
  );
}
