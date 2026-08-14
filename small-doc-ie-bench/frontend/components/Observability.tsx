"use client";

import {
  BarChart3,
  ExternalLink,
  Workflow,
  Gauge,
  ClipboardCheck,
  HardDrive,
  Activity,
} from "lucide-react";
import { GRAFANA_URL, GRAFANA_DASHBOARD_URL, INNGEST_URL, METRICS_URL } from "@/lib/env";
import {
  getReviewMetrics,
  getOcrCacheStats,
  getActivity,
  ApiError,
  type ReviewMetricsView,
  type OcrCacheStatsView,
  type ActivityEntry,
  type ActivityView,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Card, Badge } from "./ui";
import { PageHeader } from "./patterns/PageHeader";

const REVIEW_POLL_MS = 10000;
const OCR_CACHE_POLL_MS = 15000;
const ACTIVITY_POLL_MS = 15000;

/**
 * Observability = external tooling: quick-link tiles (Grafana / Inngest /
 * Prometheus) plus the docie Grafana dashboard embedded in an iframe, plus a
 * live review-queue summary tile (the actual claim/correct/approve/reject
 * workflow now lives in its own Review tab — this tile is a health signal +
 * deep-link, not the only way to reach it). One view — a prior "links"
 * sub-view was dropped: it rendered the same tiles already shown here, a
 * strict subset with nothing the combined page lacks.
 */
export function Observability({
  active = true,
  onNavigate,
}: {
  active?: boolean;
  onNavigate?: (id: "review") => void;
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
          <ActivityCard active={active} />
        </div>
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
            Blank panel? Grafana needs anonymous Viewer access and embedding
            enabled — set on the grafana service in docker-compose.yml
            (GF_AUTH_ANONYMOUS_ENABLED, GF_SECURITY_ALLOW_EMBEDDING). Rebuild
            the grafana container after changing them.
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
            Open queue →
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
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={(metrics.data?.queue_depth.pending ?? 0) > 0 ? "warn" : "ok"}>
              {metrics.data?.queue_depth.pending ?? 0} pending
            </Badge>
            <Badge tone="info">{metrics.data?.queue_depth.claimed ?? 0} claimed</Badge>
            <Badge tone="ok">{metrics.data?.queue_depth.approved ?? 0} approved</Badge>
            <Badge tone="neutral">{metrics.data?.queue_depth.rejected ?? 0} rejected</Badge>
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
        <p className="text-sm text-muted-foreground">Loading…</p>
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

function ActivityCard({ active }: { active: boolean }) {
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
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : detail ? (
        <p className="text-sm text-muted-foreground">{detail}</p>
      ) : entries.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No store: model requests tracked yet — activity is recorded when a
          request routes through a store: reference (chat/embeddings/rerank/
          extract), not on every deployment.
        </p>
      ) : (
        <div className="space-y-2">
          {shown.map((e) => {
            const replicas = replicaBadge(e);
            return (
              <div key={e.model_name} className="flex items-center justify-between gap-2 text-sm">
                <span className="truncate font-medium text-foreground">{e.model_name}</span>
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
