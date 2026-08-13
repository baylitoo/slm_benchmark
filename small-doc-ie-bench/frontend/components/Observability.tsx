"use client";

import { BarChart3, ExternalLink, Workflow, Gauge, ClipboardCheck, HardDrive } from "lucide-react";
import { GRAFANA_URL, GRAFANA_DASHBOARD_URL, INNGEST_URL, METRICS_URL } from "@/lib/env";
import {
  getReviewMetrics,
  getOcrCacheStats,
  ApiError,
  type ReviewMetricsView,
  type OcrCacheStatsView,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Card, Badge } from "./ui";
import { PageHeader } from "./patterns/PageHeader";

const REVIEW_POLL_MS = 10000;
const OCR_CACHE_POLL_MS = 15000;

/**
 * Observability = external tooling: quick-link tiles (Grafana / Inngest /
 * Prometheus) plus the docie Grafana dashboard embedded in an iframe, plus a
 * live review-queue tile. The human review workflow (POST /v1/reviews,
 * claim/correct/approve/reject) is API-only — no Studio tab for it yet — so
 * this is the only place an operator sees the backlog exists at all without
 * curling the API directly. One view — a prior "links" sub-view was dropped:
 * it rendered the same tiles already shown here, a strict subset with
 * nothing the combined page lacks.
 */
export function Observability({ active = true }: { active?: boolean }) {
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
        <div className="grid gap-4 lg:grid-cols-2">
          <ReviewQueueCard active={active} />
          <OcrCacheCard active={active} />
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

function ReviewQueueCard({ active }: { active: boolean }) {
  const metrics = usePolling<ReviewMetricsView>(getReviewMetrics, REVIEW_POLL_MS, active);
  const notEnabled = metrics.error instanceof ApiError && metrics.error.status === 422;

  return (
    <Card
      icon={<ClipboardCheck className="h-5 w-5" />}
      title="Review queue"
      subtitle="Extractions admitted for human review — low confidence, weak evidence, arithmetic mismatches, or model disagreement."
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
          <p className="text-xs text-muted-foreground">
            No Studio tab for claim/correct/decide yet — use{" "}
            <code className="rounded bg-muted px-1 py-0.5">POST /v1/reviews/&#123;id&#125;/claim</code>{" "}
            and friends directly.
          </p>
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
