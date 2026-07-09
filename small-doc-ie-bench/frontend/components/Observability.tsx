"use client";

import { BarChart3, ExternalLink, Workflow, Gauge } from "lucide-react";
import { GRAFANA_URL, INNGEST_URL, METRICS_URL } from "@/lib/env";
import { Card } from "./ui";
import { PageHeader } from "./patterns/PageHeader";

/**
 * Observability = external tooling. Two sub-views (nav-driven, presentation
 * only): "links" shows the quick-link tiles; "dashboards" shows the embedded
 * Grafana panel. All hrefs / iframe wiring are unchanged.
 */
export function Observability({ view = "dashboards" }: { view?: string }) {
  const showLinks = view === "links";
  return (
    <div>
      <PageHeader
        title="Observability"
        subtitle={
          showLinks
            ? "Jump out to the external dashboards and raw metrics."
            : "Dashboards, runs and metrics from the serving stack."
        }
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

      {showLinks ? (
        <Card
          title="Quick links"
          subtitle="External dashboards and raw metrics."
        >
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
        </Card>
      ) : (
        <Card
          title="Grafana"
          subtitle={GRAFANA_URL}
          actions={
            <a
              href={GRAFANA_URL}
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
              src={GRAFANA_URL}
              title="Grafana"
              className="h-[70vh] w-full"
              sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
            />
          </div>
          <p className="mt-2 px-1 text-xs text-muted-foreground">
            If the panel is blank, Grafana may block embedding. Set{" "}
            <code className="rounded bg-muted px-1">allow_embedding = true</code> (and an anonymous
            org/viewer) in Grafana, or open it directly via the link above.
          </p>
        </Card>
      )}
    </div>
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
