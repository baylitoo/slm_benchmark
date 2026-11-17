"use client";

import { getFleet, type FleetView } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { usePolling } from "@/lib/usePolling";
import { StatusDot } from "../ui";

export interface FleetSummary {
  total: number;
  reporting: number;
  /** Nodes that reported before and stopped; a node that never started is not stale. */
  stale: string[];
}

/** Null when there is nothing honest to show: no catalog, or no nodes yet. */
export function summarizeFleet(view: FleetView | null | undefined): FleetSummary | null {
  if (!view || !view.observed_available || view.nodes.length === 0) return null;
  return {
    total: view.nodes.length,
    reporting: view.nodes.filter((n) => n.reporting).length,
    stale: view.nodes.filter((n) => !n.reporting && !n.never_reported).map((n) => n.node_id),
  };
}

/** Top-bar count of serving nodes reporting, warning when any went silent. */
export function NodesStatus({ intervalMs = 15000 }: { intervalMs?: number }) {
  const { t } = useI18n();
  const fleet = usePolling<FleetView>(getFleet, intervalMs);
  const summary = summarizeFleet(fleet.data);
  if (!summary) return null;
  const stale = summary.stale.length;
  const title =
    stale > 0
      ? `${t("Serving nodes not reporting")}: ${summary.stale.join(", ")}`
      : t("Serving nodes reporting");
  return (
    <span
      className="hidden items-center gap-1.5 text-xs text-muted-foreground sm:flex"
      title={title}
      data-testid="nodes-status"
    >
      <StatusDot tone={stale > 0 ? "warn" : "ok"} />
      <span className="tabular-nums">
        {summary.reporting}/{summary.total} {t("nodes")}
        {stale > 0 && (
          <span className="ml-1 text-amber-600 dark:text-amber-400">
            · {stale} {t("stale")}
          </span>
        )}
      </span>
    </span>
  );
}
