"use client";

import { type PortsView as PortsViewData } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { DataTable } from "../DataTable";
import { LiveIndicator } from "../LiveIndicator";

export function PortsAdmin({ ports }: { ports: ReturnType<typeof usePolling<PortsViewData>> }) {
  const data = ports.data;
  const range = data?.range;
  const recommended = data?.recommended_next;

  return (
    <div className="rounded-md border border-border bg-muted/20 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="text-xs font-medium text-foreground">
          Port allocation
          {range && (
            <span className="ml-2 font-normal text-muted-foreground">
              window {range.start}–{range.end}
            </span>
          )}
          {recommended != null ? (
            <span className="ml-2 font-normal text-muted-foreground">
              · next free ≈ <span className="text-foreground">{recommended}</span> (hint)
            </span>
          ) : data ? (
            <span className="ml-2 font-normal text-amber-600 dark:text-amber-400">
              · window exhausted
            </span>
          ) : null}
        </div>
        <LiveIndicator
          live={ports.live}
          refreshing={ports.refreshing}
          lastUpdated={ports.lastUpdated}
          onRefresh={ports.refresh}
        />
      </div>
      <DataTable
        rows={data?.deployments ?? null}
        loading={ports.loading}
        error={ports.error}
        emptyLabel="No ports in use"
        emptyDescription="Deploy a model — its port appears here on the next refresh."
      />
      <p className="mt-2 text-xs text-muted-foreground">
        The recommended port is a hint; the worker re-checks and allocates authoritatively at
        deploy time. Leave the port field untouched to let it choose.
      </p>
    </div>
  );
}
