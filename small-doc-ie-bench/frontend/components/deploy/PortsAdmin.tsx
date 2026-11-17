"use client";

import { type PortsView as PortsViewData } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Badge, Card } from "../ui";
import { LiveIndicator } from "../LiveIndicator";
import { Table, type Column } from "../patterns/Table";
import { stateTone } from "./shared";

type PortRow = PortsViewData["deployments"][number] & { node_id?: string };

/** The api host's records plus every other node's live ports (by_node). */
function portRows(data: PortsViewData | null): PortRow[] | null {
  if (!data) return null;
  const local = new Set(data.deployments.map((r) => r.name));
  const remote = Object.entries(data.by_node ?? {}).flatMap(([node_id, rows]) =>
    rows.filter((r) => !local.has(r.name)).map((r) => ({ ...r, node_id })),
  );
  return [...data.deployments, ...remote];
}

const COLUMNS: Column<PortRow>[] = [
  {
    key: "name",
    header: "Name",
    sortAccessor: (r) => r.name ?? "",
    render: (r) => r.name ?? <span className="text-muted-foreground">—</span>,
  },
  {
    key: "node",
    header: "Node",
    sortAccessor: (r) => r.node_id ?? "",
    render: (r) => r.node_id ?? <span className="text-muted-foreground">this host</span>,
  },
  {
    key: "port",
    header: "Port",
    sortAccessor: (r) => r.port,
    render: (r) => <span className="font-mono tabular-nums">{r.port}</span>,
  },
  {
    key: "state",
    header: "State",
    sortAccessor: (r) => r.state ?? "",
    render: (r) => <Badge tone={stateTone(r.state)}>{r.state ?? "unknown"}</Badge>,
  },
];

export function PortsAdmin({ ports }: { ports: ReturnType<typeof usePolling<PortsViewData>> }) {
  const data = ports.data;
  const range = data?.range;
  const recommended = data?.recommended_next;

  return (
    <Card
      bodyClassName="p-3"
      title="Port allocation"
      subtitle={
        range || data ? (
          <>
            {range && `window ${range.start}–${range.end}`}
            {recommended != null ? (
              <>
                {" "}
                · next free ≈ <span className="text-foreground">{recommended}</span> (hint)
              </>
            ) : data ? (
              <span className="text-amber-600 dark:text-amber-400"> · window exhausted</span>
            ) : null}
          </>
        ) : undefined
      }
      actions={
        <LiveIndicator
          live={ports.live}
          refreshing={ports.refreshing}
          lastUpdated={ports.lastUpdated}
          onRefresh={ports.refresh}
        />
      }
    >
      <Table
        columns={COLUMNS}
        rows={portRows(data)}
        loading={ports.loading}
        error={ports.error}
        // Index-fallback like every sibling Table caller (DeploymentsView,
        // Benchmark, Catalog) -- port isn't guaranteed unique: an orphaned
        // deployment record not yet reaped (the class of bug #119 fixed)
        // can share a port with a newer one before cleanup catches up.
        getRowKey={(r, i) => `${r.port}-${i}`}
        emptyLabel="No ports in use"
        emptyDescription="Deploy a model — its port appears here on the next refresh."
      />
      <p className="mt-2 text-xs text-muted-foreground">
        The recommended port is a hint; the worker re-checks and allocates authoritatively at
        deploy time. Leave the port field untouched to let it choose.
      </p>
    </Card>
  );
}
