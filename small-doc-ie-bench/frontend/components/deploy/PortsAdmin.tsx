"use client";

import { type PortsView as PortsViewData } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Badge, Card } from "../ui";
import { LiveIndicator } from "../LiveIndicator";
import { Table, type Column } from "../patterns/Table";
import { stateTone } from "./shared";

interface PortRow {
  name: string | null;
  port: number;
  state: string | null;
}

const COLUMNS: Column<PortRow>[] = [
  {
    key: "name",
    header: "Name",
    sortAccessor: (r) => r.name ?? "",
    render: (r) => r.name ?? <span className="text-muted-foreground">—</span>,
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
        rows={data?.deployments ?? null}
        loading={ports.loading}
        error={ports.error}
        getRowKey={(r) => String(r.port)}
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
