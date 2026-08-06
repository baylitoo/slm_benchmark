"use client";

import { useMemo, useState } from "react";
import { Network } from "lucide-react";
import { type DeploymentRecord } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Badge, TextInput } from "../ui";
import { LiveIndicator } from "../LiveIndicator";
import { Toolbar } from "../patterns/Toolbar";
import { ResultLine } from "../patterns/ResultLine";
import { Pager } from "../patterns/Pager";
import { Table, type Column } from "../patterns/Table";
import { PAGE_SIZE, stateTone } from "./shared";

// ---------------------------------------------------------------------------
// Ports view — pure projection over the deployments already in memory.
// ---------------------------------------------------------------------------

export function PortsView({
  deployments,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
}) {
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(1);

  const all = deployments.data ?? [];
  const total = all.length;
  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return all;
    return all.filter((r) =>
      [r.spec?.name, r.spec?.launch?.port, r.state]
        .filter((x) => x != null)
        .join(" ")
        .toLowerCase()
        .includes(q),
    );
  }, [all, filter]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount);
  const paged = useMemo(
    () => filtered.slice((clampedPage - 1) * PAGE_SIZE, clampedPage * PAGE_SIZE),
    [filtered, clampedPage],
  );

  const columns: Column<DeploymentRecord>[] = [
    {
      key: "port",
      header: "Port",
      sortAccessor: (r) => r.spec?.launch?.port ?? 0,
      render: (r) => (
        <span className="font-mono tabular-nums text-xs text-foreground">
          {r.spec?.launch?.port ?? "—"}
        </span>
      ),
    },
    {
      key: "pid",
      header: "PID",
      sortAccessor: (r) => r.pid ?? 0,
      render: (r) => <span className="font-mono tabular-nums text-xs">{r.pid ?? "—"}</span>,
    },
    {
      key: "process",
      header: "Process",
      sortAccessor: (r) => r.spec?.name ?? "",
      render: (r) => <span className="text-foreground">{r.spec?.name ?? "—"}</span>,
    },
    {
      key: "state",
      header: "State",
      sortAccessor: (r) => r.state ?? "",
      render: (r) => <Badge tone={stateTone(r.state)}>{r.state ?? "unknown"}</Badge>,
    },
    {
      key: "endpoint",
      header: "Endpoint",
      className: "max-w-[18rem]",
      render: (r) =>
        r.endpoint ? (
          <span className="block truncate font-mono text-xs text-muted-foreground" title={r.endpoint}>
            {r.endpoint}
          </span>
        ) : (
          <span className="text-muted-foreground">—</span>
        ),
    },
  ];

  return (
    <div>
      <Toolbar
        onReset={() => {
          setFilter("");
          setPage(1);
        }}
        resetDisabled={filter === ""}
      >
        <TextInput
          value={filter}
          onChange={(e) => {
            setFilter(e.target.value);
            setPage(1);
          }}
          placeholder="Filter by process, port, state…"
          className="h-8 w-64 text-xs"
        />
        <div className="ml-auto">
          <LiveIndicator
            live={deployments.live}
            refreshing={deployments.refreshing}
            lastUpdated={deployments.lastUpdated}
            onRefresh={deployments.refresh}
          />
        </div>
      </Toolbar>

      <ResultLine
        shown={paged.length}
        total={total}
        noun="ports"
        onFetch={deployments.refresh}
        fetching={deployments.refreshing}
        pager={
          <Pager
            page={clampedPage}
            pageCount={pageCount}
            onPrev={() => setPage((p) => Math.max(1, p - 1))}
            onNext={() => setPage((p) => Math.min(pageCount, p + 1))}
          />
        }
      />

      <Table<DeploymentRecord>
        columns={columns}
        rows={deployments.data ? paged : null}
        getRowKey={(r, i) => `${r.spec?.launch?.port ?? "port"}-${r.spec?.name ?? i}`}
        loading={deployments.loading}
        error={deployments.error}
        emptyIcon={<Network className="h-5 w-5" />}
        emptyLabel="No ports in use"
        emptyDescription="Deploy a model — its port appears here on the next refresh."
      />
    </div>
  );
}
