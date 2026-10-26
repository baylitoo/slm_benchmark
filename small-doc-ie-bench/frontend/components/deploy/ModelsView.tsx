"use client";

import { useMemo, useState } from "react";
import { Rocket, Eye, Cpu, Boxes } from "lucide-react";
import { formatBytes, type StoreEntry } from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Badge, Button, TextInput } from "../ui";
import { LiveIndicator } from "../LiveIndicator";
import { Toolbar } from "../patterns/Toolbar";
import { ResultLine } from "../patterns/ResultLine";
import { Pager } from "../patterns/Pager";
import { Table, type Column } from "../patterns/Table";
import { PAGE_SIZE } from "./shared";

// ---------------------------------------------------------------------------
// Models view — the deployable store catalog.
// ---------------------------------------------------------------------------

export function ModelsView({
  store,
  onDeploy,
}: {
  store: ReturnType<typeof usePolling<StoreEntry[]>>;
  onDeploy: () => void;
}) {
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(1);

  const all = store.data ?? [];
  const total = all.length;
  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return all;
    return all.filter((m) =>
      [m.name, m.family, ...(m.available_backends ?? [])]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
        .includes(q),
    );
  }, [all, filter]);

  // Union across ALL entries, not the current page, so paging never changes
  // the table's shape. One known node means presence carries no information.
  const fleetSize = useMemo(() => {
    const ids = new Set<string>();
    for (const m of all) {
      const presence = m.nodes;
      if (!presence) continue;
      for (const id of [...presence.present, ...presence.missing, ...presence.unreported]) {
        ids.add(id);
      }
    }
    return ids.size;
  }, [all]);
  const showNodes = fleetSize > 1;

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount);
  const paged = useMemo(
    () => filtered.slice((clampedPage - 1) * PAGE_SIZE, clampedPage * PAGE_SIZE),
    [filtered, clampedPage],
  );

  const columns: Column<StoreEntry>[] = [
    {
      key: "name",
      header: "Name",
      sortAccessor: (m) => m.name,
      render: (m) => <span className="font-mono text-xs text-foreground">{m.name}</span>,
    },
    {
      key: "family",
      header: "Family",
      sortAccessor: (m) => m.family ?? "",
      render: (m) =>
        m.family ? (
          <Badge tone="neutral">
            <Cpu className="h-3 w-3" /> {m.family}
          </Badge>
        ) : (
          <span className="text-muted-foreground">—</span>
        ),
    },
    {
      key: "vision",
      header: "Vision",
      render: (m) =>
        m.vision ? (
          <Badge tone="info">
            <Eye className="h-3 w-3" /> vision
          </Badge>
        ) : (
          <span className="text-muted-foreground">—</span>
        ),
    },
    ...(showNodes
      ? [
          {
            key: "nodes",
            header: "Nodes",
            sortAccessor: (m: StoreEntry) => m.nodes?.present.length ?? -1,
            render: (m: StoreEntry) => <NodePresence entry={m} />,
          } satisfies Column<StoreEntry>,
        ]
      : []),
    {
      key: "size",
      header: "Size",
      sortAccessor: (m) => m.size_bytes ?? 0,
      render: (m) => <span className="font-mono tabular-nums text-xs">{formatBytes(m.size_bytes)}</span>,
    },
    {
      key: "backends",
      header: "Backends",
      render: (m) => (
        <span className="text-xs text-muted-foreground">
          {(m.available_backends ?? []).join(", ") || "—"}
        </span>
      ),
    },
    {
      key: "action",
      header: "",
      className: "text-right",
      render: () => (
        <Button
          size="sm"
          variant="secondary"
          onClick={(e) => {
            e.stopPropagation();
            onDeploy();
          }}
        >
          <Rocket className="h-3.5 w-3.5" />
          Deploy
        </Button>
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
          placeholder="Filter by name, family, backend…"
          className="h-8 w-64 text-xs"
        />
        <div className="ml-auto">
          <LiveIndicator
            live={store.live}
            refreshing={store.refreshing}
            lastUpdated={store.lastUpdated}
            onRefresh={store.refresh}
          />
        </div>
      </Toolbar>

      <ResultLine
        shown={paged.length}
        total={total}
        noun="models"
        onFetch={store.refresh}
        fetching={store.refreshing}
        pager={
          <Pager
            page={clampedPage}
            pageCount={pageCount}
            onPrev={() => setPage((p) => Math.max(1, p - 1))}
            onNext={() => setPage((p) => Math.min(pageCount, p + 1))}
          />
        }
      />

      <Table<StoreEntry>
        columns={columns}
        rows={store.data ? paged : null}
        getRowKey={(m) => m.name}
        loading={store.loading}
        error={store.error}
        emptyIcon={<Boxes className="h-5 w-5" />}
        emptyLabel="No models found"
        emptyDescription="Seed one from a local Ollama / HF reference via Add model — it'll show up here automatically."
      />
    </div>
  );
}

/**
 * Which nodes hold this model's blob.
 *
 * `missing` and `unreported` stay separate on purpose. "Reported, and does not
 * have it" calls for a seed on that node; "has not reported yet" calls for
 * waiting or checking the node. Merging them would turn not-known into
 * not-there — and the two differ at the API too: a deploy aimed at an
 * unreported node is accepted, one aimed at a missing node is refused.
 */
function NodePresence({ entry }: { entry: StoreEntry }) {
  const presence = entry.nodes;
  if (!presence) return <span className="text-muted-foreground">—</span>;
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      {presence.present.length > 0 && (
        <span title={`Blob present on: ${presence.present.join(", ")}`}>
          <Badge tone="ok">{presence.present.length} present</Badge>
        </span>
      )}
      {presence.missing.length > 0 && (
        <span
          title={`Reported without this model: ${presence.missing.join(", ")} — seed it there before deploying on it`}
        >
          <Badge tone="warn">{presence.missing.length} missing</Badge>
        </span>
      )}
      {presence.unreported.length > 0 && (
        <span title={`Has not reported its store yet: ${presence.unreported.join(", ")}`}>
          <Badge tone="neutral">{presence.unreported.length} unreported</Badge>
        </span>
      )}
    </span>
  );
}
