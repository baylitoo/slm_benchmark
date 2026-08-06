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
