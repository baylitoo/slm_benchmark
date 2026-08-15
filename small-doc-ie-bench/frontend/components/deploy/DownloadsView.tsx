"use client";

import { useMemo, useState } from "react";
import { Download, AlertCircle } from "lucide-react";
import {
  formatBytes,
  getSeedProgress,
  listSeedRuns,
  type SeedProgress,
  type SeedRunSummary,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Badge, type BadgeTone } from "../ui";
import { LiveIndicator } from "../LiveIndicator";
import { Toolbar } from "../patterns/Toolbar";
import { ResultLine } from "../patterns/ResultLine";
import { Pager } from "../patterns/Pager";
import { Table, type Column } from "../patterns/Table";
import { POLL_MS, PAGE_SIZE } from "./shared";

// ---------------------------------------------------------------------------
// Downloads view — the durable record of seed jobs (GET /v1/studio/seeds):
// what's currently downloading, what finished, and why a failed one failed.
// GET /v1/serving/seed-progress (polled from the seed forms' own
// ResultPanel while a download is in flight) still owns the live percentage
// bar; this table is what survives after that panel closes.
// ---------------------------------------------------------------------------

function statusTone(status: SeedRunSummary["status"]): BadgeTone {
  switch (status) {
    case "completed":
      return "ok";
    case "failed":
      return "err";
    default:
      return "warn"; // running
  }
}

// Progress poll interval matches PollingResult.tsx's own seed-progress
// polling (1.5s) -- this cell is the same data, just rendered as a table
// row instead of the trigger form's result panel.
const PROGRESS_POLL_MS = 1500;

/** Live percentage bar for a running download -- GET /v1/serving/seed-progress
 * polled by channel. Falls back to a plain badge until the first chunk lands
 * (no percent reported yet) or if the backend never reports one at all. */
function RunningProgressCell({ channel }: { channel: string }) {
  const progress = usePolling<SeedProgress | null>(
    () => getSeedProgress(channel),
    PROGRESS_POLL_MS,
    true,
  );
  const p = progress.data;
  if (!p || typeof p.percent !== "number") {
    return <Badge tone="warn">running</Badge>;
  }
  const pct = Math.max(0, Math.min(100, p.percent));
  return (
    <div className="w-32">
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span className="font-medium text-foreground">{pct.toFixed(0)}%</span>
        {p.received_bytes != null && p.total_bytes != null && (
          <span className="truncate pl-1">
            {formatBytes(p.received_bytes)} / {formatBytes(p.total_bytes)}
          </span>
        )}
      </div>
      <div className="mt-0.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-accent transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function DownloadsView({ active }: { active: boolean }) {
  const seeds = usePolling<SeedRunSummary[]>(listSeedRuns, POLL_MS, active);
  const [page, setPage] = useState(1);
  const [expanded, setExpanded] = useState<string | null>(null);

  const all = seeds.data ?? [];
  const pageCount = Math.max(1, Math.ceil(all.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount);
  const paged = useMemo(
    () => all.slice((clampedPage - 1) * PAGE_SIZE, clampedPage * PAGE_SIZE),
    [all, clampedPage],
  );

  const columns: Column<SeedRunSummary>[] = [
    {
      key: "name",
      header: "Name",
      render: (r) => <span className="font-mono text-xs text-foreground">{r.name || "—"}</span>,
    },
    {
      key: "kind",
      header: "Source",
      render: (r) => (
        <Badge tone="neutral">{r.kind === "hf" ? "Hugging Face" : "Ollama"}</Badge>
      ),
    },
    {
      key: "reference",
      header: "Reference",
      render: (r) => (
        <span className="font-mono text-xs text-muted-foreground">{r.reference ?? "—"}</span>
      ),
    },
    {
      key: "status",
      header: "Status",
      render: (r) =>
        r.status === "running" ? (
          <RunningProgressCell channel={r.channel} />
        ) : (
          <div className="flex items-center gap-1.5">
            <Badge tone={statusTone(r.status)}>{r.status}</Badge>
            {r.status === "failed" && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setExpanded((cur) => (cur === r.event_id ? null : r.event_id));
                }}
                className="text-muted-foreground hover:text-foreground"
                title="Show error"
              >
                <AlertCircle className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        ),
    },
    {
      key: "created_at",
      header: "Started",
      sortAccessor: (r) => r.created_at,
      render: (r) => (
        <span className="text-xs text-muted-foreground">
          {new Date(r.created_at).toLocaleString()}
        </span>
      ),
    },
  ];

  return (
    <div>
      <Toolbar>
        <div className="ml-auto">
          <LiveIndicator
            live={seeds.live}
            refreshing={seeds.refreshing}
            lastUpdated={seeds.lastUpdated}
            onRefresh={seeds.refresh}
          />
        </div>
      </Toolbar>

      <ResultLine
        shown={paged.length}
        total={all.length}
        noun="downloads"
        onFetch={seeds.refresh}
        fetching={seeds.refreshing}
        pager={
          <Pager
            page={clampedPage}
            pageCount={pageCount}
            onPrev={() => setPage((p) => Math.max(1, p - 1))}
            onNext={() => setPage((p) => Math.min(pageCount, p + 1))}
          />
        }
      />

      <Table<SeedRunSummary>
        columns={columns}
        rows={seeds.data ? paged : null}
        getRowKey={(r) => r.event_id}
        loading={seeds.loading}
        error={seeds.error}
        emptyIcon={<Download className="h-5 w-5" />}
        emptyLabel="No downloads yet"
        emptyDescription="Seed a model from Models or the Catalog — it'll show up here, live and after it settles."
        expandedKey={expanded}
        renderExpanded={(r) =>
          r.error ? (
            <pre className="scroll-thin overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-xs text-foreground/90">
              {r.error}
            </pre>
          ) : null
        }
      />
    </div>
  );
}
