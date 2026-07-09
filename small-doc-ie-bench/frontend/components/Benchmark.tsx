"use client";

import { useMemo, useState } from "react";
import { Gauge, Play, AlertCircle, Download } from "lucide-react";
import {
  triggerBenchmark,
  getBenchmarks,
  artifactUrl,
  ApiError,
  ApiUnavailable,
  type BenchmarkRun,
  type RunArtifact,
  type TriggerResponse,
} from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { useToast } from "./Toast";
import { Badge, Button, Card, Field, TextInput } from "./ui";
import { ResultPanel } from "./ResultPanel";
import { PageHeader } from "./patterns/PageHeader";
import { Toolbar } from "./patterns/Toolbar";
import { ResultLine } from "./patterns/ResultLine";
import { Pager } from "./patterns/Pager";
import { Table, type Column } from "./patterns/Table";

const PAGE_SIZE = 10;

/**
 * Benchmark = a Run console + a Results table. Two nav-driven sub-views
 * ("run" / "results"), presentation only. The `runs` fetch, `triggerBenchmark`
 * handler, and all form state are unchanged; the Results view just filters and
 * paginates the already-fetched array client-side.
 */
export function Benchmark({ view = "run" }: { view?: string }) {
  const { toast } = useToast();
  const runs = useAsync(getBenchmarks, []);

  const [dataset, setDataset] = useState("");
  const [modelProfile, setModelProfile] = useState("");
  const [schemaName, setSchemaName] = useState("invoice");
  const [concurrency, setConcurrency] = useState("1");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trigger, setTrigger] = useState<TriggerResponse | null>(null);

  async function onRun(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setTrigger(null);
    if (!dataset.trim()) {
      setError("A dataset is required to run a benchmark.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await triggerBenchmark({
        dataset: dataset.trim(),
        ...(modelProfile.trim() ? { model_profile: modelProfile.trim() } : {}),
        ...(schemaName.trim() ? { schema_name: schemaName.trim() } : {}),
        ...(concurrency.trim() ? { concurrency: Number(concurrency) } : {}),
      });
      setTrigger(res);
      toast({ title: "Benchmark started", description: dataset.trim(), tone: "success" });
      runs.reload();
    } catch (err) {
      const msg =
        err instanceof ApiUnavailable
          ? "The benchmark endpoint isn't available yet — this UI is ready for when it ships."
          : err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : "Failed to start benchmark.";
      setError(msg);
      toast({ title: "Benchmark failed", description: msg, tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  if (view === "results") {
    return <ResultsView runs={runs} />;
  }

  // ── Run console ─────────────────────────────────────────────────────────
  return (
    <div>
      <PageHeader
        title="Run benchmark"
        subtitle="POST /v1/studio/benchmark — kick off a dataset run and stream progress."
      />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card icon={<Gauge className="h-5 w-5" />} title="New run">
          <form onSubmit={onRun} className="space-y-4">
            <Field label="Dataset" required hint="Dataset name registered server-side.">
              <TextInput
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                placeholder="e.g. voxel51_invoices"
              />
            </Field>
            <div className="grid gap-4 sm:grid-cols-3">
              <Field label="Model profile" hint="Optional.">
                <TextInput
                  value={modelProfile}
                  onChange={(e) => setModelProfile(e.target.value)}
                  placeholder="(default)"
                />
              </Field>
              <Field label="Schema name">
                <TextInput
                  value={schemaName}
                  onChange={(e) => setSchemaName(e.target.value)}
                  placeholder="invoice"
                />
              </Field>
              <Field label="Concurrency">
                <TextInput
                  type="number"
                  min={1}
                  value={concurrency}
                  onChange={(e) => setConcurrency(e.target.value)}
                  placeholder="1"
                />
              </Field>
            </div>

            {error && (
              <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                {error}
              </p>
            )}

            <Button type="submit" loading={submitting}>
              <Play className="h-4 w-4" />
              {submitting ? "Starting…" : "Start benchmark"}
            </Button>
          </form>

          {trigger && (
            <div className="mt-5 border-t border-border pt-5">
              <ResultPanel trigger={trigger} noun="benchmark" />
            </div>
          )}
        </Card>

        <Card title="Latest results" subtitle="A quick peek — the full history lives under Results.">
          <ResultsSummary runs={runs.data} loading={runs.loading} />
        </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Results view — filterable, paginated table over the past runs.
// ---------------------------------------------------------------------------

function ResultsView({ runs }: { runs: ReturnType<typeof useAsync<BenchmarkRun[]>> }) {
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(1);
  const [expanded, setExpanded] = useState<string | null>(null);

  const all = runs.data ?? [];
  const total = all.length;

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return all;
    return all.filter((r) => {
      const hay = [r.dataset, r.run, r.event_id, r.status, r.model_profile]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });
  }, [all, filter]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount);
  const paged = useMemo(
    () => filtered.slice((clampedPage - 1) * PAGE_SIZE, clampedPage * PAGE_SIZE),
    [filtered, clampedPage],
  );

  const columns: Column<BenchmarkRun>[] = [
    {
      key: "run",
      header: "Run / Dataset",
      sortAccessor: (r) => r.dataset ?? r.run ?? r.event_id ?? "",
      render: (r) => (
        <span className="font-mono text-xs text-foreground">
          {r.dataset || r.run || r.event_id || "run"}
        </span>
      ),
    },
    {
      key: "status",
      header: "Status",
      sortAccessor: (r) => r.status ?? "",
      render: (r) =>
        r.status ? (
          <Badge tone={r.status === "completed" ? "ok" : r.status === "failed" ? "err" : "neutral"}>
            {r.status}
          </Badge>
        ) : (
          <span className="text-muted-foreground">—</span>
        ),
    },
    {
      key: "metrics",
      header: "Metrics",
      render: (r) => {
        const has = Boolean(r.metrics?.summary?.length);
        return <Badge tone={has ? "ok" : "neutral"}>{has ? "metrics" : "—"}</Badge>;
      },
    },
    {
      key: "created",
      header: "Created",
      sortAccessor: (r) => r.created_at ?? "",
      render: (r) => (
        <span className="tabular-nums text-xs text-muted-foreground">{r.created_at ?? "—"}</span>
      ),
    },
    {
      key: "artifacts",
      header: "Artifacts",
      render: (r) =>
        (r.artifacts ?? []).length > 0 ? (
          <ArtifactLinks artifacts={r.artifacts!} />
        ) : (
          <span className="text-muted-foreground">—</span>
        ),
    },
  ];

  return (
    <div>
      <PageHeader
        title="Runs"
        subtitle="GET /v1/studio/runs — durable, addressable benchmark runs."
      />

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
          placeholder="Filter by dataset, status, id…"
          className="h-8 w-64 text-xs"
        />
      </Toolbar>

      <ResultLine
        shown={paged.length}
        total={total}
        noun="runs"
        onFetch={runs.reload}
        fetching={runs.loading}
        pager={
          <Pager
            page={clampedPage}
            pageCount={pageCount}
            onPrev={() => setPage((p) => Math.max(1, p - 1))}
            onNext={() => setPage((p) => Math.min(pageCount, p + 1))}
          />
        }
      />

      <Table<BenchmarkRun>
        columns={columns}
        rows={runs.data ? paged : null}
        getRowKey={(r, i) => r.event_id ?? r.run ?? `run-${i}`}
        loading={runs.loading}
        error={runs.error}
        emptyLabel="No benchmark runs found"
        emptyDescription="Start a run under Benchmark → Run — completed runs and their metrics show up here."
        onRowClick={(r) => {
          const id = r.event_id ?? r.run ?? "";
          setExpanded((cur) => (cur === id ? null : id));
        }}
        expandedKey={expanded}
        renderExpanded={(r) => {
          const summary = r.metrics?.summary;
          return summary?.length ? (
            <MetricsTable summary={summary} />
          ) : (
            <p className="text-xs text-muted-foreground">
              No <code className="rounded bg-muted px-1">metrics.json</code> for this run.
            </p>
          );
        }}
      />
    </div>
  );
}

/** Compact recent-runs preview shown next to the Run form. */
function ResultsSummary({
  runs,
  loading,
}: {
  runs: BenchmarkRun[] | null;
  loading: boolean;
}) {
  if (loading && !runs) {
    return <p className="text-sm text-muted-foreground">Loading runs…</p>;
  }
  const recent = (runs ?? []).slice(0, 6);
  if (recent.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No runs yet. Start one on the left — it&apos;ll appear here and under Results.
      </p>
    );
  }
  return (
    <ul className="space-y-2">
      {recent.map((r, i) => {
        const label = r.dataset || r.run || r.event_id || "run";
        return (
          <li
            key={r.event_id ?? r.run ?? `run-${i}`}
            className="flex items-center justify-between gap-3 rounded-md border border-border bg-background px-3 py-2"
          >
            <span className="truncate font-mono text-xs text-foreground">{label}</span>
            {r.status && (
              <Badge tone={r.status === "completed" ? "ok" : r.status === "failed" ? "err" : "neutral"}>
                {r.status}
              </Badge>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/** Render the benchmark metrics `summary` (array of flat objects) as a table. */
function MetricsTable({ summary }: { summary: Record<string, unknown>[] }) {
  const columns = Array.from(new Set(summary.flatMap((r) => Object.keys(r))));
  return (
    <div className="scroll-thin overflow-auto rounded-md border border-border">
      <table className="w-full text-left text-xs">
        <thead className="bg-muted/60 uppercase tracking-wide text-muted-foreground">
          <tr>
            {columns.map((c) => (
              <th key={c} className="whitespace-nowrap px-2.5 py-2 font-medium">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {summary.map((row, i) => (
            <tr key={i} className="border-t border-border">
              {columns.map((c) => (
                <td key={c} className="whitespace-nowrap px-2.5 py-1.5 font-mono tabular-nums text-foreground/90">
                  {formatCell(row[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Download links for a run's addressable artifacts. */
function ArtifactLinks({ artifacts }: { artifacts: RunArtifact[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {artifacts.map((a) => (
        <a
          key={a.id}
          href={artifactUrl(a.uri)}
          onClick={(e) => e.stopPropagation()}
          className="inline-flex items-center gap-1.5 rounded-md border border-border bg-muted/40 px-2.5 py-1 text-xs font-medium text-foreground transition hover:bg-muted"
          download
        >
          <Download className="h-3.5 w-3.5" />
          {a.name}
        </a>
      ))}
    </div>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
