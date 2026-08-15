"use client";

import { useEffect, useMemo, useState } from "react";
import { Gauge, Play, Download, GitCompare, ShieldCheck, Workflow, ScanText } from "lucide-react";
import {
  triggerBenchmark,
  getBenchmarks,
  listDatasets,
  getDeployments,
  selectableDeployments,
  listSchemas,
  listModelProfiles,
  createPipelineProfile,
  createOcrProfile,
  artifactUrl,
  compareRuns,
  validateDataset,
  ApiError,
  ApiUnavailable,
  type BenchmarkRun,
  type RunArtifact,
  type TriggerResponse,
  type DatasetSummary,
  type DeploymentRecord,
  type ComparisonPayload,
  type ComparisonRow,
  type DatasetValidationReport,
  type ModelProfileSummary,
} from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { toUserMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";
import { useToast } from "./Toast";
import {
  Alert,
  Badge,
  type BadgeTone,
  Button,
  Card,
  Field,
  Segmented,
  Select,
  Sheet,
  TextInput,
} from "./ui";
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
  const runs = useAsync("benchmark-runs", getBenchmarks);

  // Discoverability for the three selectors below: registered datasets (what
  // `dataset` can actually reference), live/loadable deployments (what
  // `model_profile` resolves — the shared resolver accepts a deployment name
  // or `store:<name>` directly, not just a models.yaml profile), and the
  // extraction schemas (same GET /v1/schemas + Select pattern as the OCR
  // agent form). Shared SWR keys ("deployments"/"schemas") with the other
  // pages that already fetch them.
  const datasets = useAsync<DatasetSummary[]>("datasets", listDatasets);
  const deployments = useAsync<DeploymentRecord[]>("deployments", getDeployments);
  const schemas = useAsync<string[]>("schemas", listSchemas);
  const modelProfiles = useAsync<ModelProfileSummary[]>("model-profiles", listModelProfiles);
  const modelOptions = useMemo(
    () =>
      selectableDeployments(deployments.data ?? [])
        .map((d) => d.spec?.name)
        .filter((n): n is string => !!n),
    [deployments.data],
  );

  const [dataset, setDataset] = useState("");
  const [validateOpen, setValidateOpen] = useState(false);
  const [modelProfile, setModelProfile] = useState("");
  // The Select below only lists live deployments -- a models.yaml profile
  // that's never been deployed (e.g. a kind="pipeline" OCR->LLM profile,
  // which has no base_url/model of its own to deploy in the first place)
  // was otherwise unreachable from this tab. Mirrors Agents Create's own
  // "Custom reference" escape hatch for the same underlying gap.
  const [customModelProfile, setCustomModelProfile] = useState(false);
  const [pipelineSheetOpen, setPipelineSheetOpen] = useState(false);
  const [ocrSheetOpen, setOcrSheetOpen] = useState(false);
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
      const msg = toUserMessage(err, {
        unavailable: "Benchmarking isn't available on this server.",
        fallback: "Failed to start benchmark.",
      });
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
            <Field
              label="Dataset"
              required
              hint={
                datasets.data && datasets.data.length === 0
                  ? "No datasets registered — see docie_bench.benchmark.registry to add one."
                  : "A dataset registered in data/datasets.yaml."
              }
            >
              <div className="flex gap-2">
                <Select
                  className="flex-1"
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                >
                  <option value="">Select a dataset…</option>
                  {(datasets.data ?? []).map((d) => (
                    <option key={d.name} value={d.name}>
                      {d.name}
                      {d.latest ? ` (${d.latest})` : ""}
                      {d.documents != null ? ` — ${d.documents} docs` : ""}
                    </option>
                  ))}
                </Select>
                <Button
                  type="button"
                  variant="secondary"
                  disabled={!dataset}
                  onClick={() => setValidateOpen(true)}
                  title="Run duplicate-doc_id / missing-file / cross-split-leakage checks"
                >
                  <ShieldCheck className="h-4 w-4" />
                  Validate
                </Button>
              </div>
            </Field>
            <DatasetValidateSheet
              open={validateOpen}
              onClose={() => setValidateOpen(false)}
              dataset={datasets.data?.find((d) => d.name === dataset) ?? null}
            />
            <PipelineProfileSheet
              open={pipelineSheetOpen}
              onClose={() => setPipelineSheetOpen(false)}
              profiles={modelProfiles.data ?? []}
              onCreated={(name) => {
                setCustomModelProfile(true);
                setModelProfile(name);
                modelProfiles.reload();
              }}
            />
            <OcrProfileSheet
              open={ocrSheetOpen}
              onClose={() => setOcrSheetOpen(false)}
              onCreated={() => modelProfiles.reload()}
            />
            <div className="grid gap-4 sm:grid-cols-3">
              <Field
                label="Model profile"
                hint="A live deployment, a models.yaml profile by name (e.g. a pipeline OCR->LLM profile -- pick Custom), or leave empty for the default."
              >
                <Select
                  value={
                    customModelProfile
                      ? "__custom__"
                      : modelOptions.includes(modelProfile)
                        ? modelProfile
                        : ""
                  }
                  onChange={(e) => {
                    if (e.target.value === "__custom__") {
                      setCustomModelProfile(true);
                      setModelProfile("");
                    } else {
                      setCustomModelProfile(false);
                      setModelProfile(e.target.value);
                    }
                  }}
                >
                  <option value="">(default)</option>
                  {modelOptions.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                  <option value="__custom__">Custom (models.yaml profile name)…</option>
                </Select>
                {customModelProfile && (
                  <TextInput
                    className="mt-2"
                    value={modelProfile}
                    onChange={(e) => setModelProfile(e.target.value)}
                    placeholder="profile name from models.yaml"
                  />
                )}
                <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
                  <button
                    type="button"
                    onClick={() => setPipelineSheetOpen(true)}
                    className="inline-flex items-center gap-1 text-xs text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground"
                  >
                    <Workflow className="h-3 w-3" />
                    New pipeline (OCR→LLM) profile…
                  </button>
                  <button
                    type="button"
                    onClick={() => setOcrSheetOpen(true)}
                    className="inline-flex items-center gap-1 text-xs text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground"
                  >
                    <ScanText className="h-3 w-3" />
                    New OCR-only profile…
                  </button>
                </div>
              </Field>
              <Field label="Schema name">
                <Select value={schemaName} onChange={(e) => setSchemaName(e.target.value)}>
                  {(schemas.data ?? []).map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </Select>
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

            {error && <Alert tone="err">{error}</Alert>}

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
  const [compareOpen, setCompareOpen] = useState(false);

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
        // "metrics.json exists" is a metadata-availability flag, not a
        // pass/fail signal -- the real run outcome is the Status column
        // right before this one (ok/err/neutral for completed/failed/
        // other). tone="ok" here would read as "this run succeeded",
        // which isn't what it means.
        const has = Boolean(r.metrics?.summary?.length);
        return <Badge tone={has ? "info" : "neutral"}>{has ? "metrics" : "—"}</Badge>;
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
        actions={
          <Button type="button" variant="secondary" onClick={() => setCompareOpen(true)}>
            <GitCompare className="h-4 w-4" />
            Compare runs
          </Button>
        }
      />

      <ComparisonSheet open={compareOpen} onClose={() => setCompareOpen(false)} runs={all} />

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

// ---------------------------------------------------------------------------
// Comparison — POST /v1/studio/comparisons. Deliberately the smallest useful
// slice: aggregate metric deltas + budget verdict + root-cause lists. No
// per-dimension drill-down, Pareto chart, or promote-as-baseline UI here --
// the backend (PR #184) doesn't persist a baseline concept yet, and the
// aggregate table + root_causes already answer "did this get better or
// worse, and what specifically regressed" without it.
// ---------------------------------------------------------------------------

function ComparisonSheet({
  open,
  onClose,
  runs,
}: {
  open: boolean;
  onClose: () => void;
  runs: BenchmarkRun[];
}) {
  const completed = useMemo(
    () => runs.filter((r) => r.status === "completed" && r.event_id),
    [runs],
  );
  const [baselineId, setBaselineId] = useState("");
  const [candidateId, setCandidateId] = useState("");
  const [result, setResult] = useState<ComparisonPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setResult(null);
      setError(null);
    }
  }, [open]);

  async function onCompare() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await compareRuns(baselineId, candidateId));
    } catch (err) {
      setError(
        toUserMessage(err, {
          unavailable: "Run comparison isn't available on this server.",
          fallback: "Failed to compare runs.",
        }),
      );
    } finally {
      setLoading(false);
    }
  }

  function runLabel(r: BenchmarkRun): string {
    const when = r.created_at ? r.created_at.slice(0, 19).replace("T", " ") : r.event_id;
    return `${r.dataset ?? "?"} · ${r.model_profile ?? "default"} (${when})`;
  }

  const sameRun = Boolean(baselineId) && baselineId === candidateId;

  return (
    <Sheet open={open} onClose={onClose}>
      <div className="space-y-5">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Compare runs</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            POST /v1/studio/comparisons — deltas, confidence intervals and a budget
            verdict between two completed runs.
          </p>
        </div>

        {completed.length < 2 ? (
          <Alert tone="info">Need at least two completed runs to compare.</Alert>
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Baseline">
                <Select
                  value={baselineId}
                  onChange={(e) => {
                    setBaselineId(e.target.value);
                    setResult(null);
                    setError(null);
                  }}
                >
                  <option value="">Select a run…</option>
                  {completed.map((r) => (
                    <option key={r.event_id} value={r.event_id}>
                      {runLabel(r)}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Candidate">
                <Select
                  value={candidateId}
                  onChange={(e) => {
                    setCandidateId(e.target.value);
                    setResult(null);
                    setError(null);
                  }}
                >
                  <option value="">Select a run…</option>
                  {completed.map((r) => (
                    <option key={r.event_id} value={r.event_id}>
                      {runLabel(r)}
                    </option>
                  ))}
                </Select>
              </Field>
            </div>

            {sameRun && <Alert tone="warn">Pick two different runs to compare.</Alert>}

            <Button
              type="button"
              onClick={onCompare}
              loading={loading}
              disabled={!baselineId || !candidateId || sameRun}
            >
              <GitCompare className="h-4 w-4" />
              {loading ? "Comparing…" : "Compare"}
            </Button>
          </>
        )}

        {error && <Alert tone="err">{error}</Alert>}

        {result && <ComparisonResultView payload={result} />}
      </div>
    </Sheet>
  );
}

/**
 * pass/fail/warn/error share one tone map across the verdict badge and every
 * budget-check badge. "warn" (a judge-metric budget downgraded to
 * non-blocking because the judge isn't calibrated yet) and "error" (the
 * comparison -- or a budget itself -- is broken/unmatched) must render as
 * visibly distinct severities: error is a red "err", not the same amber
 * "warn" a benign calibration downgrade uses.
 */
const RESULT_STATUS_TONE: Record<string, BadgeTone> = {
  pass: "ok",
  warn: "warn",
  fail: "err",
  error: "err",
};

function ComparisonResultView({ payload }: { payload: ComparisonPayload }) {
  const aggregate = payload.comparisons.filter((c) => c.dimension === "aggregate");
  const hasRootCauses =
    payload.root_causes.documents.length > 0 || payload.root_causes.fields.length > 0;

  return (
    <div className="space-y-4 border-t border-border pt-4">
      <p className="text-xs text-muted-foreground">
        <span className="font-mono text-foreground/90">
          {payload.baseline.dataset ?? payload.baseline.event_id}
        </span>{" "}
        vs.{" "}
        <span className="font-mono text-foreground/90">
          {payload.candidate.dataset ?? payload.candidate.event_id}
        </span>
      </p>

      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">Verdict</span>
        <Badge tone={RESULT_STATUS_TONE[payload.verdict] ?? "neutral"}>{payload.verdict}</Badge>
      </div>

      {payload.compatibility_errors.length > 0 && (
        <Alert tone="err">{payload.compatibility_errors.join("; ")}</Alert>
      )}

      {aggregate.length > 0 && (
        <div className="scroll-thin overflow-auto rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="bg-muted/60 uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="whitespace-nowrap px-2.5 py-2 font-medium">Metric</th>
                <th className="whitespace-nowrap px-2.5 py-2 font-medium">Baseline</th>
                <th className="whitespace-nowrap px-2.5 py-2 font-medium">Candidate</th>
                <th className="whitespace-nowrap px-2.5 py-2 font-medium">Delta</th>
                <th className="whitespace-nowrap px-2.5 py-2 font-medium">n</th>
                <th className="whitespace-nowrap px-2.5 py-2 font-medium">Flags</th>
              </tr>
            </thead>
            <tbody>
              {aggregate.map((row) => (
                <tr key={row.metric} className="border-t border-border">
                  <td className="whitespace-nowrap px-2.5 py-1.5 font-mono text-foreground/90">
                    {row.metric}
                  </td>
                  <td className="whitespace-nowrap px-2.5 py-1.5 font-mono tabular-nums text-foreground/90">
                    {row.baseline.toFixed(3)}
                  </td>
                  <td className="whitespace-nowrap px-2.5 py-1.5 font-mono tabular-nums text-foreground/90">
                    {row.candidate.toFixed(3)}
                  </td>
                  <td
                    title={`95% CI [${row.confidence_interval_95[0].toFixed(3)}, ${row.confidence_interval_95[1].toFixed(3)}], sign-test p=${row.sign_test_p_value.toFixed(3)}`}
                    className={cn(
                      "whitespace-nowrap px-2.5 py-1.5 font-mono tabular-nums font-medium",
                      row.signed_improvement > 0
                        ? "text-emerald-500"
                        : row.signed_improvement < 0
                          ? "text-red-500"
                          : "text-foreground/90",
                    )}
                  >
                    {row.delta > 0 ? "+" : ""}
                    {row.delta.toFixed(3)}
                  </td>
                  <td className="whitespace-nowrap px-2.5 py-1.5 font-mono tabular-nums text-muted-foreground">
                    {row.paired_samples}
                  </td>
                  <td className="whitespace-nowrap px-2.5 py-1.5 text-muted-foreground">
                    {row.warnings.length > 0 ? row.warnings.join(", ") : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {payload.budget_checks.length > 0 && (
        <div>
          <h3 className="mb-1.5 text-xs font-medium text-muted-foreground">Budget checks</h3>
          <ul className="space-y-1.5">
            {payload.budget_checks.map((check, i) => (
              <li key={i} className="flex items-center gap-2 text-xs">
                <Badge tone={RESULT_STATUS_TONE[check.status] ?? "neutral"}>{check.status}</Badge>
                <span className="text-muted-foreground">
                  <span className="font-medium text-foreground/90">{check.name}</span>
                  {check.metric ? ` — ${check.metric} (${check.dimension ?? "aggregate"})` : ""}
                  {check.reason ? ` — ${check.reason}` : ""}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {hasRootCauses && (
        <div className="grid gap-4 sm:grid-cols-2">
          {payload.root_causes.documents.length > 0 && (
            <RootCauseList title="Regressed documents" rows={payload.root_causes.documents} />
          )}
          {payload.root_causes.fields.length > 0 && (
            <RootCauseList title="Regressed fields" rows={payload.root_causes.fields} />
          )}
        </div>
      )}
    </div>
  );
}

function RootCauseList({ title, rows }: { title: string; rows: ComparisonRow[] }) {
  return (
    <div>
      <h3 className="mb-1.5 text-xs font-medium text-muted-foreground">{title}</h3>
      <ul className="space-y-1">
        {rows.slice(0, 10).map((row, i) => (
          <li key={i} className="flex items-center justify-between gap-2 text-xs">
            <span className="truncate font-mono text-foreground/90">
              {Object.values(row.group)[0]}{" "}
              <span className="text-muted-foreground">({row.metric})</span>
            </span>
            <span className="tabular-nums text-red-500">{row.delta.toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pipeline profile authoring -- POST /v1/studio/model-profiles/pipeline. The
// missing counterpart to #183's "Custom (models.yaml profile name)…" escape
// hatch above: that field lets you TARGET a kind="pipeline" profile by name,
// this Sheet lets you CREATE one instead of hand-editing configs/models.yaml
// on the server. Create-only, scoped strictly to the pipeline shape (name +
// extractor + one of ocr_backend/ocr_model) -- not a general models.yaml
// editor, and not an update/delete flow (the backend doesn't support those
// yet either; see add_pipeline_profile's own docstring for why).
// ---------------------------------------------------------------------------

const OCR_BACKENDS = ["liteparse", "tesseract", "paddleocr"] as const;

function PipelineProfileSheet({
  open,
  onClose,
  profiles,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  profiles: ModelProfileSummary[];
  onCreated: (name: string) => void;
}) {
  const { toast } = useToast();
  const extractors = useMemo(() => profiles.filter((p) => p.kind === "passthrough"), [profiles]);
  const visionModels = useMemo(
    () => profiles.filter((p) => p.kind === "passthrough" && p.vision),
    [profiles],
  );

  const [name, setName] = useState("");
  const [extractor, setExtractor] = useState("");
  const [ocrSource, setOcrSource] = useState<"backend" | "model">("backend");
  const [ocrBackend, setOcrBackend] = useState<string>(OCR_BACKENDS[0]);
  const [ocrModel, setOcrModel] = useState("");
  const [language, setLanguage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setName("");
      setExtractor("");
      setOcrSource("backend");
      setOcrBackend(OCR_BACKENDS[0]);
      setOcrModel("");
      setLanguage("");
      setError(null);
    }
  }, [open]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!name.trim() || !extractor.trim()) {
      setError("Name and extractor are required.");
      return;
    }
    if (ocrSource === "model" && !ocrModel.trim()) {
      setError("Pick an OCR (vision) model.");
      return;
    }
    setSubmitting(true);
    try {
      const created = await createPipelineProfile({
        name: name.trim(),
        extractor: extractor.trim(),
        ...(ocrSource === "backend"
          ? { ocr_backend: ocrBackend }
          : { ocr_model: ocrModel.trim() }),
        ...(language.trim() ? { language: language.trim() } : {}),
      });
      toast({ title: "Pipeline profile created", description: created.name, tone: "success" });
      onCreated(created.name);
      onClose();
    } catch (err) {
      setError(
        toUserMessage(err, {
          unavailable: "Pipeline profile creation isn't available on this server.",
          fallback: "Failed to create pipeline profile.",
        }),
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Sheet open={open} onClose={onClose}>
      <form onSubmit={onSubmit} className="space-y-5">
        <div>
          <h2 className="text-sm font-semibold text-foreground">New pipeline profile</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            POST /v1/studio/model-profiles/pipeline — OCR a document, then extract with a
            passthrough LLM profile. Appends a <code className="rounded bg-muted px-1">
              kind: pipeline
            </code>{" "}
            entry to configs/models.yaml.
          </p>
        </div>

        <Field label="Name" required>
          <TextInput
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="invoice_pipeline"
          />
        </Field>

        <Field
          label="Extractor"
          required
          hint="The passthrough LLM profile that performs the structured extraction."
        >
          <Select value={extractor} onChange={(e) => setExtractor(e.target.value)}>
            <option value="">Select a profile…</option>
            {extractors.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </Select>
          {extractors.length === 0 && (
            <p className="mt-1 text-xs text-muted-foreground">
              No passthrough profiles found in configs/models.yaml.
            </p>
          )}
        </Field>

        <Field label="OCR source">
          <Segmented
            value={ocrSource}
            onChange={setOcrSource}
            options={[
              { value: "backend", label: "Built-in backend" },
              { value: "model", label: "Vision model (VLM)" },
            ]}
          />
        </Field>

        {ocrSource === "backend" ? (
          <Field label="OCR backend">
            <Select value={ocrBackend} onChange={(e) => setOcrBackend(e.target.value)}>
              {OCR_BACKENDS.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </Select>
          </Field>
        ) : (
          <Field
            label="OCR (vision) model"
            hint="A passthrough profile with vision=true — it transcribes the document instead of a built-in OCR engine."
          >
            <Select value={ocrModel} onChange={(e) => setOcrModel(e.target.value)}>
              <option value="">Select a profile…</option>
              {visionModels.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </Select>
            {visionModels.length === 0 && (
              <p className="mt-1 text-xs text-muted-foreground">
                No vision-capable passthrough profiles found.
              </p>
            )}
          </Field>
        )}

        <Field label="Language" hint="Optional — OCR backend/VLM language hint.">
          <TextInput value={language} onChange={(e) => setLanguage(e.target.value)} placeholder="en" />
        </Field>

        {error && <Alert tone="err">{error}</Alert>}

        <Button type="submit" loading={submitting}>
          <Workflow className="h-4 w-4" />
          {submitting ? "Creating…" : "Create profile"}
        </Button>
      </form>
    </Sheet>
  );
}

// ---------------------------------------------------------------------------
// OCR-only profile authoring -- POST /v1/studio/model-profiles/ocr. The sibling
// gap #188 flagged and deferred: that PR only covered kind="pipeline". A
// kind="ocr" profile has no extractor stage -- serving.solutions.OcrSolution
// runs just an OCR backend and returns its raw transcribed text as the
// completion, so this is a gateway/Playground target, not something to route
// into as this tab's benchmark model (hence no onCreated auto-select the way
// PipelineProfileSheet's does -- see the Field hint below).
// ---------------------------------------------------------------------------

function OcrProfileSheet({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (name: string) => void;
}) {
  const { toast } = useToast();

  const [name, setName] = useState("");
  const [backend, setBackend] = useState<string>(OCR_BACKENDS[0]);
  const [language, setLanguage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setName("");
      setBackend(OCR_BACKENDS[0]);
      setLanguage("");
      setError(null);
    }
  }, [open]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!name.trim()) {
      setError("Name is required.");
      return;
    }
    setSubmitting(true);
    try {
      const created = await createOcrProfile({
        name: name.trim(),
        backend,
        ...(language.trim() ? { language: language.trim() } : {}),
      });
      toast({ title: "OCR profile created", description: created.name, tone: "success" });
      onCreated(created.name);
      onClose();
    } catch (err) {
      setError(
        toUserMessage(err, {
          unavailable: "OCR profile creation isn't available on this server.",
          fallback: "Failed to create OCR profile.",
        }),
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Sheet open={open} onClose={onClose}>
      <form onSubmit={onSubmit} className="space-y-5">
        <div>
          <h2 className="text-sm font-semibold text-foreground">New OCR-only profile</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            POST /v1/studio/model-profiles/ocr — run a document through an OCR backend and
            return its transcribed text as the completion, no extraction stage. Appends a{" "}
            <code className="rounded bg-muted px-1">kind: ocr</code> entry to
            configs/models.yaml. Useful as a gateway/Playground target; not a schema-scored
            benchmark model on its own (pair it with an extractor via a pipeline profile
            instead for that).
          </p>
        </div>

        <Field label="Name" required>
          <TextInput
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="tesseract_ocr"
          />
        </Field>

        <Field label="OCR backend">
          <Select value={backend} onChange={(e) => setBackend(e.target.value)}>
            {OCR_BACKENDS.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </Select>
        </Field>

        <Field label="Language" hint="Optional — backend-specific language hint.">
          <TextInput value={language} onChange={(e) => setLanguage(e.target.value)} placeholder="en" />
        </Field>

        {error && <Alert tone="err">{error}</Alert>}

        <Button type="submit" loading={submitting}>
          <ScanText className="h-4 w-4" />
          {submitting ? "Creating…" : "Create profile"}
        </Button>
      </form>
    </Sheet>
  );
}

// ---------------------------------------------------------------------------
// Dataset validation -- POST /v1/studio/datasets/{name}/validate. Same
// smallest-real-slice call as Comparison Lab: no upload, no leakage-threshold
// tuning UI beyond the one param the backend already exposes, no schema
// builder (roadmap-v2's schema-builder half has no backend persistence to
// build a UI on top of yet -- flagged back rather than built).
// ---------------------------------------------------------------------------

function DatasetValidateSheet({
  open,
  onClose,
  dataset,
}: {
  open: boolean;
  onClose: () => void;
  dataset: DatasetSummary | null;
}) {
  const [version, setVersion] = useState("");
  const [result, setResult] = useState<DatasetValidationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setVersion(dataset?.latest ?? "");
      setResult(null);
      setError(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, dataset?.name]);

  async function onValidate() {
    if (!dataset) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await validateDataset(dataset.name, version || undefined));
    } catch (err) {
      setError(
        toUserMessage(err, {
          unavailable: "Dataset validation isn't available on this server.",
          fallback: "Failed to validate dataset.",
        }),
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <Sheet open={open} onClose={onClose}>
      <div className="space-y-5">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Validate dataset</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            POST /v1/studio/datasets/{"{name}"}/validate — duplicate doc_id, missing
            files, cross-split leakage, and document statistics.
          </p>
        </div>

        {!dataset ? (
          <Alert tone="info">Select a dataset first.</Alert>
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Dataset">
                <TextInput value={dataset.name} disabled />
              </Field>
              <Field label="Version">
                <Select value={version} onChange={(e) => setVersion(e.target.value)}>
                  {dataset.versions.length === 0 && <option value="">(latest)</option>}
                  {dataset.versions.map((v) => (
                    <option key={v} value={v}>
                      {v}
                      {v === dataset.latest ? " (latest)" : ""}
                    </option>
                  ))}
                </Select>
              </Field>
            </div>

            <Button type="button" onClick={onValidate} loading={loading}>
              <ShieldCheck className="h-4 w-4" />
              {loading ? "Validating…" : "Run validation"}
            </Button>
          </>
        )}

        {error && <Alert tone="err">{error}</Alert>}

        {result && <DatasetValidationResultView report={result} />}
      </div>
    </Sheet>
  );
}

function DatasetValidationResultView({ report }: { report: DatasetValidationReport }) {
  return (
    <div className="space-y-4 border-t border-border pt-4">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">Result</span>
        <Badge tone={report.valid ? "ok" : "err"}>{report.valid ? "valid" : "invalid"}</Badge>
        <span className="font-mono text-xs text-muted-foreground">{report.reference}</span>
      </div>

      {report.errors.length > 0 && (
        <Alert tone="err">
          <ul className="list-inside list-disc space-y-0.5">
            {report.errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </Alert>
      )}

      {report.warnings.length > 0 && (
        <Alert tone="warn">
          <ul className="list-inside list-disc space-y-0.5">
            {report.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Alert>
      )}

      {report.statistics && (
        <div>
          <h3 className="mb-1.5 text-xs font-medium text-muted-foreground">Statistics</h3>
          <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
            <Stat label="Documents" value={report.statistics.documents} />
            <Stat label="Labeled" value={report.statistics.labeled_documents} />
            <Stat
              label="Total size"
              value={`${(report.statistics.total_bytes / 1024).toFixed(1)} KB`}
            />
          </div>
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            <CountBreakdown title="Splits" counts={report.statistics.splits} />
            <CountBreakdown title="Schemas" counts={report.statistics.schemas} />
            <CountBreakdown title="Languages" counts={report.statistics.languages} />
          </div>
        </div>
      )}

      {report.leakage && (
        <div>
          <h3 className="mb-1.5 text-xs font-medium text-muted-foreground">
            Cross-split leakage (threshold {report.leakage.near_duplicate_threshold})
          </h3>
          {report.leakage.leakage_pairs === 0 ? (
            <p className="text-xs text-muted-foreground">No leakage detected.</p>
          ) : (
            <ul className="space-y-1 text-xs">
              {[...report.leakage.exact_duplicates, ...report.leakage.near_duplicates]
                .slice(0, 10)
                .map((pair, i) => (
                  <li key={i} className="flex items-center justify-between gap-2">
                    <span className="truncate font-mono text-foreground/90">
                      {pair.left_doc_id} ({pair.left_split}) ↔ {pair.right_doc_id} (
                      {pair.right_split})
                    </span>
                    <span className="tabular-nums text-red-500">
                      {pair.similarity != null ? `${(pair.similarity * 100).toFixed(0)}%` : "exact"}
                    </span>
                  </li>
                ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border border-border bg-background px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="font-mono text-sm tabular-nums text-foreground">{value}</div>
    </div>
  );
}

function CountBreakdown({ title, counts }: { title: string; counts: Record<string, number> }) {
  const entries = Object.entries(counts);
  if (entries.length === 0) return null;
  return (
    <div>
      <h4 className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {title}
      </h4>
      <ul className="space-y-0.5 text-xs">
        {entries.map(([key, count]) => (
          <li key={key} className="flex items-center justify-between gap-2">
            <span className="truncate font-mono text-foreground/90">{key}</span>
            <span className="tabular-nums text-muted-foreground">{count}</span>
          </li>
        ))}
      </ul>
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
        No runs yet. Start one with the form — it&apos;ll appear here and under Results.
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
