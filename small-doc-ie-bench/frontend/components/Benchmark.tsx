"use client";

import { useState } from "react";
import {
  Gauge,
  Play,
  History,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  FolderClosed,
} from "lucide-react";
import {
  triggerBenchmark,
  getBenchmarks,
  ApiError,
  ApiUnavailable,
  type BenchmarkRun,
  type TriggerResponse,
} from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
import { useToast } from "./Toast";
import {
  Badge,
  Button,
  Card,
  ComingSoon,
  EmptyState,
  Field,
  Skeleton,
  TextInput,
} from "./ui";
import { JsonView } from "./JsonView";
import { ResultPanel } from "./ResultPanel";

export function Benchmark() {
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

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card
        icon={<Gauge className="h-5 w-5" />}
        title="Run a benchmark"
        subtitle="POST /v1/studio/benchmark"
      >
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
            <p className="flex items-start gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
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

      <Card
        icon={<History className="h-5 w-5" />}
        title="Past runs"
        subtitle="GET /v1/serving/benchmarks"
        actions={
          <Button variant="ghost" size="sm" onClick={runs.reload} type="button">
            Reload
          </Button>
        }
      >
        <RunsList
          runs={runs.data}
          loading={runs.loading}
          error={runs.error}
        />
      </Card>
    </div>
  );
}

function RunsList({
  runs,
  loading,
  error,
}: {
  runs: BenchmarkRun[] | null;
  loading: boolean;
  error: unknown;
}) {
  const [open, setOpen] = useState<string | null>(null);

  if (loading && !runs) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    );
  }
  if (error && !runs) return <ComingSoon error={error} />;
  if (!runs || runs.length === 0) {
    return (
      <EmptyState
        icon={<FolderClosed className="h-5 w-5" />}
        title="No benchmark runs yet"
        description="Start a run on the left — completed runs and their metrics show up here."
      />
    );
  }

  return (
    <div className="scroll-thin max-h-[28rem] space-y-2 overflow-auto pr-1">
      {runs.map((run) => {
        const hasMetrics = Boolean(run.metrics?.summary?.length);
        const isOpen = open === run.run;
        return (
          <div
            key={run.run}
            className="overflow-hidden rounded-xl border border-border bg-background"
          >
            <button
              type="button"
              onClick={() => setOpen(isOpen ? null : run.run)}
              className="flex w-full items-center justify-between gap-3 px-3.5 py-3 text-left transition hover:bg-muted/40"
            >
              <div className="flex min-w-0 items-center gap-2">
                {isOpen ? (
                  <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
                ) : (
                  <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                )}
                <span className="truncate text-sm font-medium text-foreground">{run.run}</span>
              </div>
              <Badge tone={hasMetrics ? "ok" : "neutral"}>
                {hasMetrics ? "metrics" : "no metrics"}
              </Badge>
            </button>
            {isOpen && (
              <div className={cn("border-t border-border p-3.5")}>
                {hasMetrics ? (
                  <JsonView value={run.metrics?.summary} maxHeight="18rem" />
                ) : (
                  <p className="text-xs text-muted-foreground">
                    No <code className="rounded bg-muted px-1">metrics.json</code> for this run.{" "}
                    <span className="break-all">{run.path}</span>
                  </p>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
