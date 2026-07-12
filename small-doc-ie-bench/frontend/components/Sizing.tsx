"use client";

import { useMemo, useState } from "react";
import {
  Gauge,
  Scale,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Plus,
  Trash2,
  Boxes,
} from "lucide-react";
import {
  getSizing,
  whatifSizing,
  formatBytes,
  ApiError,
  ApiUnavailable,
  type SizingView,
  type SizingModelFit,
  type WhatIfPlanEntry,
  type WhatIfView,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { cn } from "@/lib/cn";
import { Badge, Button, Card, Select, TextInput } from "./ui";
import { LiveIndicator } from "./LiveIndicator";
import { Table, type Column } from "./patterns/Table";

const POLL_MS = 5000;

/**
 * Sizing tab (control-plane PR-3, design §3): "how many MORE instances fit
 * right now?" Everything rendered here is the OBSERVED surface the serving
 * reconciler publishes — the capacity bar and fit table come straight from
 * GET /v1/serving/sizing, and the what-if selector round-trips through
 * POST /v1/serving/sizing/whatif so the verdict is always the server's,
 * never re-derived client-side. The server applies the deploy path's fit-gate
 * policy — same footprint formula, same safety margin — priced against the
 * reconciler's last published snapshot (the gate itself re-measures live at
 * decision time; see serving.sizing for the honest delta).
 */
export function Sizing({ active = true }: { active?: boolean }) {
  const sizing = usePolling<SizingView>(getSizing, POLL_MS, active);
  const data = sizing.data;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end">
        <LiveIndicator
          live={sizing.live}
          refreshing={sizing.refreshing}
          lastUpdated={sizing.lastUpdated}
          onRefresh={sizing.refresh}
        />
      </div>

      {data && !data.observed_available && (
        <p className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-700 dark:text-amber-400">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            Observed capacity unavailable — {data.detail ?? "no node snapshot published"}.
            Footprints below are still priced from the store; fit counts need a live
            snapshot.
          </span>
        </p>
      )}

      <CapacityBar data={data} />
      <FitTable sizing={sizing} />
      <WhatIf data={data} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Capacity bar — total / used / margin / free, with honesty badges.
// ---------------------------------------------------------------------------

function CapacityBar({ data }: { data: SizingView | null }) {
  const node = data?.node ?? null;
  const total = node?.total_bytes ?? null;
  const free = node?.free_bytes ?? null;
  const margin = data?.safety_margin_bytes ?? null;
  const loadingReserved = data?.loading_reserved_bytes ?? 0;
  const freeEffective = data?.free_effective_bytes ?? null;
  const soft = (data?.source ?? node?.source) === "vm";
  const measuredAgo = agoLabel(node?.updated_at);

  // Segment widths in % of total; clamped so a weird reading never overflows.
  const segments = useMemo(() => {
    if (total == null || free == null || margin == null || total <= 0) return null;
    const used = Math.max(total - free, 0);
    const pct = (n: number) => Math.max(0, Math.min(100, (n / total) * 100));
    const afterMargin = Math.max(free - margin, 0);
    const reserved = Math.min(Math.max(loadingReserved, 0), afterMargin);
    return {
      used,
      usedPct: pct(used),
      marginPct: pct(Math.min(margin, Math.max(free, 0))),
      reserved,
      reservedPct: pct(reserved),
      freePct: pct(Math.max(afterMargin - reserved, 0)),
    };
  }, [total, free, margin, loadingReserved]);

  return (
    <Card
      icon={<Gauge className="h-5 w-5" />}
      title="Capacity"
      subtitle="Node RAM as the serving reconciler measured it last cycle."
      actions={
        <div className="flex items-center gap-2">
          {soft && (
            <Badge tone="warn">
              <AlertTriangle className="h-3 w-3" /> soft numbers (VM view — no cgroup limit)
            </Badge>
          )}
          {node?.reclaimable_bytes != null && node.reclaimable_bytes > 0 && (
            <Badge tone="neutral">
              reclaim-adjusted (−{formatBytes(node.reclaimable_bytes)} cache)
            </Badge>
          )}
        </div>
      }
    >
      {!segments ? (
        <p className="text-sm text-muted-foreground">
          No node snapshot yet — the capacity bar appears once the serving reconciler
          publishes one.
        </p>
      ) : (
        <div>
          <div className="flex h-3 w-full overflow-hidden rounded-full border border-border bg-muted/40">
            <div
              className="h-full bg-accent"
              style={{ width: `${segments.usedPct}%` }}
              title={`Used (working set): ${formatBytes(segments.used)}`}
            />
            <div
              className="h-full bg-amber-400/70"
              style={{ width: `${segments.marginPct}%` }}
              title={`Safety margin: ${formatBytes(margin)}`}
            />
            {segments.reserved > 0 && (
              <div
                className="h-full bg-sky-400/60"
                style={{ width: `${segments.reservedPct}%` }}
                title={`Reserved for loading models (pages not yet resident): ${formatBytes(segments.reserved)}`}
              />
            )}
            <div
              className="h-full bg-emerald-500/40"
              style={{ width: `${segments.freePct}%` }}
              title={`Deployable: ${formatBytes(Math.max(freeEffective ?? 0, 0))}`}
            />
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <LegendChip className="bg-accent" label="used" value={formatBytes(segments.used)} />
            <LegendChip
              className="bg-amber-400/70"
              label="margin"
              value={formatBytes(margin)}
            />
            {segments.reserved > 0 && (
              <LegendChip
                className="bg-sky-400/60"
                label="loading"
                value={formatBytes(segments.reserved)}
              />
            )}
            <LegendChip
              className="bg-emerald-500/40"
              label="deployable"
              value={formatBytes(Math.max(freeEffective ?? 0, 0))}
            />
            <span className="ml-auto">
              total <span className="font-medium text-foreground">{formatBytes(total)}</span>
              {" · "}running RSS{" "}
              <span className="font-medium text-foreground">
                {formatBytes(node?.sum_rss_bytes)}
              </span>
              {measuredAgo && <>{" · "}measured {measuredAgo}</>}
            </span>
          </div>
          {freeEffective != null && freeEffective < 0 && (
            <p className="mt-2 flex items-center gap-1.5 text-xs text-rose-600 dark:text-rose-400">
              <AlertCircle className="h-3.5 w-3.5" />
              The node is {formatBytes(-freeEffective)} inside its safety margin — nothing
              new fits until something unloads.
            </p>
          )}
          <p className="mt-2 text-xs text-muted-foreground">
            Free is the measured number (hot deployments&apos; RSS is already inside
            &quot;used&quot; — never double-counted; models still loading reserve only
            their not-yet-resident remainder). Margin is an explicit{" "}
            {Math.round((data?.assumptions?.margin_fraction ?? 0) * 100)}% of total held
            back before pricing anything new — the same margin the deploy path&apos;s
            fit gate enforces.
          </p>
        </div>
      )}
    </Card>
  );
}

/** "12s ago" / "3m ago" from an ISO stamp (null when absent/unparseable). */
function agoLabel(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const stamp = Date.parse(iso);
  if (Number.isNaN(stamp)) return null;
  const seconds = Math.max(0, Math.round((Date.now() - stamp) / 1000));
  return seconds < 120 ? `${seconds}s ago` : `${Math.round(seconds / 60)}m ago`;
}

function LegendChip({
  className,
  label,
  value,
}: {
  className: string;
  label: string;
  value: string;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cn("h-2.5 w-2.5 rounded-sm", className)} />
      {label} <span className="font-medium text-foreground">{value}</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Per-model fit table.
// ---------------------------------------------------------------------------

function FitTable({ sizing }: { sizing: ReturnType<typeof usePolling<SizingView>> }) {
  const rows = sizing.data?.per_model ?? null;
  // Surface the pricing assumption: uncalibrated KV is priced at THIS context
  // (the deploy default), so the operator knows what the table assumed.
  const ctx = sizing.data?.assumptions?.context_length;

  const columns: Column<SizingModelFit>[] = [
    {
      key: "name",
      header: "Model",
      sortAccessor: (m) => m.name,
      render: (m) => <span className="font-mono text-xs text-foreground">{m.name}</span>,
    },
    {
      key: "family",
      header: "Family",
      sortAccessor: (m) => m.family ?? "",
      render: (m) =>
        m.family ? (
          <Badge tone="neutral">{m.family}</Badge>
        ) : (
          <span className="text-muted-foreground">—</span>
        ),
    },
    {
      key: "footprint",
      header: "Footprint / instance",
      sortAccessor: (m) => m.footprint_bytes ?? 0,
      render: (m) =>
        m.footprint_bytes != null ? (
          <span className="inline-flex items-center gap-2">
            <span className="font-mono tabular-nums text-xs">
              {formatBytes(m.footprint_bytes)}
            </span>
            {m.calibrated ? (
              <span title={`measured steady RSS ${formatBytes(m.calibrated_bytes)}`}>
                <Badge tone="ok">calibrated</Badge>
              </span>
            ) : (
              <Badge tone="neutral">predicted</Badge>
            )}
          </span>
        ) : (
          <span
            className="text-xs text-muted-foreground"
            title={m.detail ?? "unpriceable"}
          >
            unpriceable
          </span>
        ),
    },
    {
      key: "running",
      header: "Running",
      sortAccessor: (m) => m.running_instances ?? 0,
      render: (m) => (
        <span className="font-mono tabular-nums text-xs">{m.running_instances ?? 0}</span>
      ),
    },
    {
      key: "fits",
      header: "Fits now",
      sortAccessor: (m) => m.fits_now ?? -1,
      render: (m) =>
        m.fits_now == null ? (
          <span className="text-muted-foreground" title={m.detail ?? undefined}>
            —
          </span>
        ) : (
          <Badge tone={m.fits_now > 0 ? "ok" : "err"}>
            {m.fits_now > 0 ? `${m.fits_now} more` : "0 — does not fit"}
          </Badge>
        ),
    },
  ];

  return (
    <Card
      icon={<Boxes className="h-5 w-5" />}
      title="Per-model fit"
      subtitle={`footprint = max(calibrated steady RSS, predicted weights + KV${
        ctx != null ? ` @ ${ctx.toLocaleString()}-token ctx (the deploy default)` : ""
      } + overhead + mmproj); fits = floor((free − margin − loading reserve) / footprint).`}
    >
      <Table<SizingModelFit>
        columns={columns}
        rows={rows}
        getRowKey={(m) => m.name}
        loading={sizing.loading}
        error={sizing.error}
        emptyIcon={<Boxes className="h-5 w-5" />}
        emptyLabel="No models in the store"
        emptyDescription="Seed a model on the Models view — it gets a fit estimate here."
      />
    </Card>
  );
}

// ---------------------------------------------------------------------------
// What-if selector — stage a mix, let the SERVER judge it.
// ---------------------------------------------------------------------------

interface StagedItem {
  model: string;
  instances: string; // raw input; validated on submit
  contextLength: string; // "" = server default
}

function WhatIf({ data }: { data: SizingView | null }) {
  const modelNames = useMemo(
    () => (data?.per_model ?? []).map((m) => m.name),
    [data],
  );
  const [staged, setStaged] = useState<StagedItem[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<WhatIfView | null>(null);

  function addRow() {
    setStaged((s) => [
      ...s,
      { model: modelNames[0] ?? "", instances: "1", contextLength: "" },
    ]);
    setResult(null);
  }

  function updateRow(index: number, patch: Partial<StagedItem>) {
    setStaged((s) => s.map((row, i) => (i === index ? { ...row, ...patch } : row)));
    setResult(null); // a stale verdict for an edited plan would mislead
  }

  function removeRow(index: number) {
    setStaged((s) => s.filter((_, i) => i !== index));
    setResult(null);
  }

  async function check() {
    setError(null);
    setResult(null);
    const plan: WhatIfPlanEntry[] = [];
    for (const row of staged) {
      if (!row.model) continue;
      const instances = Number(row.instances);
      if (!Number.isInteger(instances) || instances < 1) {
        setError(`Instances for ${row.model} must be a whole number ≥ 1.`);
        return;
      }
      const ctx = row.contextLength.trim() === "" ? undefined : Number(row.contextLength);
      if (ctx !== undefined && (!Number.isInteger(ctx) || ctx < 1)) {
        setError(`Context length for ${row.model} must be a whole number ≥ 1.`);
        return;
      }
      plan.push({ model: row.model, instances, ...(ctx !== undefined ? { context_length: ctx } : {}) });
    }
    if (plan.length === 0) {
      setError("Stage at least one model first.");
      return;
    }
    setSubmitting(true);
    try {
      setResult(await whatifSizing(plan));
    } catch (err) {
      setError(
        err instanceof ApiUnavailable
          ? "The what-if endpoint isn't available on this backend."
          : err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : "What-if failed.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card
      icon={<Scale className="h-5 w-5" />}
      title="What if…"
      subtitle="Stage a deployment mix — the server prices it with the fit gate's footprint math and margin."
    >
      <div className="space-y-3">
        {staged.length === 0 && (
          <p className="text-sm text-muted-foreground">
            Nothing staged. Add a model to see whether a mix would fit.
          </p>
        )}
        {staged.map((row, index) => (
          <div key={index} className="flex flex-wrap items-center gap-2">
            <Select
              value={row.model}
              onChange={(e) => updateRow(index, { model: e.target.value })}
              className="h-8 w-56 text-xs"
              aria-label="Model"
            >
              {modelNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </Select>
            <span className="text-xs text-muted-foreground">×</span>
            <TextInput
              type="number"
              min={1}
              value={row.instances}
              onChange={(e) => updateRow(index, { instances: e.target.value })}
              className="h-8 w-20 text-xs"
              aria-label="Instances"
            />
            <TextInput
              type="number"
              min={1}
              value={row.contextLength}
              onChange={(e) => updateRow(index, { contextLength: e.target.value })}
              placeholder={
                data?.assumptions?.context_length != null
                  ? `ctx (default ${data.assumptions.context_length.toLocaleString()})`
                  : "ctx (deploy default)"
              }
              className="h-8 w-36 text-xs"
              aria-label="Context length"
            />
            <button
              type="button"
              onClick={() => removeRow(index)}
              aria-label="Remove row"
              className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground transition hover:bg-muted hover:text-foreground"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        ))}

        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={addRow}
            disabled={modelNames.length === 0}
          >
            <Plus className="h-4 w-4" /> Add model
          </Button>
          <Button size="sm" onClick={check} loading={submitting} disabled={staged.length === 0}>
            <Scale className="h-4 w-4" /> Check fit
          </Button>
        </div>

        {error && (
          <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        {result && <WhatIfVerdict result={result} />}
      </div>
    </Card>
  );
}

function WhatIfVerdict({ result }: { result: WhatIfView }) {
  if (result.ok == null) {
    return (
      <p className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-700 dark:text-amber-400">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
        <span>
          Plan needs {formatBytes(result.total_predicted_bytes)}, but there is no live node
          snapshot to judge it against ({result.detail ?? "observed state unavailable"}).
        </span>
      </p>
    );
  }
  if (result.ok) {
    return (
      <p className="flex items-start gap-2 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-700 dark:text-emerald-400">
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
        <span>
          Fits — plan needs {formatBytes(result.total_predicted_bytes)};{" "}
          {formatBytes(result.remaining_bytes)} would remain above the margin.
        </span>
      </p>
    );
  }
  return (
    <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>
        Does not fit — plan needs {formatBytes(result.total_predicted_bytes)}, which is{" "}
        {formatBytes(result.deficit_bytes)} more than the deployable budget (free −
        margin).
      </span>
    </p>
  );
}
