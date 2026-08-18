"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Server,
  Plus,
  Minus,
  Layers,
  Pin,
  PinOff,
  Pencil,
  RefreshCw,
  Trash2,
} from "lucide-react";
import {
  getPorts,
  loadDeployment,
  unloadDeployment,
  pinDeployment,
  deleteDeployment,
  scaleStoreModel,
  repairDeployment,
  updateDeployment,
  getDeploymentLogs,
  deploymentModelType,
  formatBytes,
  type DeploymentLogs,
  type StoreEntry,
  type PortsView as PortsViewData,
  type DeploymentRecord,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
import { T } from "@/lib/i18n";
import { toUserMessage } from "@/lib/errors";
import { useToast } from "../Toast";
import { Alert, Badge, Button, Card, Dialog, Field, Segmented, TextInput } from "../ui";
import { LiveIndicator } from "../LiveIndicator";
import { Toolbar } from "../patterns/Toolbar";
import { ResultLine } from "../patterns/ResultLine";
import { Pager } from "../patterns/Pager";
import { Table, type Column } from "../patterns/Table";
import { POLL_MS, PAGE_SIZE, stateTone, errText, failureLabel, failureTone } from "./shared";

// ---------------------------------------------------------------------------
// Lifecycle phase (PR-4) — reconciler-observed when available, else derived
// from the record's own lifecycle state + activation.
// ---------------------------------------------------------------------------

function derivePhase(r: DeploymentRecord): string {
  const observed = r.observed?.phase;
  if (observed) return observed;
  switch ((r.state ?? "").toLowerCase()) {
    case "ready":
      return "hot";
    case "starting":
    case "degraded":
      return "loading";
    case "failed":
      return "failed";
    case "stopped":
      return r.activation === "managed" ? "evicted" : "cold";
    default:
      return "unknown";
  }
}

const PHASE_STYLES: Record<string, { dot: string; text: string; pulse?: boolean }> = {
  hot: {
    dot: "bg-emerald-500",
    text: "text-emerald-600 dark:text-emerald-400",
    pulse: true,
  },
  loading: {
    dot: "bg-amber-500",
    text: "text-amber-600 dark:text-amber-400",
    pulse: true,
  },
  cold: { dot: "bg-slate-400", text: "text-muted-foreground" },
  evicted: { dot: "bg-sky-400", text: "text-sky-600 dark:text-sky-400" },
  failed: { dot: "bg-rose-500", text: "text-rose-600 dark:text-rose-400" },
  unknown: { dot: "bg-slate-300", text: "text-muted-foreground" },
};

/** Phase chip with a live dot (pulsing while hot/loading) + pin marker. */
/** Hot ⇄ Offloaded switch. `on` = loaded; toggling calls load/unload. */
function LoadToggle({
  on,
  busy,
  disabled,
  onToggle,
}: {
  on: boolean;
  busy: boolean;
  disabled: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={disabled || busy}
      onClick={onToggle}
      title={
        on
          ? "Offload: free the RAM now (record + port kept; auto-reloads on the next request)"
          : "Load: spawn the runtime and wait until it serves"
      }
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors disabled:opacity-50",
        on ? "bg-emerald-500" : "bg-muted-foreground/40",
      )}
    >
      <span
        className={cn(
          "inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform",
          on ? "translate-x-4" : "translate-x-0.5",
          busy && "animate-pulse",
        )}
      />
    </button>
  );
}

function PhaseChip({ record }: { record: DeploymentRecord }) {
  const phase = derivePhase(record);
  const style = PHASE_STYLES[phase] ?? PHASE_STYLES.unknown;
  const hint = [
    record.pinned ? "pinned — never evicted" : null,
    record.observed?.last_error ? `error: ${record.observed.last_error}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <span
      className={cn("inline-flex items-center gap-1.5 text-xs font-medium", style.text)}
      title={hint || phase}
    >
      <span className="relative flex h-2 w-2">
        {style.pulse && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full animate-ping rounded-full opacity-60",
              style.dot,
            )}
          />
        )}
        <span className={cn("relative inline-flex h-2 w-2 rounded-full", style.dot)} />
      </span>
      {phase}
      {record.pinned && <Pin className="h-3 w-3" />}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Measured throughput. Every number in this cell came from an actual request
// sent to the deployment — nothing is estimated from the hardware. A model that
// has not been measured shows a dash, never a zero and never a guess.
// ---------------------------------------------------------------------------

// A measurement older than this is labelled stale. Mirrors the serving-side
// TTL, after which a hot deployment is measured again.
const THROUGHPUT_STALE_MS = 6 * 60 * 60 * 1000;

/** "12s" / "5m" / "3h" / "2d" since an ISO stamp (null when unparseable). */
function elapsedSince(iso: string | null | undefined): { label: string; ms: number } | null {
  if (!iso) return null;
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return null;
  const ms = Math.max(0, Date.now() - then);
  const secs = Math.round(ms / 1000);
  if (secs < 60) return { label: `${secs}s`, ms };
  const mins = Math.round(secs / 60);
  if (mins < 60) return { label: `${mins}m`, ms };
  const hrs = Math.round(mins / 60);
  if (hrs < 48) return { label: `${hrs}h`, ms };
  return { label: `${Math.round(hrs / 24)}d`, ms };
}

function Dash({ title }: { title: string }) {
  return (
    <span className="text-muted-foreground" title={title}>
      —
    </span>
  );
}

function ThroughputCell({ record }: { record: DeploymentRecord }) {
  const observed = record.observed;
  if (observed?.throughput_source === "not-applicable") {
    return (
      <Dash title="This deployment doesn't generate text, so there's no generation speed to measure." />
    );
  }
  const rate = observed?.tokens_per_second;
  if (rate == null || !(rate > 0)) {
    return (
      <Dash title="Not measured yet. Speed is measured by sending one fixed request once the model has been serving for a few minutes." />
    );
  }
  const age = elapsedSince(observed?.throughput_measured_at);
  const stale = age != null && age.ms > THROUGHPUT_STALE_MS;
  const ttft = observed?.ttft_ms;
  const detail = [
    ttft != null && ttft > 0 ? `first token ${Math.round(ttft)} ms` : null,
    age ? `${age.label} ago` : null,
    stale ? "stale" : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const hint = [
    `Measured: ${rate.toFixed(1)} tokens per second.`,
    ttft != null && ttft > 0
      ? observed?.throughput_source === "timings"
        ? `Time to the first token: ${Math.round(ttft)} ms, as reported by the runtime.`
        : `Time to the first token: ${Math.round(ttft)} ms.`
      : "Time to the first token wasn't reported by this runtime.",
    age ? `Taken ${age.label} ago.` : null,
    stale ? "This measurement is old — it will be taken again while the model is loaded." : null,
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <span className="block leading-tight" title={hint}>
      <span
        className={cn(
          "font-mono tabular-nums text-xs",
          stale ? "text-muted-foreground" : "text-foreground",
        )}
      >
        {rate.toFixed(rate >= 100 ? 0 : 1)} tok/s
      </span>
      {detail && (
        <span className="block text-[10px] text-muted-foreground">{detail}</span>
      )}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Deployments view — explicit-column table over DeploymentRecord[].
// ---------------------------------------------------------------------------

type LifecycleAction = "load" | "unload" | "pin" | "unpin" | "delete" | "repair";

interface ScalableGroup {
  base: string;
  records: DeploymentRecord[];
  total: number;
  running: number;
}

// The replica to remove when scaling a model down: the highest numeric suffix
// (`base-3` before `base-2` before the bare `base`), so scale-down peels off
// the last instance added and keeps the base while replicas remain.
function highestReplicaName(records: DeploymentRecord[]): string | null {
  const names = records
    .map((r) => r.spec?.name)
    .filter((n): n is string => Boolean(n));
  if (names.length === 0) return null;
  const suffix = (n: string) => {
    const m = n.match(/-(\d+)$/);
    return m ? parseInt(m[1], 10) : 1;
  };
  return [...names].sort((a, b) => suffix(b) - suffix(a))[0];
}

// The scale panel: one row per store model that has live deployments — deploy
// or remove an instance in place, without hunting the flat table below. Reuses
// the same endpoints the rest of the tab uses (scaleStoreModel / delete). Scale
// up is best-effort against RAM; the Sizing tab is the fit-aware surface and the
// reconciler is the runtime backstop.
function ScaledModelsPanel({
  groups,
  onChanged,
}: {
  groups: ScalableGroup[];
  onChanged: () => void;
}) {
  return (
    <Card
      icon={<Layers className="h-5 w-5" />}
      title="Scaled models"
      subtitle="Run several instances of a store model — each addressable, and load-balanced behind its model id."
      className="mb-4"
    >
      <div className="divide-y divide-border">
        {groups.map((g) => (
          <ScaleRow key={g.base} group={g} onChanged={onChanged} />
        ))}
      </div>
    </Card>
  );
}

function ScaleRow({
  group,
  onChanged,
}: {
  group: ScalableGroup;
  onChanged: () => void;
}) {
  const { toast } = useToast();
  const [busy, setBusy] = useState<"up" | "down" | null>(null);

  async function scaleUp() {
    setBusy("up");
    try {
      const res = await scaleStoreModel(group.base, group.total + 1);
      toast({
        title:
          res.adding.length > 0
            ? `Deploying 1 more of ${group.base}`
            : "Already at target",
        description: group.base,
        tone: "success",
      });
      onChanged();
    } catch (err) {
      toast({
        title: "Scale up failed",
        description: errText(err, "Scale failed."),
        tone: "error",
      });
    } finally {
      setBusy(null);
    }
  }

  async function scaleDown() {
    const victim = highestReplicaName(group.records);
    if (!victim) return;
    if (!window.confirm(`Remove one instance of "${group.base}" (delete "${victim}")?`))
      return;
    setBusy("down");
    try {
      await deleteDeployment(victim);
      toast({ title: "Removing 1 instance", description: victim, tone: "success" });
      onChanged();
    } catch (err) {
      toast({
        title: "Scale down failed",
        description: errText(err, "Delete failed."),
        tone: "error",
      });
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="flex items-center gap-3 py-2 first:pt-0 last:pb-0">
      <span
        className="min-w-0 flex-1 truncate font-mono text-xs text-foreground"
        title={group.base}
      >
        {group.base}
      </span>
      <Badge tone={group.running === group.total ? "ok" : "warn"}>
        {group.running}/{group.total} running
      </Badge>
      <div className="flex items-center gap-1">
        <Button
          size="sm"
          variant="ghost"
          loading={busy === "down"}
          disabled={busy !== null || group.total <= 1}
          title="Remove one instance"
          onClick={() => void scaleDown()}
        >
          <Minus className="h-3.5 w-3.5" />
        </Button>
        <span className="w-6 text-center font-mono tabular-nums text-xs text-foreground">
          {group.total}
        </span>
        <Button
          size="sm"
          variant="ghost"
          loading={busy === "up"}
          disabled={busy !== null}
          title="Deploy one more instance"
          onClick={() => void scaleUp()}
        >
          <Plus className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}

export function DeploymentsView({
  deployments,
  embeddingNames,
  rerankerNames,
  store,
  initialFilter,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
  embeddingNames: Set<string>;
  rerankerNames: Set<string>;
  store: ReturnType<typeof usePolling<StoreEntry[]>>;
  /** Deep-link seed (e.g. from Observability's activity tile) -- a model
   * name to pre-fill the text filter with. Re-applies on every change, not
   * just mount, so clicking a second activity row while already on this
   * tab re-filters instead of no-op'ing on the first click's value. */
  initialFilter?: string;
}) {
  const { toast } = useToast();
  const [filter, setFilter] = useState(initialFilter ?? "");
  useEffect(() => {
    if (initialFilter != null) setFilter(initialFilter);
  }, [initialFilter]);
  const [page, setPage] = useState(1);
  // Segment by semantic model type (chat SLMs / encoder analyzers / embeddings / rerankers).
  const [typeFilter, setTypeFilter] = useState<
    "all" | "chat" | "encoder" | "embedding" | "reranker"
  >("all");
  // Which row's runtime log is expanded (click a row to see WHY it failed).
  const [expanded, setExpanded] = useState<string | null>(null);
  // One in-flight lifecycle action at a time, keyed "name:action" so exactly
  // the pressed button shows its spinner.
  const [busy, setBusy] = useState<string | null>(null);
  const [editing, setEditing] = useState<DeploymentRecord | null>(null);
  const [editContextLength, setEditContextLength] = useState("");
  const [editMaxTokens, setEditMaxTokens] = useState("");
  const [editError, setEditError] = useState<string | null>(null);

  function openEditor(record: DeploymentRecord) {
    setEditing(record);
    setEditContextLength(String(record.spec?.launch?.context_length ?? 8192));
    setEditMaxTokens(
      record.spec?.launch?.max_tokens != null
        ? String(record.spec.launch.max_tokens)
        : "",
    );
    setEditError(null);
  }

  async function saveDeployment() {
    const name = editing?.spec?.name;
    if (!name) return;
    const contextLength = Number(editContextLength);
    const maxTokens = editMaxTokens.trim() ? Number(editMaxTokens) : null;
    if (!Number.isInteger(contextLength) || contextLength < 128 || contextLength > 1_048_576) {
      setEditError("Context window must be an integer between 128 and 1,048,576 tokens.");
      return;
    }
    if (
      maxTokens != null &&
      (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 131_072)
    ) {
      setEditError("Max output tokens must be an integer between 1 and 131,072.");
      return;
    }
    setBusy(`${name}:edit`);
    setEditError(null);
    try {
      await updateDeployment(name, {
        context_length: contextLength,
        max_tokens: maxTokens,
      });
      toast({
        title: "Deployment update requested",
        description: `${name} · ${contextLength.toLocaleString()} token context`,
        tone: "success",
      });
      setEditing(null);
      deployments.refresh();
    } catch (err) {
      setEditError(
        toUserMessage(err, {
          unavailable: "Deployment editing isn't available on this server.",
          fallback: "Could not update the deployment.",
        }),
      );
    } finally {
      setBusy(null);
    }
  }

  async function act(name: string, action: LifecycleAction, port?: number | null) {
    if (
      action === "delete" &&
      !window.confirm(
        `Delete deployment "${name}"? Kills the process, frees the port, removes the record.`,
      )
    ) {
      return;
    }
    setBusy(`${name}:${action}`);
    try {
      if (action === "load") await loadDeployment(name);
      else if (action === "unload") await unloadDeployment(name);
      else if (action === "delete") await deleteDeployment(name);
      else if (action === "repair") await repairDeployment(name, port ?? null);
      else await pinDeployment(name, action === "pin");
      toast({
        title:
          action === "load"
            ? "Load requested"
            : action === "unload"
              ? "Unload requested"
              : action === "delete"
                ? "Delete requested"
                : action === "repair"
                  ? port != null
                    ? `Repair on port ${port} requested`
                    : "Repair (auto-reallocate) requested"
                  : action === "pin"
                    ? "Pinned"
                    : "Unpinned",
        description: name,
        tone: "success",
      });
      deployments.refresh();
    } catch (err) {
      const msg = toUserMessage(err, {
        unavailable: "The lifecycle endpoints aren't available on this server.",
        fallback: `${action} failed.`,
      });
      toast({ title: `${action} failed`, description: msg, tone: "error" });
    } finally {
      setBusy(null);
    }
  }

  const all = deployments.data ?? [];
  const total = all.length;
  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return all.filter((r) => {
      if (typeFilter !== "all" && deploymentModelType(r, embeddingNames, rerankerNames) !== typeFilter)
        return false;
      if (!q) return true;
      const hay = [r.spec?.name, r.spec?.launch?.model, r.spec?.launch?.runtime, r.state]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });
  }, [all, filter, typeFilter]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const clampedPage = Math.min(page, pageCount);
  const paged = useMemo(
    () => filtered.slice((clampedPage - 1) * PAGE_SIZE, clampedPage * PAGE_SIZE),
    [filtered, clampedPage],
  );

  // Group deployments into their base store model to drive the scale panel.
  // A scaled model's replicas all share the launch --alias (= the base store
  // name), so alias is the grouping key; only STORE-backed groups can scale
  // (scaleStoreModel needs a store entry), so gate on the store index.
  const storeNames = useMemo(
    () => new Set((store.data ?? []).map((e) => e.name)),
    [store.data],
  );
  const scalableGroups = useMemo<ScalableGroup[]>(() => {
    const map = new Map<string, DeploymentRecord[]>();
    for (const r of all) {
      const base = r.spec?.launch?.alias || r.spec?.name;
      if (!base || !storeNames.has(base)) continue;
      const arr = map.get(base);
      if (arr) arr.push(r);
      else map.set(base, [r]);
    }
    return [...map.entries()]
      .map(([base, records]) => ({
        base,
        records,
        total: records.length,
        running: records.filter((r) => {
          const p = derivePhase(r);
          return p === "hot" || p === "loading";
        }).length,
      }))
      .sort((a, b) => a.base.localeCompare(b.base));
  }, [all, storeNames]);

  const columns: Column<DeploymentRecord>[] = [
    {
      key: "name",
      header: "Name",
      sortAccessor: (r) => r.spec?.name ?? "",
      render: (r) => <span className="font-medium text-foreground">{r.spec?.name ?? "—"}</span>,
    },
    {
      key: "model",
      header: "Model",
      sortAccessor: (r) => r.spec?.launch?.model ?? "",
      render: (r) => (
        <span className="font-mono text-xs text-foreground/90">{r.spec?.launch?.model ?? "—"}</span>
      ),
    },
    {
      key: "type",
      header: "Type",
      sortAccessor: (r) => deploymentModelType(r, embeddingNames, rerankerNames),
      render: (r) => {
        const t = deploymentModelType(r, embeddingNames, rerankerNames);
        return t === "encoder" ? (
          <Badge tone="info">Encoder</Badge>
        ) : t === "embedding" ? (
          <Badge tone="warn">Embedding</Badge>
        ) : t === "reranker" ? (
          <Badge tone="warn">Reranker</Badge>
        ) : (
          <Badge tone="neutral">Chat</Badge>
        );
      },
    },
    {
      key: "runtime",
      header: "Runtime",
      render: (r) =>
        r.spec?.launch?.runtime ? (
          <Badge tone="neutral">{r.spec.launch.runtime}</Badge>
        ) : (
          <span className="text-muted-foreground">—</span>
        ),
    },
    {
      key: "port",
      header: "Port",
      sortAccessor: (r) => r.spec?.launch?.port ?? 0,
      render: (r) => (
        <span
          className="font-mono tabular-nums text-xs"
          title={r.pid != null ? `pid: ${r.pid}` : undefined}
        >
          {r.spec?.launch?.port ?? "—"}
        </span>
      ),
    },
    {
      key: "state",
      header: "State",
      sortAccessor: (r) => r.state ?? "",
      render: (r) => {
        const failure = r.failure_kind ?? null;
        const reason = r.last_error ?? r.observed?.last_error ?? null;
        return (
          <span
            className="inline-flex items-center gap-1"
            title={
              [
                reason ? `error: ${reason}` : null,
                r.restart_count ? `restarts: ${r.restart_count}` : null,
              ]
                .filter(Boolean)
                .join(" · ") || undefined
            }
          >
            <Badge tone={stateTone(r.state)}>{r.state ?? "unknown"}</Badge>
            {failure && <Badge tone={failureTone(failure)}>{failureLabel(failure)}</Badge>}
          </span>
        );
      },
    },
    {
      key: "phase",
      header: "Phase",
      sortAccessor: (r) => derivePhase(r),
      render: (r) => <PhaseChip record={r} />,
    },
    {
      key: "speed",
      header: "Speed",
      sortAccessor: (r) => r.observed?.tokens_per_second ?? -1,
      render: (r) => <ThroughputCell record={r} />,
    },
    {
      key: "actions",
      header: "",
      className: "text-right",
      render: (r) => {
        const name = r.spec?.name;
        if (!name) return null;
        const phase = derivePhase(r);
        const running = phase === "hot" || phase === "loading";
        const transitioning =
          phase === "loading" || busy === `${name}:load` || busy === `${name}:unload`;
        return (
          <div
            className="flex items-center justify-end gap-2"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Hot ⇄ Offloaded toggle: on = loaded (hot/loading), flipping it
                off unloads (frees RAM, keeps record+port), flipping it on
                loads. Disabled mid-transition so a half-loaded model isn't
                double-toggled. */}
            <LoadToggle
              on={running}
              busy={transitioning}
              disabled={busy !== null && !transitioning}
              onToggle={() => act(name, running ? "unload" : "load")}
            />
            <Button
              size="sm"
              variant="ghost"
              disabled={busy !== null}
              title="Edit deployment runtime settings"
              onClick={() => openEditor(r)}
            >
              <Pencil className="h-3.5 w-3.5" />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              loading={busy === `${name}:pin` || busy === `${name}:unpin`}
              disabled={busy !== null}
              title={
                r.pinned
                  ? "Unpin: allow idle unload / eviction again"
                  : "Pin: never idle-unloaded or evicted"
              }
              onClick={() => act(name, r.pinned ? "unpin" : "pin")}
            >
              {r.pinned ? <PinOff className="h-3.5 w-3.5" /> : <Pin className="h-3.5 w-3.5" />}
            </Button>
            <Button
              size="sm"
              variant="danger"
              loading={busy === `${name}:delete`}
              disabled={busy !== null}
              title="Delete: stop the process, free the port, and remove the deployment"
              onClick={() => act(name, "delete")}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
        );
      },
    },
  ];

  return (
    <div>
      {scalableGroups.length > 0 && (
        <ScaledModelsPanel groups={scalableGroups} onChanged={deployments.refresh} />
      )}

      <Toolbar
        onReset={() => {
          setFilter("");
          setTypeFilter("all");
          setPage(1);
        }}
        resetDisabled={filter === "" && typeFilter === "all"}
      >
        <TextInput
          value={filter}
          onChange={(e) => {
            setFilter(e.target.value);
            setPage(1);
          }}
          placeholder="Filter by name, model, runtime…"
          className="h-8 w-64 text-xs"
        />
        <Segmented
          value={typeFilter}
          onChange={(t) => {
            setTypeFilter(t);
            setPage(1);
          }}
          options={[
            { value: "all", label: "All" },
            { value: "chat", label: "Chat" },
            { value: "encoder", label: "Encoders" },
            { value: "embedding", label: "Embeddings" },
            { value: "reranker", label: "Rerankers" },
          ]}
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
        noun="deployments"
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
        getRowKey={(r, i) => r.spec?.name ?? `dep-${i}`}
        loading={deployments.loading}
        error={deployments.error}
        emptyIcon={<Server className="h-5 w-5" />}
        emptyLabel="No deployments found"
        emptyDescription="Deploy a model with the button above — it'll appear here on the next refresh."
        expandedKey={expanded}
        onRowClick={(r) => {
          const n = r.spec?.name ?? null;
          setExpanded((cur) => (cur === n ? null : n));
        }}
        renderExpanded={(r) =>
          r.spec?.name ? (
            <div className="space-y-3">
              {/* RSS/Speed/Endpoint used to be always-visible columns --
                  collapsed here so the table stops scrolling horizontally at
                  typical widths (was 11 columns wide). Speed came back as a
                  top-level column; RSS/Endpoint stay here. */}
              <dl className="grid gap-4 text-xs sm:grid-cols-2">
                <div>
                  <dt className="text-muted-foreground">RSS</dt>
                  <dd className="mt-0.5 font-mono tabular-nums text-foreground">
                    {r.observed?.rss_bytes ? formatBytes(r.observed.rss_bytes) : "—"}
                  </dd>
                </div>
                <div className="min-w-0">
                  <dt className="text-muted-foreground">Endpoint</dt>
                  <dd
                    className="mt-0.5 truncate font-mono text-foreground"
                    title={r.endpoint ?? undefined}
                  >
                    {r.endpoint ?? "—"}
                  </dd>
                </div>
              </dl>
              <DeploymentLogsPanel
                name={r.spec.name}
                phase={derivePhase(r)}
                busy={busy}
                onRepair={(port) => act(r.spec!.name!, "repair", port)}
              />
            </div>
          ) : null
        }
      />

      <Dialog
        open={editing !== null}
        onClose={() => {
          if (busy?.endsWith(":edit")) return;
          setEditing(null);
          setEditError(null);
        }}
        title="Edit deployment"
        subtitle={editing?.spec?.name}
        footer={
          <>
            <Button
              variant="ghost"
              onClick={() => setEditing(null)}
              disabled={busy?.endsWith(":edit")}
            >
              Cancel
            </Button>
            <Button
              onClick={saveDeployment}
              loading={busy === `${editing?.spec?.name}:edit`}
            >
              Apply and restart
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <Alert tone="warn">
            <T>A running deployment restarts on the same port. Requests may be briefly unavailable while the model reloads.</T>
          </Alert>
          <dl className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <dt className="text-muted-foreground"><T>Model</T></dt>
              <dd className="mt-0.5 truncate font-mono text-foreground">
                {editing?.spec?.launch?.alias ?? editing?.spec?.launch?.model ?? "—"}
              </dd>
            </div>
            <div>
              <dt className="text-muted-foreground"><T>Runtime</T></dt>
              <dd className="mt-0.5 font-mono text-foreground">
                {editing?.spec?.launch?.runtime ?? "—"}
              </dd>
            </div>
          </dl>
          <Field
            label="Context window"
            htmlFor="edit-deployment-context"
            hint="Cannot exceed the model's supported context. Larger windows reserve more RAM."
          >
            <TextInput
              id="edit-deployment-context"
              type="number"
              min={128}
              max={1_048_576}
              step={128}
              value={editContextLength}
              onChange={(e) => setEditContextLength(e.target.value)}
            />
          </Field>
          <Field
            label="Default max output tokens"
            htmlFor="edit-deployment-max-tokens"
            hint="Optional. Blank restores the model-family default; agents and requests can override it."
          >
            <TextInput
              id="edit-deployment-max-tokens"
              type="number"
              min={1}
              max={131_072}
              value={editMaxTokens}
              onChange={(e) => setEditMaxTokens(e.target.value)}
              placeholder="family default"
            />
          </Field>
          {editError && <Alert tone="err">{editError}</Alert>}
        </div>
      </Dialog>
    </div>
  );
}

// Live log tail for a deployment: the WHY behind a failed/loading row. Polls
// while the row is expanded (paused otherwise), reading the same log file the
// supervisor captures last_error from — so an operator sees "Connection
// refused"/"binary not found"/OOM without shell access. A Repair control
// recovers a stuck row in place (reallocate the port, or set a specific one)
// instead of delete+recreate.
function DeploymentLogsPanel({
  name,
  phase,
  busy,
  onRepair,
}: {
  name: string;
  phase: string;
  busy: string | null;
  onRepair: (port?: number | null) => void;
}) {
  const logs = usePolling<DeploymentLogs>(
    () => getDeploymentLogs(name, 200),
    POLL_MS,
    true,
  );
  const data = logs.data;
  const repairing = busy === `${name}:repair`;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-muted-foreground">Runtime log · {name}</p>
        <LiveIndicator
          live={logs.live}
          refreshing={logs.refreshing}
          lastUpdated={logs.lastUpdated}
          onRefresh={logs.refresh}
        />
      </div>
      {data?.last_error && (
        <Alert tone="err">
          <span className="break-words font-mono">{data.last_error}</span>
        </Alert>
      )}
      <pre className="scroll-thin max-h-72 overflow-auto rounded-md border border-border bg-background p-3 text-[11px] leading-relaxed text-foreground/90">
        {logs.loading && !data
          ? "Loading log…"
          : (data?.lines ?? []).length > 0
            ? (data?.lines ?? []).join("\n")
            : "(no log output yet)"}
      </pre>
      <RepairControls
        name={name}
        phase={phase}
        repairing={repairing}
        disabled={busy !== null}
        onRepair={onRepair}
      />
    </div>
  );
}

// Recover a stuck deployment IN PLACE: reallocate its port (auto — steps
// around an orphan still holding the old one), or move it to a specific free
// port. Reuses the Ports view's recommendation + used-port conflict warning so
// an explicit port is never a blind guess.
function RepairControls({
  name,
  phase,
  repairing,
  disabled,
  onRepair,
}: {
  name: string;
  phase: string;
  repairing: boolean;
  disabled: boolean;
  onRepair: (port?: number | null) => void;
}) {
  const ports = useAsync<PortsViewData>("ports", getPorts);
  const [showPort, setShowPort] = useState(false);
  const [port, setPort] = useState("");
  const used = ports.data?.used ?? [];
  const conflict = port.trim() !== "" && used.includes(Number(port));
  const recommended = ports.data?.recommended_next;

  return (
    <div className="rounded-md border border-border bg-muted/20 p-3">
      <p className="mb-2 text-xs font-medium text-foreground"><T>Repair</T></p>
      <p className="mb-2 text-xs text-muted-foreground">
        {phase === "failed"
          ? "This deployment is stuck. Reallocate its port (recommended — steps around a port an orphan process still holds) or move it to a specific one. No delete/recreate; the config is kept."
          : "Move this deployment to a new port without recreating it."}
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          loading={repairing}
          disabled={disabled}
          onClick={() => onRepair(null)}
          title="Redeploy on an automatically-allocated free port (resets the restart counter)"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Reallocate port (auto)
        </Button>
        <Button
          size="sm"
          variant="secondary"
          disabled={disabled}
          onClick={() => setShowPort((s) => !s)}
        >
          Set a specific port…
        </Button>
      </div>
      {showPort && (
        <div className="mt-2 flex flex-wrap items-end gap-2">
          <Field
            label="Port"
            hint={
              recommended != null
                ? `Free ports available — e.g. ${recommended}.`
                : "Pick a free port in the serving range."
            }
          >
            <TextInput
              value={port}
              onChange={(e) => setPort(e.target.value)}
              placeholder={recommended != null ? String(recommended) : "8090"}
              inputMode="numeric"
              className="h-8 w-28 text-xs"
            />
          </Field>
          <Button
            size="sm"
            loading={repairing}
            disabled={disabled || port.trim() === "" || conflict}
            onClick={() => onRepair(Number(port))}
          >
            Repair on port {port.trim() || "…"}
          </Button>
          {conflict && (
            <p className="w-full text-xs text-rose-600 dark:text-rose-400">
              Port {port} is already in use by another deployment — pick a free one.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
