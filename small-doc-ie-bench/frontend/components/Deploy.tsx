"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Rocket,
  Server,
  Eye,
  Cpu,
  Boxes,
  Plus,
  Minus,
  Layers,
  Network,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  ShieldAlert,
  PackagePlus,
  Pin,
  PinOff,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import {
  getStore,
  getFamilies,
  getDeployments,
  getPorts,
  getHfCollection,
  getHfRepo,
  searchHf,
  inspectHf,
  deployModel,
  seedHf,
  seedOllama,
  loadDeployment,
  unloadDeployment,
  pinDeployment,
  deleteDeployment,
  scaleStoreModel,
  repairDeployment,
  getDeploymentLogs,
  deploymentModelType,
  embeddingDeploymentNames,
  formatBytes,
  type DeploymentLogs,
  ApiError,
  ApiUnavailable,
  type StoreEntry,
  type HfRepoView,
  type HfSearchCard,
  type HfInspect,
  type ModelFamily,
  type PortsView as PortsViewData,
  type DeploymentRecord,
  type TriggerResponse,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
import { toUserMessage } from "@/lib/errors";
import { useToast } from "./Toast";
import {
  Badge,
  type BadgeTone,
  Button,
  Card,
  EmptyState,
  Field,
  Select,
  Skeleton,
  Spinner,
  TextInput,
  ComingSoon,
} from "./ui";
import { DataTable } from "./DataTable";
import { LiveIndicator } from "./LiveIndicator";
import { Sizing } from "./Sizing";
import { ResultPanel } from "./ResultPanel";
import { PageHeader } from "./patterns/PageHeader";
import { Toolbar } from "./patterns/Toolbar";
import { ResultLine } from "./patterns/ResultLine";
import { Pager } from "./patterns/Pager";
import { Table, type Column } from "./patterns/Table";

const POLL_MS = 4000;
const PAGE_SIZE = 10;

type SlideOver = null | "deploy" | "seed";

/**
 * Deploy = a table-first serving console with three nav-driven sub-views
 * ("models" / "deployments" / "ports"). The Deploy + Seed forms live in
 * persistently-mounted slide-overs (visibility toggled, never unmounted) so an
 * in-flight deploy/seed and its ResultPanel survive closing the panel or
 * switching views. All pollers, handlers, and API calls are unchanged.
 */
export function Deploy({
  active = true,
  view = "deployments",
}: {
  active?: boolean;
  view?: string;
}) {
  // Auto-refreshing lists — paused when the tab is hidden OR Deploy isn't the
  // active section (every section stays mounted in the shell). Held at the top
  // level so switching sub-views never remounts a poller.
  const store = usePolling<StoreEntry[]>(getStore, POLL_MS, active);
  const deployments = usePolling<DeploymentRecord[]>(getDeployments, POLL_MS, active);
  const families = useAsync("families", getFamilies); // shared SWR key with the Playground

  const [slideOver, setSlideOver] = useState<SlideOver>(null);

  // Deployment names that are embedding models (store family flagged embedding).
  const embeddingNames = useMemo(
    () => embeddingDeploymentNames(store.data, families.data),
    [store.data, families.data],
  );

  const heading =
    view === "models"
      ? { title: "Models", subtitle: "The model store you can deploy — GGUFs, encoders, embeddings." }
      : view === "ports"
        ? { title: "Ports", subtitle: "Live port allocation across running deployments." }
        : view === "sizing"
          ? {
              title: "Sizing",
              subtitle:
                "GET /v1/serving/sizing — how many more instances fit in RAM right now.",
            }
          : { title: "Deployments", subtitle: "GET /v1/serving/deployments — live serving runtimes." };

  return (
    <div>
      <PageHeader
        title={heading.title}
        subtitle={heading.subtitle}
        actions={
          <>
            <Button variant="secondary" size="sm" onClick={() => setSlideOver("seed")}>
              <PackagePlus className="h-4 w-4" />
              Add model
            </Button>
            <Button size="sm" onClick={() => setSlideOver("deploy")}>
              <Rocket className="h-4 w-4" />
              Deploy model
            </Button>
          </>
        }
      />

      {view === "models" ? (
        <ModelsView store={store} onDeploy={() => setSlideOver("deploy")} />
      ) : view === "ports" ? (
        <PortsView deployments={deployments} />
      ) : view === "sizing" ? (
        <Sizing active={active && view === "sizing"} />
      ) : (
        <DeploymentsView
          deployments={deployments}
          embeddingNames={embeddingNames}
          store={store}
        />
      )}

      {/* Slide-overs: both forms stay mounted; only visibility toggles. */}
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/40 transition-opacity duration-200",
          slideOver ? "opacity-100" : "pointer-events-none opacity-0",
        )}
        onClick={() => setSlideOver(null)}
        aria-hidden
      />
      <SlideOverPanel open={slideOver === "deploy"} onClose={() => setSlideOver(null)}>
        <DeployForm store={store} active={active} onDeployed={() => deployments.refresh()} />
      </SlideOverPanel>
      <SlideOverPanel open={slideOver === "seed"} onClose={() => setSlideOver(null)}>
        <AddModelForm families={families.data} onSeeded={() => store.refresh()} />
      </SlideOverPanel>
    </div>
  );
}

/**
 * Right-hand slide-over. Persistently mounted; slides off-screen when closed so
 * its children (a form + any in-flight ResultPanel) keep their state.
 *
 * A11y: while closed the panel is off-screen but still in the DOM, so without
 * `inert` its buttons/inputs would stay in the tab order and reachable by
 * screen readers. `inert={!open}` makes the whole subtree non-focusable and
 * a11y-hidden when closed while preserving the translate-x slide animation and
 * the persistent mount (form state / in-flight ResultPanel survive). On open we
 * move focus into the panel (standard dialog behavior).
 */
function SlideOverPanel({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const openerRef = useRef<HTMLElement | null>(null);
  const wasOpen = useRef(open);

  useEffect(() => {
    if (open && !wasOpen.current) {
      // Opening: remember the trigger, move focus into the panel.
      openerRef.current = document.activeElement as HTMLElement | null;
      closeRef.current?.focus();
    } else if (!open && wasOpen.current) {
      // Closing: return focus to the trigger that opened it, not <body>.
      openerRef.current?.focus();
      openerRef.current = null;
    }
    wasOpen.current = open;
  }, [open]);

  return (
    <aside
      inert={!open}
      aria-hidden={!open}
      className={cn(
        "fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col bg-background shadow-elevated transition-transform duration-200",
        open ? "translate-x-0" : "translate-x-full",
      )}
    >
      <div className="flex items-center justify-end border-b border-border px-3 py-2">
        <button
          ref={closeRef}
          type="button"
          onClick={onClose}
          aria-label="Close panel"
          className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground transition hover:bg-muted hover:text-foreground"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="scroll-thin flex-1 overflow-y-auto p-4">{children}</div>
    </aside>
  );
}

// ---------------------------------------------------------------------------
// Deployment lifecycle → badge tone.
// ---------------------------------------------------------------------------

function stateTone(state?: string | null): BadgeTone {
  switch ((state ?? "").toLowerCase()) {
    case "ready":
    case "running":
    case "serving":
      return "ok";
    case "failed":
    case "error":
      return "err";
    case "starting":
    case "downloading":
    case "degraded":
      return "warn";
    default:
      return "neutral";
  }
}

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

function DeploymentsView({
  deployments,
  embeddingNames,
  store,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
  embeddingNames: Set<string>;
  store: ReturnType<typeof usePolling<StoreEntry[]>>;
}) {
  const { toast } = useToast();
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(1);
  // Segment by semantic model type (chat SLMs / encoder analyzers / embeddings).
  const [typeFilter, setTypeFilter] = useState<"all" | "chat" | "encoder" | "embedding">(
    "all",
  );
  // Which row's runtime log is expanded (click a row to see WHY it failed).
  const [expanded, setExpanded] = useState<string | null>(null);
  // One in-flight lifecycle action at a time, keyed "name:action" so exactly
  // the pressed button shows its spinner.
  const [busy, setBusy] = useState<string | null>(null);

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
      if (typeFilter !== "all" && deploymentModelType(r, embeddingNames) !== typeFilter)
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
      sortAccessor: (r) => deploymentModelType(r, embeddingNames),
      render: (r) => {
        const t = deploymentModelType(r, embeddingNames);
        return t === "encoder" ? (
          <Badge tone="info">Encoder</Badge>
        ) : t === "embedding" ? (
          <Badge tone="warn">Embedding</Badge>
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
        <span className="font-mono tabular-nums text-xs">{r.spec?.launch?.port ?? "—"}</span>
      ),
    },
    {
      key: "state",
      header: "State",
      sortAccessor: (r) => r.state ?? "",
      render: (r) => (
        <span
          title={
            [
              r.last_error ? `error: ${r.last_error}` : null,
              r.restart_count ? `restarts: ${r.restart_count}` : null,
            ]
              .filter(Boolean)
              .join(" · ") || undefined
          }
        >
          <Badge tone={stateTone(r.state)}>{r.state ?? "unknown"}</Badge>
        </span>
      ),
    },
    {
      key: "phase",
      header: "Phase",
      sortAccessor: (r) => derivePhase(r),
      render: (r) => <PhaseChip record={r} />,
    },
    {
      key: "rss",
      header: "RSS",
      sortAccessor: (r) => r.observed?.rss_bytes ?? 0,
      render: (r) =>
        r.observed?.rss_bytes ? (
          <span className="font-mono tabular-nums text-xs">
            {formatBytes(r.observed.rss_bytes)}
          </span>
        ) : (
          <span className="text-muted-foreground">—</span>
        ),
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
        <div className="inline-flex rounded-md border border-border bg-card p-0.5 text-xs">
          {(
            [
              ["all", "All"],
              ["chat", "Chat"],
              ["encoder", "Encoders"],
              ["embedding", "Embeddings"],
            ] as ["all" | "chat" | "encoder" | "embedding", string][]
          ).map(([t, label]) => (
            <button
              key={t}
              type="button"
              onClick={() => {
                setTypeFilter(t);
                setPage(1);
              }}
              className={cn(
                "rounded px-2.5 py-1 transition",
                typeFilter === t
                  ? "bg-muted text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {label}
            </button>
          ))}
        </div>
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
            <DeploymentLogsPanel
              name={r.spec.name}
              phase={derivePhase(r)}
              busy={busy}
              onRepair={(port) => act(r.spec!.name!, "repair", port)}
            />
          ) : null
        }
      />
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
        <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-600 dark:text-rose-400">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span className="break-words font-mono">{data.last_error}</span>
        </p>
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
      <p className="mb-2 text-xs font-medium text-foreground">Repair</p>
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

// ---------------------------------------------------------------------------
// Models view — the deployable store catalog.
// ---------------------------------------------------------------------------

function ModelsView({
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

// ---------------------------------------------------------------------------
// Ports view — pure projection over the deployments already in memory.
// ---------------------------------------------------------------------------

function PortsView({
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

// ---------------------------------------------------------------------------
// Deploy form (model picker + scoped runtime + advanced + progress)
// ---------------------------------------------------------------------------

function DeployForm({
  store,
  active,
  onDeployed,
}: {
  store: ReturnType<typeof usePolling<StoreEntry[]>>;
  active: boolean;
  onDeployed: () => void;
}) {
  const { toast } = useToast();
  const [selected, setSelected] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<string>(""); // "" = auto / store-entry deploy
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [name, setName] = useState("");
  // Port is DISPLAY-ONLY until the operator edits it. `portDirty` gates whether
  // we send it at all: an untouched prefill closes the page-load race (a stale
  // recommendation between poll and submit) by sending NO port, letting the
  // worker allocate authoritatively at deploy time.
  const [port, setPort] = useState("");
  const [portDirty, setPortDirty] = useState(false);
  const [contextLength, setContextLength] = useState("8192");

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trigger, setTrigger] = useState<TriggerResponse | null>(null);

  // Live port view — only polled while the Advanced panel is open (and Deploy is
  // the active, visible section), so a collapsed panel costs nothing.
  const ports = usePolling<PortsViewData>(getPorts, POLL_MS, active && showAdvanced);
  const portsData = ports.data;

  // Prefill the field from the recommended port until the operator types — never
  // clobber their edit.
  useEffect(() => {
    if (!portDirty && portsData?.recommended_next != null) {
      setPort(String(portsData.recommended_next));
    }
  }, [portsData?.recommended_next, portDirty]);

  const usedPorts = portsData?.used ?? [];
  const portConflict =
    portDirty && port.trim() !== "" && usedPorts.includes(Number(port));

  const models = store.data ?? [];
  const selectedEntry = useMemo(
    () => models.find((m) => m.name === selected) ?? null,
    [models, selected],
  );
  // Runtime picker is scoped to the chosen model's faithful backends. "encoder"
  // is NOT an operator-selectable runtime — it is implied by an analyzer
  // family and served through the store (Auto) path, so it never appears as a
  // chip that would wrongly trigger the explicit-runtime serve() route.
  const backends = (selectedEntry?.available_backends ?? []).filter((b) => b !== "encoder");
  const isEncoderEntry = (selectedEntry?.available_backends ?? []).includes("encoder");

  function pick(modelName: string) {
    setSelected(modelName);
    setRuntime(""); // reset to auto when the model changes
    setError(null);
  }

  async function onDeploy(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setTrigger(null);
    if (!selected) {
      setError("Select a model to deploy.");
      return;
    }
    setSubmitting(true);
    try {
      const payload = {
        model: selected,
        ...(runtime ? { runtime } : {}),
        ...(name.trim() ? { name: name.trim() } : {}),
        // Only send a port when the operator explicitly overrode the prefill; an
        // untouched recommendation is sent as NO port so the worker allocates
        // authoritatively (and the page-load-race stale value never ships).
        ...(portDirty && port.trim() ? { port: Number(port) } : {}),
        ...(contextLength.trim() ? { context_length: Number(contextLength) } : {}),
      };
      const res = await deployModel(payload);
      setTrigger(res);
      toast({
        title: "Deployment started",
        description: `${selected}${runtime ? ` · ${runtime}` : ""}`,
        tone: "success",
      });
      onDeployed();
    } catch (err) {
      const msg = toUserMessage(err, {
        unavailable: "Deploying isn't available on this server.",
        fallback: "Deploy failed.",
      });
      setError(msg);
      toast({ title: "Deploy failed", description: msg, tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card
      icon={<Rocket className="h-5 w-5" />}
      title="Deploy a model"
      subtitle="Pick a model from the store, choose a runtime, and serve it."
      actions={
        <LiveIndicator
          live={store.live}
          refreshing={store.refreshing}
          lastUpdated={store.lastUpdated}
          onRefresh={store.refresh}
        />
      }
    >
      <form onSubmit={onDeploy} className="space-y-5">
        {/* Model picker */}
        <div>
          <p className="mb-1.5 flex items-center gap-1 text-xs font-medium text-foreground">
            Model <span className="text-rose-500">*</span>
          </p>
          <ModelPicker
            store={store}
            selected={selected}
            onSelect={pick}
          />
        </div>

        {/* Runtime picker — scoped to the selected model's backends */}
        {selected && (
          <div className="animate-fade-in">
            <p className="mb-1.5 text-xs font-medium text-foreground">Runtime</p>
            <div
              role="radiogroup"
              aria-label="Runtime"
              className="flex flex-wrap gap-2"
            >
              <RuntimeChip
                label="Auto"
                hint="store-entry deploy"
                checked={runtime === ""}
                onClick={() => setRuntime("")}
              />
              {backends.map((b) => (
                <RuntimeChip
                  key={b}
                  label={b}
                  checked={runtime === b}
                  onClick={() => setRuntime(b)}
                />
              ))}
            </div>
            <p className="mt-1.5 text-xs text-muted-foreground">
              {isEncoderEntry
                ? "Encoder checkpoint — served by the encoder runtime automatically (Auto)."
                : backends.length > 0
                  ? "Backends compatible with this model, from its store entry."
                  : "This model lists no explicit backends — Auto lets the server choose."}
            </p>
          </div>
        )}

        {/* Advanced */}
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground transition hover:text-foreground"
          >
            {showAdvanced ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
            Advanced options
          </button>
          {showAdvanced && (
            <div className="mt-3 animate-fade-in space-y-4">
              <div className="grid gap-4 sm:grid-cols-3">
                <Field label="Deployment name" hint="Optional alias.">
                  <TextInput
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="(model name)"
                  />
                </Field>
                <Field
                  label="Port"
                  hint={
                    portDirty
                      ? "Sent as an explicit override."
                      : "Auto-allocated at deploy time (prefilled with the recommended port)."
                  }
                >
                  <TextInput
                    type="number"
                    value={port}
                    onChange={(e) => {
                      setPort(e.target.value);
                      setPortDirty(true);
                    }}
                    placeholder={
                      portsData?.recommended_next != null
                        ? String(portsData.recommended_next)
                        : "auto"
                    }
                  />
                  {portConflict && (
                    <p className="mt-1 flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400">
                      <AlertCircle className="h-3.5 w-3.5" />
                      Port {port} is already in use — the deploy will fail on bind.
                    </p>
                  )}
                </Field>
                <Field label="Context length">
                  <TextInput
                    type="number"
                    value={contextLength}
                    onChange={(e) => setContextLength(e.target.value)}
                    placeholder="8192"
                  />
                </Field>
              </div>

              <PortsAdmin ports={ports} />
            </div>
          )}
        </div>

        {error && (
          <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        <Button type="submit" loading={submitting} disabled={!selected}>
          <Rocket className="h-4 w-4" />
          {submitting ? "Deploying…" : "Deploy model"}
        </Button>
      </form>

      {trigger && (
        <div className="mt-5 border-t border-border pt-5">
          <ResultPanel trigger={trigger} noun="deployment" />
        </div>
      )}
    </Card>
  );
}

function PortsAdmin({ ports }: { ports: ReturnType<typeof usePolling<PortsViewData>> }) {
  const data = ports.data;
  const range = data?.range;
  const recommended = data?.recommended_next;

  return (
    <div className="rounded-md border border-border bg-muted/20 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="text-xs font-medium text-foreground">
          Port allocation
          {range && (
            <span className="ml-2 font-normal text-muted-foreground">
              window {range.start}–{range.end}
            </span>
          )}
          {recommended != null ? (
            <span className="ml-2 font-normal text-muted-foreground">
              · next free ≈ <span className="text-foreground">{recommended}</span> (hint)
            </span>
          ) : data ? (
            <span className="ml-2 font-normal text-amber-600 dark:text-amber-400">
              · window exhausted
            </span>
          ) : null}
        </div>
        <LiveIndicator
          live={ports.live}
          refreshing={ports.refreshing}
          lastUpdated={ports.lastUpdated}
          onRefresh={ports.refresh}
        />
      </div>
      <DataTable
        rows={data?.deployments ?? null}
        loading={ports.loading}
        error={ports.error}
        emptyLabel="No ports in use"
        emptyDescription="Deploy a model — its port appears here on the next refresh."
      />
      <p className="mt-2 text-xs text-muted-foreground">
        The recommended port is a hint; the worker re-checks and allocates authoritatively at
        deploy time. Leave the port field untouched to let it choose.
      </p>
    </div>
  );
}

function ModelPicker({
  store,
  selected,
  onSelect,
}: {
  store: ReturnType<typeof usePolling<StoreEntry[]>>;
  selected: string | null;
  onSelect: (name: string) => void;
}) {
  if (store.loading && !store.data) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-14 w-full" />
        ))}
      </div>
    );
  }
  if (store.error && !store.data) {
    // 501 = catalog not enabled; 404 = route missing on this build.
    return <ComingSoon error={store.error} />;
  }
  const models = store.data ?? [];
  if (models.length === 0) {
    return (
      <EmptyState
        icon={<Boxes className="h-5 w-5" />}
        title="No models in the store yet"
        description="Add one via the Add model button — it'll show up here automatically."
      />
    );
  }

  return (
    <div
      role="radiogroup"
      aria-label="Model"
      className="scroll-thin max-h-72 space-y-2 overflow-auto pr-1"
    >
      {models.map((m) => {
        const isSel = m.name === selected;
        return (
          <button
            key={m.name}
            type="button"
            role="radio"
            aria-checked={isSel}
            onClick={() => onSelect(m.name)}
            className={cn(
              "flex w-full items-start justify-between gap-3 rounded-md border p-3 text-left transition",
              isSel
                ? "border-accent bg-accent/5 ring-1 ring-accent/40"
                : "border-border bg-background hover:border-accent/40 hover:bg-muted/40",
            )}
          >
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-foreground">{m.name}</span>
                {m.vision && (
                  <Badge tone="info">
                    <Eye className="h-3 w-3" /> vision
                  </Badge>
                )}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                {m.family && (
                  <span className="inline-flex items-center gap-1">
                    <Cpu className="h-3 w-3" /> {m.family}
                  </span>
                )}
                <span>{formatBytes(m.size_bytes)}</span>
                {(m.available_backends ?? []).length > 0 && (
                  <span className="truncate">{(m.available_backends ?? []).join(", ")}</span>
                )}
              </div>
            </div>
            <span
              className={cn(
                "mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full border",
                isSel ? "border-accent" : "border-border",
              )}
            >
              {isSel && <span className="h-2 w-2 rounded-full bg-accent" />}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function RuntimeChip({
  label,
  hint,
  checked,
  onClick,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={checked}
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm transition",
        checked
          ? "border-accent bg-accent/10 text-accent"
          : "border-border bg-background text-muted-foreground hover:text-foreground",
      )}
    >
      {label}
      {hint && <span className="text-xs opacity-70">· {hint}</span>}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Add model — Hugging Face direct (preferred), a whole HF collection, or the
// legacy local-Ollama seed. One slide-over, three modes.
// ---------------------------------------------------------------------------

type SeedMode = "search" | "hf" | "collection" | "encoder" | "ollama";

function AddModelForm({
  families,
  onSeeded,
}: {
  families: ModelFamily[] | null;
  onSeeded: () => void;
}) {
  const [mode, setMode] = useState<SeedMode>("search");
  return (
    <div className="space-y-4">
      <div className="inline-flex rounded-lg border border-border bg-muted p-0.5 text-sm">
        {(
          [
            ["search", "Search"],
            ["hf", "By repo"],
            ["collection", "Collection"],
            ["encoder", "Encoder"],
            ["ollama", "Local Ollama"],
          ] as [SeedMode, string][]
        ).map(([m, label]) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={cn(
              "rounded-md px-3 py-1.5 transition",
              mode === m
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {label}
          </button>
        ))}
      </div>
      {/* All modes stay mounted so an in-flight download's ResultPanel survives
          switching modes (same rationale as the slide-over itself). */}
      <div hidden={mode !== "search"}>
        <HfSearchSeed families={families} onSeeded={onSeeded} />
      </div>
      <div hidden={mode !== "hf"}>
        <HfSeedForm families={families} onSeeded={onSeeded} />
      </div>
      <div hidden={mode !== "collection"}>
        <HfCollectionSeed families={families} onSeeded={onSeeded} />
      </div>
      <div hidden={mode !== "encoder"}>
        <EncoderSeedForm families={families} onSeeded={onSeeded} />
      </div>
      <div hidden={mode !== "ollama"}>
        <SeedForm families={families} onSeeded={onSeeded} />
      </div>
    </div>
  );
}

// Browse-and-deploy: search the Hub, and for a picked repo show a pre-flight
// SUPPORT VERDICT (architecture → family, read without downloading) that drives
// the Deploy button — the HuggingFace-like experience where the platform tells
// you up front what it can serve.
function verdictTone(verdict: string | undefined): BadgeTone {
  return verdict === "supported" ? "ok" : verdict === "needs_family" ? "warn" : "err";
}

function verdictLabel(verdict: string | undefined): string {
  return verdict === "needs_family" ? "needs a family" : (verdict ?? "unknown");
}

function HfSearchSeed({
  families,
  onSeeded,
}: {
  families: ModelFamily[] | null;
  onSeeded: () => void;
}) {
  const { toast } = useToast();
  const [query, setQuery] = useState("");
  const [ggufOnly, setGgufOnly] = useState(true);
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<HfSearchCard[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [selected, setSelected] = useState<string | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [inspect, setInspect] = useState<HfInspect | null>(null);
  const [name, setName] = useState("");
  const [family, setFamily] = useState("");
  const [quant, setQuant] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [trigger, setTrigger] = useState<TriggerResponse | null>(null);
  // A seed is downloading in the background until its run settles. Guards
  // navigating away from it (picking another model / re-searching) so an
  // ongoing download is never silently dropped from view.
  const [seedActive, setSeedActive] = useState(false);
  const [seedingRepo, setSeedingRepo] = useState<string | null>(null);

  function confirmLeaveSeed(): boolean {
    if (!seedActive || !seedingRepo) return true;
    return window.confirm(
      `A download is still in progress for "${seedingRepo}". It keeps running in ` +
        "the background (it appears under Models when done). Start on another model?",
    );
  }

  // A new search is a fresh start: clear any previously-selected/inspected
  // model and its (possibly completed) seed panel, so the results list and the
  // detail panel never get stuck on the last deploy.
  function resetSelection() {
    setSelected(null);
    setInspect(null);
    setTrigger(null);
    setSeedActive(false);
    setSeedingRepo(null);
    setName("");
    setFamily("");
    setQuant("");
  }

  async function runSearch(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!query.trim()) return;
    if (!confirmLeaveSeed()) return;
    resetSelection();
    setSearching(true);
    try {
      setResults(await searchHf(query.trim(), ggufOnly));
    } catch (err) {
      setError(errText(err, "Search failed."));
    } finally {
      setSearching(false);
    }
  }

  async function pick(repo: string) {
    if (repo !== selected && !confirmLeaveSeed()) return;
    setSeedActive(false);
    setSeedingRepo(null);
    setSelected(repo);
    setInspect(null);
    setTrigger(null);
    setInspecting(true);
    try {
      const v = await inspectHf(repo);
      setInspect(v);
      setName(v.suggested_name ?? "");
      setFamily(v.family ?? "");
      setQuant(v.quants?.includes("Q4_K_M") ? "Q4_K_M" : (v.quants?.[0] ?? ""));
    } catch (err) {
      setInspect({ repo, verdict: "unsupported", reason: errText(err, "Inspect failed.") });
    } finally {
      setInspecting(false);
    }
  }

  async function onSeed() {
    if (!selected || !family) return;
    setSubmitting(true);
    try {
      const res = await seedHf({
        repo: selected,
        quant: quant || null,
        quant_prefer: true,
        name: name.trim() || null,
        family,
      });
      setTrigger(res);
      setSeedActive(true);
      setSeedingRepo(selected);
      toast({ title: "Download started", description: selected, tone: "success" });
      onSeeded();
    } catch (err) {
      toast({ title: "Seed failed", description: errText(err, "Seed failed."), tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  const unsupported = inspect?.verdict === "unsupported";

  return (
    <Card
      icon={<Boxes className="h-5 w-5" />}
      title="Search Hugging Face"
      subtitle="Browse the Hub and deploy — the platform pre-checks each model's architecture and tells you what it can serve."
    >
      <form onSubmit={runSearch} className="space-y-3">
        <div className="flex items-center gap-2">
          <TextInput
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search models (e.g. lfm2, nuextract, gliner, ocr)…"
          />
          <Button type="submit" loading={searching}>
            Search
          </Button>
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={ggufOnly}
            onChange={(e) => setGgufOnly(e.target.checked)}
            className="h-3.5 w-3.5"
          />
          GGUF only (uncheck to include encoder / transformers safetensors checkpoints)
        </label>
      </form>

      {error && (
        <p className="mt-3 flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          {error}
        </p>
      )}

      {results && (
        <div className="mt-4 grid gap-4 lg:grid-cols-[1fr_1fr]">
          {/* Results list */}
          <div className="scroll-thin max-h-[26rem] space-y-1.5 overflow-y-auto pr-1">
            {results.length === 0 && (
              <p className="text-sm text-muted-foreground">No models found.</p>
            )}
            {results.map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => void pick(r.id)}
                className={cn(
                  "w-full rounded-md border px-3 py-2 text-left transition",
                  selected === r.id
                    ? "border-accent bg-accent/10"
                    : "border-border hover:bg-muted",
                )}
              >
                <p className="truncate text-sm font-medium text-foreground">{r.id}</p>
                <p className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
                  {r.downloads != null && <span>↓ {r.downloads.toLocaleString()}</span>}
                  {r.likes != null && <span>♥ {r.likes}</span>}
                  {r.gated && <Badge tone="warn">gated</Badge>}
                </p>
              </button>
            ))}
          </div>

          {/* Inspect + deploy panel */}
          <div className="rounded-md border border-border bg-muted/20 p-3">
            {!selected ? (
              <p className="text-sm text-muted-foreground">
                Pick a model to see its support verdict and deploy it.
              </p>
            ) : inspecting ? (
              <p className="flex items-center gap-2 text-sm text-muted-foreground">
                <Spinner /> Inspecting {selected}…
              </p>
            ) : inspect ? (
              <div className="space-y-3">
                <p className="truncate text-sm font-medium text-foreground">{inspect.repo}</p>
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge tone={verdictTone(inspect.verdict)}>{verdictLabel(inspect.verdict)}</Badge>
                  {inspect.architecture && (
                    <Badge tone="neutral">arch: {inspect.architecture}</Badge>
                  )}
                  {inspect.has_mmproj && <Badge tone="info">vision · mmproj</Badge>}
                </div>
                {inspect.reason && (
                  <p className="text-xs text-muted-foreground">{inspect.reason}</p>
                )}
                {inspect.runtime_note && (
                  <p className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
                    <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    Runtime: {inspect.runtime_note}
                  </p>
                )}
                {inspect.needs_trust_remote_code && (
                  <p className="flex items-start gap-2 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-700 dark:text-rose-400">
                    <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    <span>
                      <strong>Runs custom code.</strong> This checkpoint executes the
                      repo&apos;s own Python on the serving node. To load it, select the{" "}
                      <code>transformers_trust_remote_code</code> family below — a
                      deliberate choice, never the default. Only deploy code you trust.
                    </span>
                  </p>
                )}

                {!unsupported && (
                  <>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <Field label="Store name">
                        <TextInput value={name} onChange={(e) => setName(e.target.value)} />
                      </Field>
                      <Field label="Family" hint="Suggested from the architecture — override if needed.">
                        <Select value={family} onChange={(e) => setFamily(e.target.value)}>
                          <FamilyOptions families={families} />
                        </Select>
                      </Field>
                    </div>
                    {(inspect.quants?.length ?? 0) > 0 && (
                      <Field label="Quantization">
                        <div className="flex flex-wrap gap-1.5">
                          {inspect.quants!.map((q) => (
                            <button
                              key={q}
                              type="button"
                              onClick={() => setQuant(q)}
                              className={cn(
                                "rounded-md border px-2 py-1 text-xs transition",
                                quant === q
                                  ? "border-accent bg-accent/10 text-accent"
                                  : "border-border hover:bg-muted",
                              )}
                            >
                              {q}
                            </button>
                          ))}
                        </div>
                      </Field>
                    )}
                    <Button
                      type="button"
                      loading={submitting}
                      disabled={!family}
                      onClick={() => void onSeed()}
                    >
                      <Rocket className="h-4 w-4" />
                      Download &amp; seed
                    </Button>
                    {inspect.verdict === "needs_family" && (
                      <p className="text-xs text-amber-600 dark:text-amber-400">
                        No confirmed family for this architecture — pick one and try, or add a
                        family contract for it.
                      </p>
                    )}
                  </>
                )}

                {trigger && (
                  <div className="border-t border-border pt-3">
                    {seedActive && (
                      <Badge tone="info" className="mb-2">
                        download in progress · continues in the background
                      </Badge>
                    )}
                    <ResultPanel
                      trigger={trigger}
                      noun="seed"
                      onSettled={() => setSeedActive(false)}
                    />
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>
      )}
    </Card>
  );
}

// Encoders (analyzer families — safetensors, not GGUF) are SEEDED into the
// store like any other model: the snapshot downloads once (live progress),
// then deploys via the normal flow with zero network at boot. The family list
// is data-driven (families where analyzer === true), so a new analyzer family
// added on the backend shows up here with no UI change.
function EncoderSeedForm({
  families,
  onSeeded,
}: {
  families: ModelFamily[] | null;
  onSeeded: () => void;
}) {
  const { toast } = useToast();
  const analyzerFamilies = useMemo(
    () => (families ?? []).filter((f) => f.analyzer),
    [families],
  );
  const [repo, setRepo] = useState("");
  const [name, setName] = useState("");
  const [family, setFamily] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trigger, setTrigger] = useState<TriggerResponse | null>(null);

  // Default to the first analyzer family the backend reports.
  useEffect(() => {
    if (!family && analyzerFamilies.length > 0) setFamily(analyzerFamilies[0].name);
  }, [analyzerFamilies, family]);

  async function onSeed(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setTrigger(null);
    if (!repo.trim() || !name.trim()) {
      setError("Repo and store name are both required.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await seedHf({ repo: repo.trim(), name: name.trim(), family });
      setTrigger(res);
      toast({
        title: "Encoder download started",
        description: `${name.trim()} — snapshot into the store, then Deploy it.`,
        tone: "success",
      });
      onSeeded();
    } catch (err) {
      const msg = errText(err, "Encoder seed failed.");
      setError(msg);
      toast({ title: "Encoder seed failed", description: msg, tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card
      icon={<Cpu className="h-5 w-5" />}
      title="Add an encoder model"
      subtitle="Analyzer families (safetensors checkpoints served by the encoder runtime) — e.g. zero-shot NER or PII / guardrails."
    >
      <form onSubmit={onSeed} className="space-y-4">
        <Field
          label="Hugging Face repo"
          required
          hint="Any checkpoint compatible with the selected analyzer family."
        >
          <TextInput
            value={repo}
            onChange={(e) => setRepo(e.target.value)}
            placeholder="owner/checkpoint"
          />
        </Field>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Store name" required hint="How the model is listed and selected.">
            <TextInput
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my-analyzer"
            />
          </Field>
          <Field
            label="Analyzer family"
            hint="The available analyzer families reported by the serving backend."
          >
            {analyzerFamilies.length === 0 ? (
              <p className="rounded-md border border-dashed border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
                No analyzer family available on this backend.
              </p>
            ) : (
              <Select value={family} onChange={(e) => setFamily(e.target.value)}>
                {analyzerFamilies.map((f) => (
                  <option key={f.name} value={f.name}>
                    {f.name}
                    {f.encoder_backend ? ` · ${f.encoder_backend}` : ""}
                  </option>
                ))}
              </Select>
            )}
          </Field>
        </div>

        {error && (
          <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        <Button type="submit" loading={submitting} disabled={!family}>
          <Plus className="h-4 w-4" />
          Download & seed
        </Button>
        <p className="text-xs text-muted-foreground">
          The checkpoint snapshot downloads once into the store with live
          progress; deploying it afterwards is instant (no network at boot).
          Requires the encoders extra on the serving node.
        </p>
      </form>

      {trigger && (
        <div className="mt-5 border-t border-border pt-5">
          <ResultPanel trigger={trigger} noun="seed" />
        </div>
      )}
    </Card>
  );
}

// The generic capability tag for a family, derived purely from its flags — so
// any family the backend adds reads sensibly with no per-name UI change.
function familyTypeTag(f: ModelFamily): string {
  if (f.analyzer) return "encoder / analyzer";
  if (f.embedding) return "embedding · vectors (RAG)";
  // Transformers/AutoModel is the last-resort path (no GGUF) — label it as such
  // before the generic vision/chat tags so a manual pick reads honestly.
  if (f.transformers_runtime)
    return f.trust_remote_code
      ? "transformers · last resort · trusts remote code"
      : "transformers · last resort (~2-3x RAM)";
  if (f.vision || f.needs_mmproj) return "vision";
  return "chat / extraction";
}

function familyOptionLabel(f: ModelFamily): string {
  return `${f.name} — ${familyTypeTag(f)}`;
}

/** Family <option> list from the families API, with descriptive labels. */
function FamilyOptions({ families }: { families: ModelFamily[] | null }) {
  if (!families || families.length === 0) {
    return <option value="openai_chat">openai_chat — chat / extraction</option>;
  }
  return (
    <>
      {families.map((f) => (
        <option key={f.name} value={f.name}>
          {f.name} · {familyOptionLabel(f)}
        </option>
      ))}
    </>
  );
}

function isEmbeddingFamily(families: ModelFamily[] | null, name: string): boolean {
  return (families ?? []).some((f) => f.name === name && f.embedding);
}

function familyOptionsOf(families: ModelFamily[] | null): string[] {
  return families && families.length > 0 ? families.map((f) => f.name) : ["openai_chat"];
}

function HfSeedForm({
  families,
  onSeeded,
}: {
  families: ModelFamily[] | null;
  onSeeded: () => void;
}) {
  const { toast } = useToast();
  const [repo, setRepo] = useState("");
  const [inspecting, setInspecting] = useState(false);
  const [repoView, setRepoView] = useState<HfRepoView | null>(null);
  const [quant, setQuant] = useState<string>("");
  const [name, setName] = useState("");
  const [family, setFamily] = useState("openai_chat");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trigger, setTrigger] = useState<TriggerResponse | null>(null);

  const quants = useMemo(
    () =>
      (repoView?.ggufs ?? []).filter((g) => !g.is_mmproj && !g.is_multipart),
    [repoView],
  );
  const hasMmproj = (repoView?.ggufs ?? []).some((g) => g.is_mmproj);

  async function inspect() {
    setError(null);
    setRepoView(null);
    if (!repo.trim()) {
      setError("Enter a Hugging Face repo id (owner/Name-GGUF).");
      return;
    }
    setInspecting(true);
    try {
      const view = await getHfRepo(repo.trim());
      setRepoView(view);
      setName(view.suggested_name);
      const usable = view.ggufs.filter((g) => !g.is_mmproj && !g.is_multipart);
      setQuant(usable.find((g) => g.quant === "Q4_K_M")?.quant ?? usable[0]?.quant ?? "");
    } catch (err) {
      setError(errText(err, "Could not inspect the repo."));
    } finally {
      setInspecting(false);
    }
  }

  async function onSeed(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setTrigger(null);
    if (!repoView) {
      void inspect();
      return;
    }
    setSubmitting(true);
    try {
      const res = await seedHf({
        repo: repoView.repo,
        quant: quant || null,
        name: name.trim() || null,
        family,
      });
      setTrigger(res);
      toast({
        title: "Download started",
        description: `${repoView.repo}${quant ? ` · ${quant}` : ""}`,
        tone: "success",
      });
      onSeeded();
    } catch (err) {
      const msg = errText(err, "Seeding failed.");
      setError(msg);
      toast({ title: "Seed failed", description: msg, tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card
      icon={<PackagePlus className="h-5 w-5" />}
      title="Add from Hugging Face"
      subtitle="Any GGUF repo on the Hub — inspect, pick a quant, download into the store."
    >
      <form onSubmit={onSeed} className="space-y-4">
        <Field label="Repo" required hint='e.g. "LiquidAI/LFM2.5-350M-Instruct-GGUF"'>
          <div className="flex items-center gap-2">
            <TextInput
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
              placeholder="owner/Model-GGUF"
            />
            <Button
              type="button"
              variant="secondary"
              loading={inspecting}
              onClick={() => void inspect()}
            >
              Inspect
            </Button>
          </div>
        </Field>

        {repoView && (
          <>
            <div>
              <p className="mb-1.5 text-xs font-medium text-foreground">
                Quantization ({quants.length} available)
              </p>
              <div className="flex max-h-40 flex-wrap gap-2 overflow-y-auto">
                {quants.map((g) => (
                  <button
                    key={g.filename}
                    type="button"
                    onClick={() => setQuant(g.quant ?? "")}
                    className={cn(
                      "rounded-md border px-2.5 py-1.5 text-xs transition",
                      quant === g.quant
                        ? "border-accent bg-accent/10 text-accent"
                        : "border-border bg-card text-foreground hover:bg-muted",
                    )}
                  >
                    {g.quant ?? g.filename}
                    {g.size_bytes != null && (
                      <span className="ml-1 text-muted-foreground">
                        {formatBytes(g.size_bytes)}
                      </span>
                    )}
                  </button>
                ))}
              </div>
              {hasMmproj && (
                <p className="mt-1 text-xs text-muted-foreground">
                  This repo ships a vision projector (mmproj) — downloaded
                  automatically for families that need one.
                </p>
              )}
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Store name" required>
                <TextInput value={name} onChange={(e) => setName(e.target.value)} />
              </Field>
              <Field
                label="Family"
                hint={
                  isEmbeddingFamily(families, family)
                    ? "Embedding model — served with --embedding, used via /v1/embeddings (Playground → Embed)."
                    : "How the model is launched and prompted. Pick 'embedding' for a vector model."
                }
              >
                <Select value={family} onChange={(e) => setFamily(e.target.value)}>
                  <FamilyOptions families={families} />
                </Select>
              </Field>
            </div>
          </>
        )}

        {error && (
          <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        <Button type="submit" loading={submitting}>
          <Plus className="h-4 w-4" />
          {repoView ? "Download & seed" : "Inspect repo"}
        </Button>
      </form>

      {trigger && (
        <div className="mt-5 border-t border-border pt-5">
          <ResultPanel trigger={trigger} noun="seed" />
        </div>
      )}
    </Card>
  );
}

function HfCollectionSeed({
  families,
  onSeeded,
}: {
  families: ModelFamily[] | null;
  onSeeded: () => void;
}) {
  const { toast } = useToast();
  const [slug, setSlug] = useState("");
  const [loading, setLoading] = useState(false);
  const [title, setTitle] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [family, setFamily] = useState("openai_chat");
  const [quant, setQuant] = useState("Q4_K_M");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [triggers, setTriggers] = useState<{ repo: string; trigger: TriggerResponse }[]>([]);

  async function load() {
    setError(null);
    setTitle(null);
    setModels([]);
    if (!slug.trim()) {
      setError("Paste a collection URL or owner/slug-hash.");
      return;
    }
    setLoading(true);
    try {
      const view = await getHfCollection(slug.trim());
      setTitle(view.title);
      setModels(view.models);
      setSelected(new Set(view.models.filter((m) => /gguf/i.test(m))));
    } catch (err) {
      setError(errText(err, "Could not load the collection."));
    } finally {
      setLoading(false);
    }
  }

  function toggle(repo: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(repo)) next.delete(repo);
      else next.add(repo);
      return next;
    });
  }

  async function seedSelected() {
    setError(null);
    setSubmitting(true);
    const fired: { repo: string; trigger: TriggerResponse }[] = [];
    try {
      for (const repo of models.filter((m) => selected.has(m))) {
        const trigger = await seedHf({
          repo,
          quant: quant || null,
          quant_prefer: true, // batch: quant is a preference, fall back per repo
          family,
        });
        fired.push({ repo, trigger });
      }
      setTriggers(fired);
      toast({
        title: `Seeding ${fired.length} model(s)`,
        description: title ?? slug,
        tone: "success",
      });
      onSeeded();
    } catch (err) {
      setTriggers(fired); // keep panels for what DID fire
      const msg = errText(err, "Collection seed failed.");
      setError(msg);
      toast({ title: "Collection seed failed", description: msg, tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card
      icon={<Boxes className="h-5 w-5" />}
      title="Seed a collection"
      subtitle="A provider-curated Hub collection (e.g. LiquidAI's) — pick the repos, seed them all."
    >
      <div className="space-y-4">
        <Field
          label="Collection"
          required
          hint="URL or owner/slug-hash, e.g. LiquidAI/lfm25-collection-hash"
        >
          <div className="flex items-center gap-2">
            <TextInput
              value={slug}
              onChange={(e) => setSlug(e.target.value)}
              placeholder="https://huggingface.co/collections/…"
            />
            <Button type="button" variant="secondary" loading={loading} onClick={() => void load()}>
              Load
            </Button>
          </div>
        </Field>

        {title && (
          <>
            <p className="text-sm font-medium text-foreground">
              {title} · {models.length} model(s)
            </p>
            <div className="max-h-52 space-y-1 overflow-y-auto rounded-md border border-border p-2">
              {models.map((repo) => (
                <label
                  key={repo}
                  className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-xs text-foreground/90 hover:bg-muted"
                >
                  <input
                    type="checkbox"
                    checked={selected.has(repo)}
                    onChange={() => toggle(repo)}
                    className="h-3.5 w-3.5"
                  />
                  <span className="truncate">{repo}</span>
                  {!/gguf/i.test(repo) && (
                    <Badge tone="warn" className="ml-auto shrink-0">
                      may lack GGUF
                    </Badge>
                  )}
                </label>
              ))}
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Quant preference" hint="Applied to every repo; falls back to its best available.">
                <TextInput value={quant} onChange={(e) => setQuant(e.target.value)} />
              </Field>
              <Field label="Family" hint="One family for the whole batch.">
                <Select value={family} onChange={(e) => setFamily(e.target.value)}>
                  <FamilyOptions families={families} />
                </Select>
              </Field>
            </div>
            <Button
              type="button"
              loading={submitting}
              disabled={selected.size === 0}
              onClick={() => void seedSelected()}
            >
              <Plus className="h-4 w-4" />
              Seed {selected.size} model(s)
            </Button>
          </>
        )}

        {error && (
          <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        {triggers.map(({ repo, trigger }) => (
          <details key={trigger.channel} className="rounded-md border border-border p-3" open>
            <summary className="cursor-pointer text-xs font-medium text-foreground">
              {repo}
            </summary>
            <div className="mt-3">
              <ResultPanel trigger={trigger} noun="seed" />
            </div>
          </details>
        ))}
      </div>
    </Card>
  );
}

function errText(err: unknown, fallback: string): string {
  return toUserMessage(err, { fallback });
}

// ---------------------------------------------------------------------------
// Seed form (populate the store from a local Ollama reference)
// ---------------------------------------------------------------------------

function SeedForm({
  families,
  onSeeded,
}: {
  families: ModelFamily[] | null;
  onSeeded: () => void;
}) {
  const { toast } = useToast();
  const [reference, setReference] = useState("");
  const [name, setName] = useState("");
  const [family, setFamily] = useState("openai_chat");
  const [mmproj, setMmproj] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trigger, setTrigger] = useState<TriggerResponse | null>(null);

  // Vision families (e.g. nuextract3) are served by llama-server with a projector;
  // surface an explicit mmproj input so a GGUF pull without a projector layer is
  // still deployable (the server refuses a needs_mmproj seed with no projector).
  const selectedFamily = useMemo(
    () => (families ?? []).find((f) => f.name === family) ?? null,
    [families, family],
  );
  const needsMmproj = selectedFamily?.needs_mmproj === true;

  async function onSeed(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setTrigger(null);
    if (!reference.trim() || !name.trim()) {
      setError("Reference and store name are both required.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await seedOllama({
        reference: reference.trim(),
        name: name.trim(),
        family,
        ...(mmproj.trim() ? { mmproj: mmproj.trim() } : {}),
      });
      setTrigger(res);
      toast({ title: "Seeding started", description: name.trim(), tone: "success" });
      onSeeded();
    } catch (err) {
      const msg = toUserMessage(err, {
        unavailable: "Adding models isn't available on this server.",
        fallback: "Seeding failed.",
      });
      setError(msg);
      toast({ title: "Seed failed", description: msg, tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  const familyOptions = families && families.length > 0
    ? families.map((f) => f.name)
    : ["openai_chat"];

  // Keep the selected family in sync with what the backend actually offers, so
  // the <select> never shows option 0 while state holds a stale default.
  useEffect(() => {
    if (familyOptions.length > 0 && !familyOptions.includes(family)) {
      setFamily(familyOptions[0]);
    }
  }, [familyOptions, family]);

  return (
    <Card
      icon={<PackagePlus className="h-5 w-5" />}
      title="Add model"
      subtitle="Seed the store from a local Ollama / HF reference."
    >
      <form onSubmit={onSeed} className="space-y-4">
        <Field label="Reference" required hint='e.g. "qwen2.5:1.5b"'>
          <TextInput
            value={reference}
            onChange={(e) => setReference(e.target.value)}
            placeholder="qwen2.5:1.5b"
          />
        </Field>
        <Field label="Store name" required hint="How it's listed in the model store.">
          <TextInput
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="qwen2.5-1.5b"
          />
        </Field>
        <Field label="Family">
          <Select value={family} onChange={(e) => setFamily(e.target.value)}>
            <FamilyOptions families={families} />
          </Select>
        </Field>

        {needsMmproj && (
          <Field
            label="Vision projector (mmproj)"
            hint="Path to an mmproj GGUF, if the pulled model ships none. Reachable inside the serving container."
          >
            <TextInput
              value={mmproj}
              onChange={(e) => setMmproj(e.target.value)}
              placeholder="/models/nuextract3/mmproj.gguf"
            />
          </Field>
        )}

        {error && (
          <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        <Button type="submit" variant="secondary" loading={submitting}>
          <Plus className="h-4 w-4" />
          {submitting ? "Seeding…" : "Seed from Ollama"}
        </Button>
      </form>

      {trigger && (
        <div className="mt-5 border-t border-border pt-5">
          <ResultPanel trigger={trigger} noun="seed" />
        </div>
      )}
    </Card>
  );
}
