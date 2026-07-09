"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Rocket,
  Server,
  Eye,
  Cpu,
  Boxes,
  Plus,
  Network,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  PackagePlus,
  X,
} from "lucide-react";
import {
  getStore,
  getFamilies,
  getDeployments,
  getPorts,
  deployModel,
  seedOllama,
  formatBytes,
  ApiError,
  ApiUnavailable,
  type StoreEntry,
  type ModelFamily,
  type PortsView as PortsViewData,
  type DeploymentRecord,
  type TriggerResponse,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
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
  TextInput,
  ComingSoon,
} from "./ui";
import { DataTable } from "./DataTable";
import { LiveIndicator } from "./LiveIndicator";
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
  const families = useAsync(getFamilies, []); // static-ish; one-shot fetch

  const [slideOver, setSlideOver] = useState<SlideOver>(null);

  const heading =
    view === "models"
      ? { title: "Models", subtitle: "GET /v1/serving/store — the GGUF catalog you can deploy." }
      : view === "ports"
        ? { title: "Ports", subtitle: "Live port allocation across running deployments." }
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
      ) : (
        <DeploymentsView deployments={deployments} />
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
        <SeedForm families={families.data} onSeeded={() => store.refresh()} />
      </SlideOverPanel>
    </div>
  );
}

/**
 * Right-hand slide-over. Persistently mounted; slides off-screen when closed so
 * its children (a form + any in-flight ResultPanel) keep their state.
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
  return (
    <aside
      aria-hidden={!open}
      className={cn(
        "fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col bg-background shadow-elevated transition-transform duration-200",
        open ? "translate-x-0" : "translate-x-full",
      )}
    >
      <div className="flex items-center justify-end border-b border-border px-3 py-2">
        <button
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
// Deployments view — explicit-column table over DeploymentRecord[].
// ---------------------------------------------------------------------------

function DeploymentsView({
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
    return all.filter((r) => {
      const hay = [r.spec?.name, r.spec?.launch?.model, r.spec?.launch?.runtime, r.state]
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
          placeholder="Filter by name, model, runtime…"
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
      />
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
  // Runtime picker is scoped to the chosen model's faithful backends.
  const backends = selectedEntry?.available_backends ?? [];

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
      const msg =
        err instanceof ApiUnavailable
          ? "The deploy endpoint isn't available yet — this UI is ready for when it ships."
          : err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : "Deploy failed.";
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
              {backends.length > 0
                ? "Backends are scoped to this model (from its store entry's available_backends)."
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
        description="Seed one from a local Ollama model using the form on the right — it'll show up here automatically."
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
      const msg =
        err instanceof ApiUnavailable
          ? "The seed endpoint isn't available yet — this UI is ready for when it ships."
          : err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : "Seeding failed.";
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
            {familyOptions.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
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
