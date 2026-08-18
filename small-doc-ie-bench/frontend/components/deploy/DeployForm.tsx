"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Rocket,
  Eye,
  Cpu,
  Boxes,
  ChevronDown,
  ChevronRight,
  AlertCircle,
} from "lucide-react";
import {
  getPorts,
  deployModel,
  formatBytes,
  type StoreEntry,
  type PortsView as PortsViewData,
  type TriggerResponse,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { cn } from "@/lib/cn";
import { T } from "@/lib/i18n";
import { toUserMessage } from "@/lib/errors";
import { useToast } from "../Toast";
import {
  Alert,
  Badge,
  Button,
  Card,
  EmptyState,
  Field,
  Skeleton,
  TextInput,
  ComingSoon,
} from "../ui";
import { LiveIndicator } from "../LiveIndicator";
import { ResultPanel } from "../ResultPanel";
import { POLL_MS } from "./shared";
import { PortsAdmin } from "./PortsAdmin";

// ---------------------------------------------------------------------------
// Deploy form (model picker + scoped runtime + advanced + progress)
// ---------------------------------------------------------------------------

export function DeployForm({
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
            <p className="mb-1.5 text-xs font-medium text-foreground"><T>Runtime</T></p>
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
            <T>Advanced options</T>
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

        {error && <Alert tone="err">{error}</Alert>}

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
