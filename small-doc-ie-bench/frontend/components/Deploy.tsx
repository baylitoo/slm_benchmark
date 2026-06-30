"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Rocket,
  Server,
  Eye,
  Cpu,
  Boxes,
  Plus,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  PackagePlus,
} from "lucide-react";
import {
  getStore,
  getFamilies,
  getDeployments,
  deployModel,
  seedOllama,
  formatBytes,
  ApiError,
  ApiUnavailable,
  type StoreEntry,
  type TriggerResponse,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
import { useToast } from "./Toast";
import {
  Badge,
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

const POLL_MS = 4000;

export function Deploy({ active = true }: { active?: boolean }) {
  // Auto-refreshing lists — paused when the tab is hidden OR Deploy isn't the
  // active section (every section stays mounted in the shell).
  const store = usePolling<StoreEntry[]>(getStore, POLL_MS, active);
  const deployments = usePolling(getDeployments, POLL_MS, active);
  const families = useAsync(getFamilies, []); // static-ish; one-shot fetch

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        <DeployForm store={store} onDeployed={() => deployments.refresh()} />
        <SeedForm families={families.data} onSeeded={() => store.refresh()} />
      </div>

      <Card
        icon={<Server className="h-5 w-5" />}
        title="Deployments"
        subtitle="GET /v1/serving/deployments"
        actions={
          <LiveIndicator
            live={deployments.live}
            refreshing={deployments.refreshing}
            lastUpdated={deployments.lastUpdated}
            onRefresh={deployments.refresh}
          />
        }
      >
        <DataTable
          rows={deployments.data}
          loading={deployments.loading}
          error={deployments.error}
          emptyLabel="No active deployments"
          emptyDescription="Deploy a model above — it'll appear here on the next refresh."
        />
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Deploy form (model picker + scoped runtime + advanced + progress)
// ---------------------------------------------------------------------------

function DeployForm({
  store,
  onDeployed,
}: {
  store: ReturnType<typeof usePolling<StoreEntry[]>>;
  onDeployed: () => void;
}) {
  const { toast } = useToast();
  const [selected, setSelected] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<string>(""); // "" = auto / store-entry deploy
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [name, setName] = useState("");
  const [port, setPort] = useState("8088");
  const [contextLength, setContextLength] = useState("8192");

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trigger, setTrigger] = useState<TriggerResponse | null>(null);

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
        ...(port.trim() ? { port: Number(port) } : {}),
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
            <div className="mt-3 grid animate-fade-in gap-4 sm:grid-cols-3">
              <Field label="Deployment name" hint="Optional alias.">
                <TextInput
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="(model name)"
                />
              </Field>
              <Field label="Port">
                <TextInput
                  type="number"
                  value={port}
                  onChange={(e) => setPort(e.target.value)}
                  placeholder="8088"
                />
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
          )}
        </div>

        {error && (
          <p className="flex items-start gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
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
              "flex w-full items-start justify-between gap-3 rounded-xl border p-3 text-left transition",
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
        "inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition",
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
  families: { name: string }[] | null;
  onSeeded: () => void;
}) {
  const { toast } = useToast();
  const [reference, setReference] = useState("");
  const [name, setName] = useState("");
  const [family, setFamily] = useState("openai_chat");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trigger, setTrigger] = useState<TriggerResponse | null>(null);

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

        {error && (
          <p className="flex items-start gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
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
