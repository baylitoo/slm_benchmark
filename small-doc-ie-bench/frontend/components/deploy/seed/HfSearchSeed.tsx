"use client";

import { useMemo, useState } from "react";
import { Rocket, Boxes, ShieldAlert } from "lucide-react";
import {
  formatBytes,
  HF_PARAMETER_RANGES,
  searchHf,
  inspectHf,
  seedHf,
  type HfSearchCard,
  type HfInspect,
  type ModelFamily,
  type TriggerResponse,
} from "@/lib/api";
import { cn } from "@/lib/cn";
import { useToast } from "../../Toast";
import {
  Alert,
  Badge,
  type BadgeTone,
  Button,
  Card,
  Field,
  Select,
  Spinner,
  TextInput,
} from "../../ui";
import { ResultPanel } from "../../ResultPanel";
import { errText, FamilyOptions } from "../shared";
import {
  ExistingModelsDialog,
  existingStoreNames as findExistingNames,
} from "./ExistingModelsDialog";

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

function readinessTone(readiness: HfInspect["readiness"]): BadgeTone {
  return readiness === "ready" ? "ok" : readiness === "blocked" ? "err" : "warn";
}

function readinessLabel(readiness: HfInspect["readiness"]): string {
  return readiness === "ready"
    ? "ready to deploy"
    : readiness === "blocked"
      ? "blocked"
      : "review required";
}

export function HfSearchSeed({
  families,
  existingStoreNames,
  onSeeded,
}: {
  families: ModelFamily[] | null;
  existingStoreNames: string[];
  onSeeded: () => void;
}) {
  const { toast } = useToast();
  const [query, setQuery] = useState("");
  const [ggufOnly, setGgufOnly] = useState(true);
  const [parameterRange, setParameterRange] = useState("");
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
  const [confirmExisting, setConfirmExisting] = useState(false);
  const conflicts = findExistingNames(existingStoreNames, [name]);

  const artifact = useMemo(() => {
    const options = inspect?.artifact_options ?? [];
    return (
      options.find((option) => option.quant === quant) ??
      options.find((option) => option.recommended) ??
      options[0]
    );
  }, [inspect, quant]);
  const selectedFit = artifact?.fits_node ?? inspect?.fits_node;
  const activeBlockers = (inspect?.blockers ?? []).filter((blocker) => {
    if (blocker.code === "insufficient_memory") return selectedFit === false;
    if (blocker.code === "remote_code_approval_required") {
      return family !== "transformers_trust_remote_code";
    }
    return true;
  });
  const effectiveReadiness: HfInspect["readiness"] = activeBlockers.length
    ? "blocked"
    : (inspect?.warnings?.length ?? 0) > 0
      ? "caution"
      : "ready";

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
      setResults(await searchHf(query.trim(), ggufOnly, parameterRange || undefined));
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
      setQuant(v.recommended_quant ?? v.quants?.[0] ?? "");
    } catch (err) {
      setInspect({ repo, verdict: "unsupported", reason: errText(err, "Inspect failed.") });
    } finally {
      setInspecting(false);
    }
  }

  async function startSeed() {
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
      setSeedActive(conflicts.length === 0);
      setSeedingRepo(selected);
      toast({
        title: conflicts.length > 0 ? "Using existing model" : "Download started",
        description: selected,
        tone: "success",
      });
      onSeeded();
    } catch (err) {
      toast({ title: "Seed failed", description: errText(err, "Seed failed."), tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  function onSeed() {
    if (conflicts.length > 0) {
      setConfirmExisting(true);
      return;
    }
    void startSeed();
  }

  const unsupported = inspect?.verdict === "unsupported";
  const hardBlocked = unsupported || activeBlockers.length > 0;

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
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={ggufOnly}
              onChange={(e) => setGgufOnly(e.target.checked)}
              className="h-3.5 w-3.5"
            />
            GGUF only (uncheck to include encoder / transformers safetensors checkpoints)
          </label>
          <Select
            value={parameterRange}
            onChange={(e) => setParameterRange(e.target.value)}
            className="h-8 w-40 text-xs"
            aria-label="Model parameters"
          >
            {HF_PARAMETER_RANGES.map((range) => (
              <option key={range.value} value={range.value}>
                {range.label}
              </option>
            ))}
          </Select>
        </div>
      </form>

      {error && (
        <Alert tone="err" className="mt-3">
          {error}
        </Alert>
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
                  {inspect.readiness && (
                    <Badge tone={readinessTone(effectiveReadiness)}>
                      {readinessLabel(effectiveReadiness)}
                    </Badge>
                  )}
                  <Badge tone={verdictTone(inspect.verdict)}>{verdictLabel(inspect.verdict)}</Badge>
                  {inspect.architecture && (
                    <Badge tone="neutral">arch: {inspect.architecture}</Badge>
                  )}
                  {inspect.runtime && <Badge tone="info">{inspect.runtime}</Badge>}
                  {inspect.has_mmproj && <Badge tone="info">vision · mmproj</Badge>}
                </div>
                {inspect.reason && (
                  <p className="text-xs text-muted-foreground">{inspect.reason}</p>
                )}
                {inspect.runtime_note && !(inspect.warnings?.length ?? 0) && (
                  <Alert tone="warn">Runtime: {inspect.runtime_note}</Alert>
                )}
                {activeBlockers.length > 0 && (
                  <Alert tone="err">
                    {activeBlockers.map((blocker) => (
                      <span key={blocker.code} className="block">
                        {blocker.message}
                      </span>
                    ))}
                  </Alert>
                )}
                {(inspect.warnings?.length ?? 0) > 0 && (
                  <Alert tone="warn">
                    {inspect.warnings!.map((warning) => (
                      <span key={warning.code} className="block">
                        {warning.message}
                      </span>
                    ))}
                  </Alert>
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
                    {artifact && (
                      <div className="grid grid-cols-2 gap-2 rounded-md border border-border bg-card p-3 text-xs sm:grid-cols-4">
                        <PreflightMetric
                          label="Download"
                          value={formatBytes(artifact.download_size_bytes)}
                        />
                        <PreflightMetric
                          label={`RAM · ${inspect.context_length?.toLocaleString() ?? "?"} ctx`}
                          value={formatBytes(artifact.estimated_ram_bytes)}
                        />
                        <PreflightMetric
                          label="Node budget"
                          value={formatBytes(artifact.node_available_bytes)}
                        />
                        <PreflightMetric
                          label="Fit"
                          value={
                            selectedFit === true
                              ? "Fits"
                              : selectedFit === false
                                ? "Does not fit"
                                : "Unknown"
                          }
                          tone={selectedFit === true ? "ok" : selectedFit === false ? "err" : undefined}
                        />
                      </div>
                    )}
                    <div className="grid gap-3 sm:grid-cols-2">
                      <Field label="Store name">
                        <TextInput value={name} onChange={(e) => setName(e.target.value)} />
                      </Field>
                      <Field label="Family" hint="Suggested from the architecture — override if needed.">
                        <Select
                          value={family}
                          onChange={(e) => setFamily(e.target.value)}
                          aria-label="Model family"
                        >
                          <FamilyOptions families={families} />
                        </Select>
                      </Field>
                    </div>
                    {(inspect.artifact_options?.filter((option) => option.quant).length ?? 0) > 0 && (
                      <Field
                        label="Quantization"
                        hint="Each choice updates the download, RAM, and fit estimate above."
                      >
                        <div className="flex flex-wrap gap-1.5">
                          {inspect.artifact_options!
                            .filter((option) => option.quant)
                            .map((option) => (
                            <button
                              key={option.filename ?? option.label}
                              type="button"
                              onClick={() => setQuant(option.quant ?? "")}
                              className={cn(
                                "rounded-md border px-2 py-1 text-xs transition",
                                quant === option.quant
                                  ? "border-accent bg-accent/10 text-accent"
                                  : "border-border hover:bg-muted",
                              )}
                            >
                              {option.label}
                              {option.fits_node === true
                                ? " · fits"
                                : option.fits_node === false
                                  ? " · too large"
                                  : ""}
                            </button>
                          ))}
                        </div>
                      </Field>
                    )}
                    {(artifact?.required_files.length ?? 0) > 0 && (
                      <details className="rounded-md border border-border px-3 py-2 text-xs">
                        <summary className="cursor-pointer font-medium text-foreground">
                          Required files ({artifact!.required_files.length})
                        </summary>
                        <ul className="mt-2 space-y-1 text-muted-foreground">
                          {artifact!.required_files.map((file) => (
                            <li key={`${file.role}:${file.filename}`} className="flex justify-between gap-3">
                              <span className="min-w-0 truncate" title={file.filename}>
                                {file.role}: {file.filename}
                              </span>
                              <span className="shrink-0 font-mono">
                                {formatBytes(file.size_bytes)}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                    {(inspect.recommendations?.length ?? 0) > 0 && (
                      <Alert tone="info">
                        {inspect.recommendations!.map((recommendation) => (
                          <span key={recommendation} className="block">
                            {recommendation}
                          </span>
                        ))}
                      </Alert>
                    )}
                    <Button
                      type="button"
                      loading={submitting}
                      disabled={!family || hardBlocked}
                      onClick={onSeed}
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
      <ExistingModelsDialog
        names={conflicts}
        open={confirmExisting}
        onClose={() => setConfirmExisting(false)}
        onContinue={() => {
          setConfirmExisting(false);
          void startSeed();
        }}
      />
    </Card>
  );
}

function PreflightMetric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "ok" | "err";
}) {
  return (
    <div>
      <p className="text-muted-foreground">{label}</p>
      <p
        className={cn(
          "mt-0.5 font-medium text-foreground",
          tone === "ok" && "text-emerald-600 dark:text-emerald-400",
          tone === "err" && "text-rose-600 dark:text-rose-400",
        )}
      >
        {value}
      </p>
    </div>
  );
}
