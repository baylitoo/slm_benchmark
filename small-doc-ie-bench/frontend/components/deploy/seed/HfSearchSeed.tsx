"use client";

import { useState } from "react";
import { Rocket, Boxes, AlertCircle, ShieldAlert } from "lucide-react";
import {
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

export function HfSearchSeed({
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
