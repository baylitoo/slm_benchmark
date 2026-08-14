"use client";

import { useMemo, useState } from "react";
import { PackagePlus, Plus } from "lucide-react";
import {
  getHfRepo,
  seedHf,
  formatBytes,
  type HfRepoView,
  type ModelFamily,
  type TriggerResponse,
} from "@/lib/api";
import { cn } from "@/lib/cn";
import { useToast } from "../../Toast";
import { Alert, Button, Card, Field, Select, TextInput } from "../../ui";
import { ResultPanel } from "../../ResultPanel";
import { errText, FamilyOptions } from "../shared";

function isEmbeddingFamily(families: ModelFamily[] | null, name: string): boolean {
  return (families ?? []).some((f) => f.name === name && f.embedding);
}

export function HfSeedForm({
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

        {error && <Alert tone="err">{error}</Alert>}

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
