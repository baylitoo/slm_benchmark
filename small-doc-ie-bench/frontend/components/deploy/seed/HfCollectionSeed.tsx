"use client";

import { useState } from "react";
import { Boxes, Plus } from "lucide-react";
import {
  getHfCollection,
  seedHf,
  type ModelFamily,
  type TriggerResponse,
} from "@/lib/api";
import { useToast } from "../../Toast";
import { Alert, Badge, Button, Card, Field, Select, TextInput } from "../../ui";
import { ResultPanel } from "../../ResultPanel";
import { errText, FamilyOptions } from "../shared";

export function HfCollectionSeed({
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

        {error && <Alert tone="err">{error}</Alert>}

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
