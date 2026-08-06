"use client";

import { useEffect, useMemo, useState } from "react";
import { Cpu, Plus, AlertCircle } from "lucide-react";
import { seedHf, type ModelFamily, type TriggerResponse } from "@/lib/api";
import { useToast } from "../../Toast";
import { Button, Card, Field, Select, TextInput } from "../../ui";
import { ResultPanel } from "../../ResultPanel";
import { errText } from "../shared";

// Encoders (analyzer families — safetensors, not GGUF) are SEEDED into the
// store like any other model: the snapshot downloads once (live progress),
// then deploys via the normal flow with zero network at boot. The family list
// is data-driven (families where analyzer === true), so a new analyzer family
// added on the backend shows up here with no UI change.
export function EncoderSeedForm({
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
