"use client";

import { useEffect, useMemo, useState } from "react";
import { PackagePlus, Plus, AlertCircle } from "lucide-react";
import { seedOllama, type ModelFamily, type TriggerResponse } from "@/lib/api";
import { toUserMessage } from "@/lib/errors";
import { useToast } from "../../Toast";
import { Button, Card, Field, Select, TextInput } from "../../ui";
import { ResultPanel } from "../../ResultPanel";
import { FamilyOptions } from "../shared";

// ---------------------------------------------------------------------------
// Seed form (populate the store from a local Ollama reference)
// ---------------------------------------------------------------------------

export function SeedForm({
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
