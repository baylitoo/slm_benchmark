"use client";

import { useState } from "react";
import { type ModelFamily } from "@/lib/api";
import { cn } from "@/lib/cn";
import { HfSearchSeed } from "./HfSearchSeed";
import { HfSeedForm } from "./HfSeedForm";
import { HfCollectionSeed } from "./HfCollectionSeed";
import { EncoderSeedForm } from "./EncoderSeedForm";
import { SeedForm } from "./OllamaSeedForm";

// ---------------------------------------------------------------------------
// Add model — Hugging Face direct (preferred), a whole HF collection, or the
// legacy local-Ollama seed. One slide-over, three modes.
// ---------------------------------------------------------------------------

type SeedMode = "search" | "hf" | "collection" | "encoder" | "ollama";

export function AddModelForm({
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
