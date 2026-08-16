"use client";

import { AlertTriangle } from "lucide-react";
import { Button, Dialog } from "../../ui";

export function defaultStoreName(repo: string): string {
  const tail = (repo.split("/").pop() ?? repo)
    .toLowerCase()
    .replace(/[-_.]?gguf$/i, "");
  return tail.replace(/[^a-z0-9._-]+/g, "-").replace(/^[.-]+|[.-]+$/g, "") || "model";
}

export function existingStoreNames(names: string[], candidates: string[]): string[] {
  const stored = new Set(names);
  return [
    ...new Set(candidates.map((name) => name.trim()).filter((name) => stored.has(name))),
  ];
}

export function ExistingModelsDialog({
  names,
  open,
  onClose,
  onContinue,
}: {
  names: string[];
  open: boolean;
  onClose: () => void;
  onContinue: () => void;
}) {
  const one = names.length === 1;
  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={
        one ? "This store name already exists" : `${names.length} store names already exist`
      }
      footer={
        <>
          <Button type="button" variant="secondary" onClick={onClose}>
            Go back
          </Button>
          <Button type="button" onClick={onContinue}>
            Use existing {one ? "model" : "models"}
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        <div className="flex items-start gap-3 rounded-lg border border-amber-500/25 bg-amber-500/10 p-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-foreground">
              No new files will be downloaded or replaced.
            </p>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              Continuing will keep the model already stored under{" "}
              {one ? "this name" : "these names"}. Go back if you meant to download a
              different model or quantization.
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {names.map((name) => (
            <code
              key={name}
              className="rounded-md border border-border bg-muted px-2 py-1 text-xs text-foreground"
            >
              {name}
            </code>
          ))}
        </div>
      </div>
    </Dialog>
  );
}
