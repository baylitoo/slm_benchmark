"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/cn";

/** Pretty-prints any JSON-serializable value with a copy button. */
export function JsonView({
  value,
  maxHeight = "24rem",
}: {
  value: unknown;
  maxHeight?: string;
}) {
  const [copied, setCopied] = useState(false);
  const text = safeStringify(value);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard may be unavailable (insecure context) */
    }
  }

  return (
    <div className="group relative">
      <button
        onClick={copy}
        type="button"
        className={cn(
          "absolute right-2 top-2 z-10 inline-flex items-center gap-1 rounded-md border border-border bg-card px-2 py-1 text-xs text-muted-foreground transition hover:text-foreground",
          "opacity-0 focus-visible:opacity-100 group-hover:opacity-100",
        )}
      >
        {copied ? (
          <>
            <Check className="h-3.5 w-3.5 text-emerald-500" /> Copied
          </>
        ) : (
          <>
            <Copy className="h-3.5 w-3.5" /> Copy
          </>
        )}
      </button>
      <pre
        className="scroll-thin overflow-auto rounded-xl border border-border bg-muted/40 p-4 font-mono text-xs leading-relaxed text-foreground"
        style={{ maxHeight }}
      >
        <code>{text}</code>
      </pre>
    </div>
  );
}

function safeStringify(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
