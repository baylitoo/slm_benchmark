"use client";

import { useEffect, useRef, useState } from "react";
import { Clock3, Loader2 } from "lucide-react";
import { getRuns, statusIs, type InngestRun } from "@/lib/api";
import { JsonView } from "./JsonView";
import { Badge } from "./ui";

/** Best-effort human error out of an Inngest run's output/error shape. */
function runErrorMessage(run: InngestRun | undefined): string | null {
  if (!run) return null;
  const out = run.output as { message?: string; error?: { message?: string } } | undefined;
  if (out?.error?.message) return out.error.message;
  if (out?.message) return out.message;
  if (typeof run.output === "string") return run.output;
  return null;
}

const POLL_MS = 1500;

export function PollingResult({
  eventId,
  noun = "result",
}: {
  eventId: string;
  noun?: string;
}) {
  const [runs, setRuns] = useState<InngestRun[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRuns(null);
    setError(null);
    setDone(false);

    async function tick() {
      try {
        const list = await getRuns(eventId);
        if (cancelled) return;
        setRuns(list);
        const settled =
          list.length > 0 && list.every((r) => statusIs(r, "Completed", "Failed", "Cancelled"));
        if (settled) {
          setDone(true);
          return;
        }
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Polling failed");
        // Keep retrying transient failures.
      }
      if (!cancelled) timer.current = setTimeout(tick, POLL_MS);
    }

    tick();
    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [eventId]);

  const primary = runs?.[0];
  const completed = primary && statusIs(primary, "Completed");
  const failed = primary && statusIs(primary, "Failed", "Cancelled");
  const failMsg = failed ? runErrorMessage(primary) : null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="warn">
          <Clock3 className="h-3 w-3" /> polling fallback
        </Badge>
        {primary?.status && (
          <Badge tone={completed ? "ok" : failed ? "err" : "info"}>{primary.status}</Badge>
        )}
        {!done && !error && (
          <Badge tone="neutral">checking every {POLL_MS / 1000}s…</Badge>
        )}
      </div>

      {error && (
        <p className="text-xs text-amber-500 dark:text-amber-400">{error} — retrying…</p>
      )}

      <div>
        <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {noun}
        </p>
        {/* Human status, not a raw Inngest run dump. Realtime carries the live
            download progress; when it drops, polling can only report the
            coarse run status — so show THAT clearly instead of JSON. */}
        {completed && primary?.output !== undefined ? (
          <JsonView value={primary.output} />
        ) : failed ? (
          <p className="rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
            The {noun} failed{failMsg ? `: ${failMsg}` : "."}
          </p>
        ) : (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {primary?.status
              ? `${noun} is ${String(primary.status).toLowerCase()} — this can take a while for a large download.`
              : "Waiting for the run to start…"}
          </p>
        )}
      </div>
    </div>
  );
}
