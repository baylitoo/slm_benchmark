"use client";

import { useEffect, useRef, useState } from "react";
import { Clock3 } from "lucide-react";
import { getRuns, statusIs, type InngestRun } from "@/lib/api";
import { JsonView } from "./JsonView";
import { Badge } from "./ui";

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

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="warn">
          <Clock3 className="h-3 w-3" /> polling fallback
        </Badge>
        {primary?.status && (
          <Badge tone={completed ? "ok" : statusIs(primary, "Failed") ? "err" : "info"}>
            {primary.status}
          </Badge>
        )}
        {!done && !error && (
          <Badge tone="neutral">polling every {POLL_MS / 1000}s…</Badge>
        )}
      </div>

      {error && (
        <p className="text-xs text-amber-500 dark:text-amber-400">{error} — retrying…</p>
      )}

      <div>
        <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {noun}
        </p>
        {completed && primary?.output !== undefined ? (
          <JsonView value={primary.output} />
        ) : runs && runs.length > 0 ? (
          <JsonView value={runs} maxHeight="16rem" />
        ) : (
          <p className="text-sm text-muted-foreground">Waiting for the run to start…</p>
        )}
      </div>
    </div>
  );
}
