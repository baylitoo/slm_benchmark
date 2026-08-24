"use client";

import { useEffect, useRef, useState } from "react";
import { Clock3, Loader2 } from "lucide-react";
import {
  getRuns,
  getSeedProgress,
  statusIs,
  type InngestRun,
  type SeedProgress,
} from "@/lib/api";
import { JsonView } from "./JsonView";
import { RoutingSummary } from "./RoutingSummary";
import { ProgressView } from "./ProgressBar";
import { T, useI18n } from "@/lib/i18n";
import { Alert, Badge } from "./ui";

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
  channel,
  noun = "result",
  onSettled,
}: {
  eventId: string;
  /** The run channel — enables the pollable download-progress bar (seed jobs). */
  channel?: string;
  noun?: string;
  onSettled?: () => void;
}) {
  const { t } = useI18n();
  const [runs, setRuns] = useState<InngestRun[] | null>(null);
  const [progress, setProgress] = useState<SeedProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRuns(null);
    setProgress(null);
    setError(null);
    setDone(false);

    async function tick() {
      try {
        const list = await getRuns(eventId);
        if (cancelled) return;
        setRuns(list);
        // Poll the pollable progress sidecar too — this is what gives the
        // polling fallback the same percentage bar realtime shows. Best-effort:
        // a missing endpoint / null progress just leaves the bar hidden.
        if (channel) {
          try {
            const p = await getSeedProgress(channel);
            if (!cancelled) setProgress(p);
          } catch {
            /* no progress endpoint / not a seed — ignore */
          }
        }
        const settled =
          list.length > 0 && list.every((r) => statusIs(r, "Completed", "Failed", "Cancelled"));
        if (settled) {
          setDone(true);
          onSettled?.();
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
  }, [eventId, channel]);

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
          <div className="space-y-2">
            <RoutingSummary result={primary.output} />
            <JsonView value={primary.output} />
          </div>
        ) : failed ? (
          <Alert tone="err">The {noun} failed{failMsg ? `: ${failMsg}` : "."}</Alert>
        ) : progress && typeof progress.percent === "number" ? (
          // Realtime is down, but the pollable sidecar gives us the live
          // percentage — render the same bar the realtime stream would.
          <div className="space-y-2">
            <ProgressView value={progress} />
            <p className="text-xs text-muted-foreground">
              <T>Downloading — continues in the background if you navigate away.</T>
            </p>
          </div>
        ) : (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            {primary?.status
              ? t("{noun} is {status} — this can take a while for a large download.", {
                  noun: t(noun),
                  status: String(primary.status).toLowerCase(),
                })
              : t("Waiting for the run to start…")}
          </p>
        )}
      </div>
    </div>
  );
}
