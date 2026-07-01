"use client";

import { useEffect, useState } from "react";
import { getRealtimeToken, type RealtimeToken, type TriggerResponse } from "@/lib/api";
import { RealtimeResult } from "./RealtimeResult";
import { PollingResult } from "./PollingResult";
import { Spinner } from "./ui";

type Mode =
  | { kind: "connecting" }
  | { kind: "realtime"; token: RealtimeToken }
  | { kind: "polling" };

/**
 * Orchestrates live progress for any triggered job (extract / deploy /
 * benchmark — they share the `TriggerResponse` shape). Tries realtime first
 * (mint a token), and falls back to polling /runs if the token route is
 * unavailable (501) or errors. Each branch renders a distinct component so the
 * realtime hook is only ever mounted when we actually have a token.
 */
export function ResultPanel({
  trigger,
  noun = "result",
}: {
  trigger: TriggerResponse;
  noun?: string;
}) {
  const [mode, setMode] = useState<Mode>({ kind: "connecting" });

  useEffect(() => {
    let cancelled = false;
    setMode({ kind: "connecting" });

    getRealtimeToken(trigger.channel, trigger.topics)
      .then((token) => {
        if (!cancelled) setMode({ kind: "realtime", token });
      })
      .catch(() => {
        // 501 / 404 / network — degrade to polling.
        if (!cancelled) setMode({ kind: "polling" });
      });

    return () => {
      cancelled = true;
    };
  }, [trigger.channel, trigger.topics]);

  if (mode.kind === "connecting") {
    return (
      <p className="flex items-center gap-2 text-sm text-muted-foreground">
        <Spinner /> Connecting to live updates…
      </p>
    );
  }

  if (mode.kind === "realtime") {
    return (
      <RealtimeResult
        channel={trigger.channel}
        topics={trigger.topics}
        initialToken={mode.token}
        eventId={trigger.event_ids[0]}
        noun={noun}
      />
    );
  }

  return <PollingResult eventId={trigger.event_ids[0]} noun={noun} />;
}
