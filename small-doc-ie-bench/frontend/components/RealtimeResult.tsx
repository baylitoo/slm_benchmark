"use client";

import { useMemo } from "react";
import {
  useInngestSubscription,
  InngestSubscriptionState,
} from "@inngest/realtime/hooks";
import type { Realtime } from "@inngest/realtime";
import { Radio } from "lucide-react";
import { getRealtimeToken, type RealtimeToken } from "@/lib/api";
import { JsonView } from "./JsonView";
import { Badge, type BadgeTone } from "./ui";

// The token is minted by the Python backend, so its TS type can't be inferred
// here. We cast through `unknown` to the hook's expected token type.
type AnyToken = Realtime.Subscribe.Token;

function stateTone(state: InngestSubscriptionState): BadgeTone {
  switch (state) {
    case InngestSubscriptionState.Active:
      return "ok";
    case InngestSubscriptionState.Error:
    case InngestSubscriptionState.Closed:
      return "err";
    default:
      return "info";
  }
}

export function RealtimeResult({
  channel,
  topics,
  initialToken,
  noun = "result",
}: {
  channel: string;
  topics: string[];
  initialToken: RealtimeToken;
  noun?: string;
}) {
  const refreshToken = useMemo(
    () => async () =>
      (await getRealtimeToken(channel, topics)) as unknown as AnyToken,
    [channel, topics],
  );

  const { data, error, state } = useInngestSubscription({
    token: initialToken as unknown as AnyToken,
    refreshToken,
    enabled: true,
    key: channel,
  });

  // Reduce the message stream into the latest value seen per topic.
  const byTopic = useMemo(() => {
    const acc: Record<string, unknown> = {};
    for (const msg of data ?? []) {
      const m = msg as { kind?: string; topic?: string; data?: unknown };
      if (m.kind !== "data" || !m.topic) continue;
      acc[m.topic] = m.data;
    }
    return acc;
  }, [data]);

  const result = byTopic["result"];
  const status = byTopic["status"];
  const progress = byTopic["progress"];
  const errTopic = byTopic["error"];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="info">
          <Radio className="h-3 w-3" /> realtime
        </Badge>
        <Badge tone={stateTone(state)}>{state}</Badge>
        {result !== undefined && <Badge tone="ok">{noun} received</Badge>}
        {errTopic !== undefined && <Badge tone="err">error</Badge>}
      </div>

      {error && (
        <p className="text-xs text-rose-500 dark:text-rose-400">
          Subscription error: {error.message}
        </p>
      )}

      {status !== undefined && (
        <Section label="Status">
          <JsonView value={status} maxHeight="8rem" />
        </Section>
      )}
      {progress !== undefined && (
        <Section label="Progress">
          <JsonView value={progress} maxHeight="8rem" />
        </Section>
      )}
      {errTopic !== undefined && (
        <Section label="Error">
          <JsonView value={errTopic} maxHeight="8rem" />
        </Section>
      )}

      <Section label={noun}>
        {result !== undefined ? (
          <JsonView value={result} />
        ) : (
          <p className="text-sm text-muted-foreground">Waiting for the {noun}…</p>
        )}
      </Section>
    </div>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      {children}
    </div>
  );
}
