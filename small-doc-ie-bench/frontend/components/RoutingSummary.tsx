"use client";

import { ArrowRight, Route } from "lucide-react";
import { Badge } from "./ui";
import { T } from "@/lib/i18n";

/**
 * A live extraction's routing audit, made legible: when the result was
 * produced by a routing policy (``result.routing`` is set), show which
 * stage answered and the escalation chain that got there -- instead of
 * leaving the caller to dig it out of the raw JSON. Renders nothing for a
 * single-model result (no ``routing``), so both result panels can include
 * it unconditionally.
 */

interface StageAudit {
  stage: string;
  decision: string;
  reason?: string;
  avg_confidence?: number | null;
  latency_ms?: number;
  total_tokens?: number;
  status?: string;
}

interface RoutingAudit {
  policy?: string;
  selected_stage?: string | null;
  terminal_decision?: string;
  attempts?: number;
  fallback_count?: number;
  latency_ms?: number;
  total_tokens?: number;
  budget_exhausted?: boolean;
  stages?: StageAudit[];
}

function pickRouting(result: unknown): RoutingAudit | null {
  if (!result || typeof result !== "object") return null;
  const routing = (result as { routing?: unknown }).routing;
  if (!routing || typeof routing !== "object") return null;
  return routing as RoutingAudit;
}

function decisionTone(decision: string): "ok" | "warn" | "err" | "neutral" {
  if (decision === "accept") return "ok";
  if (decision === "fail") return "err";
  if (decision === "escalate" || decision === "fallback" || decision === "retry") return "warn";
  return "neutral";
}

export function RoutingSummary({ result }: { result: unknown }) {
  const routing = pickRouting(result);
  if (!routing) return null;
  const stages = routing.stages ?? [];
  return (
    <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1 font-medium text-foreground">
          <Route className="h-3.5 w-3.5" />
          <T>Routing policy</T>
          {routing.policy && <span className="font-mono">{routing.policy}</span>}
        </span>
        {routing.selected_stage && (
          <span className="text-muted-foreground">
            <T>answered by</T> <span className="font-mono text-foreground">{routing.selected_stage}</span>
          </span>
        )}
        {typeof routing.attempts === "number" && (
          <Badge tone={routing.attempts > 1 ? "warn" : "neutral"}>
            {routing.attempts} {routing.attempts === 1 ? "attempt" : "attempts"}
          </Badge>
        )}
        {routing.budget_exhausted && <Badge tone="err"><T>budget exhausted</T></Badge>}
        {typeof routing.total_tokens === "number" && (
          <span className="text-muted-foreground tabular-nums">{routing.total_tokens} tok</span>
        )}
      </div>
      {stages.length > 0 && (
        <div className="mt-1.5 flex flex-wrap items-center gap-1">
          {stages.map((s, i) => (
            <span key={`${s.stage}-${i}`} className="inline-flex items-center gap-1">
              {i > 0 && <ArrowRight className="h-3 w-3 text-muted-foreground" />}
              <span
                className="inline-flex items-center gap-1 rounded border border-border bg-card px-1.5 py-0.5"
                title={[
                  s.reason,
                  s.avg_confidence != null ? `confidence ${s.avg_confidence.toFixed(2)}` : null,
                  s.latency_ms != null ? `${s.latency_ms} ms` : null,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              >
                <span className="font-mono">{s.stage}</span>
                <Badge tone={decisionTone(s.decision)}>{s.decision}</Badge>
                {s.avg_confidence != null && (
                  <span className="tabular-nums text-muted-foreground">
                    {s.avg_confidence.toFixed(2)}
                  </span>
                )}
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
