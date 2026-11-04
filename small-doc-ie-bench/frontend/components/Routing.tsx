"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Plus, Trash2, X } from "lucide-react";
import {
  ApiError,
  createRoutingPolicy,
  deleteRoutingPolicy,
  getDeployments,
  listRoutingPolicies,
  type DeploymentRecord,
  type RoutingPolicySummary,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { toUserMessage } from "@/lib/errors";
import { useToast } from "./Toast";
import { Alert, Badge, Button, Card } from "./ui";
import { LiveIndicator } from "./LiveIndicator";

const POLL_MS = 5000;

/**
 * Routing policies, built from the models this deployment actually has.
 *
 * A policy is an ORDERED LADDER: try the first model, and if its result does
 * not clear that stage's gate, fall through to the next. The last stage
 * escalates rather than falling through, because there is nothing left to fall
 * through to. That order is the whole policy — everything else on this page is
 * the gate that decides whether a stage's answer is good enough to stop at.
 *
 * Until this existed the only way to make one was hand-writing the JSON, which
 * meant knowing the rule vocabulary before you could try the feature.
 *
 * CANDIDATES ARE DEPLOYED, NOT HOT. A stage names a model, and resolution
 * happens per request, so a cold or evicted deployment is a perfectly good
 * stage — it loads when the ladder reaches it. Filtering to what is hot right
 * now would hide most of the catalogue and make the ladder look emptier than
 * the machine is. State is shown as a badge so the cost of a cold stage is
 * visible without being forbidden.
 *
 * STAGES NAME MODELS, NOT DEPLOYMENT RECORDS. `replica_group` is the launch
 * alias a model's replicas share, which is what resolution routes on, so a
 * policy keeps working when a record is replaced or scaled. Naming records
 * instead would break a policy on every redeploy.
 */
export function Routing({ active = true }: { active?: boolean }) {
  const deployments = usePolling<DeploymentRecord[]>(getDeployments, POLL_MS, active);
  const policies = usePolling<RoutingPolicySummary[]>(listRoutingPolicies, POLL_MS, active);
  const { toast } = useToast();

  const [ladder, setLadder] = useState<Stage[]>([]);
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);

  const candidates = useMemo(
    () => candidatesFrom(deployments.data ?? []),
    [deployments.data],
  );
  const chosen = new Set(ladder.map((stage) => stage.model));
  const available = candidates.filter((candidate) => !chosen.has(candidate.model));
  const policy = useMemo(() => buildPolicy(ladder), [ladder]);

  function move(index: number, by: number) {
    const target = index + by;
    if (target < 0 || target >= ladder.length) return;
    const next = [...ladder];
    [next[index], next[target]] = [next[target], next[index]];
    setLadder(next);
  }

  async function save() {
    const trimmed = name.trim();
    if (!trimmed || ladder.length === 0) return;
    setSaving(true);
    try {
      await createRoutingPolicy(trimmed, policy as Record<string, unknown>);
      toast({ title: `Saved routing policy "${trimmed}"`, tone: "success" });
      setName("");
      setLadder([]);
      policies.refresh();
    } catch (error) {
      // A 409 is the name, a 422 is the shape. Both are actionable and the
      // server's own detail says which, so it is surfaced rather than replaced.
      toast({
        title: "Could not save the policy",
        description:
          error instanceof ApiError && error.status === 409
            ? `A routing policy called "${trimmed}" already exists.`
            : toUserMessage(error),
        tone: "error",
      });
    } finally {
      setSaving(false);
    }
  }

  async function remove(policyName: string) {
    try {
      await deleteRoutingPolicy(policyName);
      toast({ title: `Deleted "${policyName}"`, tone: "success" });
      policies.refresh();
    } catch (error) {
      toast({ title: "Delete failed", description: toUserMessage(error), tone: "error" });
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end">
        <LiveIndicator
          live={deployments.live}
          refreshing={deployments.refreshing}
          lastUpdated={deployments.lastUpdated}
          onRefresh={() => {
            deployments.refresh();
            policies.refresh();
          }}
        />
      </div>

      {candidates.length === 0 && deployments.data != null && (
        <Alert tone="warn">
          No models are deployed, so there is nothing to route between. Deploy at
          least two from the Deploy tab — a ladder of one is just that model.
        </Alert>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Available models" subtitle="Deployed on this stack. Cold is fine.">
          <div className="space-y-1">
            {available.length === 0 && (
              <p className="py-2 text-sm text-muted">
                {candidates.length === 0
                  ? "Nothing deployed yet."
                  : "Every deployed model is already in the ladder."}
              </p>
            )}
            {available.map((candidate) => (
              <div
                key={candidate.model}
                className="flex items-center justify-between gap-2 rounded border border-border px-3 py-2"
              >
                <div className="min-w-0">
                  <div className="truncate font-mono text-sm">{candidate.model}</div>
                  <div className="text-xs text-muted">
                    {candidate.replicas > 1 ? `${candidate.replicas} replicas · ` : ""}
                    {candidate.runtime ?? "unknown runtime"}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <PhaseBadge phase={candidate.phase} state={candidate.state} />
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => setLadder([...ladder, { model: candidate.model }])}
                    aria-label={`Add ${candidate.model} to the ladder`}
                  >
                    <Plus className="size-3.5" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card
          title="Ladder"
          subtitle="Tried in order. The last stage escalates instead of falling through."
        >
          <div className="space-y-2">
            {ladder.length === 0 && (
              <p className="py-2 text-sm text-muted">
                Add a model to start. The first one handles every request; later
                ones only see what the stage above them did not accept.
              </p>
            )}
            {ladder.map((stage, index) => (
              <div key={stage.model} className="rounded border border-border px-3 py-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="shrink-0 font-mono text-xs text-muted">{index + 1}</span>
                    <span className="truncate font-mono text-sm">{stage.model}</span>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={index === 0}
                      onClick={() => move(index, -1)}
                      aria-label={`Move ${stage.model} earlier`}
                    >
                      <ArrowUp className="size-3.5" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={index === ladder.length - 1}
                      onClick={() => move(index, 1)}
                      aria-label={`Move ${stage.model} later`}
                    >
                      <ArrowDown className="size-3.5" />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setLadder(ladder.filter((s) => s.model !== stage.model))}
                      aria-label={`Remove ${stage.model}`}
                    >
                      <X className="size-3.5" />
                    </Button>
                  </div>
                </div>
                <label className="mt-2 flex items-center gap-2 text-xs text-muted">
                  Accept above confidence
                  <input
                    id={`confidence-${stage.model}`}
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={stage.minConfidence ?? ""}
                    placeholder="any"
                    onChange={(event) => {
                      const raw = event.target.value;
                      const parsed = raw === "" ? undefined : Number(raw);
                      setLadder(
                        ladder.map((s, i) =>
                          i === index ? { ...s, minConfidence: parsed } : s,
                        ),
                      );
                    }}
                    className="w-20 rounded border border-border bg-surface px-2 py-1 font-mono"
                  />
                  {index < ladder.length - 1 ? "else fall through" : "else escalate"}
                </label>
              </div>
            ))}
          </div>

          {ladder.some((stage) => stage.minConfidence != null) && (
            <Alert tone="warn" className="mt-3">
              A confidence gate only fires for models that emit per-field
              confidence. Where none is reported the gate never matches, so that
              stage always falls through — which is a slower ladder, not a wrong
              one.
            </Alert>
          )}

          <div className="mt-4 flex items-end gap-2">
            <label className="flex-1 text-xs text-muted">
              Policy name
              <input
                id="routing-policy-name"
                value={name}
                maxLength={64}
                placeholder="cpu-cascade-1"
                onChange={(event) => setName(event.target.value)}
                className="mt-1 w-full rounded border border-border bg-surface px-2 py-1.5 font-mono text-sm"
              />
            </label>
            <Button
              onClick={() => void save()}
              disabled={saving || ladder.length === 0 || name.trim() === ""}
            >
              {saving ? "Saving…" : "Save policy"}
            </Button>
          </div>
        </Card>
      </div>

      {ladder.length > 0 && (
        <Card
          title="What gets saved"
          subtitle="The same shape a hand-written policy file has."
        >
          <pre className="overflow-x-auto rounded bg-surface-sunken p-3 font-mono text-xs">
            {JSON.stringify(policy, null, 2)}
          </pre>
        </Card>
      )}

      <Card title="Saved policies" subtitle="Selectable from a benchmark run.">
        <div className="space-y-1">
          {(policies.data ?? []).length === 0 && (
            <p className="py-2 text-sm text-muted">None saved yet.</p>
          )}
          {(policies.data ?? []).map((saved) => (
            <div
              key={saved.name}
              className="flex items-center justify-between gap-2 rounded border border-border px-3 py-2"
            >
              <div className="min-w-0">
                <div className="truncate font-mono text-sm">{saved.name}</div>
                <div className="text-xs text-muted">{stageSummary(saved.policy)}</div>
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => void remove(saved.name)}
                aria-label={`Delete ${saved.name}`}
              >
                <Trash2 className="size-3.5" />
              </Button>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

export interface Stage {
  model: string;
  /** Omitted means "accept any valid extraction from this stage". */
  minConfidence?: number;
}

export interface Candidate {
  model: string;
  replicas: number;
  state?: string;
  phase?: string | null;
  runtime?: string;
}

/**
 * One candidate per ROUTABLE model, not per deployment record.
 *
 * Replicas of one model share a `replica_group`, and that alias is what
 * resolution routes on — so three replicas are one choice here, and the record
 * names behind them never reach the policy. A record with no group (an
 * unscaled deployment) is its own group.
 *
 * The liveliest replica wins the badge: a model with one hot and two cold
 * replicas is reachable without a load, and showing it as cold would misstate
 * what the stage costs.
 */
export function candidatesFrom(records: DeploymentRecord[]): Candidate[] {
  const byModel = new Map<string, Candidate>();
  for (const record of records) {
    const model = record.replica_group ?? record.spec?.name;
    if (!model) continue;
    const phase = record.observed?.phase ?? null;
    const existing = byModel.get(model);
    if (!existing) {
      byModel.set(model, {
        model,
        replicas: 1,
        state: record.state,
        phase,
        runtime: record.spec?.launch?.runtime,
      });
      continue;
    }
    existing.replicas += 1;
    if (rank(phase) > rank(existing.phase)) {
      existing.phase = phase;
      existing.state = record.state;
    }
  }
  return [...byModel.values()].sort((a, b) => a.model.localeCompare(b.model));
}

function rank(phase: string | null | undefined): number {
  if (phase === "hot") return 3;
  if (phase === "loading") return 2;
  if (phase == null) return 1;
  return 0;
}

/**
 * The ladder as a RoutingPolicy.
 *
 * Every stage accepts a successful, schema-valid extraction; the optional
 * confidence gate narrows that. Earlier stages fall through, the last one
 * escalates — there is nothing below it to fall to, and silently accepting a
 * failed last stage would turn a routing policy into a single model with extra
 * latency.
 *
 * `max_stages` is the ladder's own length rather than a tunable: a budget
 * smaller than the ladder makes later stages unreachable, which reads as the
 * policy being broken rather than as a budget doing its job.
 */
export function buildPolicy(ladder: Stage[]): Record<string, unknown> {
  return {
    version: "1",
    stages: ladder.map((stage, index) => {
      const last = index === ladder.length - 1;
      const gated = stage.minConfidence != null;
      return {
        name: stage.model,
        rules: [
          {
            when: {
              status: "success",
              validation_valid: true,
              ...(gated ? { min_confidence: stage.minConfidence } : {}),
            },
            decision: "accept",
            reason: gated
              ? `${stage.model} cleared the confidence gate`
              : `${stage.model} returned a valid extraction`,
          },
        ],
        default_decision: last ? "escalate" : "fallback",
        default_reason: last
          ? `${stage.model} was the last stage and did not produce an accepted extraction`
          : `${stage.model} did not clear its gate`,
      };
    }),
    budget: { max_stages: ladder.length },
    exhausted_decision: "escalate",
    exhausted_reason: "no stage produced an accepted extraction",
  };
}

function stageSummary(policy: Record<string, unknown>): string {
  const stages = policy?.stages;
  if (!Array.isArray(stages) || stages.length === 0) return "no stages";
  const names = stages
    .map((stage) => (typeof stage === "object" && stage ? String((stage as { name?: unknown }).name ?? "?") : "?"))
    .join(" → ");
  return names;
}

function PhaseBadge({ phase, state }: { phase?: string | null; state?: string }) {
  if (phase === "hot") return <Badge tone="ok">hot</Badge>;
  if (phase === "loading") return <Badge tone="warn">loading</Badge>;
  if (phase === "evicted") return <Badge>evicted</Badge>;
  if (phase === "failed" || state === "failed") return <Badge tone="err">failed</Badge>;
  return <Badge>{state ?? "cold"}</Badge>;
}
