// The fleet view: every serving node, its capacity, and what it holds.
//
// `/v1/serving/resources` answers "how much room is there", which is a question
// about ONE node and is what the Sizing tab consumes. This answers "what is the
// fleet" — the per-node rows live in `serving_node` and nothing surfaced them
// before.

import { request } from "./core";
import type { NodeSnapshot } from "./sizing";

/** One serving node, as `GET /v1/serving/nodes` reports it. */
export interface FleetNode extends NodeSnapshot {
  node_id: string;
  /** Whether this node counts as alive — the SAME staleness rule the router
   * uses to demote a node's placements, so the Studio and the routing layer
   * cannot disagree about which nodes are up. */
  reporting: boolean;
  /** True when the node has a row but has never published a measurement: a
   * machine that desired state was staged for and that has not started. Its
   * capacity numbers are zeroes and must not be priced against. Distinct from
   * `reporting: false`, which also covers a node that reported and stopped. */
  never_reported: boolean;
  /** Seconds since its last publish; null when never published or unparseable. */
  age_s?: number | null;
  /** How many placements name this node as their owner, running or not. A
   * stopped deployment keeps its row and its port reservation, so this is what
   * the node is responsible for rather than what it is doing. */
  deployments: number;
  /** Of those, how many have a live endpoint. The difference is the stopped
   * ones. Absent on an older api. */
  deployments_live?: number;
  /** The store models whose blobs this node actually holds. `null` means the
   * node has not reported its store yet — which is NOT the same as holding
   * nothing, and the UI must not conflate them. */
  store_models?: string[] | null;
  /** Desired-state generation last written for this node; null when none. */
  desired_generation?: number | null;
  /** When an operator declared this node drained; null when it is admitting. */
  drained_at?: string | null;
  /** Where the node is between admitting traffic and being free to stop.
   *
   * `draining` and `drained` are separate because an operator waiting on a
   * machine needs to know which one they are in: draining means requests are
   * still finishing on it, drained means it has released everything. Derived
   * server-side from the LIVE placement count rather than from the node's
   * grace period, which is computed on the node itself. */
  drain_state?: "admitting" | "draining" | "drained" | string;
  /** Operator override of DocIE's RAM share on this node; null = none set. */
  mem_budget_bytes?: number | null;
  /** Version to send back as `expected_version` when changing the override. */
  mem_budget_version?: number | null;
  mem_budget_updated_at?: string | null;
  mem_budget_actor?: string | null;
  /** Kernel memory limit (cgroup memory.max); null when nothing enforces one. */
  ceiling_bytes?: number | null;
  /** What DocIE's container is using now. */
  working_set_bytes?: number | null;
  /** The budget the node is actually sizing against. */
  effective_budget_bytes?: number | null;
  /** "catalog" | "catalog_cache" | "setting" | "none". */
  budget_source?: string | null;
  /** True when the node could not read the catalog and kept its last budget. */
  budget_stale?: boolean | null;
}

/** Response of the drain/resume endpoints. */
export interface DrainResponse {
  node_id: string;
  drained_at: string | null;
  desired_generation: number;
  /** Models whose only live placement was on this node, so draining it took
   * them offline. Only non-empty when the refusal was acknowledged. */
  models_offline?: string[];
  detail?: string | null;
}

/**
 * Take a node out of the routing rotation.
 *
 * Refuses with 409 when the node holds the last live placement of a model,
 * naming them; `acknowledgeOutage` repeats the request past that refusal. The
 * refusal is overridable rather than absolute because taking a machine out
 * even at the cost of a model is a legitimate thing to want, and only the
 * operator knows whether it is.
 */
export async function drainNode(
  nodeId: string,
  opts: { acknowledgeOutage?: boolean } = {},
): Promise<DrainResponse> {
  return request<DrainResponse>(`/v1/serving/nodes/${encodeURIComponent(nodeId)}/drain`, {
    method: "POST",
    body: JSON.stringify({ acknowledge_outage: opts.acknowledgeOutage ?? false }),
  });
}

/** Admit a node again. Its reconciler restores what the drain stopped. */
export async function resumeNode(nodeId: string): Promise<DrainResponse> {
  return request<DrainResponse>(`/v1/serving/nodes/${encodeURIComponent(nodeId)}/resume`, {
    method: "POST",
  });
}

export interface FleetView {
  nodes: FleetNode[];
  observed_available: boolean;
  /** How old a snapshot may be before `reporting` goes false. */
  stale_after_s?: number | null;
  detail?: string | null;
}

export async function getFleet(): Promise<FleetView> {
  return request<FleetView>("/v1/serving/nodes");
}

export interface NodeBudgetResponse {
  node_id: string;
  mem_budget_bytes: number | null;
  mem_budget_version: number | null;
  mem_budget_updated_at: string | null;
  mem_budget_actor: string | null;
  detail?: string | null;
}

/**
 * Set or clear a node's RAM budget override. `expectedVersion` is the
 * `mem_budget_version` last read (null when none was ever set); a mismatch is
 * a 409. Needs an operator key.
 */
export async function setNodeBudget(
  nodeId: string,
  budgetBytes: number | null,
  expectedVersion: number | null,
): Promise<NodeBudgetResponse> {
  return request<NodeBudgetResponse>(`/v1/serving/nodes/${encodeURIComponent(nodeId)}/budget`, {
    method: "POST",
    body: JSON.stringify({ budget_bytes: budgetBytes, expected_version: expectedVersion }),
  });
}
