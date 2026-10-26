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
  /** How many placements name this node as their owner. */
  deployments: number;
  /** The store models whose blobs this node actually holds. `null` means the
   * node has not reported its store yet — which is NOT the same as holding
   * nothing, and the UI must not conflate them. */
  store_models?: string[] | null;
  /** Desired-state generation last written for this node; null when none. */
  desired_generation?: number | null;
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
