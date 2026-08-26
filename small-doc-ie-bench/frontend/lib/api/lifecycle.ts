// Deployment lifecycle actions (PR-4). Each fires a serving/* event at the
// single-replica serving service and returns the event ids to poll.

import { request } from "./core";
import type { LifecycleActionResponse } from "./serving";

/** Cold-start a deployment (idempotent server-side; may evict LRU victims). */
export function loadDeployment(name: string): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}/load`,
    { method: "POST" },
  );
}

/** Evict a deployment: process killed, record + port + row kept (phase=evicted). */
export function unloadDeployment(name: string): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}/unload`,
    { method: "POST" },
  );
}

/** Set/clear the eviction shield. */
export function pinDeployment(
  name: string,
  pinned: boolean,
): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}/pin`,
    { method: "POST", body: JSON.stringify({ pinned }) },
  );
}

/** Real teardown: kills the process, frees the port, deletes the row. */
export function deleteDeployment(name: string): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
}

/** Recover a stuck/failed deployment on a (re)allocated port (no delete+recreate).
 *  `port` omitted / null = auto-reallocate a free port (steps around an orphan
 *  still holding the old one); an explicit port is honored verbatim. */
export function repairDeployment(
  name: string,
  port?: number | null,
): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}/repair`,
    { method: "POST", body: JSON.stringify(port != null ? { port } : {}) },
  );
}

export interface DeploymentUpdate {
  context_length: number;
  /** Null clears the deployment override and restores the family default. */
  max_tokens: number | null;
}

/** Replace editable launch defaults in place. A hot runtime is restarted on
 * its existing port; a stopped/offloaded deployment remains stopped. */
export function updateDeployment(
  name: string,
  update: DeploymentUpdate,
): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/deployments/${encodeURIComponent(name)}`,
    { method: "PATCH", body: JSON.stringify(update) },
  );
}

/** Change a live deployment's context window with zero downtime: the server
 * drains a new-sized shadow instance into READY before routing to it and
 * stopping the old one -- no restart gap. A stopped/offloaded deployment has
 * no process to drain and is edited in place instead. Only llama.cpp
 * deployments accept this (422 otherwise); a 422 also carries the RAM
 * deficit up front (footprint/needed/available/shortfall) when the new
 * context would not fit, without touching the running deployment. */
export function resizeDeployment(
  name: string,
  contextLength: number,
): Promise<LifecycleActionResponse> {
  return request<LifecycleActionResponse>(
    `/v1/serving/store/${encodeURIComponent(name)}/resize`,
    { method: "POST", body: JSON.stringify({ context_length: contextLength }) },
  );
}

/** A deployment's runtime log tail (GET /v1/serving/deployments/{name}/logs). */
export interface DeploymentLogs {
  name: string;
  /** Reconciler one-line failure summary, if any. */
  last_error?: string | null;
  /** Raw stdout/stderr tail (most recent last). */
  lines: string[];
}

export function getDeploymentLogs(name: string, lines = 200): Promise<DeploymentLogs> {
  return request<DeploymentLogs>(
    `/v1/serving/deployments/${encodeURIComponent(name)}/logs?lines=${lines}`,
  );
}
