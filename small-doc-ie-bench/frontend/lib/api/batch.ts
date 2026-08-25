// Batch extraction: N documents through one schema + model as one durable job
// with per-document state (POST /v1/studio/extract/batch, GET /v1/studio/batches).
// Documents are sent once (a zip or inline items); the server stashes them in
// the shared artifact store and the worker runs each as its own durable step,
// so a batch that dies on document 150 resumes at 150 and one bad PDF records
// its error without killing the rest. Results download as JSONL (lossless) or
// CSV (flattened) through the batch's own tenant-scoped route.

import { downloadArtifact } from "./benchmark";
import { request, type TriggerResponse } from "./core";

export interface BatchDocumentInput {
  filename: string;
  content_b64: string;
}

export interface BatchExtractRequest {
  name?: string;
  /** A base64 zip of documents (.pdf/.png/.jpg/.tif/.txt; junk skipped). */
  zip_b64?: string;
  /** Or inline documents. Exactly one of zip_b64 / documents. */
  documents?: BatchDocumentInput[];
  schema_name?: string;
  dynamic_schema_name?: string;
  /** Live-deployment selector (a DeploymentRecord spec.name). */
  deployment?: string;
  model_profile?: string;
  /** A saved routing policy: every document runs the policy's escalation
   * ladder. Mutually exclusive with deployment/model_profile. */
  routing_policy?: string;
  ocr_backend?: string;
  language?: string;
  /** Optional completion webhook: the worker POSTs the settled run summary
   * (status, counts, result URIs) here. With callback_secret, the body is
   * HMAC-SHA256-signed into X-DocIE-Signature. */
  callback_url?: string;
  callback_secret?: string;
}

export function triggerBatchExtract(payload: BatchExtractRequest): Promise<TriggerResponse> {
  return request<TriggerResponse>("/v1/studio/extract/batch", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export type BatchStatus = "running" | "completed" | "failed";
export type BatchItemStatus = "pending" | "done" | "failed";

export interface BatchArtifactRef {
  name: string;
  relkey: string;
  media_type: string;
  size_bytes: number;
  sha256: string;
}

export interface BatchItemView {
  position: number;
  filename: string;
  status: BatchItemStatus;
  /** The full ExtractionResponse for a done item. */
  result: Record<string, unknown> | null;
  error: string | null;
  latency_ms: number | null;
  updated_at: string;
}

/** A batch as listed (GET /v1/studio/batches) -- no items. */
export interface BatchRunSummary {
  event_id: string;
  channel: string;
  tenant_id: string;
  name: string;
  schema_name: string;
  model_selector: string | null;
  status: BatchStatus;
  total_items: number;
  done_items: number;
  failed_items: number;
  error: string | null;
  artifacts: BatchArtifactRef[];
  created_at: string;
  updated_at: string;
}

/** A batch with every item (GET /v1/studio/batches/{event_id}). */
export interface BatchRunDetail extends BatchRunSummary {
  items: BatchItemView[];
}

export function listBatches(): Promise<BatchRunSummary[]> {
  return request<BatchRunSummary[]>("/v1/studio/batches");
}

export function getBatch(eventId: string): Promise<BatchRunDetail> {
  return request<BatchRunDetail>(`/v1/studio/batches/${encodeURIComponent(eventId)}`);
}

/** Save a batch's results file. Authenticated blob download (X-API-Key), same
 * mechanism as benchmark artifacts -- a plain <a href> can't carry the key. */
export function downloadBatchResults(
  batch: Pick<BatchRunSummary, "event_id" | "name">,
  fmt: "jsonl" | "csv",
): Promise<void> {
  const uri = `/v1/studio/batches/${encodeURIComponent(batch.event_id)}/results.${fmt}`;
  return downloadArtifact(uri, `${batch.name || "batch"}.${fmt}`);
}

/** Optional model override for a retry — e.g. re-run failures with a
 * stronger model or a policy. Empty = the original batch's selectors. */
export interface RetryFailedRequest {
  deployment?: string;
  model_profile?: string;
  routing_policy?: string;
}

/** Re-run ONLY a settled batch's failed documents, as a new batch (the
 * documents are re-read server-side from durable storage — no re-upload). */
export function retryBatchFailed(
  eventId: string,
  override: RetryFailedRequest = {},
): Promise<TriggerResponse> {
  return request<TriggerResponse>(
    `/v1/studio/batches/${encodeURIComponent(eventId)}/retry-failed`,
    { method: "POST", body: JSON.stringify(override) },
  );
}

// -- schedules (recurring re-runs of a batch's stored documents) -------------

export type BatchScheduleInterval = "hourly" | "daily" | "weekly" | "every_n_minutes";

/** A saved recurring batch: re-runs the source batch's durably stored
 * documents on an interval. A once-a-minute worker cron fires each due
 * schedule as a NORMAL batch (it appears in the batches list like any other),
 * then advances next_run_at. last_error records a firing that couldn't happen
 * (source batch deleted, blobs swept). */
export interface BatchScheduleView {
  id: string;
  tenant_id: string;
  name: string;
  source_event_id: string;
  schema_name: string;
  /** Selectors each firing carries (model_profile/deployment/routing_policy/
   * ocr_backend/language/dynamic_schema_name). */
  selectors: Record<string, string>;
  interval: BatchScheduleInterval;
  every_n_minutes: number | null;
  enabled: boolean;
  next_run_at: string;
  last_run_at: string | null;
  last_event_id: string | null;
  last_error: string | null;
  created_at: string;
  updated_at: string;
}

export interface BatchScheduleCreateRequest {
  name?: string;
  /** The batch whose stored documents every firing re-runs. */
  source_event_id: string;
  interval: BatchScheduleInterval;
  /** Required when interval is "every_n_minutes"; floor 15. */
  every_n_minutes?: number;
  enabled?: boolean;
  /** Optional model override (mutually exclusive trio) — replaces the source
   * batch's model selector, mirroring retry-failed. */
  deployment?: string;
  model_profile?: string;
  routing_policy?: string;
}

export interface BatchSchedulePatchRequest {
  name?: string;
  enabled?: boolean;
  interval?: BatchScheduleInterval;
  every_n_minutes?: number;
}

export function listBatchSchedules(): Promise<BatchScheduleView[]> {
  return request<BatchScheduleView[]>("/v1/studio/batch-schedules");
}

export function createBatchSchedule(
  payload: BatchScheduleCreateRequest,
): Promise<BatchScheduleView> {
  return request<BatchScheduleView>("/v1/studio/batch-schedules", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateBatchSchedule(
  scheduleId: string,
  patch: BatchSchedulePatchRequest,
): Promise<BatchScheduleView> {
  return request<BatchScheduleView>(
    `/v1/studio/batch-schedules/${encodeURIComponent(scheduleId)}`,
    { method: "PATCH", body: JSON.stringify(patch) },
  );
}

export function deleteBatchSchedule(scheduleId: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(
    `/v1/studio/batch-schedules/${encodeURIComponent(scheduleId)}`,
    { method: "DELETE" },
  );
}

/** Fire the schedule immediately (exactly the event the cron would send)
 * without touching its cadence. */
export function runBatchScheduleNow(scheduleId: string): Promise<TriggerResponse> {
  return request<TriggerResponse>(
    `/v1/studio/batch-schedules/${encodeURIComponent(scheduleId)}/run-now`,
    { method: "POST" },
  );
}
