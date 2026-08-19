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
  ocr_backend?: string;
  language?: string;
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
