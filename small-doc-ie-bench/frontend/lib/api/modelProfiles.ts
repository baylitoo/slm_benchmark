// configs/models.yaml profile listing + kind=pipeline/ocr authoring.

import { request } from "./core";

/** A models.yaml profile (GET /v1/studio/model-profiles) — what the Benchmark
 * tab's "Model profile" field, and a new pipeline profile's extractor/ocr_model
 * pickers, can actually reference instead of a free-text guess. */
export interface ModelProfileSummary {
  name: string;
  kind: "passthrough" | "ocr" | "pipeline";
  vision: boolean;
  model: string;
}

export function listModelProfiles(): Promise<ModelProfileSummary[]> {
  return request<ModelProfileSummary[]>("/v1/studio/model-profiles");
}

export interface CreatePipelineProfileRequest {
  name: string;
  extractor: string;
  ocr_backend?: string;
  ocr_model?: string;
  language?: string;
}

export interface PipelineProfileResult {
  name: string;
  kind: string;
  options: Record<string, string>;
}

/** Author a kind="pipeline" (OCR->LLM) profile into configs/models.yaml — the
 * missing counterpart to referencing one by name (Benchmark's "Custom" field,
 * #183). Create-only; a 409 means the name is already taken. */
export function createPipelineProfile(
  payload: CreatePipelineProfileRequest,
): Promise<PipelineProfileResult> {
  return request<PipelineProfileResult>("/v1/studio/model-profiles/pipeline", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface CreateOcrProfileRequest {
  name: string;
  backend: string;
  language?: string;
}

/** Author a kind="ocr" (OCR-only, no LLM stage) profile into configs/models.yaml
 * — the sibling gap #188 explicitly deferred: that PR only covered kind="pipeline".
 * The resulting profile returns a backend's raw transcribed text as the completion
 * (no extraction stage) — not a schema-scored benchmark model; reachable directly
 * through the gateway by name, but no Studio UI surface invokes one yet.
 * Create-only; a 409 means the name is taken. */
export function createOcrProfile(
  payload: CreateOcrProfileRequest,
): Promise<PipelineProfileResult> {
  return request<PipelineProfileResult>("/v1/studio/model-profiles/ocr", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** Remove a kind="pipeline"/kind="ocr" profile from configs/models.yaml — the
 * counterpart to createPipelineProfile. Backend refuses (422) for any other
 * kind, since a passthrough profile can still be referenced by a live
 * deployment or another pipeline profile's extractor/ocr_model. */
export function deletePipelineProfile(name: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(
    `/v1/studio/model-profiles/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
}
