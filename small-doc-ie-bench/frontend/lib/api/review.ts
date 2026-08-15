// Review workflow (Review tab — GET/POST /v1/reviews/*). Every mutation is
// optimistic-concurrency-controlled: the server rejects with 409 if the
// caller's `expected_version` doesn't match the task's current `version`
// (someone else acted on it since the caller last fetched it) — callers
// should reload the task and let the operator retry rather than silently
// overwrite. `reviewer_id` is REQUIRED by the request schema (non-empty,
// validated server-side) but ignored by the handler in favor of the
// authenticated tenant identity — see api.py's claim_review_task et al.
// ("B2: provenance = authenticated principal"). This Studio has no login
// flow, so REVIEWER_ID is a fixed placeholder that only needs to satisfy
// that validation.

import { request } from "./core";

export type ReviewStatus = "pending" | "claimed" | "approved" | "rejected";

export interface ReviewReason {
  code:
    | "invalid"
    | "low_confidence"
    | "weak_evidence"
    | "arithmetic_mismatch"
    | "model_disagreement"
    | "learning_value"
    | "manual";
  score: number;
  detail: string;
}

export interface FieldCorrection {
  field_path: string;
  value: unknown;
  comment?: string | null;
}

export interface ReviewCorrectionView {
  id: number;
  revision: number;
  reviewer_id: string;
  created_at: string;
  corrections: Record<string, unknown>[];
  corrected_prediction: Record<string, unknown>;
  comment: string | null;
}

export interface ReviewEventView {
  id: number;
  event_type: string;
  actor_id: string | null;
  from_status: string | null;
  to_status: string | null;
  task_version: number;
  created_at: string;
  details: Record<string, unknown>;
}

export interface ReviewTaskView {
  id: number;
  source_request_id: string;
  schema_name: string;
  model_profile: string;
  document_hash: string | null;
  status: ReviewStatus;
  priority: number;
  priority_reasons: ReviewReason[];
  original_prediction: Record<string, unknown>;
  latest_prediction: Record<string, unknown>;
  validation_errors: string[];
  validation_warnings: string[];
  dynamic_schema: Record<string, unknown> | null;
  metadata: Record<string, string>;
  claimed_by: string | null;
  claim_expires_at: string | null;
  version: number;
  created_at: string;
  updated_at: string;
  decided_at: string | null;
  decided_by: string | null;
  decision_comment: string | null;
  corrections: ReviewCorrectionView[];
  events: ReviewEventView[];
  /** Computed fresh on every fetch (arithmetic reconciliation), never
   * persisted — a one-click starting point for the correction form, not
   * applied automatically. */
  suggested_corrections: FieldCorrection[];
}

const REVIEWER_ID = "studio-operator";

export function listReviews(params?: {
  status?: ReviewStatus;
  limit?: number;
}): Promise<ReviewTaskView[]> {
  const qs = new URLSearchParams();
  if (params?.status) qs.set("status", params.status);
  if (params?.limit) qs.set("limit", String(params.limit));
  const suffix = qs.toString() ? `?${qs}` : "";
  return request<ReviewTaskView[]>(`/v1/reviews${suffix}`);
}

export function getReview(taskId: number): Promise<ReviewTaskView> {
  return request<ReviewTaskView>(`/v1/reviews/${taskId}`);
}

export function claimReview(
  taskId: number,
  expectedVersion: number,
): Promise<ReviewTaskView> {
  return request<ReviewTaskView>(`/v1/reviews/${taskId}/claim`, {
    method: "POST",
    body: JSON.stringify({ reviewer_id: REVIEWER_ID, expected_version: expectedVersion }),
  });
}

export function releaseReview(
  taskId: number,
  expectedVersion: number,
  comment?: string,
): Promise<ReviewTaskView> {
  return request<ReviewTaskView>(`/v1/reviews/${taskId}/release`, {
    method: "POST",
    body: JSON.stringify({
      reviewer_id: REVIEWER_ID,
      expected_version: expectedVersion,
      comment: comment || undefined,
    }),
  });
}

export function correctReview(
  taskId: number,
  expectedVersion: number,
  corrections: FieldCorrection[],
  comment?: string,
): Promise<ReviewTaskView> {
  return request<ReviewTaskView>(`/v1/reviews/${taskId}/correct`, {
    method: "POST",
    body: JSON.stringify({
      reviewer_id: REVIEWER_ID,
      expected_version: expectedVersion,
      corrections,
      comment: comment || undefined,
    }),
  });
}

export function approveReview(
  taskId: number,
  expectedVersion: number,
  comment?: string,
): Promise<ReviewTaskView> {
  return request<ReviewTaskView>(`/v1/reviews/${taskId}/approve`, {
    method: "POST",
    body: JSON.stringify({
      reviewer_id: REVIEWER_ID,
      expected_version: expectedVersion,
      comment: comment || undefined,
    }),
  });
}

export function rejectReview(
  taskId: number,
  expectedVersion: number,
  comment?: string,
): Promise<ReviewTaskView> {
  return request<ReviewTaskView>(`/v1/reviews/${taskId}/reject`, {
    method: "POST",
    body: JSON.stringify({
      reviewer_id: REVIEWER_ID,
      expected_version: expectedVersion,
      comment: comment || undefined,
    }),
  });
}
