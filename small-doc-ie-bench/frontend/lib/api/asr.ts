// Durable speech-to-text jobs and their downloadable transcript artifacts.

import { downloadArtifact } from "./benchmark";
import { request } from "./core";
import { isLiveDeployment } from "./helpers";
import type { DeploymentRecord } from "./serving";

export type ASRJobStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "completed_with_errors"
  | "failed";

export type ASRItemStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface ASRSegment {
  id?: number;
  start?: number;
  end?: number;
  text?: string;
  [key: string]: unknown;
}

export interface ASRTranscriptionResult {
  text?: string;
  language?: string | null;
  duration?: number;
  segments?: ASRSegment[];
  processing_seconds?: number;
  real_time_factor?: number | null;
  model?: string;
  backend?: string;
  [key: string]: unknown;
}

export interface ASRMetrics {
  completed_items?: number;
  failed_items?: number;
  scored_items?: number;
  audio_seconds?: number;
  processing_seconds?: number;
  real_time_factor?: number | null;
  word_errors?: number;
  reference_words?: number;
  wer?: number | null;
  character_errors?: number;
  reference_characters?: number;
  cer?: number | null;
  [key: string]: unknown;
}

export interface ASRArtifact {
  id: string;
  name: string;
  kind: "text" | "verbose_json" | "srt" | "vtt" | "manifest" | string;
  sha256: string;
  size_bytes: number;
  media_type: string;
  uri: string;
}

export interface ASRJobItem {
  position: number;
  filename: string;
  input_sha256: string;
  input_size_bytes: number;
  raw_available: boolean;
  mime_type: string;
  reference: string | null;
  status: ASRItemStatus;
  detected_language: string | null;
  duration_seconds: number | null;
  processing_seconds: number | null;
  result: ASRTranscriptionResult | null;
  metrics: ASRMetrics | null;
  error: string | null;
  attempts: number;
  started_at: string | null;
  completed_at: string | null;
  artifacts: ASRArtifact[];
}

export interface ASRJob {
  event_id: string;
  channel: string;
  deployment: string;
  model: string;
  status: ASRJobStatus;
  total_items: number;
  completed_items: number;
  failed_items: number;
  options: { language?: string; prompt?: string; temperature?: number; [key: string]: unknown };
  metrics: ASRMetrics | null;
  error: string | null;
  raw_retention: "delete_after_completion" | "retain_7d" | "retain_30d";
  raw_expires_at: string | null;
  cancel_requested_at: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  updated_at: string;
  artifacts: ASRArtifact[];
  items: ASRJobItem[];
}

export interface ASRRecordingInput {
  filename: string;
  content_b64: string;
  reference?: string;
}

export interface CreateASRJobRequest {
  model: string;
  recordings: ASRRecordingInput[];
  language?: string;
  prompt?: string;
  temperature?: number;
  raw_audio_retention?: ASRJob["raw_retention"];
  idempotency_key?: string;
}

export interface CreateASRJobResponse {
  job_id: string;
  status: string;
  channel: string;
  topics: string[];
  status_uri: string;
  deduplicated: boolean;
}

export const ASR_REALTIME_TOPICS = ["status", "progress", "result", "error"];

export function createASRJob(payload: CreateASRJobRequest): Promise<CreateASRJobResponse> {
  return request<CreateASRJobResponse>("/v1/audio/transcription-jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listASRJobs(limit = 100): Promise<ASRJob[]> {
  return request<ASRJob[]>(`/v1/audio/transcription-jobs?limit=${limit}`);
}

export function getASRJob(jobId: string): Promise<ASRJob> {
  return request<ASRJob>(`/v1/audio/transcription-jobs/${encodeURIComponent(jobId)}`);
}

export function cancelASRJob(jobId: string): Promise<ASRJob> {
  return request<ASRJob>(`/v1/audio/transcription-jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
}

export function downloadASRArtifact(artifact: ASRArtifact): Promise<void> {
  return downloadArtifact(artifact.uri, artifact.name);
}

/** Only ready, reachable ASR deployments may enter a transcription request. */
export function liveASRDeployments(records: DeploymentRecord[]): DeploymentRecord[] {
  return records.filter(
    (record) => record.spec?.launch?.runtime === "asr" && isLiveDeployment(record),
  );
}

export function isTerminalASRStatus(status: ASRJobStatus): boolean {
  return ["cancelled", "completed", "completed_with_errors", "failed"].includes(status);
}
