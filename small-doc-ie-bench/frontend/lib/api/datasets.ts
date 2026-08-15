// Registered-dataset listing + validation (duplicate doc_id, missing files,
// cross-split leakage, stats).

import { request } from "./core";

/** A registered benchmark dataset (GET /v1/studio/datasets) — the values a
 * Benchmark run's `dataset` field can actually reference. */
export interface DatasetSummary {
  name: string;
  description: string | null;
  latest: string | null;
  versions: string[];
  documents: number | null;
}

/** Registered datasets, so the Benchmark form can offer a picker instead of a
 * free-text guess at a name that may not be registered. */
export function listDatasets(): Promise<DatasetSummary[]> {
  return request<DatasetSummary[]>("/v1/studio/datasets");
}

interface DatasetLeakagePair {
  left_doc_id: string;
  left_split: string;
  right_doc_id: string;
  right_split: string;
  similarity?: number;
}

export interface DatasetLeakageReport {
  near_duplicate_threshold: number;
  exact_duplicates: DatasetLeakagePair[];
  near_duplicates: DatasetLeakagePair[];
  leakage_pairs: number;
}

export interface DatasetStatistics {
  documents: number;
  total_bytes: number;
  schemas: Record<string, number>;
  languages: Record<string, number>;
  splits: Record<string, number>;
  ground_truth_fields: Record<string, number>;
  labeled_documents: number;
}

// dataset_hash/statistics/leakage are ABSENT (not merely empty) when validate_dataset
// short-circuits on a structural error (duplicate doc_id, missing file) before it ever
// reaches the hash/leakage/stats computation -- see build_comparison_payload's sibling
// docstring in studio_api.py for the same "one report, but not every field always
// present" contract.
export interface DatasetValidationReport {
  reference: string;
  version: string | null;
  valid: boolean;
  errors: string[];
  warnings: string[];
  dataset_hash?: string;
  statistics?: DatasetStatistics;
  leakage?: DatasetLeakageReport;
}

/** Run the CLI's dataset validate/inspect checks against an already-registered
 * dataset version (duplicate doc_id, missing files, cross-split leakage, stats). */
export function validateDataset(
  name: string,
  version?: string,
): Promise<DatasetValidationReport> {
  const params = new URLSearchParams();
  if (version) params.set("version", version);
  const qs = params.toString();
  return request<DatasetValidationReport>(
    `/v1/studio/datasets/${encodeURIComponent(name)}/validate${qs ? `?${qs}` : ""}`,
    { method: "POST" },
  );
}
