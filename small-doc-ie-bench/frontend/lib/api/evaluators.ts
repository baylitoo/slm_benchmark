// Evaluators (GET/POST/PUT/DELETE /v1/studio/evaluators): judges that score
// agent runs after the fact and send flagged runs to the review queue.

import { request } from "./core";

export type EvaluatorJudge = "decision" | "llm_judge";
export type QuestionType = "choice" | "score" | "noul";
export type ReadScope = "run" | "user" | "answer";
export type ReviewRuleWhen =
  | "noul_above"
  | "noul_below"
  | "score_above"
  | "score_below"
  | "choice_in";

export interface EvaluatorQuestion {
  id: string;
  type: QuestionType;
  instructions: string;
  /** choice: {option: description|null}; score: levels lowest first; noul: optional. */
  criteria?: Record<string, string | null> | string[] | null;
}

export interface ReviewRule {
  question: string;
  when: ReviewRuleWhen;
  value: number | string[];
}

export interface EvaluatorSpec {
  name: string;
  description?: string;
  judge: EvaluatorJudge;
  model: string;
  questions: EvaluatorQuestion[];
  agents: string[];
  sample_rate: number;
  review_when: ReviewRule[];
  review_on_error: boolean;
  max_transcript_chars?: number;
  /** run: the whole exchange; user: the user's messages only; answer: the final answer. */
  reads?: ReadScope;
  enabled: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface EvaluatorTemplate {
  id: string;
  title: string;
  category: string;
  summary: string;
  recommended?: boolean;
  judge: EvaluatorJudge;
  reads?: ReadScope;
  questions: EvaluatorQuestion[];
  review_when: ReviewRule[];
}

export interface EvaluatorCategory {
  id: string;
  title: string;
  summary?: string;
}

export interface EvaluatorVerdict {
  evaluator: string;
  question_id: string;
  type: QuestionType;
  provenance: "measured" | "self_reported";
  choice: string | null;
  score: number | null;
  noul: number | null;
  probabilities: Record<string, number> | null;
  confidence: number | null;
  reason: string | null;
  error: string | null;
}

export interface EvaluationResult {
  run_id: string;
  agent: string;
  verdicts: EvaluatorVerdict[];
  review_reasons: string[];
  created_at: string;
}

export interface AgentRunView {
  id: string;
  agent: string;
  tenant_id: string;
  messages: { role: string; content: unknown }[];
  output: string;
  tool_calls: { tool?: string; status?: string }[];
  evaluators: string[];
  created_at: string;
  expires_at: string;
}

export function listEvaluators(): Promise<EvaluatorSpec[]> {
  return request<EvaluatorSpec[]>("/v1/studio/evaluators");
}

export function getEvaluatorTemplates(): Promise<{
  categories: EvaluatorCategory[];
  templates: EvaluatorTemplate[];
}> {
  return request("/v1/studio/evaluators/templates");
}

export function createEvaluator(spec: EvaluatorSpec): Promise<EvaluatorSpec> {
  return request<EvaluatorSpec>("/v1/studio/evaluators", {
    method: "POST",
    body: JSON.stringify(spec),
  });
}

export function updateEvaluator(spec: EvaluatorSpec): Promise<EvaluatorSpec> {
  return request<EvaluatorSpec>(`/v1/studio/evaluators/${encodeURIComponent(spec.name)}`, {
    method: "PUT",
    body: JSON.stringify(spec),
  });
}

export function deleteEvaluator(name: string): Promise<{ deleted: string }> {
  return request(`/v1/studio/evaluators/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export function listEvaluationResults(name: string, limit = 50): Promise<EvaluationResult[]> {
  return request<EvaluationResult[]>(
    `/v1/studio/evaluators/${encodeURIComponent(name)}/results?limit=${limit}`,
  );
}

export function getAgentRun(runId: string): Promise<AgentRunView> {
  return request<AgentRunView>(`/v1/studio/agent-runs/${encodeURIComponent(runId)}`);
}
