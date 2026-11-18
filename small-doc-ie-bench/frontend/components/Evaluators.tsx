"use client";

import { useMemo, useState } from "react";
import { Code2, ListChecks, Plus, Scale, Search, Sparkles, Trash2 } from "lucide-react";
import {
  createEvaluator,
  decisionDeploymentNames,
  deleteEvaluator,
  deploymentModelType,
  deploymentServesAny,
  getAgents,
  getDeployments,
  getEvaluatorTemplates,
  getStore,
  listEvaluationResults,
  listEvaluators,
  modelPickerOptions,
  selectableDeployments,
  updateEvaluator,
  type AgentView,
  type DeploymentRecord,
  type StoreEntry,
  type EvaluationResult,
  type EvaluatorJudge,
  type EvaluatorQuestion,
  type EvaluatorSpec,
  type EvaluatorTemplate,
  type QuestionType,
  type ReadScope,
  type ReviewRule,
  type ReviewRuleWhen,
} from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { toUserMessage } from "@/lib/errors";
import { cn } from "@/lib/cn";
import { useToast } from "./Toast";
import {
  Alert,
  Badge,
  Button,
  Dialog,
  Field,
  Segmented,
  Select,
  Skeleton,
  TextArea,
  TextInput,
} from "./ui";
import { PageHeader } from "./patterns/PageHeader";
import { Table, type Column } from "./patterns/Table";

const JUDGE_LABEL: Record<EvaluatorJudge, string> = {
  decision: "Decision model",
  llm_judge: "LLM-as-a-judge",
};

// ---------------------------------------------------------------------------
// Question drafts: the editor's text form of EvaluatorQuestion.
// ---------------------------------------------------------------------------

export interface QuestionDraft {
  id: string;
  type: QuestionType;
  instructions: string;
  /** choice: "option" or "option: description" per line; score: one level per line. */
  options: string;
}

function lines(text: string): string[] {
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
}

export function toDraft(q: EvaluatorQuestion): QuestionDraft {
  let options = "";
  if (q.type === "choice" && q.criteria && !Array.isArray(q.criteria)) {
    options = Object.entries(q.criteria)
      .map(([k, v]) => (v ? `${k}: ${v}` : k))
      .join("\n");
  } else if (q.type === "score" && Array.isArray(q.criteria)) {
    options = q.criteria.join("\n");
  }
  return { id: q.id, type: q.type, instructions: q.instructions, options };
}

export function fromDraft(d: QuestionDraft): EvaluatorQuestion {
  const base = { id: d.id.trim(), type: d.type, instructions: d.instructions.trim() };
  if (d.type === "choice") {
    const criteria: Record<string, string | null> = {};
    for (const line of lines(d.options)) {
      const at = line.indexOf(":");
      const key = (at >= 0 ? line.slice(0, at) : line).trim();
      const desc = at >= 0 ? line.slice(at + 1).trim() : "";
      if (key) criteria[key] = desc || null;
    }
    return { ...base, criteria };
  }
  if (d.type === "score") return { ...base, criteria: lines(d.options) };
  return base;
}

interface Draft {
  isNew: boolean;
  name: string;
  description: string;
  judge: EvaluatorJudge;
  reads: ReadScope;
  model: string;
  agents: string[];
  samplePercent: number;
  questions: QuestionDraft[];
  rules: ReviewRule[];
  reviewOnError: boolean;
  enabled: boolean;
}

function draftFromTemplate(t: EvaluatorTemplate): Draft {
  return {
    isNew: true,
    name: t.questions.length ? t.id : "",
    description: t.summary,
    judge: t.judge,
    reads: t.reads ?? "run",
    model: "",
    agents: [],
    samplePercent: 100,
    questions: t.questions.length
      ? t.questions.map(toDraft)
      : [{ id: "q1", type: "noul", instructions: "", options: "" }],
    rules: t.review_when,
    reviewOnError: false,
    enabled: true,
  };
}

function draftFromSpec(s: EvaluatorSpec): Draft {
  return {
    isNew: false,
    name: s.name,
    description: s.description ?? "",
    judge: s.judge,
    reads: s.reads ?? "run",
    model: s.model,
    agents: s.agents,
    samplePercent: Math.round(s.sample_rate * 100),
    questions: s.questions.map(toDraft),
    rules: s.review_when,
    reviewOnError: s.review_on_error,
    enabled: s.enabled,
  };
}

function specFromDraft(d: Draft): EvaluatorSpec {
  return {
    name: d.name.trim(),
    description: d.description,
    judge: d.judge,
    reads: d.reads,
    model: d.model,
    agents: d.agents,
    sample_rate: Math.min(100, Math.max(0, d.samplePercent)) / 100,
    questions: d.questions.map(fromDraft),
    review_when: d.rules,
    review_on_error: d.reviewOnError,
    enabled: d.enabled,
  };
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function Evaluators({ active = true }: { active?: boolean }) {
  void active;
  const evaluators = useAsync<EvaluatorSpec[]>("evaluators", listEvaluators);
  const templates = useAsync("evaluator-templates", getEvaluatorTemplates);
  const agents = useAsync<AgentView[]>("agents", getAgents);
  const deployments = useAsync<DeploymentRecord[]>("deployments", getDeployments);
  const store = useAsync<StoreEntry[]>("store", getStore);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [showGallery, setShowGallery] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);

  const list = evaluators.data ?? [];
  const galleryOpen = showGallery || (evaluators.data != null && list.length === 0);

  const columns: Column<EvaluatorSpec>[] = [
    {
      key: "name",
      header: "Evaluator",
      sortAccessor: (e) => e.name,
      render: (e) => (
        <div className="min-w-0">
          <p className="truncate font-medium text-foreground">{e.name}</p>
          {e.description && (
            <p className="truncate text-xs text-muted-foreground">{e.description}</p>
          )}
        </div>
      ),
    },
    {
      key: "judge",
      header: "Judge",
      render: (e) => <Badge tone="info">{JUDGE_LABEL[e.judge]}</Badge>,
    },
    { key: "model", header: "Model", render: (e) => <span className="text-xs">{e.model}</span> },
    {
      key: "agents",
      header: "Agents",
      render: (e) =>
        e.agents.length ? (
          <span className="text-xs">{e.agents.join(", ")}</span>
        ) : (
          <span className="text-xs text-muted-foreground">none</span>
        ),
    },
    {
      key: "sample",
      header: "Sample",
      render: (e) => <span className="text-xs tabular-nums">{Math.round(e.sample_rate * 100)}%</span>,
    },
    {
      key: "enabled",
      header: "Status",
      render: (e) => (
        <Badge tone={e.enabled ? "ok" : "neutral"}>{e.enabled ? "enabled" : "disabled"}</Badge>
      ),
    },
  ];

  return (
    <div>
      <PageHeader
        title="Evaluators"
        subtitle="Score agent runs after they finish. A run that trips a rule goes to Review → Agent runs."
        actions={
          <Button size="sm" onClick={() => setShowGallery(true)}>
            <Plus className="h-4 w-4" />
            Evaluator
          </Button>
        }
      />

      {evaluators.error ? (
        <Alert tone="err">{toUserMessage(evaluators.error, { fallback: "Could not load evaluators." })}</Alert>
      ) : evaluators.loading && !evaluators.data ? (
        <Skeleton className="h-32" />
      ) : galleryOpen ? (
        <Gallery
          templates={templates.data?.templates ?? []}
          categories={templates.data?.categories ?? []}
          loading={templates.loading}
          onPick={(t) => setDraft(draftFromTemplate(t))}
          onClose={list.length ? () => setShowGallery(false) : undefined}
        />
      ) : (
        <Table
          columns={columns}
          rows={list}
          loading={false}
          error={null}
          getRowKey={(e) => e.name}
          emptyLabel="No evaluators"
          expandedKey={expanded}
          onRowClick={(e) => setExpanded((cur) => (cur === e.name ? null : e.name))}
          renderExpanded={(e) => (
            <EvaluatorDetail
              spec={e}
              onEdit={() => setDraft(draftFromSpec(e))}
              onDeleted={() => {
                setExpanded(null);
                evaluators.reload();
              }}
            />
          )}
        />
      )}

      {draft && (
        <EditorDialog
          draft={draft}
          agents={(agents.data ?? []).filter((a) => a.kind !== "proxy_security")}
          deployments={selectableDeployments(deployments.data ?? [])}
          store={store.data}
          onClose={() => setDraft(null)}
          onSaved={() => {
            setDraft(null);
            setShowGallery(false);
            evaluators.reload();
          }}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Template gallery
// ---------------------------------------------------------------------------

function TemplateCard({ t, onPick }: { t: EvaluatorTemplate; onPick: () => void }) {
  return (
    <button
      type="button"
      onClick={onPick}
      className="flex w-full items-start gap-3 rounded-lg border border-border bg-card p-4 text-left transition hover:border-primary/50 hover:bg-muted/30"
    >
      <span className="mt-0.5 rounded-md bg-muted p-1.5 text-muted-foreground">
        {t.judge === "decision" ? <ListChecks className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex flex-wrap items-center gap-2">
          <span className="font-medium text-foreground">{t.title}</span>
          <Badge tone="neutral">{JUDGE_LABEL[t.judge]}</Badge>
          {t.recommended && <Badge tone="ok">Recommended</Badge>}
        </span>
        <span className="mt-1 block text-sm text-muted-foreground">{t.summary}</span>
      </span>
    </button>
  );
}

function Gallery({
  templates,
  categories,
  loading,
  onPick,
  onClose,
}: {
  templates: EvaluatorTemplate[];
  categories: { id: string; title: string; summary?: string }[];
  loading: boolean;
  onPick: (t: EvaluatorTemplate) => void;
  onClose?: () => void;
}) {
  const [query, setQuery] = useState("");
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return templates;
    return templates.filter((t) => `${t.title} ${t.summary}`.toLowerCase().includes(q));
  }, [templates, query]);

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="flex items-center gap-3">
        <span className="rounded-lg bg-muted p-2 text-muted-foreground">
          <Scale className="h-5 w-5" />
        </span>
        <h2 className="flex-1 text-lg font-semibold">Get started with Evaluators</h2>
        {onClose && (
          <Button size="sm" variant="secondary" onClick={onClose}>
            Back to my evaluators
          </Button>
        )}
      </div>
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <TextInput
          aria-label="Search templates"
          placeholder="Search templates…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="pl-9"
        />
      </div>
      {loading && <Skeleton className="h-40" />}
      {categories.map((c) => {
        const items = shown.filter((t) => t.category === c.id);
        if (!items.length) return null;
        const rows = c.id === "scratch";
        return (
          <section key={c.id} className="space-y-2">
            <div>
              <h3 className="text-sm font-semibold text-foreground">{c.title}</h3>
              {c.summary && <p className="text-xs text-muted-foreground">{c.summary}</p>}
            </div>
            <div className={cn("grid gap-3", rows ? "grid-cols-1" : "sm:grid-cols-2")}>
              {items.map((t) => (
                <TemplateCard key={t.id} t={t} onPick={() => onPick(t)} />
              ))}
            </div>
          </section>
        );
      })}
      {!loading && shown.length === 0 && (
        <p className="text-sm text-muted-foreground">No template matches “{query}”.</p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Evaluator detail: latest results
// ---------------------------------------------------------------------------

function verdictText(r: EvaluationResult): string {
  return r.verdicts
    .map((v) => {
      if (v.error) return `${v.question_id}: failed`;
      if (v.type === "noul" && v.noul != null) return `${v.question_id}: yes ${(v.noul * 100).toFixed(0)}%`;
      if (v.type === "score" && v.score != null) return `${v.question_id}: ${v.score.toFixed(2)}`;
      return `${v.question_id}: ${v.choice ?? "—"}`;
    })
    .join(" · ");
}

function EvaluatorDetail({
  spec,
  onEdit,
  onDeleted,
}: {
  spec: EvaluatorSpec;
  onEdit: () => void;
  onDeleted: () => void;
}) {
  const { toast } = useToast();
  const results = useAsync<EvaluationResult[]>(`evaluator-results:${spec.name}`, () =>
    listEvaluationResults(spec.name),
  );
  const [busy, setBusy] = useState(false);

  async function remove() {
    setBusy(true);
    try {
      await deleteEvaluator(spec.name);
      onDeleted();
    } catch (err) {
      toast({ title: "Delete failed", description: toUserMessage(err), tone: "error" });
    } finally {
      setBusy(false);
    }
  }

  const rows = results.data ?? [];
  const flagged = rows.filter((r) => r.review_reasons.length > 0).length;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="secondary" onClick={onEdit}>
          Edit
        </Button>
        <Button size="sm" variant="danger" loading={busy} onClick={() => void remove()}>
          <Trash2 className="h-3.5 w-3.5" />
          Delete
        </Button>
        <span className="text-xs text-muted-foreground">
          {rows.length} recent run{rows.length === 1 ? "" : "s"} scored · {flagged} sent to review
        </span>
      </div>
      {results.loading && !results.data ? (
        <Skeleton className="h-16" />
      ) : rows.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No runs scored yet. Runs are scored after a targeted agent answers.
        </p>
      ) : (
        <div className="space-y-1">
          {rows.slice(0, 20).map((r) => (
            <div
              key={`${r.run_id}-${r.created_at}`}
              className="flex flex-wrap items-center gap-2 rounded-md border border-border px-3 py-1.5 text-xs"
            >
              <span className="text-muted-foreground">{new Date(r.created_at).toLocaleString()}</span>
              <span className="font-medium">{r.agent}</span>
              <span className="flex-1 truncate">{verdictText(r)}</span>
              {r.review_reasons.length > 0 && <Badge tone="warn">review</Badge>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

const RULES_FOR: Record<QuestionType, { value: ReviewRuleWhen; label: string }[]> = {
  noul: [
    { value: "noul_below", label: "yes-probability below" },
    { value: "noul_above", label: "yes-probability above" },
  ],
  score: [
    { value: "score_below", label: "level below" },
    { value: "score_above", label: "level above" },
  ],
  choice: [{ value: "choice_in", label: "answer is one of" }],
};

function EditorDialog({
  draft: initial,
  agents,
  deployments,
  store,
  onClose,
  onSaved,
}: {
  draft: Draft;
  agents: AgentView[];
  deployments: DeploymentRecord[];
  store?: StoreEntry[] | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const { toast } = useToast();
  const [d, setD] = useState<Draft>(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A decision judge must be a decision model (others answer 501); an LLM
  // judge must be a chat model.
  const modelOptions = useMemo(() => {
    const decision = decisionDeploymentNames(store);
    const judges = deployments.filter((x) =>
      d.judge === "decision"
        ? deploymentServesAny(x, decision)
        : !deploymentServesAny(x, decision) && deploymentModelType(x) === "chat",
    );
    const options = modelPickerOptions(judges);
    if (d.model && !options.some((o) => o.value === d.model)) {
      options.unshift({ value: d.model, label: d.model });
    }
    return options;
  }, [deployments, store, d.judge, d.model]);

  function patch(p: Partial<Draft>) {
    setD((cur) => ({ ...cur, ...p }));
  }
  function patchQuestion(i: number, p: Partial<QuestionDraft>) {
    setD((cur) => ({
      ...cur,
      questions: cur.questions.map((q, j) => (j === i ? { ...q, ...p } : q)),
    }));
  }

  async function save() {
    setError(null);
    if (!d.name.trim()) return setError("Give the evaluator a name.");
    if (!d.model) return setError("Pick the model that judges.");
    setSaving(true);
    try {
      const spec = specFromDraft(d);
      await (d.isNew ? createEvaluator(spec) : updateEvaluator(spec));
      toast({ title: d.isNew ? "Evaluator created" : "Evaluator saved", tone: "success" });
      onSaved();
    } catch (err) {
      setError(toUserMessage(err, { fallback: "Could not save the evaluator." }));
    } finally {
      setSaving(false);
    }
  }

  const questionIds = d.questions.map((q) => q.id).filter(Boolean);

  return (
    <Dialog
      open
      onClose={onClose}
      title={d.isNew ? "New evaluator" : `Edit ${d.name}`}
      subtitle={JUDGE_LABEL[d.judge]}
      className="max-w-2xl"
      footer={
        <div className="flex justify-end gap-2">
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button loading={saving} onClick={() => void save()}>
            Save
          </Button>
        </div>
      }
    >
      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Name" hint="Lowercase letters, digits, - and _.">
            <TextInput
              value={d.name}
              disabled={!d.isNew}
              onChange={(e) => patch({ name: e.target.value.toLowerCase() })}
            />
          </Field>
          <Field label="Judge">
            <Segmented
              value={d.judge}
              onChange={(judge) => patch({ judge })}
              options={[
                { value: "decision", label: "Decision model" },
                { value: "llm_judge", label: "LLM-as-a-judge" },
              ]}
            />
          </Field>
        </div>
        <Field label="Description">
          <TextInput value={d.description} onChange={(e) => patch({ description: e.target.value })} />
        </Field>
        <Field
          label="Model"
          hint={
            d.judge === "decision"
              ? "A decision-model deployment (e.g. ggml-org/Laya-GGUF). Other models answer 501."
              : "A chat deployment; it answers each question with a reason."
          }
        >
          <Select value={d.model} onChange={(e) => patch({ model: e.target.value })}>
            <option value="">Select a deployment…</option>
            {modelOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </Select>
        </Field>
        <Field
          label="Reads"
          hint="A question about the user should not see the answer: a refusal reads as an attack."
        >
          <Segmented
            value={d.reads}
            onChange={(reads) => patch({ reads })}
            options={[
              { value: "run", label: "Whole run" },
              { value: "user", label: "User messages" },
              { value: "answer", label: "Final answer" },
            ]}
          />
        </Field>
        <Field
          label="Agents to score"
          hint="A run is kept until it is scored, or until its review is decided."
        >
          <div className="flex flex-wrap gap-2">
            {agents.length === 0 && (
              <span className="text-xs text-muted-foreground">No agents yet.</span>
            )}
            {agents.map((a) => {
              const on = d.agents.includes(a.name);
              return (
                <button
                  key={a.name}
                  type="button"
                  aria-pressed={on}
                  onClick={() =>
                    patch({
                      agents: on ? d.agents.filter((n) => n !== a.name) : [...d.agents, a.name],
                    })
                  }
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs transition",
                    on
                      ? "border-primary bg-primary/10 text-foreground"
                      : "border-border text-muted-foreground hover:text-foreground",
                  )}
                >
                  {a.name}
                </button>
              );
            })}
          </div>
        </Field>
        <Field label="Sample" hint="Share of runs scored, 0–100%.">
          <TextInput
            type="number"
            min={0}
            max={100}
            value={String(d.samplePercent)}
            onChange={(e) => patch({ samplePercent: Number(e.target.value) || 0 })}
          />
        </Field>

        <div className="space-y-2">
          <p className="text-sm font-medium">Questions</p>
          {d.questions.map((q, i) => (
            <div key={i} className="space-y-2 rounded-md border border-border p-3">
              <div className="flex flex-wrap items-end gap-2">
                <Field label="Id">
                  <TextInput
                    aria-label={`Question ${i + 1} id`}
                    value={q.id}
                    onChange={(e) => patchQuestion(i, { id: e.target.value })}
                  />
                </Field>
                <Field label="Type">
                  <Select
                    aria-label={`Question ${i + 1} type`}
                    value={q.type}
                    onChange={(e) => patchQuestion(i, { type: e.target.value as QuestionType })}
                  >
                    <option value="noul">yes / no</option>
                    <option value="choice">choice</option>
                    <option value="score">score</option>
                  </Select>
                </Field>
                <Button
                  size="sm"
                  variant="secondary"
                  aria-label={`Remove question ${i + 1}`}
                  onClick={() =>
                    patch({ questions: d.questions.filter((_, j) => j !== i) })
                  }
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
              <Field label="Question">
                <TextInput
                  value={q.instructions}
                  onChange={(e) => patchQuestion(i, { instructions: e.target.value })}
                />
              </Field>
              {q.type !== "noul" && (
                <Field
                  label={q.type === "choice" ? "Options" : "Levels"}
                  hint={
                    q.type === "choice"
                      ? "One per line: option, or option: description."
                      : "2 to 10 levels, one per line, lowest first."
                  }
                >
                  <TextArea
                    rows={3}
                    value={q.options}
                    onChange={(e) => patchQuestion(i, { options: e.target.value })}
                  />
                </Field>
              )}
            </div>
          ))}
          <Button
            size="sm"
            variant="secondary"
            onClick={() =>
              patch({
                questions: [
                  ...d.questions,
                  { id: `q${d.questions.length + 1}`, type: "noul", instructions: "", options: "" },
                ],
              })
            }
          >
            Add question
          </Button>
        </div>

        <div className="space-y-2">
          <p className="text-sm font-medium">Send to review when</p>
          {d.rules.map((r, i) => {
            const qtype = d.questions.find((q) => q.id === r.question)?.type ?? "noul";
            return (
              <div key={i} className="flex flex-wrap items-center gap-2">
                <Select
                  aria-label={`Rule ${i + 1} question`}
                  value={r.question}
                  onChange={(e) => {
                    const next = d.questions.find((q) => q.id === e.target.value);
                    const when = RULES_FOR[next?.type ?? "noul"][0].value;
                    const rules = [...d.rules];
                    rules[i] = {
                      question: e.target.value,
                      when,
                      value: when === "choice_in" ? [] : 0.5,
                    };
                    patch({ rules });
                  }}
                >
                  {questionIds.map((id) => (
                    <option key={id} value={id}>
                      {id}
                    </option>
                  ))}
                </Select>
                <Select
                  aria-label={`Rule ${i + 1} condition`}
                  value={r.when}
                  onChange={(e) => {
                    const rules = [...d.rules];
                    rules[i] = { ...r, when: e.target.value as ReviewRuleWhen };
                    patch({ rules });
                  }}
                >
                  {RULES_FOR[qtype].map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </Select>
                <TextInput
                  aria-label={`Rule ${i + 1} value`}
                  className="w-40"
                  value={Array.isArray(r.value) ? r.value.join(", ") : String(r.value)}
                  onChange={(e) => {
                    const rules = [...d.rules];
                    rules[i] = {
                      ...r,
                      value:
                        r.when === "choice_in"
                          ? e.target.value
                              .split(",")
                              .map((v) => v.trim())
                              .filter(Boolean)
                          : Number(e.target.value),
                    };
                    patch({ rules });
                  }}
                />
                <Button
                  size="sm"
                  variant="secondary"
                  aria-label={`Remove rule ${i + 1}`}
                  onClick={() => patch({ rules: d.rules.filter((_, j) => j !== i) })}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            );
          })}
          <div className="flex flex-wrap items-center gap-3">
            <Button
              size="sm"
              variant="secondary"
              disabled={!questionIds.length}
              onClick={() => {
                const first = d.questions[0];
                const when = RULES_FOR[first.type][0].value;
                patch({
                  rules: [
                    ...d.rules,
                    { question: first.id, when, value: when === "choice_in" ? [] : 0.5 },
                  ],
                });
              }}
            >
              Add rule
            </Button>
            <label className="flex items-center gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={d.reviewOnError}
                onChange={(e) => patch({ reviewOnError: e.target.checked })}
              />
              Also when the judge fails
            </label>
            <label className="flex items-center gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={d.enabled}
                onChange={(e) => patch({ enabled: e.target.checked })}
              />
              Enabled
            </label>
          </div>
        </div>

        {error && <Alert tone="err">{error}</Alert>}
        {d.judge === "llm_judge" && (
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Code2 className="h-3.5 w-3.5" />
            An LLM judge&apos;s answer is self-reported; a decision model&apos;s probability is measured.
          </p>
        )}
      </div>
    </Dialog>
  );
}
