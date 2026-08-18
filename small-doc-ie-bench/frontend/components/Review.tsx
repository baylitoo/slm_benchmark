"use client";

import { useEffect, useMemo, useState } from "react";
import { ClipboardCheck, Check, X, Undo2, Search } from "lucide-react";
import {
  approveReview,
  claimReview,
  correctReview,
  getReviewEvidence,
  listReviews,
  rejectReview,
  releaseReview,
  ApiError,
  type FieldCorrection,
  type OCRBlock,
  type ReviewEvidenceView,
  type ReviewStatus,
  type ReviewTaskView,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { toUserMessage } from "@/lib/errors";
import { useToast } from "./Toast";
import { Alert, Badge, Button, Segmented, Skeleton, TextInput, type BadgeTone } from "./ui";
import { JsonView } from "./JsonView";
import { PageHeader } from "./patterns/PageHeader";
import { Toolbar } from "./patterns/Toolbar";
import { ResultLine } from "./patterns/ResultLine";
import { Table, type Column } from "./patterns/Table";
import { T } from "@/lib/i18n";

const POLL_MS = 8000;

const STATUS_TONE: Record<ReviewStatus, BadgeTone> = {
  pending: "warn",
  claimed: "info",
  approved: "ok",
  rejected: "err",
};

const STATUS_FILTERS: { value: ReviewStatus | "all"; label: string }[] = [
  { value: "pending", label: "Pending" },
  { value: "claimed", label: "Claimed" },
  { value: "approved", label: "Approved" },
  { value: "rejected", label: "Rejected" },
  { value: "all", label: "All" },
];

/**
 * Review = the human review queue (claim/correct/approve/reject), previously
 * API-only (see Observability's now-trimmed tile — this is the "no Studio tab
 * yet" gap it used to point at). One view: the status filter lives on the
 * queue table itself, nothing to split into sub-tabs.
 */
export function Review({ active = true }: { active?: boolean }) {
  const { toast } = useToast();
  const [status, setStatus] = useState<ReviewStatus | "all">("pending");
  const [expanded, setExpanded] = useState<number | null>(null);
  // A mutation (claim, in particular) commonly moves a task OUT of the
  // active status filter -- claiming a task removes it from the default
  // "Pending" view mid-workflow, which would otherwise unmount the detail
  // panel (and its Release/Approve/Reject buttons) the instant it's needed.
  // Pin the freshest known copy of whatever row is expanded so it keeps
  // rendering regardless of the filter, until the operator collapses it or
  // picks a different row.
  const [pinnedTask, setPinnedTask] = useState<ReviewTaskView | null>(null);

  const queue = usePolling<ReviewTaskView[]>(
    () => listReviews(status === "all" ? undefined : { status }),
    POLL_MS,
    active,
  );
  const notEnabled = queue.error instanceof ApiError && queue.error.status === 422;

  const displayRows = useMemo(() => {
    if (!queue.data || !pinnedTask) return queue.data;
    const idx = queue.data.findIndex((t) => t.id === pinnedTask.id);
    if (idx === -1) return [pinnedTask, ...queue.data];
    // pinnedTask came straight from a mutation's own response, so it's
    // fresher than whatever the last poll returned for this row.
    const copy = [...queue.data];
    copy[idx] = pinnedTask;
    return copy;
  }, [queue.data, pinnedTask]);

  function selectRow(t: ReviewTaskView) {
    setExpanded((current) => (current === t.id ? null : t.id));
    setPinnedTask(null); // a fresh selection uses the queue's live copy
  }

  // A 409 means someone else (or another tab) acted on the task since this
  // list was fetched -- expected_version is now stale. Refresh and tell the
  // operator, rather than let the mutation silently no-op or the stale row
  // linger with action buttons that will just 409 again.
  async function handleConflict(err: unknown): Promise<boolean> {
    if (err instanceof ApiError && err.status === 409) {
      toast({
        title: "Task changed since you loaded it",
        description: "Someone else acted on it first — the queue has been refreshed.",
        tone: "error",
      });
      setPinnedTask(null);
      queue.refresh();
      return true;
    }
    return false;
  }

  const columns: Column<ReviewTaskView>[] = [
    {
      key: "id",
      header: "ID",
      sortAccessor: (t) => t.id,
      render: (t) => <span className="font-mono text-xs">#{t.id}</span>,
    },
    {
      key: "source",
      header: "Source",
      render: (t) => (
        <div className="min-w-0">
          <p className="truncate font-medium text-foreground">{t.source_request_id}</p>
          <p className="truncate text-xs text-muted-foreground">
            {t.schema_name} · {t.model_profile}
          </p>
        </div>
      ),
    },
    {
      key: "status",
      header: "Status",
      sortAccessor: (t) => t.status,
      render: (t) => <Badge tone={STATUS_TONE[t.status]}>{t.status}</Badge>,
    },
    {
      key: "priority",
      header: "Priority",
      sortAccessor: (t) => t.priority,
      render: (t) => (
        <span
          className="font-mono text-xs tabular-nums"
          title={t.priority_reasons.map((r) => `${r.code}: ${r.detail}`).join("\n") || undefined}
        >
          {t.priority.toFixed(2)}
        </span>
      ),
    },
    {
      key: "claimed_by",
      header: "Claimed by",
      render: (t) => t.claimed_by ?? <span className="text-muted-foreground">—</span>,
    },
    {
      key: "created_at",
      header: "Created",
      sortAccessor: (t) => t.created_at,
      render: (t) => (
        <span className="text-xs text-muted-foreground">
          {new Date(t.created_at).toLocaleString()}
        </span>
      ),
    },
  ];

  const rows = displayRows ?? [];

  return (
    <div>
      <PageHeader
        title="Review"
        subtitle="Extractions admitted for human review — low confidence, weak evidence, arithmetic mismatches, or model disagreement."
      />

      {notEnabled ? (
        <Alert tone="info">
          Not enabled — the review workflow needs DATABASE_URL set (persistence is what
          the queue is stored in).
        </Alert>
      ) : (
        <>
          <Toolbar>
            <Segmented
              value={status}
              onChange={(next) => {
                setStatus(next);
                // A pinned task's status can differ from the newly-selected
                // filter (that's the whole point of pinning) -- carrying it
                // across an unrelated filter switch would show it inside a
                // list it no longer belongs to.
                setExpanded(null);
                setPinnedTask(null);
              }}
              options={STATUS_FILTERS}
            />
          </Toolbar>

          <ResultLine
            shown={rows.length}
            total={rows.length}
            noun="tasks"
            onFetch={queue.refresh}
            fetching={queue.refreshing}
          />

          <Table
            columns={columns}
            rows={displayRows}
            loading={queue.loading}
            error={queue.error}
            getRowKey={(t) => String(t.id)}
            emptyLabel="Queue is empty"
            emptyDescription={
              status === "all" ? "No tasks have been admitted for review yet." : `No ${status} tasks.`
            }
            emptyIcon={<ClipboardCheck className="h-5 w-5" />}
            expandedKey={expanded != null ? String(expanded) : null}
            onRowClick={selectRow}
            renderExpanded={(t) => (
              <TaskDetail
                task={t}
                onChanged={(updated) => {
                  // A mutation started on this row before the operator
                  // collapsed it or moved to a different one can still
                  // resolve after that happened -- only pin if this row is
                  // still the one actually expanded.
                  if (expanded === updated.id) setPinnedTask(updated);
                  queue.refresh();
                }}
                onConflict={handleConflict}
              />
            )}
          />
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Schema-generated field editor: walks latest_prediction for wrapper objects
// (TextField/DateField/NumberField -- {value, confidence, evidence_ids} -- or
// MoneyField -- {amount, currency, confidence, evidence_ids}) at any depth,
// including through list indices, so line_items.0.description.value edits
// exactly like a top-level field. Replaces a free-text "type the dotted
// path yourself" form -- the editor already knows every correctable path,
// because it's reading them off the same JSON the correction eventually
// writes back into (matches the backend's structural path validation, see
// review.py's _validate_correction_paths).
// ---------------------------------------------------------------------------

interface FieldGroup {
  path: string;
  label: string;
  kind: "scalar" | "money";
  value: unknown;
  amount: unknown;
  currency: unknown;
  confidence: number;
  evidenceIds: string[];
}

function isFieldWrapper(node: unknown): node is Record<string, unknown> {
  if (typeof node !== "object" || node === null || Array.isArray(node)) return false;
  const obj = node as Record<string, unknown>;
  return (
    typeof obj.confidence === "number" &&
    Array.isArray(obj.evidence_ids) &&
    ("value" in obj || "amount" in obj)
  );
}

function walkFieldGroups(node: unknown, path: string[] = [], label: string[] = []): FieldGroup[] {
  if (isFieldWrapper(node)) {
    const evidenceIds = (node.evidence_ids as unknown[]).filter(
      (x): x is string => typeof x === "string",
    );
    const confidence = node.confidence as number;
    const fullPath = path.join(".");
    const fullLabel = label.slice(-2).join(" › ") || fullPath;
    if ("amount" in node) {
      return [
        {
          path: fullPath,
          label: fullLabel,
          kind: "money",
          value: undefined,
          amount: node.amount,
          currency: node.currency,
          confidence,
          evidenceIds,
        },
      ];
    }
    return [
      {
        path: fullPath,
        label: fullLabel,
        kind: "scalar",
        value: node.value,
        amount: undefined,
        currency: undefined,
        confidence,
        evidenceIds,
      },
    ];
  }
  if (Array.isArray(node)) {
    return node.flatMap((item, i) =>
      walkFieldGroups(item, [...path, String(i)], [...label, `#${i + 1}`]),
    );
  }
  if (node && typeof node === "object") {
    return Object.entries(node as Record<string, unknown>).flatMap(([key, value]) =>
      walkFieldGroups(value, [...path, key], [...label, key]),
    );
  }
  return [];
}

type EditSub = "value" | "amount" | "currency";

function FieldEditor({
  task,
  selectedPath,
  onSelect,
  onSubmit,
  busy,
}: {
  task: ReviewTaskView;
  selectedPath: string | null;
  onSelect: (group: FieldGroup) => void;
  onSubmit: (corrections: FieldCorrection[]) => Promise<void>;
  busy: boolean;
}) {
  const groups = useMemo(() => walkFieldGroups(task.latest_prediction), [task.latest_prediction]);
  const [edits, setEdits] = useState<Record<string, string>>({});

  // A submitted correction (or a claim/release) bumps task.version -- the
  // fields it touched are now reflected in latest_prediction itself, so any
  // in-progress edit text is stale and must be dropped, not resubmitted.
  useEffect(() => setEdits({}), [task.id, task.version]);

  const suggestionByKey = useMemo(() => {
    const map = new Map<string, FieldCorrection>();
    for (const s of task.suggested_corrections) map.set(s.field_path, s);
    return map;
  }, [task.suggested_corrections]);

  function originalOf(group: FieldGroup, sub: EditSub): unknown {
    return sub === "value" ? group.value : sub === "amount" ? group.amount : group.currency;
  }

  function editedValue(group: FieldGroup, sub: EditSub): string {
    const key = `${group.path}.${sub}`;
    if (key in edits) return edits[key];
    const raw = originalOf(group, sub);
    return raw == null ? "" : String(raw);
  }

  function setEditedValue(group: FieldGroup, sub: EditSub, raw: string) {
    setEdits((prev) => ({ ...prev, [`${group.path}.${sub}`]: raw }));
  }

  function applySuggestion(group: FieldGroup) {
    const sub: EditSub = group.kind === "money" ? "amount" : "value";
    const suggestion = suggestionByKey.get(`${group.path}.${sub}`);
    if (suggestion) setEditedValue(group, sub, String(suggestion.value));
  }

  function parseEdit(sub: EditSub, raw: string): unknown {
    if (raw === "") return null;
    if (sub === "currency") return raw;
    try {
      return JSON.parse(raw);
    } catch {
      return raw;
    }
  }

  function buildCorrections(): FieldCorrection[] {
    const out: FieldCorrection[] = [];
    for (const group of groups) {
      const subs: EditSub[] = group.kind === "money" ? ["amount", "currency"] : ["value"];
      for (const sub of subs) {
        const key = `${group.path}.${sub}`;
        if (!(key in edits)) continue;
        const parsed = parseEdit(sub, edits[key]);
        if (JSON.stringify(parsed) === JSON.stringify(originalOf(group, sub))) continue;
        out.push({ field_path: key, value: parsed });
      }
    }
    return out;
  }

  const pending = buildCorrections();
  const claimed = task.status === "claimed";

  return (
    <div className="space-y-1.5">
      {groups.length === 0 && (
        <p className="text-xs text-muted-foreground"><T>No correctable fields on this prediction.</T></p>
      )}
      {groups.map((group) => {
        const isSelected = selectedPath === group.path;
        const hasSuggestion =
          claimed && suggestionByKey.has(`${group.path}.${group.kind === "money" ? "amount" : "value"}`);
        return (
          <div
            key={group.path}
            onClick={() => onSelect(group)}
            className={`cursor-pointer rounded-md border px-3 py-2 text-sm transition ${
              isSelected ? "border-accent bg-accent/5" : "border-border hover:border-accent/40"
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-mono text-xs text-muted-foreground">
                {group.label}
              </span>
              <div className="flex shrink-0 items-center gap-1.5">
                {hasSuggestion && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      applySuggestion(group);
                    }}
                    className="rounded-full border border-accent/30 bg-accent/10 px-2 py-0.5 text-[10px] text-accent transition hover:bg-accent/20"
                  >
                    <T>Apply suggestion</T>
                  </button>
                )}
                <Badge tone={group.confidence < 0.5 ? "warn" : "neutral"}>
                  {(group.confidence * 100).toFixed(0)}%
                </Badge>
                {group.evidenceIds.length > 0 && (
                  <Search className="h-3 w-3 text-muted-foreground" aria-label="Has evidence" />
                )}
              </div>
            </div>
            {group.kind === "money" ? (
              <div className="mt-1 flex gap-2" onClick={(e) => e.stopPropagation()}>
                <TextInput
                  className="h-8 font-mono text-sm"
                  value={editedValue(group, "amount")}
                  disabled={!claimed}
                  onChange={(e) => setEditedValue(group, "amount", e.target.value)}
                />
                <TextInput
                  className="h-8 w-20 font-mono text-sm uppercase"
                  value={editedValue(group, "currency")}
                  disabled={!claimed}
                  onChange={(e) => setEditedValue(group, "currency", e.target.value)}
                />
              </div>
            ) : (
              <TextInput
                className="mt-1 h-8 font-mono text-sm"
                value={editedValue(group, "value")}
                disabled={!claimed}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => setEditedValue(group, "value", e.target.value)}
              />
            )}
          </div>
        );
      })}
      {claimed && (
        <div className="pt-1.5">
          <Button
            size="sm"
            loading={busy}
            disabled={pending.length === 0}
            onClick={() => void onSubmit(pending)}
          >
            Submit {pending.length} correction{pending.length === 1 ? "" : "s"}
          </Button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Evidence panel: the OCR blocks a selected field's evidence_ids point at.
// Blocks carry a bounding box but the Studio never persists page images
// (see settings.review_evidence_retention) -- so instead of a real page
// scan, this draws a synthetic layout: one box per OCR block positioned at
// its own bbox, sized to the page's own text extents. Approximates "where on
// the page" without ever storing a pixel of the source document. Falls back
// to a plain highlighted text list for blocks with no bbox (manual/inferred
// text has none).
// ---------------------------------------------------------------------------

function EvidencePanel({
  taskId,
  evidenceAvailable,
  selectedEvidenceIds,
}: {
  taskId: number;
  evidenceAvailable: boolean;
  selectedEvidenceIds: string[];
}) {
  const [evidence, setEvidence] = useState<ReviewEvidenceView | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setEvidence(null);
    setError(null);
    if (!evidenceAvailable) return;
    let cancelled = false;
    setLoading(true);
    getReviewEvidence(taskId)
      .then((view) => {
        if (!cancelled) setEvidence(view);
      })
      .catch((err) => {
        if (!cancelled) setError(err);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [taskId, evidenceAvailable]);

  if (!evidenceAvailable) {
    return (
      <p className="text-xs text-muted-foreground">
        <T>No OCR evidence was persisted for this task (a vision-only extraction has no OCR step, or evidence retention was off when it ran).</T>
      </p>
    );
  }
  if (loading) return <Skeleton className="h-40 w-full" />;
  if (error) {
    return <Alert tone="err">{toUserMessage(error, { fallback: "Could not load evidence." })}</Alert>;
  }
  if (!evidence || evidence.blocks.length === 0) {
    return <p className="text-xs text-muted-foreground"><T>No OCR blocks on this document.</T></p>;
  }

  const selected = new Set(selectedEvidenceIds);
  const byPage = new Map<number, OCRBlock[]>();
  for (const block of evidence.blocks) {
    const list = byPage.get(block.page) ?? [];
    list.push(block);
    byPage.set(block.page, list);
  }
  const pages = [...byPage.keys()].sort((a, b) => a - b);

  return (
    <div className="max-h-[32rem] space-y-4 overflow-y-auto pr-1">
      {pages.map((page) => {
        const blocks = byPage.get(page) ?? [];
        const withBbox = blocks.filter((b) => b.bbox);
        return (
          <div key={page}>
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Page {page}
            </p>
            {withBbox.length === 0 ? (
              <div className="space-y-1">
                {blocks.map((b) => (
                  <p
                    key={b.id}
                    className={`rounded px-1.5 py-0.5 text-xs ${
                      selected.has(b.id) ? "bg-accent/20 text-foreground" : "text-muted-foreground"
                    }`}
                  >
                    {b.text}
                  </p>
                ))}
              </div>
            ) : (
              <svg
                viewBox={`0 0 ${Math.max(...withBbox.map((b) => b.bbox!.x1))} ${Math.max(
                  ...withBbox.map((b) => b.bbox!.y1),
                )}`}
                className="w-full rounded border border-border bg-muted/10"
              >
                {withBbox.map((b) => (
                  <rect
                    key={b.id}
                    x={b.bbox!.x0}
                    y={b.bbox!.y0}
                    width={Math.max(b.bbox!.x1 - b.bbox!.x0, 1)}
                    height={Math.max(b.bbox!.y1 - b.bbox!.y0, 1)}
                    vectorEffect="non-scaling-stroke"
                    className={
                      selected.has(b.id)
                        ? "fill-accent/40 stroke-accent"
                        : "fill-foreground/5 stroke-border"
                    }
                    strokeWidth={selected.has(b.id) ? 1.5 : 0.5}
                  >
                    <title>{b.text}</title>
                  </rect>
                ))}
              </svg>
            )}
          </div>
        );
      })}
    </div>
  );
}

function TaskDetail({
  task,
  onChanged,
  onConflict,
}: {
  task: ReviewTaskView;
  onChanged: (updated: ReviewTaskView) => void;
  onConflict: (err: unknown) => Promise<boolean>;
}) {
  const { toast } = useToast();
  const [busy, setBusy] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<FieldGroup | null>(null);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => setSelectedGroup(null), [task.id]);

  async function run(action: () => Promise<ReviewTaskView>) {
    setBusy(true);
    try {
      const updated = await action();
      onChanged(updated);
    } catch (err) {
      if (!(await onConflict(err))) {
        toast({
          title: "Action failed",
          description: toUserMessage(err, { fallback: "Action failed." }),
          tone: "error",
        });
      }
    } finally {
      setBusy(false);
    }
  }

  async function submitCorrections(corrections: FieldCorrection[]) {
    if (corrections.length === 0) return;
    await run(() => correctReview(task.id, task.version, corrections));
  }

  return (
    <div className="space-y-4">
      {(task.validation_errors.length > 0 || task.validation_warnings.length > 0) && (
        <div className="space-y-1.5">
          {task.validation_errors.map((e, i) => (
            <Alert key={`err-${i}`} tone="err">
              {e}
            </Alert>
          ))}
          {task.validation_warnings.map((w, i) => (
            <Alert key={`warn-${i}`} tone="warn">
              {w}
            </Alert>
          ))}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <T>Fields</T>
          </p>
          <FieldEditor
            task={task}
            selectedPath={selectedGroup?.path ?? null}
            onSelect={setSelectedGroup}
            onSubmit={submitCorrections}
            busy={busy}
          />
        </div>
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Evidence{selectedGroup ? ` — ${selectedGroup.label}` : ""}
          </p>
          <EvidencePanel
            taskId={task.id}
            evidenceAvailable={task.evidence_available}
            selectedEvidenceIds={selectedGroup?.evidenceIds ?? []}
          />
        </div>
      </div>

      {task.corrections.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            <T>Correction history</T>
          </p>
          <div className="space-y-1.5">
            {task.corrections.map((c) => (
              <div
                key={c.id}
                className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs"
              >
                <span className="text-muted-foreground">
                  {new Date(c.created_at).toLocaleString()} · {c.reviewer_id}
                </span>
                {c.comment && <p className="mt-0.5 text-foreground/90">{c.comment}</p>}
              </div>
            ))}
          </div>
        </div>
      )}

      <button
        type="button"
        onClick={() => setShowRaw((v) => !v)}
        className="text-xs text-muted-foreground transition hover:text-foreground"
      >
        {showRaw ? "Hide" : "Show"} raw JSON
      </button>
      {showRaw && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              <T>Original prediction</T>
            </p>
            <JsonView value={task.original_prediction} maxHeight="16rem" />
          </div>
          <div>
            <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              <T>Latest prediction</T>
            </p>
            <JsonView value={task.latest_prediction} maxHeight="16rem" />
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {task.status === "pending" && (
          <Button
            size="sm"
            loading={busy}
            onClick={() => void run(() => claimReview(task.id, task.version))}
          >
            <Check className="h-3.5 w-3.5" />
            Claim
          </Button>
        )}
        {task.status === "claimed" && (
          <>
            <Button
              size="sm"
              variant="secondary"
              loading={busy}
              onClick={() => void run(() => releaseReview(task.id, task.version))}
            >
              <Undo2 className="h-3.5 w-3.5" />
              Release
            </Button>
            <Button
              size="sm"
              loading={busy}
              onClick={() => void run(() => approveReview(task.id, task.version))}
            >
              <Check className="h-3.5 w-3.5" />
              Approve
            </Button>
            <Button
              size="sm"
              variant="danger"
              loading={busy}
              onClick={() => void run(() => rejectReview(task.id, task.version))}
            >
              <X className="h-3.5 w-3.5" />
              Reject
            </Button>
          </>
        )}
        {(task.status === "approved" || task.status === "rejected") && (
          <p className="text-xs text-muted-foreground">
            Decided {task.decided_at ? new Date(task.decided_at).toLocaleString() : "—"}
            {task.decided_by && ` by ${task.decided_by}`}
            {task.decision_comment && ` — "${task.decision_comment}"`}
          </p>
        )}
      </div>
    </div>
  );
}
