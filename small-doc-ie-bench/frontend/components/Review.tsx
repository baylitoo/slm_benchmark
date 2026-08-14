"use client";

import { useMemo, useState } from "react";
import {
  ClipboardCheck,
  Check,
  X,
  Undo2,
  PlusCircle,
  Trash2,
} from "lucide-react";
import {
  approveReview,
  claimReview,
  correctReview,
  listReviews,
  rejectReview,
  releaseReview,
  ApiError,
  type FieldCorrection,
  type ReviewStatus,
  type ReviewTaskView,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { toUserMessage } from "@/lib/errors";
import { useToast } from "./Toast";
import { Alert, Badge, Button, Card, Field, Segmented, TextInput, type BadgeTone } from "./ui";
import { JsonView } from "./JsonView";
import { PageHeader } from "./patterns/PageHeader";
import { Toolbar } from "./patterns/Toolbar";
import { ResultLine } from "./patterns/ResultLine";
import { Table, type Column } from "./patterns/Table";

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
            <Segmented value={status} onChange={setStatus} options={STATUS_FILTERS} />
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
                  setPinnedTask(updated);
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
  const [pending, setPending] = useState<FieldCorrection[]>([]);
  const [fieldPath, setFieldPath] = useState("");
  const [value, setValue] = useState("");

  function addPending() {
    if (!fieldPath.trim()) return;
    // Best-effort JSON parse so a number/bool/null corrects with the right
    // type; falls back to the raw string, the common case (a fixed invoice
    // number, a corrected date) doesn't need quoting.
    let parsed: unknown = value;
    try {
      parsed = JSON.parse(value);
    } catch {
      /* not JSON -- keep it as the raw string */
    }
    setPending((p) => [...p, { field_path: fieldPath.trim(), value: parsed }]);
    setFieldPath("");
    setValue("");
  }

  function removePending(index: number) {
    setPending((p) => p.filter((_, i) => i !== index));
  }

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

  async function submitCorrections() {
    if (pending.length === 0) return;
    await run(() => correctReview(task.id, task.version, pending));
    setPending([]);
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
            Original prediction
          </p>
          <JsonView value={task.original_prediction} maxHeight="16rem" />
        </div>
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Latest prediction
          </p>
          <JsonView value={task.latest_prediction} maxHeight="16rem" />
        </div>
      </div>

      {task.status === "claimed" && task.suggested_corrections.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Suggested corrections
          </p>
          <div className="flex flex-wrap gap-1.5">
            {task.suggested_corrections.map((s, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setPending((p) => [...p, s])}
                className="rounded-full border border-accent/30 bg-accent/10 px-2.5 py-1 font-mono text-xs text-accent transition hover:bg-accent/20"
              >
                {s.field_path} → {JSON.stringify(s.value)}
              </button>
            ))}
          </div>
        </div>
      )}

      {task.corrections.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Correction history
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

      {task.status === "claimed" && (
        <Card
          title="Add a correction"
          subtitle="Field path + value; add several before submitting as one batch."
          bodyClassName="space-y-3"
        >
          <div className="grid gap-3 sm:grid-cols-[1fr_1fr_auto]">
            <Field label="Field path" hint="e.g. total_ttc.value">
              <TextInput
                value={fieldPath}
                onChange={(e) => setFieldPath(e.target.value)}
                placeholder="field.path"
              />
            </Field>
            <Field label="Value" hint="JSON if it parses, else a raw string">
              <TextInput
                value={value}
                onChange={(e) => setValue(e.target.value)}
                placeholder='123.45 or "text"'
              />
            </Field>
            <div className="flex items-end">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={addPending}
                disabled={!fieldPath.trim()}
              >
                <PlusCircle className="h-3.5 w-3.5" />
                Add
              </Button>
            </div>
          </div>

          {pending.length > 0 && (
            <div className="space-y-1.5">
              {pending.map((c, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between gap-2 rounded-md border border-border bg-muted/20 px-3 py-1.5 text-xs"
                >
                  <span className="min-w-0 truncate font-mono">
                    {c.field_path} = {JSON.stringify(c.value)}
                  </span>
                  <button
                    type="button"
                    onClick={() => removePending(i)}
                    aria-label="Remove"
                    className="shrink-0 text-muted-foreground hover:text-foreground"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
              <Button size="sm" loading={busy} onClick={() => void submitCorrections()}>
                Submit {pending.length} correction{pending.length === 1 ? "" : "s"}
              </Button>
            </div>
          )}
        </Card>
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
