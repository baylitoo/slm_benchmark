"use client";

import { useEffect, useState } from "react";
import { AlertCircle, CalendarClock, Pause, Play, Trash2, Zap } from "lucide-react";
import {
  createBatchSchedule,
  deleteBatchSchedule,
  listBatchSchedules,
  runBatchScheduleNow,
  updateBatchSchedule,
  type BatchRunSummary,
  type BatchScheduleInterval,
  type BatchScheduleView,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useToast } from "../Toast";
import { Alert, Badge, Button, Card, Field, Select, TextInput } from "../ui";
import { Table, type Column } from "../patterns/Table";

// ---------------------------------------------------------------------------
// Batch schedules — recurring re-runs of a batch's stored documents. A saved
// schedule references a source batch; a once-a-minute worker cron fires each
// due schedule as a NORMAL batch (it lands in the batches table above like
// any other run), then advances next_run_at. Creation starts from a settled
// batch row's "Schedule" action in BatchView, which hands the source batch
// down here as `source`.
// ---------------------------------------------------------------------------

const POLL_MS = 10000;
const MIN_EVERY_N = 15;

const INTERVAL_OPTIONS: { value: BatchScheduleInterval; label: string }[] = [
  { value: "hourly", label: "Hourly" },
  { value: "daily", label: "Daily" },
  { value: "weekly", label: "Weekly" },
  { value: "every_n_minutes", label: "Every N minutes" },
];

export function intervalLabel(s: Pick<BatchScheduleView, "interval" | "every_n_minutes">): string {
  if (s.interval === "every_n_minutes") return `every ${s.every_n_minutes ?? "?"} min`;
  return s.interval;
}

function when(iso: string | null): string {
  return iso ? new Date(iso).toLocaleString() : "—";
}

export function BatchSchedules({
  source,
  onSourceHandled,
  active = true,
}: {
  /** A settled batch picked in BatchView ("Schedule" row action) — opens the
   * create form prefilled from it. */
  source: BatchRunSummary | null;
  /** Called once the create form is submitted or dismissed. */
  onSourceHandled: () => void;
  active?: boolean;
}) {
  const { toast } = useToast();
  const schedules = usePolling<BatchScheduleView[]>(listBatchSchedules, POLL_MS, active);

  // -- create form (opened by a source batch pick) ---------------------------
  const [name, setName] = useState("");
  const [interval, setInterval_] = useState<BatchScheduleInterval>("daily");
  const [everyN, setEveryN] = useState(60);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (source) {
      setName(`re-run: ${source.name}`);
      setError(null);
    }
  }, [source]);

  async function onCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!source) return;
    setCreating(true);
    setError(null);
    try {
      await createBatchSchedule({
        source_event_id: source.event_id,
        name: name.trim() || undefined,
        interval,
        every_n_minutes: interval === "every_n_minutes" ? everyN : undefined,
      });
      toast({ title: "Schedule created", description: intervalLabel({ interval, every_n_minutes: everyN }), tone: "success" });
      onSourceHandled();
      schedules.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create the schedule.");
    } finally {
      setCreating(false);
    }
  }

  async function toggle(s: BatchScheduleView) {
    try {
      await updateBatchSchedule(s.id, { enabled: !s.enabled });
      toast({
        title: s.enabled ? "Schedule paused" : "Schedule enabled",
        description: s.name,
        tone: "success",
      });
      schedules.refresh();
    } catch (err) {
      toast({
        title: "Update failed",
        description: err instanceof Error ? err.message : "Update failed.",
        tone: "error",
      });
    }
  }

  async function runNow(s: BatchScheduleView) {
    try {
      const res = await runBatchScheduleNow(s.id);
      toast({ title: "Scheduled batch started", description: res.channel, tone: "success" });
      schedules.refresh();
    } catch (err) {
      toast({
        title: "Run failed to start",
        description: err instanceof Error ? err.message : "Run failed.",
        tone: "error",
      });
    }
  }

  async function remove(s: BatchScheduleView) {
    try {
      await deleteBatchSchedule(s.id);
      toast({ title: "Schedule deleted", description: s.name, tone: "success" });
      schedules.refresh();
    } catch (err) {
      toast({
        title: "Delete failed",
        description: err instanceof Error ? err.message : "Delete failed.",
        tone: "error",
      });
    }
  }

  const columns: Column<BatchScheduleView>[] = [
    {
      key: "name",
      header: "Schedule",
      render: (s) => (
        <div className="min-w-0">
          <p className="truncate font-medium text-foreground">{s.name}</p>
          <p className="truncate font-mono text-[11px] text-muted-foreground">
            {s.schema_name}
            {s.selectors.routing_policy
              ? ` · policy:${s.selectors.routing_policy}`
              : s.selectors.deployment || s.selectors.model_profile
                ? ` · ${s.selectors.deployment ?? s.selectors.model_profile}`
                : ""}
          </p>
        </div>
      ),
    },
    {
      key: "interval",
      header: "Interval",
      render: (s) => <span className="text-xs text-foreground">{intervalLabel(s)}</span>,
    },
    {
      key: "status",
      header: "Status",
      sortAccessor: (s) => (s.enabled ? "enabled" : "paused"),
      render: (s) => (
        <div className="flex items-center gap-1.5">
          <Badge tone={s.enabled ? "ok" : "neutral"}>{s.enabled ? "enabled" : "paused"}</Badge>
          {s.last_error && (
            <span className="flex items-center text-rose-500" title={s.last_error}>
              <AlertCircle className="h-3.5 w-3.5" />
            </span>
          )}
        </div>
      ),
    },
    {
      key: "next_run_at",
      header: "Next run",
      sortAccessor: (s) => s.next_run_at,
      render: (s) => (
        <span className="text-xs text-muted-foreground">
          {s.enabled ? when(s.next_run_at) : "—"}
        </span>
      ),
    },
    {
      key: "last_run_at",
      header: "Last run",
      sortAccessor: (s) => s.last_run_at ?? "",
      render: (s) => <span className="text-xs text-muted-foreground">{when(s.last_run_at)}</span>,
    },
    {
      key: "actions",
      header: "",
      className: "text-right",
      render: (s) => (
        <div className="flex justify-end gap-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void runNow(s)}
            title="Fire this schedule now (does not change its cadence)"
          >
            <Zap className="h-3.5 w-3.5" /> Run now
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void toggle(s)}
            title={s.enabled ? "Pause this schedule" : "Resume this schedule"}
          >
            {s.enabled ? (
              <>
                <Pause className="h-3.5 w-3.5" /> Pause
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5" /> Resume
              </>
            )}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void remove(s)}
            title="Delete this schedule (already-run batches are kept)"
          >
            <Trash2 className="h-3.5 w-3.5" /> Delete
          </Button>
        </div>
      ),
    },
  ];

  return (
    <Card
      icon={<CalendarClock className="h-5 w-5" />}
      title="Schedules"
      subtitle="Recurring re-runs of a batch's stored documents — pick a settled batch above (“Schedule”) to create one. Each firing appears in the batches list as a normal run."
      className="mt-6"
    >
      {source && (
        <form
          onSubmit={onCreate}
          className="mb-4 grid gap-4 rounded-lg border border-border bg-muted/30 p-4 md:grid-cols-[1fr_1fr]"
        >
          <div className="md:col-span-2 text-sm text-foreground">
            Re-run the {source.total_items} document{source.total_items === 1 ? "" : "s"} of{" "}
            <span className="font-medium">{source.name}</span> on a schedule.
          </div>
          <Field label="Name">
            <TextInput value={name} onChange={(e) => setName(e.target.value)} placeholder="nightly re-run" />
          </Field>
          <Field label="Interval" required>
            <div className="flex items-center gap-2">
              <Select
                value={interval}
                onChange={(e) => setInterval_(e.target.value as BatchScheduleInterval)}
                aria-label="Interval"
              >
                {INTERVAL_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </Select>
              {interval === "every_n_minutes" && (
                <TextInput
                  type="number"
                  min={MIN_EVERY_N}
                  value={everyN}
                  onChange={(e) => setEveryN(Number(e.target.value))}
                  aria-label="Minutes between runs"
                  className="w-24"
                />
              )}
            </div>
          </Field>
          {error && (
            <div className="md:col-span-2">
              <Alert tone="err">{error}</Alert>
            </div>
          )}
          <div className="flex gap-2 md:col-span-2">
            <Button type="submit" loading={creating}>
              <CalendarClock className="h-4 w-4" /> Create schedule
            </Button>
            <Button type="button" variant="secondary" onClick={onSourceHandled}>
              Cancel
            </Button>
          </div>
        </form>
      )}

      <Table
        columns={columns}
        rows={schedules.data}
        loading={schedules.loading}
        error={schedules.error}
        getRowKey={(s) => s.id}
        emptyLabel="No schedules yet"
        emptyDescription="Use “Schedule” on a settled batch to re-run its documents automatically."
        emptyIcon={<CalendarClock className="h-5 w-5" />}
      />
    </Card>
  );
}
