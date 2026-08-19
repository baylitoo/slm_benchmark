"use client";

import { useMemo, useRef, useState } from "react";
import { AlertCircle, Download, FileStack, Upload } from "lucide-react";
import {
  ApiError,
  ApiUnavailable,
  downloadBatchResults,
  fileToBase64,
  getBatch,
  getDeployments,
  listBatches,
  selectableDeployments,
  triggerBatchExtract,
  type BatchDocumentInput,
  type BatchItemView,
  type BatchRunDetail,
  type BatchRunSummary,
  type DeploymentRecord,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useAsync } from "@/lib/useAsync";
import { useToast } from "../Toast";
import { Alert, Badge, Button, Card, Field, Select, Spinner, TextInput, type BadgeTone } from "../ui";
import { PageHeader } from "../patterns/PageHeader";
import { ResultLine } from "../patterns/ResultLine";
import { Table, type Column } from "../patterns/Table";
import { JsonView } from "../JsonView";

// ---------------------------------------------------------------------------
// Batch view — "extract from MY invoices": N documents through one schema +
// model as one durable job with per-document state. Submit a zip or pick
// files; the table below is the durable record (GET /v1/studio/batches):
// live done/failed counts while running, per-item results/errors on expand,
// JSONL/CSV download when settled. Mirrors deploy/DownloadsView.
// ---------------------------------------------------------------------------

const POLL_MS = 4000;
const DETAIL_POLL_MS = 3000;
const ZIP_TYPES = new Set(["application/zip", "application/x-zip-compressed"]);

function statusTone(status: BatchRunSummary["status"]): BadgeTone {
  if (status === "completed") return "ok";
  if (status === "failed") return "err";
  return "warn";
}

function itemTone(status: BatchItemView["status"]): BadgeTone {
  if (status === "done") return "ok";
  if (status === "failed") return "err";
  return "neutral";
}

/** done/failed/total as a bar + counts -- the row's own denormalized state,
 * refreshed by the list poll; no per-row side-channel needed. */
function ProgressCell({ r }: { r: BatchRunSummary }) {
  const total = Math.max(r.total_items, 1);
  const done = (r.done_items / total) * 100;
  const failed = (r.failed_items / total) * 100;
  return (
    <div className="w-40">
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span className="font-medium text-foreground tabular-nums">
          {r.done_items + r.failed_items}/{r.total_items}
        </span>
        {r.failed_items > 0 && (
          <span className="text-rose-500 tabular-nums">{r.failed_items} failed</span>
        )}
      </div>
      <div className="mt-0.5 flex h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div className="h-full bg-accent transition-all duration-300" style={{ width: `${done}%` }} />
        <div className="h-full bg-rose-500 transition-all duration-300" style={{ width: `${failed}%` }} />
      </div>
    </div>
  );
}

function BatchDetail({ eventId }: { eventId: string }) {
  const detail = usePolling<BatchRunDetail>(() => getBatch(eventId), DETAIL_POLL_MS, true);
  const [open, setOpen] = useState<number | null>(null);
  const b = detail.data;
  if (!b) {
    return detail.error ? (
      <Alert tone="err">Could not load this batch.</Alert>
    ) : (
      <Spinner className="h-4 w-4" />
    );
  }
  const columns: Column<BatchItemView>[] = [
    {
      key: "position",
      header: "#",
      render: (i) => <span className="font-mono text-xs tabular-nums">{i.position + 1}</span>,
    },
    {
      key: "filename",
      header: "Document",
      render: (i) => <span className="font-mono text-xs">{i.filename}</span>,
    },
    {
      key: "status",
      header: "Status",
      render: (i) => <Badge tone={itemTone(i.status)}>{i.status}</Badge>,
    },
    {
      key: "latency",
      header: "Time",
      render: (i) => (
        <span className="text-xs text-muted-foreground tabular-nums">
          {i.latency_ms != null ? `${i.latency_ms} ms` : "—"}
        </span>
      ),
    },
    {
      key: "detail",
      header: "",
      render: (i) =>
        i.status === "failed" ? (
          <span className="flex items-center gap-1 text-xs text-rose-500" title={i.error ?? ""}>
            <AlertCircle className="h-3.5 w-3.5 shrink-0" />
            <span className="max-w-[28rem] truncate">{i.error}</span>
          </span>
        ) : i.status === "done" ? (
          <button
            type="button"
            className="text-xs text-accent hover:underline"
            onClick={(e) => {
              e.stopPropagation();
              setOpen((cur) => (cur === i.position ? null : i.position));
            }}
          >
            {open === i.position ? "Hide" : "Show"} result
          </button>
        ) : null,
    },
  ];
  return (
    <div className="space-y-3">
      {b.error && <Alert tone="err">{b.error}</Alert>}
      <Table
        columns={columns}
        rows={b.items}
        getRowKey={(i) => String(i.position)}
        emptyLabel="No documents"
        expandedKey={open != null ? String(open) : null}
        renderExpanded={(i) => <JsonView value={i.result} maxHeight="20rem" />}
      />
    </div>
  );
}

export function BatchView({ active = true }: { active?: boolean }) {
  const { toast } = useToast();
  const batches = usePolling<BatchRunSummary[]>(listBatches, POLL_MS, active);
  const deployments = useAsync<DeploymentRecord[]>("batch-deployments", getDeployments);
  const [expanded, setExpanded] = useState<string | null>(null);

  // -- submit form ---------------------------------------------------------
  const [name, setName] = useState("");
  const [schemaName, setSchemaName] = useState("invoice");
  const [deployment, setDeployment] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const deploymentNames = useMemo(
    () =>
      selectableDeployments(deployments.data ?? [])
        .filter((d) => d.spec?.launch?.runtime !== "encoder")
        .map((d) => d.spec?.name)
        .filter((n): n is string => Boolean(n)),
    [deployments.data],
  );

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (files.length === 0) {
      setError("Pick a zip of documents, or one or more document files.");
      return;
    }
    setSubmitting(true);
    try {
      const single = files.length === 1 ? files[0] : null;
      const isZip =
        single != null && (ZIP_TYPES.has(single.type) || single.name.toLowerCase().endsWith(".zip"));
      const payload = isZip
        ? { zip_b64: await fileToBase64(single) }
        : {
            documents: (await Promise.all(
              files.map(async (f) => ({ filename: f.name, content_b64: await fileToBase64(f) })),
            )) as BatchDocumentInput[],
          };
      const res = await triggerBatchExtract({
        ...payload,
        name: name.trim() || undefined,
        schema_name: schemaName.trim() || "invoice",
        deployment: deployment || undefined,
      });
      toast({ title: "Batch started", description: res.channel, tone: "success" });
      setFiles([]);
      setName("");
      if (fileInput.current) fileInput.current.value = "";
      batches.refresh();
    } catch (err) {
      const msg =
        err instanceof ApiUnavailable
          ? "The batch endpoint isn't reachable. Is the backend running?"
          : err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : "Something went wrong.";
      setError(msg);
      toast({ title: "Batch failed to start", description: msg, tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  async function download(r: BatchRunSummary, fmt: "jsonl" | "csv") {
    try {
      await downloadBatchResults(r, fmt);
    } catch (err) {
      toast({
        title: "Download failed",
        description: err instanceof Error ? err.message : "Download failed.",
        tone: "error",
      });
    }
  }

  const columns: Column<BatchRunSummary>[] = [
    {
      key: "name",
      header: "Batch",
      render: (r) => (
        <div className="min-w-0">
          <p className="truncate font-medium text-foreground">{r.name}</p>
          <p className="truncate font-mono text-[11px] text-muted-foreground">
            {r.schema_name}
            {r.model_selector ? ` · ${r.model_selector}` : ""}
          </p>
        </div>
      ),
    },
    {
      key: "status",
      header: "Status",
      sortAccessor: (r) => r.status,
      render: (r) => <Badge tone={statusTone(r.status)}>{r.status}</Badge>,
    },
    { key: "progress", header: "Progress", render: (r) => <ProgressCell r={r} /> },
    {
      key: "created_at",
      header: "Started",
      sortAccessor: (r) => r.created_at,
      render: (r) => (
        <span className="text-xs text-muted-foreground">{new Date(r.created_at).toLocaleString()}</span>
      ),
    },
    {
      key: "results",
      header: "",
      className: "text-right",
      render: (r) =>
        r.artifacts.length > 0 ? (
          <div className="flex justify-end gap-1" onClick={(e) => e.stopPropagation()}>
            <Button size="sm" variant="ghost" onClick={() => void download(r, "jsonl")} title="Lossless per-document results">
              <Download className="h-3.5 w-3.5" /> JSONL
            </Button>
            <Button size="sm" variant="ghost" onClick={() => void download(r, "csv")} title="Flattened for a spreadsheet">
              <Download className="h-3.5 w-3.5" /> CSV
            </Button>
          </div>
        ) : null,
    },
  ];

  const rows = batches.data ?? [];

  return (
    <div>
      <PageHeader
        title="Batch extraction"
        subtitle="Many documents, one schema, one model — as a single durable job with per-document results. A bad document is recorded and the rest continue."
      />

      <Card
        icon={<FileStack className="h-5 w-5" />}
        title="New batch"
        subtitle="Drop a zip (.pdf/.png/.jpg/.tif/.txt inside; anything else is skipped) or pick several files."
        className="mb-4"
      >
        <form onSubmit={onSubmit} className="grid gap-4 md:grid-cols-[1fr_1fr]">
          <Field label="Documents" required hint={files.length ? `${files.length} file${files.length === 1 ? "" : "s"} selected` : "One .zip, or several documents."}>
            <div className="flex items-center gap-2">
              <input
                ref={fileInput}
                type="file"
                multiple
                accept=".zip,.pdf,.png,.jpg,.jpeg,.tif,.tiff,.txt"
                className="hidden"
                onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
                aria-label="Batch documents"
              />
              <Button type="button" variant="secondary" size="sm" onClick={() => fileInput.current?.click()}>
                <Upload className="h-3.5 w-3.5" /> Choose files
              </Button>
              {files.length > 0 && (
                <span className="truncate text-xs text-muted-foreground">
                  {files.length === 1 ? files[0].name : files.map((f) => f.name).slice(0, 3).join(", ") + (files.length > 3 ? ", …" : "")}
                </span>
              )}
            </div>
          </Field>
          <Field label="Name" hint="Optional label for the list.">
            <TextInput value={name} onChange={(e) => setName(e.target.value)} placeholder="Q3 invoices" />
          </Field>
          <Field label="Schema" required>
            <TextInput value={schemaName} onChange={(e) => setSchemaName(e.target.value)} placeholder="invoice" />
          </Field>
          <Field label="Deployment" hint="Applies to every document. Empty = backend default.">
            <Select value={deployment} onChange={(e) => setDeployment(e.target.value)} aria-label="Deployment">
              <option value="">(default)</option>
              {deploymentNames.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </Select>
          </Field>
          {error && (
            <div className="md:col-span-2">
              <Alert tone="err">{error}</Alert>
            </div>
          )}
          <div className="md:col-span-2">
            <Button type="submit" loading={submitting} disabled={files.length === 0}>
              <FileStack className="h-4 w-4" /> Start batch
            </Button>
          </div>
        </form>
      </Card>

      <ResultLine shown={rows.length} total={rows.length} noun="batches" onFetch={batches.refresh} fetching={batches.refreshing} />
      <Table
        columns={columns}
        rows={batches.data}
        loading={batches.loading}
        error={batches.error}
        getRowKey={(r) => r.event_id}
        emptyLabel="No batches yet"
        emptyDescription="Start one above — results appear here as documents finish."
        emptyIcon={<FileStack className="h-5 w-5" />}
        expandedKey={expanded}
        onRowClick={(r) => setExpanded((cur) => (cur === r.event_id ? null : r.event_id))}
        renderExpanded={(r) => <BatchDetail eventId={r.event_id} />}
      />
    </div>
  );
}
