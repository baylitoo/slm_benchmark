"use client";

import { useEffect, useMemo, useState } from "react";
import { FileText, Play, Sparkles, Upload, AlertCircle } from "lucide-react";
import {
  triggerExtract,
  getDeployments,
  selectableDeployments,
  fileToBase64,
  ApiError,
  ApiUnavailable,
  type TriggerResponse,
  type ExtractRequest,
  type DeploymentRecord,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { cn } from "@/lib/cn";
import { useToast } from "./Toast";
import { Button, Card, Field, Select, TextArea, TextInput, Badge } from "./ui";
import { ResultPanel } from "./ResultPanel";

type InputMode = "text" | "file";

const DEPLOY_POLL_MS = 4000;

export function Playground({ active = true }: { active?: boolean }) {
  const { toast } = useToast();
  const [inputMode, setInputMode] = useState<InputMode>("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [schemaName, setSchemaName] = useState("invoice");
  const [selectedDeployment, setSelectedDeployment] = useState<string>("");
  const [ocrBackend, setOcrBackend] = useState("");
  const [language, setLanguage] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trigger, setTrigger] = useState<TriggerResponse | null>(null);

  // Live deployments, sourced from the same endpoint the Deploy tab uses
  // (GET /v1/serving/deployments). Polling is paused while the tab is hidden.
  const deployments = usePolling<DeploymentRecord[]>(getDeployments, DEPLOY_POLL_MS, active);
  const ready = useMemo(
    () => selectableDeployments(deployments.data ?? []),
    [deployments.data],
  );
  const readyNames = useMemo(
    () => ready.map((d) => d.spec?.name ?? "").filter(Boolean),
    [ready],
  );

  // Pre-select the first ready deployment so an explicit `deployment` is always
  // sent when one exists; resync if the current pick disappears from the list.
  useEffect(() => {
    if (readyNames.length === 0) {
      if (selectedDeployment !== "") setSelectedDeployment("");
      return;
    }
    if (!readyNames.includes(selectedDeployment)) {
      setSelectedDeployment(readyNames[0]);
    }
  }, [readyNames, selectedDeployment]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setTrigger(null);

    const payload: ExtractRequest = { schema_name: schemaName || "invoice" };
    // Send ONLY the deployment selector (its value is a DeploymentRecord
    // spec.name); never model_profile. Empty selection → backend default.
    if (selectedDeployment) payload.deployment = selectedDeployment;
    if (ocrBackend.trim()) payload.ocr_backend = ocrBackend.trim();
    if (language.trim()) payload.language = language.trim();

    try {
      if (inputMode === "text") {
        if (!text.trim()) {
          setError("Paste some document text first.");
          return;
        }
        payload.text = text;
      } else {
        if (!file) {
          setError("Choose a PDF or image file first.");
          return;
        }
        payload.content_b64 = await fileToBase64(file);
        payload.filename = file.name;
      }

      setSubmitting(true);
      const res = await triggerExtract(payload);
      setTrigger(res);
      toast({ title: "Extraction started", description: res.channel, tone: "success" });
    } catch (e) {
      const msg =
        e instanceof ApiUnavailable
          ? "The extract endpoint isn't reachable. Is the backend running and NEXT_PUBLIC_API_BASE correct?"
          : e instanceof ApiError
            ? e.message
            : e instanceof Error
              ? e.message
              : "Something went wrong.";
      setError(msg);
      toast({ title: "Extraction failed", description: msg, tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card
        icon={<Sparkles className="h-5 w-5" />}
        title="Extract"
        subtitle="Paste text or upload a document, then run extraction."
      >
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="inline-flex rounded-lg border border-border bg-muted p-0.5 text-sm">
            {(["text", "file"] as InputMode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setInputMode(m)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 transition",
                  inputMode === m
                    ? "bg-card text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {m === "text" ? <FileText className="h-4 w-4" /> : <Upload className="h-4 w-4" />}
                {m === "text" ? "Paste text" : "Upload file"}
              </button>
            ))}
          </div>

          {inputMode === "text" ? (
            <Field label="Document text">
              <TextArea
                rows={10}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste the raw document text here…"
              />
            </Field>
          ) : (
            <Field label="Document file" hint="PDF or image; encoded to base64 in your browser.">
              <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-muted/30 px-4 py-8 text-center transition hover:border-accent hover:bg-muted/50">
                <Upload className="h-6 w-6 text-muted-foreground" />
                <span className="text-sm text-foreground">
                  {file ? file.name : "Click to choose a PDF or image"}
                </span>
                {file && (
                  <span className="text-xs text-muted-foreground">
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                )}
                <input
                  type="file"
                  accept=".pdf,image/*"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  className="sr-only"
                />
              </label>
            </Field>
          )}

          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Schema name">
              <TextInput
                value={schemaName}
                onChange={(e) => setSchemaName(e.target.value)}
                placeholder="invoice"
              />
            </Field>
            <Field label="Deployment" hint="Live runtime to route this extraction to.">
              <DeploymentSelect
                deployments={deployments}
                ready={ready}
                value={selectedDeployment}
                onChange={setSelectedDeployment}
              />
            </Field>
            <Field label="OCR backend" hint="Optional — for file uploads.">
              <TextInput
                value={ocrBackend}
                onChange={(e) => setOcrBackend(e.target.value)}
                placeholder="(default)"
              />
            </Field>
            <Field label="Language" hint="Optional ISO code.">
              <TextInput
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                placeholder="(auto)"
              />
            </Field>
          </div>

          {error && (
            <p className="flex items-start gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              {error}
            </p>
          )}

          <Button type="submit" loading={submitting}>
            <Play className="h-4 w-4" />
            {submitting ? "Submitting…" : "Run extraction"}
          </Button>
        </form>
      </Card>

      <Card
        icon={<Play className="h-5 w-5" />}
        title="Live result"
        subtitle="Realtime stream when available, polling otherwise."
        actions={trigger ? <Badge tone="info">{trigger.channel}</Badge> : undefined}
      >
        {trigger ? (
          <ResultPanel trigger={trigger} noun="extraction" />
        ) : (
          <p className="text-sm text-muted-foreground">
            Run an extraction to see live progress and the resulting JSON here.
          </p>
        )}
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Deployment selector — a dropdown of live (ready) deployments. Falls back to
// clear, non-crashing states for loading / unavailable / empty.
// ---------------------------------------------------------------------------

function DeploymentSelect({
  deployments,
  ready,
  value,
  onChange,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
  ready: DeploymentRecord[];
  value: string;
  onChange: (name: string) => void;
}) {
  // First load, nothing cached yet.
  if (deployments.loading && !deployments.data) {
    return (
      <Select value="" disabled>
        <option value="">Loading deployments…</option>
      </Select>
    );
  }

  // Endpoint missing (404/501 on older builds) or otherwise errored, and we
  // have no data to fall back on: leave the selector empty so the backend
  // default applies, and explain why.
  if (deployments.error && !deployments.data) {
    return (
      <p className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
        <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        Deployments unavailable — is the serving API up? The server default will
        be used.
      </p>
    );
  }

  if (ready.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        No live deployments — deploy one in the Deploy tab. The server default
        will be used.
      </p>
    );
  }

  return (
    <Select value={value} onChange={(e) => onChange(e.target.value)}>
      {ready.map((d) => {
        const name = d.spec?.name ?? "";
        const model = d.spec?.launch?.model ?? "?";
        const runtime = d.spec?.launch?.runtime ?? "?";
        return (
          <option key={name} value={name}>
            {`${name} · ${model} (${runtime})`}
          </option>
        );
      })}
    </Select>
  );
}
