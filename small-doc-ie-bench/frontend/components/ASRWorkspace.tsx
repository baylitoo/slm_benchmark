"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useInngestSubscription,
  InngestSubscriptionState,
} from "@inngest/realtime/hooks";
import type { Realtime } from "@inngest/realtime";
import {
  AlertCircle,
  AudioLines,
  Download,
  FileAudio,
  Radio,
  Square,
  Trash2,
  Upload,
} from "lucide-react";
import {
  ASR_REALTIME_TOPICS,
  ApiError,
  ApiUnavailable,
  cancelASRJob,
  createASRJob,
  downloadASRArtifact,
  fileToBase64,
  formatBytes,
  getASRJob,
  getDeployments,
  getRealtimeToken,
  isTerminalASRStatus,
  listASRJobs,
  liveASRDeployments,
  type ASRArtifact,
  type ASRJob,
  type ASRJobItem,
  type ASRJobStatus,
  type DeploymentRecord,
  type RealtimeToken,
} from "@/lib/api";
import { cn } from "@/lib/cn";
import { usePolling } from "@/lib/usePolling";
import { useToast } from "./Toast";
import {
  Alert,
  Badge,
  type BadgeTone,
  Button,
  Card,
  EmptyState,
  Field,
  Select,
  Spinner,
  TextArea,
  TextInput,
} from "./ui";
import { PageHeader } from "./patterns/PageHeader";
import { ResultLine } from "./patterns/ResultLine";
import { Table, type Column } from "./patterns/Table";

const JOB_POLL_MS = 4000;
const DETAIL_POLL_MS = 2500;
const LAST_JOB_KEY = "docie-studio-asr-last-job";
const AUDIO_ACCEPT = ".wav,.flac,.mp3,.mp4,.m4a,.ogg,.webm";
const ALLOWED_SUFFIXES = new Set(AUDIO_ACCEPT.split(","));

interface FileDraft {
  file: File;
  reference: string;
}

function statusTone(status: ASRJobStatus): BadgeTone {
  if (status === "completed") return "ok";
  if (status === "completed_with_errors" || status === "cancelling") return "warn";
  if (status === "failed" || status === "cancelled") return "err";
  return "info";
}

function itemTone(status: ASRJobItem["status"]): BadgeTone {
  if (status === "completed") return "ok";
  if (status === "failed" || status === "cancelled") return "err";
  return status === "running" ? "info" : "neutral";
}

function percent(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(2)}%`;
}

function decimal(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(3);
}

function seconds(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const minutes = Math.floor(value / 60);
  const rest = value - minutes * 60;
  return minutes > 0 ? `${minutes}:${rest.toFixed(1).padStart(4, "0")}` : `${rest.toFixed(2)}s`;
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiUnavailable) {
    return "The durable speech-to-text service is unavailable. Check the backend and job database.";
  }
  if (error instanceof ApiError || error instanceof Error) return error.message;
  return "Something went wrong.";
}

function JobProgress({ job }: { job: ASRJob }) {
  const settled = job.completed_items + job.failed_items;
  const total = Math.max(job.total_items, 1);
  const completedWidth = (job.completed_items / total) * 100;
  const failedWidth = (job.failed_items / total) * 100;
  return (
    <div className="min-w-36">
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span className="tabular-nums text-foreground">{settled}/{job.total_items}</span>
        {job.failed_items > 0 && <span className="text-rose-500">{job.failed_items} failed</span>}
      </div>
      <div
        className="mt-1 flex h-1.5 overflow-hidden rounded-full bg-muted"
        role="progressbar"
        aria-label={`Transcription progress for ${job.event_id}`}
        aria-valuenow={settled}
        aria-valuemax={total}
      >
        <span className="bg-emerald-500 transition-all" style={{ width: `${completedWidth}%` }} />
        <span className="bg-rose-500 transition-all" style={{ width: `${failedWidth}%` }} />
      </div>
    </div>
  );
}

export function ASRWorkspace({ active = true }: { active?: boolean }) {
  const { toast } = useToast();
  const deployments = usePolling<DeploymentRecord[]>(getDeployments, JOB_POLL_MS, active);
  const jobs = usePolling<ASRJob[]>(() => listASRJobs(100), JOB_POLL_MS, active);
  const asrDeployments = useMemo(
    () => liveASRDeployments(deployments.data ?? []),
    [deployments.data],
  );
  const deploymentNames = useMemo(
    () => asrDeployments.map((record) => record.spec?.name).filter((name): name is string => !!name),
    [asrDeployments],
  );

  const [model, setModel] = useState("");
  const [language, setLanguage] = useState("");
  const [prompt, setPrompt] = useState("");
  const [temperature, setTemperature] = useState("0");
  const [retention, setRetention] = useState<ASRJob["raw_retention"]>(
    "delete_after_completion",
  );
  const [files, setFiles] = useState<FileDraft[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const saved = window.localStorage.getItem(LAST_JOB_KEY);
    if (saved) setSelectedId(saved);
  }, []);

  useEffect(() => {
    if (deploymentNames.length === 0) {
      if (model) setModel("");
    } else if (!deploymentNames.includes(model)) {
      setModel(deploymentNames[0]);
    }
  }, [deploymentNames, model]);

  function selectJob(jobId: string) {
    setSelectedId(jobId);
    window.localStorage.setItem(LAST_JOB_KEY, jobId);
  }

  function pickFiles(next: FileList | null) {
    setFormError(null);
    const picked = Array.from(next ?? []);
    if (picked.length > 100) {
      setFormError("A transcription job can contain at most 100 recordings.");
      return;
    }
    const unsupported = picked.find((file) => {
      const dot = file.name.lastIndexOf(".");
      return dot < 0 || !ALLOWED_SUFFIXES.has(file.name.slice(dot).toLowerCase());
    });
    if (unsupported) {
      setFormError(`${unsupported.name} is not a supported audio file.`);
      return;
    }
    setFiles(picked.map((file) => ({ file, reference: "" })));
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setFormError(null);
    if (!model || !deploymentNames.includes(model)) {
      setFormError("Select a live ASR deployment from the catalog.");
      return;
    }
    if (files.length === 0) {
      setFormError("Choose at least one audio recording.");
      return;
    }
    setSubmitting(true);
    try {
      const recordings = await Promise.all(
        files.map(async ({ file, reference }) => ({
          filename: file.name,
          content_b64: await fileToBase64(file),
          ...(reference.trim() ? { reference: reference.trim() } : {}),
        })),
      );
      const parsedTemperature = Number(temperature);
      const response = await createASRJob({
        model,
        recordings,
        ...(language.trim() ? { language: language.trim() } : {}),
        ...(prompt.trim() ? { prompt: prompt.trim() } : {}),
        temperature: Number.isFinite(parsedTemperature) ? parsedTemperature : 0,
        raw_audio_retention: retention,
      });
      selectJob(response.job_id);
      setFiles([]);
      if (fileInput.current) fileInput.current.value = "";
      jobs.refresh();
      toast({
        title: response.deduplicated ? "Existing transcription recovered" : "Transcription queued",
        description: response.job_id,
        tone: "success",
      });
    } catch (error) {
      const message = errorMessage(error);
      setFormError(message);
      toast({ title: "Could not start transcription", description: message, tone: "error" });
    } finally {
      setSubmitting(false);
    }
  }

  const columns: Column<ASRJob>[] = [
    {
      key: "created",
      header: "Started",
      sortAccessor: (job) => job.created_at,
      render: (job) => (
        <div>
          <p className="text-xs text-foreground">{new Date(job.created_at).toLocaleString()}</p>
          <p className="max-w-48 truncate font-mono text-[10px] text-muted-foreground">
            {job.event_id}
          </p>
        </div>
      ),
    },
    {
      key: "deployment",
      header: "Deployment",
      sortAccessor: (job) => job.deployment,
      render: (job) => <span className="font-mono text-xs">{job.deployment}</span>,
    },
    {
      key: "status",
      header: "Status",
      sortAccessor: (job) => job.status,
      render: (job) => <Badge tone={statusTone(job.status)}>{job.status.replaceAll("_", " ")}</Badge>,
    },
    { key: "progress", header: "Recordings", render: (job) => <JobProgress job={job} /> },
    {
      key: "evaluation",
      header: "Evaluation",
      render: (job) => (
        <span className="text-xs tabular-nums text-muted-foreground">
          {job.metrics?.scored_items ? `WER ${percent(job.metrics.wer)} · CER ${percent(job.metrics.cer)}` : "—"}
        </span>
      ),
    },
    {
      key: "open",
      header: "",
      className: "text-right",
      render: (job) => (
        <Button
          type="button"
          size="sm"
          variant="ghost"
          onClick={(event) => {
            event.stopPropagation();
            selectJob(job.event_id);
          }}
          aria-label={`Open transcription job ${job.event_id}`}
        >
          Open
        </Button>
      ),
    },
  ];

  return (
    <div>
      <PageHeader
        title="Speech to text"
        subtitle="Transcribe one recording or a batch as a durable job, then inspect timestamps, artifacts, and evaluation quality."
        actions={<Badge tone="info"><AudioLines className="h-3.5 w-3.5" /> durable jobs</Badge>}
      />

      <div className="grid gap-5 xl:grid-cols-[minmax(22rem,0.8fr)_minmax(0,1.2fr)]">
        <Card
          title="New transcription"
          subtitle="Every upload uses the recoverable background-job path."
          icon={<Upload className="h-4 w-4" />}
        >
          <form className="space-y-4" onSubmit={submit}>
            <Field label="Live ASR deployment" htmlFor="asr-deployment" required>
              <Select
                id="asr-deployment"
                value={model}
                onChange={(event) => setModel(event.target.value)}
                disabled={deployments.loading || deploymentNames.length === 0}
              >
                {deploymentNames.length === 0 && <option value="">No live ASR deployments</option>}
                {deploymentNames.map((name) => <option key={name} value={name}>{name}</option>)}
              </Select>
            </Field>
            {Boolean(deployments.error) && <Alert tone="err">Could not load the deployment catalog.</Alert>}
            {!deployments.loading && !deployments.error && deploymentNames.length === 0 && (
              <Alert tone="warn">
                Deploy and start an ASR model first. Stopped, loading, and non-ASR deployments are not selectable.
              </Alert>
            )}

            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Language" htmlFor="asr-language" hint="Optional ISO code, for example en or fr.">
                <TextInput
                  id="asr-language"
                  value={language}
                  onChange={(event) => setLanguage(event.target.value)}
                  placeholder="Auto-detect"
                  maxLength={16}
                />
              </Field>
              <Field label="Temperature" htmlFor="asr-temperature" hint="0 is deterministic; allowed range is 0–1.">
                <TextInput
                  id="asr-temperature"
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={temperature}
                  onChange={(event) => setTemperature(event.target.value)}
                />
              </Field>
            </div>
            <Field label="Prompt" htmlFor="asr-prompt" hint="Optional vocabulary or transcription context.">
              <TextArea
                id="asr-prompt"
                rows={2}
                maxLength={4000}
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="Names, acronyms, or domain vocabulary"
              />
            </Field>
            <Field label="Raw audio retention" htmlFor="asr-retention">
              <Select
                id="asr-retention"
                value={retention}
                onChange={(event) => setRetention(event.target.value as ASRJob["raw_retention"])}
              >
                <option value="delete_after_completion">Delete after completion (recommended)</option>
                <option value="retain_7d">Retain for 7 days</option>
                <option value="retain_30d">Retain for 30 days</option>
              </Select>
            </Field>

            <Field label="Audio recordings" htmlFor="asr-recordings" required hint="WAV, FLAC, MP3, MP4/M4A, OGG, or WebM. Up to 100 files.">
              <input
                ref={fileInput}
                id="asr-recordings"
                type="file"
                multiple
                accept={AUDIO_ACCEPT}
                onChange={(event) => pickFiles(event.target.files)}
                className="block w-full rounded-md border border-dashed border-input bg-muted/20 px-3 py-4 text-sm text-foreground file:mr-3 file:rounded-md file:border-0 file:bg-accent file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-accent-foreground"
              />
            </Field>

            {files.length > 0 && (
              <div className="space-y-2" aria-label="Selected recordings">
                {files.map((draft, index) => (
                  <div key={`${draft.file.name}-${draft.file.lastModified}-${index}`} className="rounded-md border border-border bg-muted/20 p-3">
                    <div className="flex items-center gap-2">
                      <FileAudio className="h-4 w-4 shrink-0 text-accent" />
                      <span className="min-w-0 flex-1 truncate text-xs font-medium">{draft.file.name}</span>
                      <span className="text-[11px] text-muted-foreground">{formatBytes(draft.file.size)}</span>
                      <button
                        type="button"
                        aria-label={`Remove ${draft.file.name}`}
                        className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-rose-500"
                        onClick={() => setFiles((current) => current.filter((_, i) => i !== index))}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                    <TextArea
                      className="mt-2 min-h-16"
                      rows={2}
                      aria-label={`Reference transcript for ${draft.file.name}`}
                      placeholder="Optional reference transcript for WER / CER"
                      value={draft.reference}
                      onChange={(event) => setFiles((current) => current.map((item, i) => i === index ? { ...item, reference: event.target.value } : item))}
                    />
                  </div>
                ))}
              </div>
            )}

            {formError && <Alert tone="err">{formError}</Alert>}
            <Button
              type="submit"
              loading={submitting}
              disabled={files.length === 0 || !model || deploymentNames.length === 0}
              className="w-full"
            >
              <AudioLines className="h-4 w-4" />
              {submitting ? "Uploading recordings…" : `Start transcription${files.length > 1 ? ` (${files.length})` : ""}`}
            </Button>
          </form>
        </Card>

        <div className="min-w-0">
          {selectedId ? (
            <ASRJobDetailPane
              key={selectedId}
              jobId={selectedId}
              initial={jobs.data?.find((job) => job.event_id === selectedId) ?? null}
              active={active}
              onUpdated={jobs.refresh}
            />
          ) : (
            <Card title="Transcription result" icon={<AudioLines className="h-4 w-4" />}>
              <EmptyState
                title="No transcription selected"
                description="Start a durable job or select one from history. Your last selection is restored after refresh."
                icon={<FileAudio className="h-5 w-5" />}
              />
            </Card>
          )}
        </div>
      </div>

      <Card
        className="mt-5"
        title="Transcription history"
        subtitle="Tenant-scoped durable jobs remain available across refreshes and reconnects."
        icon={<FileAudio className="h-4 w-4" />}
      >
        <ResultLine
          shown={jobs.data?.length ?? 0}
          total={jobs.data?.length ?? 0}
          noun="jobs"
          onFetch={jobs.refresh}
          fetching={jobs.refreshing}
        />
        <Table
          columns={columns}
          rows={jobs.data}
          loading={jobs.loading}
          error={jobs.error}
          getRowKey={(job) => job.event_id}
          onRowClick={(job) => selectJob(job.event_id)}
          emptyLabel="No speech-to-text jobs"
          emptyDescription="Upload a recording above to create the first durable transcription."
          emptyIcon={<FileAudio className="h-5 w-5" />}
        />
      </Card>
    </div>
  );
}

function ASRJobDetailPane({
  jobId,
  initial,
  active,
  onUpdated,
}: {
  jobId: string;
  initial: ASRJob | null;
  active: boolean;
  onUpdated: () => void;
}) {
  const { toast } = useToast();
  const detail = usePolling<ASRJob>(() => getASRJob(jobId), DETAIL_POLL_MS, active);
  const job = detail.data ?? initial;
  const [expanded, setExpanded] = useState<number | null>(null);
  const [cancelling, setCancelling] = useState(false);

  const refreshDetail = detail.refresh;
  const refresh = useCallback(() => {
    refreshDetail();
    onUpdated();
  }, [refreshDetail, onUpdated]);

  async function cancel() {
    setCancelling(true);
    try {
      await cancelASRJob(jobId);
      refresh();
      toast({ title: "Cancellation requested", description: jobId, tone: "success" });
    } catch (error) {
      toast({ title: "Could not cancel job", description: errorMessage(error), tone: "error" });
    } finally {
      setCancelling(false);
    }
  }

  if (!job) {
    return (
      <Card title="Transcription result" icon={<AudioLines className="h-4 w-4" />}>
        {detail.error ? <Alert tone="err">{errorMessage(detail.error)}</Alert> : <div className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner /> Recovering durable job…</div>}
      </Card>
    );
  }

  const terminal = isTerminalASRStatus(job.status);
  const metrics = job.metrics;

  return (
    <Card
      title={job.items.length === 1 ? job.items[0].filename : `${job.items.length} recordings`}
      subtitle={`${job.deployment} · ${job.event_id}`}
      icon={<AudioLines className="h-4 w-4" />}
      actions={
        <div className="flex flex-wrap items-center gap-2">
          <RealtimeRefresh channel={job.channel} enabled={active && !terminal} onMessage={refresh} />
          <Badge tone={statusTone(job.status)}>{job.status.replaceAll("_", " ")}</Badge>
          {!terminal && (
            <Button size="sm" variant="danger" loading={cancelling} onClick={() => void cancel()}>
              <Square className="h-3 w-3" /> Cancel
            </Button>
          )}
        </div>
      }
    >
      <div className="space-y-5">
        {Boolean(detail.error) && <Alert tone="warn">Live refresh failed; showing the last durable state.</Alert>}
        {job.error && <Alert tone="err">{job.error}</Alert>}
        <JobProgress job={job} />

        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Metric label="WER" value={metrics?.scored_items ? percent(metrics.wer) : "—"} hint={metrics?.scored_items ? `${metrics.scored_items} scored` : "Add references"} />
          <Metric label="CER" value={metrics?.scored_items ? percent(metrics.cer) : "—"} hint={metrics?.scored_items ? `${metrics.character_errors ?? 0} edits` : "Add references"} />
          <Metric label="Real-time factor" value={decimal(metrics?.real_time_factor)} hint="processing ÷ audio" />
          <Metric label="Audio" value={seconds(metrics?.audio_seconds)} hint={`${job.completed_items} completed`} />
        </div>

        {job.artifacts.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Batch artifacts</p>
            <ArtifactButtons artifacts={job.artifacts} />
          </div>
        )}

        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Recordings</p>
          {job.items.map((item) => (
            <div key={item.position} className="overflow-hidden rounded-md border border-border">
              <button
                type="button"
                className="flex w-full items-center gap-3 bg-muted/20 px-3 py-3 text-left hover:bg-muted/40"
                aria-expanded={expanded === item.position}
                onClick={() => setExpanded((current) => current === item.position ? null : item.position)}
              >
                {item.status === "failed" ? <AlertCircle className="h-4 w-4 shrink-0 text-rose-500" /> : <FileAudio className="h-4 w-4 shrink-0 text-accent" />}
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium">{item.filename}</span>
                  <span className="block text-[11px] text-muted-foreground">
                    {item.detected_language ?? "language pending"} · {seconds(item.duration_seconds)} · {item.attempts} attempt{item.attempts === 1 ? "" : "s"}
                  </span>
                </span>
                {item.metrics && <span className="hidden text-xs tabular-nums text-muted-foreground sm:block">WER {percent(item.metrics.wer)}</span>}
                <Badge tone={itemTone(item.status)}>{item.status}</Badge>
              </button>
              {expanded === item.position && <ASRItemDetail item={item} />}
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/20 p-3">
      <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-lg font-semibold tabular-nums text-foreground">{value}</p>
      <p className="text-[10px] text-muted-foreground">{hint}</p>
    </div>
  );
}

function ASRItemDetail({ item }: { item: ASRJobItem }) {
  const segments = item.result?.segments ?? [];
  return (
    <div className="space-y-4 border-t border-border bg-background p-4">
      {item.error && <Alert tone="err">{item.error}</Alert>}
      {item.result?.text ? (
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Transcript</p>
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">{item.result.text}</p>
        </div>
      ) : item.status === "completed" ? (
        <p className="text-sm text-muted-foreground">The transcription completed with empty text.</p>
      ) : (
        <p className="text-sm text-muted-foreground">Transcript not available yet.</p>
      )}
      {segments.length > 0 && (
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Timestamped segments</p>
          <ol className="max-h-64 space-y-1 overflow-y-auto pr-1">
            {segments.map((segment, index) => (
              <li key={`${segment.id ?? index}-${segment.start ?? 0}`} className="grid grid-cols-[6.5rem_1fr] gap-3 rounded bg-muted/30 px-2.5 py-2 text-xs">
                <span className="font-mono tabular-nums text-muted-foreground">{seconds(segment.start)} → {seconds(segment.end)}</span>
                <span className="text-foreground">{segment.text}</span>
              </li>
            ))}
          </ol>
        </div>
      )}
      {item.reference && (
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Reference</p>
          <p className="whitespace-pre-wrap text-xs text-muted-foreground">{item.reference}</p>
        </div>
      )}
      {item.artifacts.length > 0 && <ArtifactButtons artifacts={item.artifacts} />}
    </div>
  );
}

function ArtifactButtons({ artifacts }: { artifacts: ASRArtifact[] }) {
  const { toast } = useToast();
  const [pending, setPending] = useState<string | null>(null);
  async function download(artifact: ASRArtifact) {
    setPending(artifact.id);
    try {
      await downloadASRArtifact(artifact);
    } catch (error) {
      toast({ title: `Download failed: ${artifact.name}`, description: errorMessage(error), tone: "error" });
    } finally {
      setPending(null);
    }
  }
  return (
    <div className="flex flex-wrap gap-2" aria-label="Transcript downloads">
      {artifacts.map((artifact) => (
        <Button
          key={artifact.id}
          type="button"
          size="sm"
          variant="secondary"
          loading={pending === artifact.id}
          onClick={() => void download(artifact)}
          title={`${artifact.name} · ${formatBytes(artifact.size_bytes)}`}
        >
          <Download className="h-3.5 w-3.5" /> {artifact.kind.replace("verbose_json", "JSON").toUpperCase()}
        </Button>
      ))}
    </div>
  );
}

type AnyToken = Realtime.Subscribe.Token;

function RealtimeRefresh({ channel, enabled, onMessage }: { channel: string; enabled: boolean; onMessage: () => void }) {
  const [token, setToken] = useState<RealtimeToken | null>(null);
  useEffect(() => {
    let cancelled = false;
    setToken(null);
    if (!enabled) return;
    void getRealtimeToken(channel, ASR_REALTIME_TOPICS)
      .then((next) => { if (!cancelled) setToken(next); })
      .catch(() => { /* Polling remains authoritative when realtime is unavailable. */ });
    return () => { cancelled = true; };
  }, [channel, enabled]);
  if (!enabled) return null;
  if (!token) return <Badge tone="neutral"><Radio className="h-3 w-3" /> polling</Badge>;
  return <RealtimeSubscription channel={channel} token={token} onMessage={onMessage} />;
}

function RealtimeSubscription({ channel, token, onMessage }: { channel: string; token: RealtimeToken; onMessage: () => void }) {
  const refreshToken = useCallback(
    async () => (await getRealtimeToken(channel, ASR_REALTIME_TOPICS)) as unknown as AnyToken,
    [channel],
  );
  const { data, state } = useInngestSubscription({
    token: token as unknown as AnyToken,
    refreshToken,
    enabled: true,
    key: channel,
  });
  const count = data?.length ?? 0;
  useEffect(() => {
    if (count > 0) onMessage();
  }, [count, onMessage]);
  const live = state === InngestSubscriptionState.Active;
  return (
    <Badge tone={live ? "ok" : state === InngestSubscriptionState.Error ? "warn" : "info"}>
      <Radio className={cn("h-3 w-3", live && "animate-pulse")} />
      {live ? "live" : state === InngestSubscriptionState.Error ? "polling fallback" : "connecting"}
    </Badge>
  );
}
