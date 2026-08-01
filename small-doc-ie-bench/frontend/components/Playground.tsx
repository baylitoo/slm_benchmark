"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  FileText,
  Fingerprint,
  MessageSquare,
  Play,
  Send,
  Sparkles,
  Trash2,
  Upload,
  AlertCircle,
} from "lucide-react";
import {
  triggerExtract,
  chatCompletion,
  embed,
  embeddingDeploymentNames,
  getDeployments,
  getStore,
  getFamilies,
  selectableDeployments,
  isLiveDeployment,
  fileToBase64,
  ApiError,
  ApiUnavailable,
  type TriggerResponse,
  type ExtractRequest,
  type DeploymentRecord,
  type StoreEntry,
  type ModelFamily,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
import { useToast } from "./Toast";
import { Button, Card, Field, Select, TextArea, TextInput, Badge, Spinner } from "./ui";
import { ResultPanel } from "./ResultPanel";
import { PageHeader } from "./patterns/PageHeader";

type InputMode = "text" | "file";
type PlaygroundMode = "extract" | "chat" | "embed";

const DEPLOY_POLL_MS = 4000;

export function Playground({ active = true }: { active?: boolean }) {
  const { toast } = useToast();
  const [mode, setMode] = useState<PlaygroundMode>("extract");
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

  // Routable deployments, sourced from the same endpoint the Deploy tab uses
  // (GET /v1/serving/deployments): live ones PLUS evicted `managed` ones — a
  // request to an evicted deployment auto-reloads it (PR-4 cold-start-on-
  // demand), so it must stay selectable here or the flagship flow would be
  // unreachable from the UI. Polling is paused while the tab is hidden.
  const deployments = usePolling<DeploymentRecord[]>(getDeployments, DEPLOY_POLL_MS, active);
  const store = useAsync<StoreEntry[]>(getStore, []);
  const families = useAsync<ModelFamily[]>(getFamilies, []);
  const embeddingNames = useMemo(
    () => embeddingDeploymentNames(store.data, families.data),
    [store.data, families.data],
  );
  // Encoders (analyzers) AND embedding models are excluded from extract/chat:
  // they don't answer chat/extraction prompts. Embedding models live in the
  // Embed mode instead.
  const selectable = useMemo(
    () =>
      selectableDeployments(deployments.data ?? []).filter(
        (d) =>
          d.spec?.launch?.runtime !== "encoder" &&
          !(d.spec?.name && embeddingNames.has(d.spec.name)),
      ),
    [deployments.data, embeddingNames],
  );
  const selectableNames = useMemo(
    () => selectable.map((d) => d.spec?.name ?? "").filter(Boolean),
    [selectable],
  );

  // Pre-select the first selectable deployment so an explicit `deployment` is
  // always sent when one exists; resync if the current pick disappears.
  useEffect(() => {
    if (selectableNames.length === 0) {
      if (selectedDeployment !== "") setSelectedDeployment("");
      return;
    }
    if (!selectableNames.includes(selectedDeployment)) {
      setSelectedDeployment(selectableNames[0]);
    }
  }, [selectableNames, selectedDeployment]);

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
    <div>
      <PageHeader
        title="Playground"
        subtitle={
          mode === "chat"
            ? "Classic queries: chat directly with any live deployment."
            : mode === "embed"
              ? "Embeddings computed locally — vectors never leave the infra (RAG-ready)."
              : "Paste text or upload a document, route it to a live deployment, and watch the extraction stream."
        }
        actions={
          <div className="inline-flex rounded-lg border border-border bg-muted p-0.5 text-sm">
            {(
              [
                ["extract", "Extract", Sparkles],
                ["chat", "Chat", MessageSquare],
                ["embed", "Embed", Fingerprint],
              ] as [PlaygroundMode, string, typeof Sparkles][]
            ).map(([m, label, Icon]) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 transition",
                  mode === m
                    ? "bg-card text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                {label}
              </button>
            ))}
          </div>
        }
      />

      {/* Both modes stay mounted (hidden, never unmounted) so an in-flight
          extraction stream or a chat history survives switching modes. */}
      <div hidden={mode !== "chat"}>
        <ChatPanel deployments={deployments} selectable={selectable} />
      </div>
      <div hidden={mode !== "embed"}>
        <EmbedPanel
          deployments={deployments.data ?? []}
          embeddingNames={embeddingNames}
        />
      </div>
      <div hidden={mode !== "extract"} className="grid gap-6 lg:grid-cols-2">
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
            <Field
              label="Deployment"
              hint="Runtime to route this extraction to. Evicted deployments reload on request (first request waits for the model load)."
            >
              <DeploymentSelect
                deployments={deployments}
                selectable={selectable}
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
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chat mode — classic free-form queries against a live deployment, through
// the generic OpenAI surface (POST /v1/chat/completions, model = deployment).
// ---------------------------------------------------------------------------

interface ChatMsg {
  role: "user" | "assistant";
  content: string;
}

function ChatPanel({
  deployments,
  selectable,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
  selectable: DeploymentRecord[];
}) {
  // Chat owns its OWN selection: it only offers LIVE deployments, so sharing
  // the Extract-side selection (which legitimately includes evicted
  // auto-reload targets) would let the select DISPLAY one model while the
  // request carries another — the exact mismatch this state split fixes.
  const [model, setModel] = useState("");
  const liveOnly = useMemo(() => selectable.filter(isLiveDeployment), [selectable]);
  const liveNames = useMemo(
    () => liveOnly.map((d) => d.spec?.name ?? "").filter(Boolean),
    [liveOnly],
  );
  useEffect(() => {
    if (liveNames.length === 0) {
      if (model !== "") setModel("");
      return;
    }
    if (!liveNames.includes(model)) setModel(liveNames[0]);
  }, [liveNames, model]);

  const [system, setSystem] = useState("");
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<ChatMsg[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [msgs, busy]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    if (!model) {
      setError("No live deployment selected — deploy a model in the Deploy tab first.");
      return;
    }
    setError(null);
    const next: ChatMsg[] = [...msgs, { role: "user", content: text }];
    setMsgs(next);
    setInput("");
    setBusy(true);
    try {
      const payload = [
        ...(system.trim() ? [{ role: "system", content: system.trim() }] : []),
        ...next,
      ];
      const res = await chatCompletion(model, payload);
      const content = res.choices?.[0]?.message?.content ?? "(empty response)";
      setMsgs([...next, { role: "assistant", content }]);
    } catch (e) {
      const msg =
        e instanceof ApiError || e instanceof ApiUnavailable || e instanceof Error
          ? e.message
          : "Chat request failed.";
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      icon={<MessageSquare className="h-5 w-5" />}
      title="Chat"
      subtitle="Free-form conversation with the selected deployment (multi-turn, history kept locally)."
    >
      <div className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Deployment"
            hint="Live deployments only — evicted ones need a Load from the Deploy tab first."
          >
            <DeploymentSelect
              deployments={deployments}
              selectable={liveOnly}
              value={model}
              onChange={setModel}
            />
          </Field>
          <Field label="System prompt" hint="Optional — applied to the whole conversation.">
            <TextInput
              value={system}
              onChange={(e) => setSystem(e.target.value)}
              placeholder="You are…"
            />
          </Field>
        </div>

        <div className="scroll-thin max-h-[50vh] min-h-40 space-y-3 overflow-y-auto rounded-md border border-border bg-muted/20 p-4">
          {msgs.length === 0 && (
            <p className="text-sm text-muted-foreground">
              Ask anything — the request goes straight to the deployment through
              the OpenAI-compatible surface.
            </p>
          )}
          {msgs.map((m, i) => (
            <div
              key={i}
              className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}
            >
              <div
                className={cn(
                  "max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm",
                  m.role === "user"
                    ? "bg-accent text-accent-foreground"
                    : "border border-border bg-card text-foreground",
                )}
              >
                {m.content}
              </div>
            </div>
          ))}
          {busy && (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <Spinner /> Waiting for the model…
            </p>
          )}
          <div ref={bottomRef} />
        </div>

        {error && (
          <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        <div className="flex items-end gap-2">
          <TextArea
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            placeholder="Type a message (Enter to send, Shift+Enter for a new line)…"
          />
          <Button type="button" loading={busy} onClick={() => void send()}>
            <Send className="h-4 w-4" />
            Send
          </Button>
          <Button
            type="button"
            variant="secondary"
            disabled={msgs.length === 0 || busy}
            onClick={() => {
              setMsgs([]);
              setError(null);
            }}
            title="Clear the conversation"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Embed mode — local embeddings + cosine similarity (RAG demo). Vectors are
// computed on this node; nothing leaves the infra.
// ---------------------------------------------------------------------------

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

function EmbedPanel({
  deployments,
  embeddingNames,
}: {
  deployments: DeploymentRecord[];
  embeddingNames: Set<string>;
}) {
  const embedDeployments = useMemo(
    () =>
      deployments.filter(
        (d) => d.spec?.name && embeddingNames.has(d.spec.name) && isLiveDeployment(d),
      ),
    [deployments, embeddingNames],
  );
  const names = useMemo(
    () => embedDeployments.map((d) => d.spec?.name ?? "").filter(Boolean),
    [embedDeployments],
  );

  const [model, setModel] = useState("");
  const [textA, setTextA] = useState("Facture 5 400 € TTC, échéance 30 jours.");
  const [textB, setTextB] = useState("Invoice total 5400 EUR, net 30 payment terms.");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ dims: number; sim: number; a: number[] } | null>(null);

  useEffect(() => {
    if (names.length === 0) {
      if (model !== "") setModel("");
      return;
    }
    if (!names.includes(model)) setModel(names[0]);
  }, [names, model]);

  async function run() {
    if (!model) {
      setError("No live embedding deployment — deploy an embedding model (family: embedding).");
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const res = await embed(model, [textA, textB]);
      const a = res.data?.[0]?.embedding ?? [];
      const b = res.data?.[1]?.embedding ?? [];
      setResult({ dims: a.length, sim: cosine(a, b), a });
    } catch (e) {
      setError(
        e instanceof ApiError || e instanceof ApiUnavailable || e instanceof Error
          ? e.message
          : "Embedding request failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      icon={<Fingerprint className="h-5 w-5" />}
      title="Embeddings"
      subtitle="Two texts in, cosine similarity out — the retrieval primitive, computed on-node."
    >
      <div className="space-y-4">
        <Field
          label="Embedding deployment"
          hint="Deploy an embedding GGUF (e.g. LiquidAI/LFM2.5-Embedding-350M-GGUF) with family 'embedding'."
        >
          {names.length === 0 ? (
            <p className="rounded-lg border border-dashed border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
              No live embedding deployment — add one under Serving → Models
              (family: embedding), then Deploy it.
            </p>
          ) : (
            <Select value={model} onChange={(e) => setModel(e.target.value)}>
              {names.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </Select>
          )}
        </Field>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Text A">
            <TextArea rows={3} value={textA} onChange={(e) => setTextA(e.target.value)} />
          </Field>
          <Field label="Text B">
            <TextArea rows={3} value={textB} onChange={(e) => setTextB(e.target.value)} />
          </Field>
        </div>

        {error && (
          <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        <Button type="button" loading={busy} onClick={() => void run()} disabled={!model}>
          <Fingerprint className="h-4 w-4" />
          Embed & compare
        </Button>

        {result && (
          <div className="space-y-3 rounded-md border border-border bg-muted/20 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="info">{result.dims} dims</Badge>
              <Badge tone={result.sim > 0.7 ? "ok" : result.sim > 0.4 ? "warn" : "neutral"}>
                cosine similarity {result.sim.toFixed(4)}
              </Badge>
            </div>
            <div>
              <p className="mb-1 text-xs font-medium text-muted-foreground">
                Text A vector (first 12 dims)
              </p>
              <pre className="scroll-thin overflow-x-auto rounded-md border border-border bg-card p-3 text-xs text-foreground/90">
                [{result.a.slice(0, 12).map((v) => v.toFixed(4)).join(", ")}
                {result.a.length > 12 ? ", …" : ""}]
              </pre>
            </div>
            <p className="text-xs text-muted-foreground">
              Both vectors were computed by the local deployment — the text
              never left this node.
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Deployment selector — a dropdown of routable deployments: live ones plus
// evicted/loading `managed` ones (a request auto-reloads those; PR-4). Falls
// back to clear, non-crashing states for loading / unavailable / empty.
// ---------------------------------------------------------------------------

function DeploymentSelect({
  deployments,
  selectable,
  value,
  onChange,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
  selectable: DeploymentRecord[];
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

  if (selectable.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        No routable deployments — deploy one in the Deploy tab. The server
        default will be used.
      </p>
    );
  }

  return (
    <Select value={value} onChange={(e) => onChange(e.target.value)}>
      {selectable.map((d) => {
        const name = d.spec?.name ?? "";
        const model = d.spec?.launch?.model ?? "?";
        const runtime = d.spec?.launch?.runtime ?? "?";
        const suffix = isLiveDeployment(d)
          ? ""
          : d.state === "stopped"
            ? " · evicted — loads on request"
            : " · loading";
        return (
          <option key={name} value={name}>
            {`${name} · ${model} (${runtime})${suffix}`}
          </option>
        );
      })}
    </Select>
  );
}
