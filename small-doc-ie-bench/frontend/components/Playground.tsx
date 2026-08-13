"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Boxes,
  FileText,
  Fingerprint,
  Image as ImageIcon,
  ListOrdered,
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
  rerank,
  embeddingDeploymentNames,
  rerankerDeploymentNames,
  visionDeploymentNames,
  getDeployments,
  getStore,
  getFamilies,
  selectableDeployments,
  isLiveDeployment,
  fileToBase64,
  renderDocument,
  ApiError,
  ApiUnavailable,
  ModelLoading,
  type TriggerResponse,
  type ExtractRequest,
  type DeploymentRecord,
  type StoreEntry,
  type ModelFamily,
  type RerankResponse,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
import { useToast } from "./Toast";
import { Button, Card, Field, Select, TextArea, TextInput, Badge, Spinner } from "./ui";
import { ResultPanel } from "./ResultPanel";
import { PageHeader } from "./patterns/PageHeader";

type InputMode = "text" | "file";
type PlaygroundMode = "extract" | "chat" | "vision" | "embed" | "rerank";

// Deep-link callback threaded from AppShell so first-run empty states can send
// the user straight to Models to deploy a model. Optional everywhere: when it
// is absent the empty states degrade to instructive text only.
type NavigateToDeploy = (id: "deploy", view?: string) => void;

const DEPLOY_POLL_MS = 4000;

export function Playground({
  active = true,
  onNavigate,
}: {
  active?: boolean;
  onNavigate?: NavigateToDeploy;
}) {
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
  // Poll the store (not a one-shot fetch): the vision/embed model pickers derive
  // from it, so a model seeded/deployed after the Playground opened must appear
  // without a page reload — same cadence as the deployments list.
  const store = usePolling<StoreEntry[]>(getStore, DEPLOY_POLL_MS, active);
  const families = useAsync<ModelFamily[]>("families", getFamilies);
  const embeddingNames = useMemo(
    () => embeddingDeploymentNames(store.data, families.data),
    [store.data, families.data],
  );
  const rerankerNames = useMemo(
    () => rerankerDeploymentNames(store.data, families.data),
    [store.data, families.data],
  );
  const visionNames = useMemo(() => visionDeploymentNames(store.data), [store.data]);
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

  // Genuine first-run empty: deployments loaded cleanly but none can serve an
  // extraction. Kept distinct from loading / endpoint-error (those legitimately
  // fall back to the server default and must NOT disable Run).
  const extractNoModel =
    !deployments.loading && !deployments.error && selectable.length === 0;

  return (
    <div>
      <PageHeader
        title="Playground"
        subtitle={
          mode === "chat"
            ? "Chat directly with any live deployment."
            : mode === "vision"
              ? "Send an image to a vision deployment (OCR / description) — free-text answer."
              : mode === "embed"
                ? "Compute embeddings with your deployed models (RAG-ready)."
                : mode === "rerank"
                  ? "Score documents against a query with a reranker deployment (retrieval, ranked)."
                  : "Paste text or upload a document, route it to a live deployment, and watch the extraction stream."
        }
        actions={
          <div className="inline-flex rounded-lg border border-border bg-muted p-0.5 text-sm">
            {(
              [
                ["extract", "Extract", Sparkles],
                ["chat", "Chat", MessageSquare],
                ["vision", "Vision", ImageIcon],
                ["embed", "Embed", Fingerprint],
                ["rerank", "Rerank", ListOrdered],
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
        <ChatPanel deployments={deployments} selectable={selectable} onNavigate={onNavigate} />
      </div>
      <div hidden={mode !== "vision"}>
        <VisionPanel
          deployments={deployments.data ?? []}
          visionNames={visionNames}
          onNavigate={onNavigate}
        />
      </div>
      <div hidden={mode !== "embed"}>
        <EmbedPanel
          deployments={deployments.data ?? []}
          embeddingNames={embeddingNames}
          onNavigate={onNavigate}
        />
      </div>
      <div hidden={mode !== "rerank"}>
        <RerankPanel
          deployments={deployments.data ?? []}
          rerankerNames={rerankerNames}
          onNavigate={onNavigate}
        />
      </div>
      <div hidden={mode !== "extract"}>
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
            <Field
              label="Deployment"
              hint="Runtime to route this extraction to. Evicted deployments reload on request (first request waits for the model load)."
            >
              <DeploymentSelect
                deployments={deployments}
                selectable={selectable}
                value={selectedDeployment}
                onChange={setSelectedDeployment}
                emptyNoun="chat"
                onNavigate={onNavigate}
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

          <div className="space-y-1.5">
            <Button type="submit" loading={submitting} disabled={extractNoModel}>
              <Play className="h-4 w-4" />
              {submitting ? "Submitting…" : "Run extraction"}
            </Button>
            {extractNoModel && (
              <p className="text-xs text-muted-foreground">
                Deploy a chat model before running an extraction.
              </p>
            )}
          </div>
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
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chat mode — classic free-form queries against a live deployment, through
// the generic OpenAI surface (POST /v1/chat/completions, model = deployment).
// ---------------------------------------------------------------------------

interface ChatMsg {
  role: "user" | "assistant" | "status";
  content: string;
}

// A cold store: model's first request can take a while to boot. Auto-retry a
// bounded number of times (the backend's load trigger is idempotent — see
// loadDeployment's own doc comment) rather than making the user notice the
// status message and resend by hand.
const MAX_LOAD_RETRIES = 3;

function ChatPanel({
  deployments,
  selectable,
  onNavigate,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
  selectable: DeploymentRecord[];
  onNavigate?: NavigateToDeploy;
}) {
  // Chat owns its OWN selection: it only offers LIVE deployments, so sharing
  // the Extract-side selection (which legitimately includes evicted
  // auto-reload targets) would let the select DISPLAY one model while the
  // request carries another — the exact mismatch this state split fixes.
  // Evicted deployments stay selectable — a request to one auto-reloads it
  // (load-on-demand), same convention as the Extract and Vision pickers.
  // Prefer a live model as the default so the FIRST message a user sends
  // doesn't automatically eat a cold-start wait, but still let a cold one be
  // picked deliberately.
  const [model, setModel] = useState("");
  const chatNames = useMemo(
    () => selectable.map((d) => d.spec?.name ?? "").filter(Boolean),
    [selectable],
  );
  const liveNames = useMemo(
    () =>
      selectable
        .filter(isLiveDeployment)
        .map((d) => d.spec?.name ?? "")
        .filter(Boolean),
    [selectable],
  );
  useEffect(() => {
    if (chatNames.length === 0) {
      if (model !== "") setModel("");
      return;
    }
    if (!chatNames.includes(model)) setModel(liveNames[0] ?? chatNames[0]);
  }, [chatNames, liveNames, model]);

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
      setError("No deployment selected — deploy a model under Serving → Models first.");
      return;
    }
    setError(null);
    const next: ChatMsg[] = [...msgs, { role: "user", content: text }];
    setMsgs(next);
    setInput("");
    setBusy(true);
    const payload = [
      ...(system.trim() ? [{ role: "system", content: system.trim() }] : []),
      ...next,
    ];
    await attempt(next, payload, 0);
    setBusy(false);
  }

  async function attempt(
    next: ChatMsg[],
    payload: { role: string; content: unknown }[],
    retryCount: number,
  ) {
    try {
      const res = await chatCompletion(model, payload);
      const content = res.choices?.[0]?.message?.content ?? "(empty response)";
      setMsgs([...next, { role: "assistant", content }]);
    } catch (e) {
      if (e instanceof ModelLoading) {
        const willRetry = retryCount < MAX_LOAD_RETRIES;
        setMsgs([
          ...next,
          {
            role: "status",
            content: willRetry
              ? `${e.message} Retrying automatically…`
              : `${e.message} Still starting — send your message again in a bit.`,
          },
        ]);
        if (willRetry) {
          const waitMs = Math.min(Math.max(e.etaSeconds, 2), 30) * 1000;
          await new Promise((resolve) => setTimeout(resolve, waitMs));
          await attempt(next, payload, retryCount + 1);
        }
        return;
      }
      const msg =
        e instanceof ApiError || e instanceof ApiUnavailable || e instanceof Error
          ? e.message
          : "Chat request failed.";
      setError(msg);
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
            hint="Evicted deployments reload on request (first message waits for the load)."
          >
            <DeploymentSelect
              deployments={deployments}
              selectable={selectable}
              value={model}
              onChange={setModel}
              emptyNoun="chat"
              onNavigate={onNavigate}
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
          {msgs.map((m, i) =>
            m.role === "status" ? (
              <p
                key={i}
                className="flex items-center gap-2 text-xs italic text-muted-foreground"
              >
                <Spinner /> {m.content}
              </p>
            ) : (
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
            ),
          )}
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
          <Button
            type="button"
            loading={busy}
            disabled={chatNames.length === 0}
            onClick={() => void send()}
          >
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
// ---------------------------------------------------------------------------
// Vision mode — send an image to a vision deployment (lfm2_vl / nuextract3 /
// vision_ocr) as an OpenAI multimodal chat request; the answer is free text
// (OCR, description, caption). Closes the deploy→use loop for vision models.
// ---------------------------------------------------------------------------

const VISION_PRESETS = [
  "Extract all the text from this document (OCR).",
  "Describe this image in detail.",
  "What is written in this document? Return it verbatim.",
];

function VisionPanel({
  deployments,
  visionNames,
  onNavigate,
}: {
  deployments: DeploymentRecord[];
  visionNames: Set<string>;
  onNavigate?: NavigateToDeploy;
}) {
  // Include evicted (managed) deployments, not just live ones — a request to an
  // evicted vision deployment auto-reloads it (the first request waits for the
  // load), exactly like the Extract picker. Requiring isLiveDeployment hid a
  // deployed-but-idle VLM as "no vision model deployed yet".
  const selectableVision = useMemo(
    () =>
      selectableDeployments(deployments).filter(
        (d) => d.spec?.name && visionNames.has(d.spec.name),
      ),
    [deployments, visionNames],
  );
  const names = useMemo(
    () => selectableVision.map((d) => d.spec?.name ?? "").filter(Boolean),
    [selectableVision],
  );

  const [model, setModel] = useState("");
  const [prompt, setPrompt] = useState(VISION_PRESETS[0]);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  // PDF rasterization DPI — higher = sharper text (better for dense documents /
  // small vision models), larger payload. Only used for PDFs.
  const [dpi, setDpi] = useState(200);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A cold store: model triggering a load isn't a failure — style it apart
  // from an actual error so "warming up" doesn't read as "something broke".
  const [errorIsLoading, setErrorIsLoading] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);

  useEffect(() => {
    if (names.length === 0) {
      if (model !== "") setModel("");
      return;
    }
    if (!names.includes(model)) setModel(names[0]);
  }, [names, model]);

  async function onFile(f: File | null) {
    setAnswer(null);
    setError(null);
    if (preview?.startsWith("blob:")) URL.revokeObjectURL(preview);
    setPreview(null);
    setFile(f);
    if (!f) return;
    if (f.type.startsWith("image/")) {
      setPreview(URL.createObjectURL(f));
      return;
    }
    // PDF: rasterize page 1 (low DPI, single page) for a real visual preview —
    // an <img> can't show a PDF directly. Best-effort: on failure the run still
    // renders the pages when sent.
    setPreviewLoading(true);
    try {
      const b64 = await fileToBase64(f);
      const { images } = await renderDocument(b64, f.name, 150, 1);
      setPreview(images[0] ?? null);
    } catch {
      setPreview(null);
    } finally {
      setPreviewLoading(false);
    }
  }

  async function run() {
    if (!model) {
      setError("No live vision deployment — deploy a vision model (family lfm2_vl / vision_ocr).");
      return;
    }
    if (!file) {
      setError("Choose an image or PDF first.");
      return;
    }
    setError(null);
    setErrorIsLoading(false);
    setBusy(true);
    setAnswer(null);
    try {
      const b64 = await fileToBase64(file);
      // Vision models take IMAGES, not PDFs (llama-server rejects a
      // data:application/pdf URL). A PDF is rasterized to PNG page images
      // server-side first; a plain image is sent as-is.
      const isPdf =
        file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
      const imageUrls = isPdf
        ? (await renderDocument(b64, file.name, dpi)).images
        : [`data:${file.type || "image/png"};base64,${b64}`];
      if (imageUrls.length === 0) {
        setError("The document produced no page images.");
        return;
      }
      const res = await chatCompletion(model, [
        {
          role: "user",
          content: [
            { type: "text", text: prompt.trim() || "Describe this image." },
            ...imageUrls.map((url) => ({
              type: "image_url" as const,
              image_url: { url },
            })),
          ],
        },
      ]);
      setAnswer(res.choices?.[0]?.message?.content ?? "(empty response)");
    } catch (e) {
      setErrorIsLoading(e instanceof ModelLoading);
      setError(
        e instanceof ApiError || e instanceof ApiUnavailable || e instanceof Error
          ? e.message
          : "Vision request failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      icon={<ImageIcon className="h-5 w-5" />}
      title="Vision"
      subtitle="Upload an image, ask a question — the vision deployment answers in free text (OCR / description)."
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-4">
          <Field
            label="Vision deployment"
            hint="Deploy a vision model (Serving → Models: family lfm2_vl / nuextract3 / vision_ocr)."
          >
            {names.length === 0 ? (
              <EmptyModelState noun="vision" onNavigate={onNavigate} />
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

          <Field label="Image / document" hint="PNG, JPG or PDF — encoded in your browser.">
            <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-muted/30 px-4 py-6 text-center transition hover:border-accent hover:bg-muted/50">
              <Upload className="h-6 w-6 text-muted-foreground" />
              <span className="text-sm text-foreground">
                {file ? file.name : "Click to choose an image or PDF"}
              </span>
              <input
                type="file"
                accept=".pdf,image/*"
                onChange={(e) => onFile(e.target.files?.[0] ?? null)}
                className="sr-only"
              />
            </label>
          </Field>

          {previewLoading && (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <Spinner /> Rendering preview…
            </p>
          )}
          {preview && (
            // A PDF's preview is its rasterized page 1 (an <img> can't show a PDF).
            <img
              src={preview}
              alt="preview"
              className="max-h-56 w-full rounded-md border border-border object-contain"
            />
          )}
          {file && !file.type.startsWith("image/") && (
            <div className="space-y-2 rounded-md border border-border bg-muted/40 px-3 py-2">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <FileText className="h-4 w-4 shrink-0" />
                <span className="truncate">{file.name}</span>
                <span className="ml-auto shrink-0 text-xs">PDF · page 1 shown; all pages sent</span>
              </div>
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>Render quality</span>
                <Select
                  value={String(dpi)}
                  onChange={(e) => setDpi(Number(e.target.value))}
                  className="h-7 w-auto text-xs"
                >
                  <option value="150">150 DPI — faster, smaller</option>
                  <option value="200">200 DPI — recommended</option>
                  <option value="300">300 DPI — sharpest, heaviest</option>
                </Select>
                <span className="text-muted-foreground/80">
                  higher = sharper text, larger request
                </span>
              </label>
            </div>
          )}

          <Field label="Prompt">
            <TextArea rows={2} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </Field>
          <div className="flex flex-wrap gap-1.5">
            {VISION_PRESETS.map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => setPrompt(p)}
                className="rounded-md border border-border px-2 py-1 text-xs text-muted-foreground transition hover:bg-muted hover:text-foreground"
              >
                {p.length > 32 ? `${p.slice(0, 32)}…` : p}
              </button>
            ))}
          </div>

          {error &&
            (errorIsLoading ? (
              <p className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-700 dark:text-amber-400">
                <Spinner /> {error}
              </p>
            ) : (
              <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                {error}
              </p>
            ))}

          <Button type="button" loading={busy} disabled={!model || !file} onClick={() => void run()}>
            <Play className="h-4 w-4" />
            Run
          </Button>
        </div>

        <div>
          <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Answer
          </p>
          {busy ? (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner /> The vision model is reading the image… (CPU vision is slow)
            </p>
          ) : answer != null ? (
            <pre className="scroll-thin max-h-[28rem] overflow-auto whitespace-pre-wrap rounded-md border border-border bg-muted/40 p-3 text-sm leading-relaxed text-foreground/90">
              {answer}
            </pre>
          ) : (
            <p className="text-sm text-muted-foreground">
              Pick a vision deployment, upload an image, and run to see the answer.
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}

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
  onNavigate,
}: {
  deployments: DeploymentRecord[];
  embeddingNames: Set<string>;
  onNavigate?: NavigateToDeploy;
}) {
  // Same as the vision picker: an evicted embedding deployment reloads on the
  // next request, so include selectable (live + managed-evicted), not only live.
  const embedDeployments = useMemo(
    () =>
      selectableDeployments(deployments).filter(
        (d) => d.spec?.name && embeddingNames.has(d.spec.name),
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
            <EmptyModelState noun="embedding" onNavigate={onNavigate} />
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
              Both vectors were computed by the selected deployment.
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}

function RerankPanel({
  deployments,
  rerankerNames,
  onNavigate,
}: {
  deployments: DeploymentRecord[];
  rerankerNames: Set<string>;
  onNavigate?: NavigateToDeploy;
}) {
  // Same as the embed picker: an evicted reranker deployment reloads on the
  // next request, so include selectable (live + managed-evicted), not only live.
  const rerankDeployments = useMemo(
    () =>
      selectableDeployments(deployments).filter(
        (d) => d.spec?.name && rerankerNames.has(d.spec.name),
      ),
    [deployments, rerankerNames],
  );
  const names = useMemo(
    () => rerankDeployments.map((d) => d.spec?.name ?? "").filter(Boolean),
    [rerankDeployments],
  );

  const [model, setModel] = useState("");
  const [query, setQuery] = useState("What is the invoice total?");
  const [docs, setDocs] = useState(
    "FACTURE total 5 400 € TTC, échéance 30 jours.\nWeather forecast for tomorrow: light rain.\nInvoice total 5400 EUR, net 30 payment terms.",
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ doc: string; score: number }[] | null>(null);

  useEffect(() => {
    if (names.length === 0) {
      if (model !== "") setModel("");
      return;
    }
    if (!names.includes(model)) setModel(names[0]);
  }, [names, model]);

  async function run() {
    if (!model) {
      setError("No live reranker deployment — deploy a reranker model (family: reranker).");
      return;
    }
    const documents = docs.split("\n").map((d) => d.trim()).filter(Boolean);
    if (documents.length === 0) {
      setError("Add at least one document to rank, one per line.");
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const res: RerankResponse = await rerank(model, query, documents);
      const ranked = (res.results ?? [])
        .map((r) => ({
          doc: documents[r.index ?? -1] ?? "?",
          score: r.relevance_score ?? 0,
        }))
        .sort((a, b) => b.score - a.score);
      setResult(ranked);
    } catch (e) {
      setError(
        e instanceof ApiError || e instanceof ApiUnavailable || e instanceof Error
          ? e.message
          : "Rerank request failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      icon={<ListOrdered className="h-5 w-5" />}
      title="Rerank"
      subtitle="A query and a list of documents in, relevance-ranked out — the retrieval second stage, computed on-node."
    >
      <div className="space-y-4">
        <Field
          label="Reranker deployment"
          hint="Deploy a reranker GGUF (e.g. LiquidAI/LFM2.5-ColBERT-350M-GGUF) with family 'reranker'."
        >
          {names.length === 0 ? (
            <EmptyModelState noun="reranker" onNavigate={onNavigate} />
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

        <Field label="Query">
          <TextInput value={query} onChange={(e) => setQuery(e.target.value)} />
        </Field>
        <Field label="Documents" hint="One document per line.">
          <TextArea rows={5} value={docs} onChange={(e) => setDocs(e.target.value)} />
        </Field>

        {error && (
          <p className="flex items-start gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-600 dark:text-rose-400">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        <Button type="button" loading={busy} onClick={() => void run()} disabled={!model}>
          <ListOrdered className="h-4 w-4" />
          Rerank
        </Button>

        {result && (
          <div className="space-y-2 rounded-md border border-border bg-muted/20 p-4">
            {result.map((r, i) => (
              <div
                key={i}
                className="flex items-start gap-2 rounded-md border border-border bg-card p-2 text-sm"
              >
                <Badge tone={i === 0 ? "ok" : r.score > 0 ? "info" : "neutral"}>
                  {r.score.toFixed(4)}
                </Badge>
                <p className="text-foreground/90">{r.doc}</p>
              </div>
            ))}
            <p className="text-xs text-muted-foreground">
              Sorted by relevance_score, highest first — the same order a retrieval pipeline
              would keep.
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

// First-run empty state: no usable model deployed for the current mode. Names
// the mode, points at Models → Search Hugging Face, and (when a nav callback is
// threaded in) offers a one-click jump to deploy one. Degrades to text-only
// when `onNavigate` is absent.
function EmptyModelState({
  noun,
  onNavigate,
}: {
  noun: "chat" | "vision" | "embedding" | "reranker";
  onNavigate?: NavigateToDeploy;
}) {
  return (
    <div className="space-y-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-3 text-amber-700 dark:text-amber-400">
      <div className="flex items-start gap-2">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
        <div>
          <p className="text-sm font-medium">No {noun} model deployed yet.</p>
          <p className="mt-0.5 text-xs text-amber-700/90 dark:text-amber-400/90">
            Deploy one from Models → Search Hugging Face.
          </p>
        </div>
      </div>
      {onNavigate && (
        <Button
          type="button"
          size="sm"
          variant="secondary"
          onClick={() => onNavigate("deploy", "models")}
        >
          <Boxes className="h-4 w-4" />
          Deploy a model
        </Button>
      )}
    </div>
  );
}

function DeploymentSelect({
  deployments,
  selectable,
  value,
  onChange,
  emptyNoun = "chat",
  onNavigate,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
  selectable: DeploymentRecord[];
  value: string;
  onChange: (name: string) => void;
  emptyNoun?: "chat" | "vision" | "embedding";
  onNavigate?: NavigateToDeploy;
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
    return <EmptyModelState noun={emptyNoun} onNavigate={onNavigate} />;
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
