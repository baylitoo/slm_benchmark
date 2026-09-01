"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Boxes,
  FileText,
  Fingerprint,
  ListOrdered,
  MessageSquare,
  Play,
  Send,
  Sparkles,
  Swords,
  Trash2,
  Upload,
  AlertCircle,
  FilePlus2,
  Gauge,
  Paperclip,
  Pencil,
  RotateCcw,
  Wrench,
  X,
} from "lucide-react";
import {
  triggerExtract,
  chatCompletionMcpStream,
  chatCompletionStream,
  listMcpServers,
  type McpRegisteredServer,
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
  uploadSessionDocument,
  listDynamicSchemas,
  listSchemas,
  getSchemaFields,
  listRoutingPolicies,
  ApiError,
  ApiUnavailable,
  ModelLoading,
  type TriggerResponse,
  type ExtractRequest,
  type DeploymentRecord,
  type StoreEntry,
  type ModelFamily,
  type RerankResponse,
  type DynamicSchemaSummary,
  type RoutingPolicySummary,
  type AgentToolCallTrace,
  type AgentUsageTrace,
  type AgentContextBudgetTrace,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
import { T, useI18n } from "@/lib/i18n";
import { useToast } from "./Toast";
import {
  Alert,
  Button,
  Card,
  Field,
  Segmented,
  Select,
  TextArea,
  TextInput,
  Badge,
  Spinner,
} from "./ui";
import { ResultPanel } from "./ResultPanel";
import { PageHeader } from "./patterns/PageHeader";
import { SchemaBuilderSheet } from "./SchemaBuilderSheet";
import { ToolCallItem } from "./ToolCallTrace";

type PlaygroundMode = "chat" | "arena" | "embedrerank";

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
  const [mode, setMode] = useState<PlaygroundMode>("chat");

  // Auto-refreshing lists shared by every mode — held at the top level so
  // switching modes never re-fetches or remounts a poller.
  const deployments = usePolling<DeploymentRecord[]>(getDeployments, DEPLOY_POLL_MS, active);
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
  // Encoders (analyzers) AND embedding models are excluded from chat/arena:
  // they don't answer chat prompts. Embedding/reranker models live in the
  // Embed/Rerank mode instead. Vision and extraction ride the SAME deployment
  // list — Chat now handles both inline, so there's no separate narrowing.
  const selectable = useMemo(
    () =>
      selectableDeployments(deployments.data ?? []).filter(
        (d) =>
          d.spec?.launch?.runtime !== "encoder" &&
          !(d.spec?.name && embeddingNames.has(d.spec.name)),
      ),
    [deployments.data, embeddingNames],
  );

  return (
    <div>
      <PageHeader
        title="Playground"
        subtitle={
          mode === "chat"
            ? "Chat, attach a file or image, or turn on extraction — one place for every synchronous request."
            : mode === "arena"
              ? "Send one prompt to two deployments side by side and compare their answers."
              : "Compute embeddings or rerank documents with your deployed models (RAG-ready)."
        }
        actions={
          <div className="inline-flex rounded-lg border border-border bg-muted p-0.5 text-sm">
            {(
              [
                ["chat", "Chat", MessageSquare],
                ["arena", "Arena", Swords],
                ["embedrerank", "Embed/Rerank", Fingerprint],
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

      {/* Every mode stays mounted (hidden, never unmounted) so an in-flight
          extraction stream or a chat/arena history survives switching modes. */}
      <div hidden={mode !== "chat"}>
        <ChatPanel
          deployments={deployments}
          selectable={selectable}
          store={store}
          onNavigate={onNavigate}
        />
      </div>
      <div hidden={mode !== "arena"}>
        <ArenaPanel deployments={deployments} selectable={selectable} onNavigate={onNavigate} />
      </div>
      <div hidden={mode !== "embedrerank"}>
        <EmbedRerankPanel
          deployments={deployments.data ?? []}
          embeddingNames={embeddingNames}
          rerankerNames={rerankerNames}
          onNavigate={onNavigate}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chat mode — free-form conversation with a live deployment through the
// generic OpenAI surface (POST /v1/chat/completions), PLUS everything that
// used to be its own Playground tab:
//   - Vision: attach an image or PDF and it rides the same message as
//     multimodal content — no separate mode, no separate model list.
//   - Extraction: a toggle that redirects Send to POST /v1/extract (schema +
//     model-source selection identical to the old standalone Extract form);
//     each extraction becomes its own inline result card in the timeline,
//     interleaved freely with ordinary chat turns.
// ---------------------------------------------------------------------------

interface ChatMsg {
  role: "user" | "assistant" | "status" | "extraction";
  content: string;
  /** Only set when role === "extraction". */
  trigger?: TriggerResponse;
  /** Only set on an assistant turn that ran MCP tools (selectedMcp). */
  toolCalls?: AgentToolCallTrace[];
  /** Reasoning-capable model's "why" for a round (calling a tool, or the
   * final answer) -- only set when the chat template emits one separately
   * from content/tool_calls. One entry per round that had any. */
  reasoning?: string[];
  /** toolCalls and reasoning above, interleaved in the order the SSE events
   * actually arrived -- rendered as one chronological trace instead of two
   * separate blocks that lose which reasoning step preceded which call. */
  trace?: TraceEntry[];
  /** The server-injected system-prompt addendum (TOOL_DISCIPLINE_DIRECTIVE,
   * plus any eager-list context) that run_tool_loop folds on top of the
   * caller's own system prompt -- fires once per request, not once per
   * round, so it's a field on the message rather than a TraceEntry. */
  systemAddendum?: string;
  /** Fires AT MOST ONCE per exchange (#344): cumulative usage crossed the
   * resolved deployment's context-budget warning threshold -- a standing
   * risk for the rest of THIS exchange, not a per-round log entry, so it's
   * a field on the message (like systemAddendum) rather than a TraceEntry. */
  contextBudgetWarning?: AgentContextBudgetTrace;
}

type TraceEntry =
  | { kind: "reasoning"; text: string }
  | { kind: "tool_call"; call: AgentToolCallTrace }
  | { kind: "usage"; usage: AgentUsageTrace };

// A cold store: model's first request can take a while to boot. Auto-retry a
// bounded number of times (the backend's load trigger is idempotent — see
// loadDeployment's own doc comment) rather than making the user notice the
// status message and resend by hand.
const MAX_LOAD_RETRIES = 3;

const VISION_PRESETS = [
  "Extract all the text from this document (OCR).",
  "Describe this image in detail.",
  "What is written in this document? Return it verbatim.",
];

// Exported for tests: rendered by Playground with its polled deployment state.
export function ChatPanel({
  deployments,
  selectable,
  store,
  onNavigate,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
  selectable: DeploymentRecord[];
  store: ReturnType<typeof usePolling<StoreEntry[]>>;
  onNavigate?: NavigateToDeploy;
}) {
  const { t } = useI18n();
  const { toast } = useToast();
  // Chat owns its OWN selection: it only offers LIVE deployments by default,
  // so sharing a broader selection (which legitimately includes evicted
  // auto-reload targets) would let the select DISPLAY one model while the
  // request carries another — the exact mismatch this state split fixes.
  // Evicted deployments stay selectable — a request to one auto-reloads it
  // (load-on-demand), same convention throughout the Studio.
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

  const visionNames = useMemo(() => visionDeploymentNames(store.data), [store.data]);
  const modelHasVision = visionNames.has(model);
  // Explicit toggle, not implicit-forever: an image/PDF attachment only
  // rides the message as image_url content when this is on. Defaults to
  // following the deployment's own capability (vision model -> on), but the
  // user can flip it off even for a vision model (it's the expensive path)
  // -- and it MUST be off for a non-vision model, since llama-server 500s on
  // image content with no mmproj rather than silently ignoring it.
  const [visionEnabled, setVisionEnabled] = useState(false);
  useEffect(() => {
    setVisionEnabled(modelHasVision);
  }, [model, modelHasVision]);

  const [system, setSystem] = useState("");
  const mcpServers = useAsync<McpRegisteredServer[]>("mcp-servers", listMcpServers);
  const [selectedMcp, setSelectedMcp] = useState<string[]>([]);
  // Server-issued once the first attachment is uploaded for docs-search
  // (#296) -- carried for the rest of this conversation so every later
  // upload/chat turn lands in the SAME session directory. Never invented
  // client-side; reset with the rest of the chat state on Clear. A ref, not
  // state: sendChat awaits the upload then immediately calls attempt() in
  // the SAME turn, which needs the just-issued id synchronously -- state
  // set this render wouldn't be visible until the next one.
  const docsSearchSessionIdRef = useRef<string | undefined>(undefined);
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<ChatMsg[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // Index of a user message currently being edited (its content is loaded
  // into the main input box). null when nothing is being edited. Consumed by
  // sendChat: submitting while this is set truncates msgs to everything
  // BEFORE this index instead of appending to the full history (#343).
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  // The exact { next, payload } sendChat last handed to attempt() -- kept so
  // Regenerate can replay the identical request (same messages array) rather
  // than reconstructing it, which would risk drifting from what was actually
  // sent (e.g. a multimodal attachment on that turn) (#343).
  const lastRequestRef = useRef<{
    next: ChatMsg[];
    payload: { role: string; content: unknown }[];
  } | null>(null);

  // --- Attachment (vision + extraction file input share one attach slot) ---
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  // PDF rasterization DPI — higher = sharper text (better for dense documents
  // / small vision models), larger payload. Only used for a PDF attachment in
  // CHAT mode (extraction rasterizes server-side via its own OCR backend).
  const [dpi, setDpi] = useState(200);

  function clearAttachment() {
    if (preview?.startsWith("blob:")) URL.revokeObjectURL(preview);
    setPreview(null);
    setFile(null);
  }

  async function onAttach(f: File | null) {
    clearAttachment();
    setFile(f);
    if (!f) return;
    if (f.type.startsWith("image/")) {
      setPreview(URL.createObjectURL(f));
      return;
    }
    // PDF: rasterize page 1 (low DPI, single page) for a real visual preview
    // — an <img> can't show a PDF directly. Best-effort: on failure the run
    // still renders the pages when sent.
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

  // --- Extraction toggle ---
  const [extractionOn, setExtractionOn] = useState(false);
  const [schemaName, setSchemaName] = useState("invoice");
  const [dynamicSchemaName, setDynamicSchemaName] = useState("");
  const [schemaSheetOpen, setSchemaSheetOpen] = useState(false);
  const schemas = useAsync<string[]>("schemas", listSchemas);
  const dynamicSchemas = useAsync<DynamicSchemaSummary[]>("dynamic-schemas", listDynamicSchemas);
  const routingPolicies = useAsync<RoutingPolicySummary[]>(
    "routing-policies",
    listRoutingPolicies,
  );
  // Field names for whichever schema is currently selected above -- used to
  // build a schema-aware vision preset (see VISION_PRESETS below). A schema
  // counts as "selected" only once it resolves against the lists this same
  // picker already fetched (schemas.data / dynamicSchemas.data); this keeps
  // the fresh-mount / no-schemas-configured case free of a preset that names
  // fields nothing on the server actually recognizes.
  const dynamicSchemaEntry = dynamicSchemaName
    ? (dynamicSchemas.data ?? []).find((s) => s.name === dynamicSchemaName)
    : undefined;
  const builtinSchemaKnown =
    !dynamicSchemaName && !!schemaName && (schemas.data ?? []).includes(schemaName);
  const builtinSchemaFields = useAsync<string[]>(
    `schema-fields:${builtinSchemaKnown ? schemaName : ""}`,
    () => (builtinSchemaKnown ? getSchemaFields(schemaName) : Promise.resolve([])),
  );
  const selectedSchemaFields =
    dynamicSchemaEntry?.spec.fields.map((f) => f.name) ??
    (builtinSchemaKnown ? (builtinSchemaFields.data ?? []) : []);
  // "single" routes to the deployment picked above; "policy" runs a saved
  // routing policy (cheap stage first, escalate on the policy's own
  // confidence rules) — mutually exclusive with a single deployment, same as
  // the backend's own contract. Only meaningful while extraction is on.
  const [modelSource, setModelSource] = useState<"single" | "policy">("single");
  const [selectedPolicy, setSelectedPolicy] = useState<string>("");
  const [ocrBackend, setOcrBackend] = useState("");
  const [language, setLanguage] = useState("");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [msgs, busy]);

  function isPdfFile(f: File): boolean {
    return f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf");
  }

  async function runExtraction() {
    if (busy) return;
    const trimmed = input.trim();
    const attached = file;
    if (!trimmed && !attached) {
      setError("Paste some document text first, or attach a file.");
      return;
    }
    const payload: ExtractRequest = dynamicSchemaName
      ? { schema_name: schemaName || "invoice", dynamic_schema_name: dynamicSchemaName }
      : { schema_name: schemaName || "invoice" };
    if (modelSource === "policy") {
      if (!selectedPolicy) {
        setError("Pick a routing policy, or switch back to a single model.");
        return;
      }
      payload.routing_policy = selectedPolicy;
    } else if (model) {
      payload.deployment = model;
    }
    if (ocrBackend.trim()) payload.ocr_backend = ocrBackend.trim();
    if (language.trim()) payload.language = language.trim();

    let displayLabel: string;
    try {
      if (attached) {
        payload.content_b64 = await fileToBase64(attached);
        payload.filename = attached.name;
        displayLabel = `📎 ${attached.name}`;
      } else {
        payload.text = trimmed;
        displayLabel = trimmed;
      }
    } catch {
      setError("Could not read the attached file.");
      return;
    }

    setError(null);
    setInput("");
    clearAttachment();
    setBusy(true);
    try {
      const res = await triggerExtract(payload);
      setMsgs((prev) => [
        ...prev,
        { role: "user", content: displayLabel },
        { role: "extraction", content: dynamicSchemaName || schemaName || "invoice", trigger: res },
      ]);
      toast({ title: "Extraction started", description: res.channel, tone: "success" });
    } catch (e) {
      const msg =
        e instanceof ApiUnavailable
          ? "The extract endpoint isn't reachable. Is the backend running and NEXT_PUBLIC_API_BASE correct?"
          : e instanceof ApiError || e instanceof Error
            ? e.message
            : "Something went wrong.";
      setError(msg);
      toast({ title: "Extraction failed", description: msg, tone: "error" });
    } finally {
      setBusy(false);
    }
  }

  async function sendChat() {
    if (busy) return;
    const trimmed = input.trim();
    const attached = file;
    if (!trimmed && !attached) return;
    if (!model) {
      setError("No deployment selected — deploy a model under Serving → Models first.");
      return;
    }
    setError(null);

    // Editing an earlier user message (#343): everything from that message
    // onward is dropped -- the edited text becomes the new next message, as
    // if the conversation forked at that point. A plain send (editingIndex
    // null) keeps the full history.
    const baseMsgs = editingIndex !== null ? msgs.slice(0, editingIndex) : msgs;

    // Prior turns replay as plain text (an attachment from an earlier turn
    // isn't re-sent — only its display label is kept, same convention every
    // OpenAI-style chat playground uses for image history).
    const priorHistory = baseMsgs
      .filter((m): m is ChatMsg & { role: "user" | "assistant" } =>
        m.role === "user" || m.role === "assistant",
      )
      .map((m) => ({ role: m.role, content: m.content }));

    // Vision is now an explicit toggle (defaults to following the
    // deployment's own capability, see the effect above) -- an image
    // attachment with vision off has nothing to ride on (docs-search only
    // accepts .pdf/.txt), and a PDF with vision off AND docs-search
    // unselected has no path to reach the model either. Both are refused
    // up front rather than silently sending a useless/erroring attachment
    // (llama-server 500s on image content with no mmproj rather than
    // ignoring it).
    if (attached && !visionEnabled) {
      if (!isPdfFile(attached)) {
        setError(
          "This deployment has no vision (or Vision is off) — image attachments need Vision on.",
        );
        return;
      }
      if (!selectedMcp.includes("docs-search")) {
        setError(
          "Vision is off and docs-search isn't selected — turn Vision on, or select docs-search so this PDF can be read.",
        );
        return;
      }
    }

    let newContent: unknown = trimmed;
    let displayLabel = trimmed;
    if (attached) {
      try {
        const b64 = await fileToBase64(attached);
        // Additive, not instead-of, when vision IS on: vision still reads
        // the rendered page images below regardless. Only a real .pdf is
        // worth indexing for docs-search (#296) — an image attachment has
        // no text to search, and isn't a suffix docs-search accepts anyway.
        if (isPdfFile(attached) && selectedMcp.includes("docs-search")) {
          try {
            const uploaded = await uploadSessionDocument(
              b64,
              attached.name,
              docsSearchSessionIdRef.current,
            );
            docsSearchSessionIdRef.current = uploaded.session_id;
          } catch {
            // Best-effort: docs-search just won't see this file, vision
            // still answers from the page images below either way.
          }
        }
        if (visionEnabled) {
          const imageUrls = isPdfFile(attached)
            ? (await renderDocument(b64, attached.name, dpi)).images
            : [`data:${attached.type || "image/png"};base64,${b64}`];
          if (imageUrls.length === 0) {
            setError("The document produced no page images.");
            return;
          }
          newContent = [
            { type: "text", text: trimmed || "Describe this image." },
            ...imageUrls.map((url) => ({ type: "image_url" as const, image_url: { url } })),
          ];
        } else {
          // Vision off, PDF, docs-search selected (guarded above): the
          // upload just made the real file searchable -- no image content
          // to send, the model reads it via docs-search's tools instead.
          newContent = trimmed || "Look up the attached document via docs-search.";
        }
        displayLabel = trimmed || `📎 ${attached.name}`;
      } catch {
        setError("Could not read the attached file.");
        return;
      }
    }

    const next: ChatMsg[] = [...baseMsgs, { role: "user", content: displayLabel }];
    setMsgs(next);
    setInput("");
    setEditingIndex(null);
    clearAttachment();
    setBusy(true);
    const payload = [
      ...(system.trim() ? [{ role: "system", content: system.trim() }] : []),
      ...priorHistory,
      { role: "user", content: newContent },
    ];
    lastRequestRef.current = { next, payload };
    await attempt(next, payload, 0);
    setBusy(false);
  }

  // Regenerate (#343): replays the exact same request the last completed
  // turn sent -- same messages array, no new params -- and REPLACES the last
  // assistant message with the new response instead of appending a
  // duplicate. attempt() already does that replacement whenever the message
  // at `next.length` exists, which is exactly the last assistant reply here.
  async function regenerate() {
    if (busy) return;
    const last = lastRequestRef.current;
    if (!last) return;
    setError(null);
    setBusy(true);
    await attempt(last.next, last.payload, 0);
    setBusy(false);
  }

  function startEdit(i: number) {
    if (busy) return;
    const target = msgs[i];
    if (!target || target.role !== "user") return;
    setEditingIndex(i);
    setInput(target.content);
    clearAttachment();
    inputRef.current?.focus();
  }

  function cancelEdit() {
    setEditingIndex(null);
    setInput("");
  }

  async function attempt(
    next: ChatMsg[],
    payload: { role: string; content: unknown }[],
    retryCount: number,
  ) {
    try {
      if (selectedMcp.length > 0) {
        // Each tool call arrives as its own SSE event the instant it
        // finishes -- the trace renders progressively instead of the whole
        // exchange completing silently behind "Waiting for the model…".
        const liveToolCalls: AgentToolCallTrace[] = [];
        const liveReasoning: string[] = [];
        const liveUsage: AgentUsageTrace[] = [];
        const liveTrace: TraceEntry[] = [];
        let liveSystemAddendum: string | undefined;
        let liveContextBudget: AgentContextBudgetTrace | undefined;
        const patchLiveMsg = () => {
          setMsgs((prev) => {
            const patch = {
              content: "",
              toolCalls: [...liveToolCalls],
              trace: [...liveTrace],
              ...(liveReasoning.length > 0 ? { reasoning: [...liveReasoning] } : {}),
              ...(liveSystemAddendum ? { systemAddendum: liveSystemAddendum } : {}),
              ...(liveContextBudget ? { contextBudgetWarning: liveContextBudget } : {}),
            };
            return prev.length <= next.length
              ? [...next, { role: "assistant", ...patch }]
              : prev.map((m, i) => (i === next.length ? { ...m, ...patch } : m));
          });
        };
        const onToolCall = (call: AgentToolCallTrace) => {
          liveToolCalls.push(call);
          liveTrace.push({ kind: "tool_call", call });
          patchLiveMsg();
        };
        const onReasoning = (text: string) => {
          liveReasoning.push(text);
          liveTrace.push({ kind: "reasoning", text });
          patchLiveMsg();
        };
        const onSystemAddendum = (text: string) => {
          liveSystemAddendum = text;
          patchLiveMsg();
        };
        const onUsage = (usage: AgentUsageTrace) => {
          liveUsage.push(usage);
          liveTrace.push({ kind: "usage", usage });
          patchLiveMsg();
        };
        const onContextBudget = (budget: AgentContextBudgetTrace) => {
          liveContextBudget = budget;
          patchLiveMsg();
        };
        const res = await chatCompletionMcpStream(
          model,
          payload,
          selectedMcp,
          onToolCall,
          docsSearchSessionIdRef.current,
          onReasoning,
          onSystemAddendum,
          onUsage,
          onContextBudget,
        );
        const answer = res.choices?.[0]?.message?.content ?? "";
        const toolCalls = res.docie_agent?.tool_calls ?? liveToolCalls;
        // The live trace is only trustworthy when every reported tool call
        // actually streamed as its own onToolCall event -- if the caller
        // resolved with a completion's docie_agent.tool_calls that never
        // streamed (e.g. a non-streaming response), fall back to reasoning
        // then tool calls in report order rather than silently dropping them.
        const finalTrace: TraceEntry[] =
          toolCalls.length === liveToolCalls.length
            ? liveTrace
            : [
                ...liveReasoning.map((text): TraceEntry => ({ kind: "reasoning", text })),
                ...liveUsage.map((usage): TraceEntry => ({ kind: "usage", usage })),
                ...toolCalls.map((call): TraceEntry => ({ kind: "tool_call", call })),
              ];
        const finalMsg: ChatMsg = {
          role: "assistant",
          content: answer || t("(empty response)"),
          ...(toolCalls.length > 0 ? { toolCalls } : {}),
          ...(liveReasoning.length > 0 ? { reasoning: liveReasoning } : {}),
          ...(finalTrace.length > 0 ? { trace: finalTrace } : {}),
          ...(liveSystemAddendum ? { systemAddendum: liveSystemAddendum } : {}),
          ...(liveContextBudget ? { contextBudgetWarning: liveContextBudget } : {}),
        };
        setMsgs((prev) =>
          prev.length <= next.length
            ? [...next, finalMsg]
            : prev.map((m, i) => (i === next.length ? finalMsg : m)),
        );
        return;
      }
      let content = "";
      const appendToken = (token: string) => {
        content += token;
        setMsgs((prev) =>
          prev.length <= next.length
            ? [...next, { role: "assistant", content }]
            : prev.map((m, i) => (i === next.length ? { role: "assistant", content } : m)),
        );
      };
      await chatCompletionStream(model, payload, appendToken);
      if (!content) setMsgs([...next, { role: "assistant", content: t("(empty response)") }]);
    } catch (e) {
      if (e instanceof ModelLoading) {
        const willRetry = retryCount < MAX_LOAD_RETRIES;
        setMsgs([
          ...next,
          {
            role: "status",
            content: willRetry
              ? `${e.message} ${t("Retrying automatically…")}`
              : `${e.message} ${t("Still starting — send your message again in a bit.")}`,
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

  function submit() {
    void (extractionOn ? runExtraction() : sendChat());
  }

  const showVisionExtras = !extractionOn && file && !file.type.startsWith("image/");

  return (
    <Card
      icon={<MessageSquare className="h-5 w-5" />}
      title="Chat"
      subtitle="Free-form conversation, file/image attachments, and extraction — all in one timeline."
    >
      <div className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          {extractionOn && modelSource === "policy" ? (
            <Field
              label="Routing policy"
              hint="The document goes to the first stage and escalates to later stages on the policy's confidence/validity rules and budgets. The result carries the routing audit."
            >
              <Select
                value={selectedPolicy}
                onChange={(e) => setSelectedPolicy(e.target.value)}
                aria-label="Routing policy"
              >
                <option value="">
                  {routingPolicies.data && routingPolicies.data.length === 0
                    ? "(no saved policies — create one in Benchmark)"
                    : "(pick a policy)"}
                </option>
                {(routingPolicies.data ?? []).map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
              </Select>
            </Field>
          ) : (
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
                disabled={busy}
              />
            </Field>
          )}
          <Field label="System prompt" hint="Optional — applied to the whole conversation.">
            <TextInput
              value={system}
              onChange={(e) => setSystem(e.target.value)}
              placeholder="You are…"
              disabled={extractionOn}
            />
          </Field>
        </div>

        <div className="flex flex-wrap items-center gap-3 rounded-md border border-border bg-muted/20 px-3 py-2">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={extractionOn}
              onChange={(e) => {
                setExtractionOn(e.target.checked);
                // Extraction turns don't participate in edit/truncate — drop
                // any in-progress chat-message edit rather than leaving it
                // to silently apply (or not) to a mode it wasn't meant for.
                if (e.target.checked) cancelEdit();
              }}
              className="h-4 w-4 rounded border-border"
            />
            <Sparkles className="h-4 w-4 text-muted-foreground" />
            <span className="font-medium"><T>Extraction</T></span>
          </label>
          <span className="text-xs text-muted-foreground">
            <T>
              On: Send routes to constrained-generation extraction (schema-validated JSON) instead
              of a chat reply.
            </T>
          </span>
        </div>

        {extractionOn && (
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Schema">
              <div className="flex items-center gap-2">
                <Select
                  className="flex-1"
                  aria-label="Schema"
                  value={dynamicSchemaName ? `d:${dynamicSchemaName}` : `s:${schemaName}`}
                  onChange={(e) => {
                    const [kind, name] = [
                      e.target.value.slice(0, 2),
                      e.target.value.slice(2),
                    ];
                    if (kind === "d:") {
                      setDynamicSchemaName(name);
                    } else {
                      setSchemaName(name);
                      setDynamicSchemaName("");
                    }
                  }}
                >
                  <optgroup label="Built-in">
                    {(schemas.data ?? []).map((s) => (
                      <option key={s} value={`s:${s}`}>
                        {s}
                      </option>
                    ))}
                  </optgroup>
                  {(dynamicSchemas.data ?? []).length > 0 && (
                    <optgroup label="Saved">
                      {(dynamicSchemas.data ?? []).map((s) => (
                        <option key={s.name} value={`d:${s.name}`}>
                          {s.name}
                        </option>
                      ))}
                    </optgroup>
                  )}
                </Select>
                <button
                  type="button"
                  onClick={() => setSchemaSheetOpen(true)}
                  className="inline-flex shrink-0 items-center gap-1 text-xs text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground"
                >
                  <FilePlus2 className="h-3 w-3" />
                  <T>New schema…</T>
                </button>
              </div>
            </Field>
            <SchemaBuilderSheet
              open={schemaSheetOpen}
              onClose={() => setSchemaSheetOpen(false)}
              onCreated={(name) => {
                setDynamicSchemaName(name);
                dynamicSchemas.reload();
              }}
            />
            <Field
              label="Model source"
              hint="Route to the deployment above, or to a saved routing policy instead."
            >
              <Segmented
                value={modelSource}
                onChange={setModelSource}
                options={[
                  { value: "single", label: "Single model" },
                  { value: "policy", label: "Routing policy" },
                ]}
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
        )}

        {model && !extractionOn && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-muted-foreground">
              <T>Vision:</T>
            </span>
            <button
              type="button"
              aria-pressed={visionEnabled}
              disabled={busy || !modelHasVision}
              onClick={() => setVisionEnabled((v) => !v)}
              title={
                modelHasVision
                  ? t("An attached image/PDF rides the message as page images.")
                  : t("This deployment has no vision support (no mmproj).")
              }
              className={cn(
                "rounded-full border px-2.5 py-0.5 text-xs transition-colors",
                visionEnabled
                  ? "border-accent bg-accent text-accent-foreground"
                  : "border-border bg-card text-muted-foreground hover:text-foreground",
                !modelHasVision && "cursor-not-allowed opacity-50",
              )}
            >
              {visionEnabled ? <T>on</T> : <T>off</T>}
            </button>
          </div>
        )}

        {(mcpServers.data ?? []).length > 0 && !extractionOn && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-muted-foreground">
              <T>Tools:</T>
            </span>
            {(mcpServers.data ?? []).map((server) => {
              const on = selectedMcp.includes(server.name);
              return (
                <button
                  key={server.name}
                  type="button"
                  aria-pressed={on}
                  disabled={busy}
                  onClick={() =>
                    setSelectedMcp((prev) =>
                      on ? prev.filter((n) => n !== server.name) : [...prev, server.name],
                    )
                  }
                  className={cn(
                    "rounded-full border px-2.5 py-0.5 text-xs transition-colors",
                    on
                      ? "border-accent bg-accent text-accent-foreground"
                      : "border-border bg-card text-muted-foreground hover:text-foreground",
                  )}
                >
                  {server.name}
                </button>
              );
            })}
            {selectedMcp.length > 0 && (
              <span className="text-xs text-muted-foreground">
                <T>tool calls stream live; the final answer arrives in one piece</T>
              </span>
            )}
          </div>
        )}

        <div className="scroll-thin max-h-[50vh] min-h-40 space-y-3 overflow-y-auto rounded-md border border-border bg-muted/20 p-4">
          {msgs.length === 0 && (
            <p className="text-sm text-muted-foreground">
              <T>
                Ask anything, attach an image or PDF, or turn on extraction — all through the same
                deployment.
              </T>
            </p>
          )}
          {msgs.map((m, i) =>
            m.role === "status" ? (
              <p key={i} className="flex items-center gap-2 text-xs italic text-muted-foreground">
                <Spinner /> {m.content}
              </p>
            ) : m.role === "extraction" ? (
              <div key={i} className="rounded-lg border border-border bg-card">
                <div className="flex items-center gap-2 border-b border-border px-3 py-1.5 text-xs text-muted-foreground">
                  <Sparkles className="h-3.5 w-3.5" />
                  <T>Extraction</T> · {m.content}
                </div>
                <div className="p-3">
                  {m.trigger && <ResultPanel trigger={m.trigger} noun="extraction" />}
                </div>
              </div>
            ) : (
              <div
                key={i}
                className={cn(
                  "flex flex-col gap-1.5",
                  m.role === "user" ? "items-end" : "items-start",
                )}
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
                {m.role === "user" && (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => startEdit(i)}
                    className="inline-flex items-center gap-1 text-xs text-muted-foreground transition hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                    title={t("Edit this message and resend (drops everything after it)")}
                  >
                    <Pencil className="h-3 w-3" />
                    <T>Edit</T>
                  </button>
                )}
                {m.role === "assistant" && i === msgs.length - 1 && (
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void regenerate()}
                    className="inline-flex items-center gap-1 text-xs text-muted-foreground transition hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                    title={t("Resend the same request and replace this answer")}
                  >
                    <RotateCcw className="h-3 w-3" />
                    <T>Regenerate</T>
                  </button>
                )}
                {m.contextBudgetWarning && (
                  <Alert tone="warn" className="w-full max-w-[85%]">
                    <span>
                      <T>Context budget warning:</T> {m.contextBudgetWarning.cumulative_tokens} /{" "}
                      {m.contextBudgetWarning.context_length}{" "}
                      <T>tokens used</T> ({Math.round(m.contextBudgetWarning.threshold_fraction * 100)}%{" "}
                      <T>threshold</T>) —{" "}
                      <T>this exchange is at risk of overflowing the deployment's context window.</T>
                    </span>
                  </Alert>
                )}
                {m.systemAddendum && (
                  <details className="w-full max-w-[85%] rounded-md border border-border bg-muted/20 text-xs">
                    <summary className="cursor-pointer px-2 py-1 font-medium text-muted-foreground hover:text-foreground">
                      <T>System-prompt addendum</T>
                    </summary>
                    <pre className="scroll-thin max-h-48 overflow-auto whitespace-pre-wrap border-t border-border px-2 py-1.5 text-foreground/80">
                      {m.systemAddendum}
                    </pre>
                  </details>
                )}
                {m.trace && m.trace.length > 0 && (
                  <div className="w-full max-w-[85%] space-y-1.5">
                    <p className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
                      <Wrench className="h-3.5 w-3.5" />
                      <T>Trace</T>
                    </p>
                    <ol className="space-y-1.5">
                      {(() => {
                        let toolIndex = -1;
                        return m.trace.map((entry, i) =>
                          entry.kind === "reasoning" ? (
                            <li
                              key={i}
                              className="rounded-md border border-border bg-muted/20 px-2 py-1 text-xs italic text-muted-foreground"
                            >
                              {entry.text}
                            </li>
                          ) : entry.kind === "usage" ? (
                            <li
                              key={i}
                              className="flex items-center gap-1.5 rounded-md border border-border bg-muted/20 px-2 py-1 text-xs text-muted-foreground"
                            >
                              <Gauge className="h-3.5 w-3.5" />
                              <span>
                                {entry.usage.round.prompt_tokens ?? 0} <T>prompt tokens</T> ·{" "}
                                {entry.usage.cumulative.total_tokens ?? 0} <T>total</T>
                              </span>
                            </li>
                          ) : (
                            <ToolCallItem key={i} call={entry.call} index={++toolIndex} />
                          ),
                        );
                      })()}
                    </ol>
                  </div>
                )}
              </div>
            ),
          )}
          {busy && (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <Spinner /> <T>Waiting for the model…</T>
            </p>
          )}
          <div ref={bottomRef} />
        </div>

        {previewLoading && (
          <p className="flex items-center gap-2 text-xs text-muted-foreground">
            <Spinner /> <T>Rendering preview…</T>
          </p>
        )}
        {preview && (
          <div className="flex items-start gap-3 rounded-md border border-border bg-muted/20 p-2">
            <img
              src={preview}
              alt="attachment preview"
              className="h-20 w-20 shrink-0 rounded-md border border-border object-cover"
            />
            <div className="min-w-0 flex-1 space-y-1">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="truncate">{file?.name}</span>
                <button
                  type="button"
                  onClick={clearAttachment}
                  className="shrink-0 text-muted-foreground hover:text-foreground"
                  aria-label={t("Remove attachment")}
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
              {showVisionExtras && (
                <label className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <span><T>Render quality</T></span>
                  <Select
                    value={String(dpi)}
                    onChange={(e) => setDpi(Number(e.target.value))}
                    className="h-7 w-auto text-xs"
                  >
                    <option value="150">150 DPI — faster, smaller</option>
                    <option value="200">200 DPI — recommended</option>
                    <option value="300">300 DPI — sharpest, heaviest</option>
                  </Select>
                </label>
              )}
              {!extractionOn && file && (
                <div className="flex flex-wrap gap-1">
                  {[
                    ...VISION_PRESETS,
                    ...(selectedSchemaFields.length > 0
                      ? [`Extract: ${selectedSchemaFields.join(", ")}`]
                      : []),
                  ].map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setInput(p)}
                      className="rounded-md border border-border px-2 py-0.5 text-xs text-muted-foreground transition hover:bg-muted hover:text-foreground"
                    >
                      {p.length > 32 ? `${p.slice(0, 32)}…` : p}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {error && <Alert tone="err">{error}</Alert>}

        {editingIndex !== null && (
          <div className="flex items-center justify-between gap-2 rounded-md border border-accent/40 bg-accent/10 px-3 py-1.5 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <Pencil className="h-3.5 w-3.5" />
              <T>
                Editing an earlier message — sending will drop everything after it.
              </T>
            </span>
            <button
              type="button"
              onClick={cancelEdit}
              className="shrink-0 font-medium text-muted-foreground hover:text-foreground"
            >
              <T>Cancel</T>
            </button>
          </div>
        )}

        <div className="flex items-end gap-2">
          <label
            className={cn(
              "flex h-[4.5rem] shrink-0 cursor-pointer items-center justify-center rounded-md border border-dashed border-border px-3 text-muted-foreground transition hover:border-accent hover:text-foreground",
              file && "border-accent text-accent",
            )}
            title="Attach an image or PDF"
          >
            <Paperclip className="h-4 w-4" />
            <input
              type="file"
              accept=".pdf,image/*"
              onChange={(e) => void onAttach(e.target.files?.[0] ?? null)}
              className="sr-only"
            />
          </label>
          <TextArea
            ref={inputRef}
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={
              extractionOn
                ? "Paste document text, or attach a file (Enter to run)…"
                : editingIndex !== null
                  ? "Edit your message (Enter to resend, Shift+Enter for a new line)…"
                  : "Type a message, attach an image/PDF (Enter to send, Shift+Enter for a new line)…"
            }
          />
          <Button type="button" loading={busy} disabled={chatNames.length === 0} onClick={submit}>
            {extractionOn ? <Play className="h-4 w-4" /> : <Send className="h-4 w-4" />}
            {extractionOn ? "Run extraction" : "Send"}
          </Button>
          <Button
            type="button"
            variant="secondary"
            disabled={msgs.length === 0 || busy}
            onClick={() => {
              setMsgs([]);
              setError(null);
              docsSearchSessionIdRef.current = undefined;
              lastRequestRef.current = null;
              setEditingIndex(null);
              setInput("");
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
// Arena mode — one prompt, two deployments, answers streamed side by side.
// Both sides receive the exact same message list (system + shared history +
// the new prompt) concurrently. Multi-turn: every turn stores BOTH answers,
// and a per-turn "Continue from" control picks which side's answer feeds the
// next turn as assistant history (default: left) — the conversation stays
// coherent while still exposing where the two models diverge.
// ---------------------------------------------------------------------------

type ArenaSide = 0 | 1;

interface ArenaAnswer {
  /** Deployment this side used when the turn was sent (the picker may change later). */
  model: string;
  content: string;
  status: "streaming" | "loading" | "done" | "error";
  error?: string;
  elapsedMs: number;
}

interface ArenaTurn {
  prompt: string;
  /** [left, right] */
  answers: [ArenaAnswer, ArenaAnswer];
  /** Which side's answer feeds later turns as assistant history. */
  historySide: ArenaSide;
}

function formatElapsed(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

// Exported for tests: rendered by Playground with its polled deployment state.
export function ArenaPanel({
  deployments,
  selectable,
  onNavigate,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
  selectable: DeploymentRecord[];
  onNavigate?: NavigateToDeploy;
}) {
  const { t } = useI18n();
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

  // Two independent selections, one per side. Same live-first preference as
  // Chat, and the right side defaults to a DIFFERENT model when one exists —
  // comparing a model against itself is allowed, just not the default.
  const [modelA, setModelA] = useState("");
  const [modelB, setModelB] = useState("");
  useEffect(() => {
    if (chatNames.length === 0) {
      if (modelA !== "") setModelA("");
      if (modelB !== "") setModelB("");
      return;
    }
    const preferred = [
      ...liveNames,
      ...chatNames.filter((n) => !liveNames.includes(n)),
    ];
    if (!chatNames.includes(modelA)) setModelA(preferred[0]);
    if (!chatNames.includes(modelB)) setModelB(preferred[1] ?? preferred[0]);
  }, [chatNames, liveNames, modelA, modelB]);

  const [system, setSystem] = useState("");
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<ArenaTurn[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [turns, busy]);

  function patchAnswer(turnIndex: number, side: ArenaSide, patch: Partial<ArenaAnswer>) {
    setTurns((prev) =>
      prev.map((turn, i) =>
        i === turnIndex
          ? {
              ...turn,
              answers: turn.answers.map((a, j) =>
                j === side ? { ...a, ...patch } : a,
              ) as [ArenaAnswer, ArenaAnswer],
            }
          : turn,
      ),
    );
  }

  async function runSide(
    turnIndex: number,
    side: ArenaSide,
    model: string,
    payload: { role: string; content: unknown }[],
    retryCount = 0,
  ): Promise<void> {
    const started = performance.now();
    try {
      let content = "";
      await chatCompletionStream(model, payload, (token) => {
        content += token;
        patchAnswer(turnIndex, side, {
          content,
          status: "streaming",
          elapsedMs: performance.now() - started,
        });
      });
      patchAnswer(turnIndex, side, {
        content: content || t("(empty response)"),
        status: "done",
        elapsedMs: performance.now() - started,
      });
    } catch (e) {
      if (e instanceof ModelLoading && retryCount < MAX_LOAD_RETRIES) {
        // Same bounded cold-start retry as Chat: the backend's load trigger
        // is idempotent, so re-sending after the ETA is safe. The elapsed
        // clock restarts per attempt — it measures the answering attempt,
        // not the cold-start wait.
        patchAnswer(turnIndex, side, { status: "loading" });
        const waitMs = Math.min(Math.max(e.etaSeconds, 2), 30) * 1000;
        await new Promise((resolve) => setTimeout(resolve, waitMs));
        return runSide(turnIndex, side, model, payload, retryCount + 1);
      }
      const msg =
        e instanceof ModelLoading
          ? `${e.message} ${t("Still starting — send your message again in a bit.")}`
          : e instanceof ApiError || e instanceof ApiUnavailable || e instanceof Error
            ? e.message
            : "Chat request failed.";
      patchAnswer(turnIndex, side, {
        status: "error",
        error: msg,
        elapsedMs: performance.now() - started,
      });
    }
  }

  // Shared history: each finished turn contributes its user prompt plus ONE
  // assistant answer — the side picked by that turn's "Continue from" control,
  // falling back to the other side when the picked one failed. A turn where
  // both sides failed contributes nothing (it never happened, history-wise).
  function historyMessages(): { role: string; content: string }[] {
    const out: { role: string; content: string }[] = [];
    for (const turn of turns) {
      const picked = turn.answers[turn.historySide];
      const other = turn.answers[turn.historySide === 0 ? 1 : 0];
      const answer =
        picked.status === "done" ? picked : other.status === "done" ? other : null;
      if (!answer) continue;
      out.push({ role: "user", content: turn.prompt });
      out.push({ role: "assistant", content: answer.content });
    }
    return out;
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    if (!modelA || !modelB) {
      setError("No deployment selected — deploy a model under Serving → Models first.");
      return;
    }
    setError(null);
    const payload = [
      ...(system.trim() ? [{ role: "system", content: system.trim() }] : []),
      ...historyMessages(),
      { role: "user", content: text },
    ];
    const turnIndex = turns.length;
    const blank = (model: string): ArenaAnswer => ({
      model,
      content: "",
      status: "streaming",
      elapsedMs: 0,
    });
    setTurns((prev) => [
      ...prev,
      { prompt: text, historySide: 0, answers: [blank(modelA), blank(modelB)] },
    ]);
    setInput("");
    setBusy(true);
    // Both sides get the exact same payload, concurrently. Each side settles
    // on its own — one failing never cancels the other.
    await Promise.all([
      runSide(turnIndex, 0, modelA, payload),
      runSide(turnIndex, 1, modelB, payload),
    ]);
    setBusy(false);
  }

  return (
    <Card
      icon={<Swords className="h-5 w-5" />}
      title="Arena"
      subtitle="One prompt, two deployments — the same conversation runs against both, answers stream side by side."
    >
      <div className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Left model" hint="Feeds the shared history by default.">
            <DeploymentSelect
              deployments={deployments}
              selectable={selectable}
              value={modelA}
              onChange={setModelA}
              emptyNoun="chat"
              onNavigate={onNavigate}
              disabled={busy}
            />
          </Field>
          <Field label="Right model">
            <DeploymentSelect
              deployments={deployments}
              selectable={selectable}
              value={modelB}
              onChange={setModelB}
              emptyNoun="chat"
              onNavigate={onNavigate}
              disabled={busy}
            />
          </Field>
        </div>
        <Field label="System prompt" hint="Optional — applied to both sides.">
          <TextInput
            value={system}
            onChange={(e) => setSystem(e.target.value)}
            placeholder="You are…"
          />
        </Field>

        <div className="scroll-thin max-h-[55vh] min-h-40 space-y-4 overflow-y-auto rounded-md border border-border bg-muted/20 p-4">
          {turns.length === 0 && (
            <p className="text-sm text-muted-foreground">
              <T>Send one prompt to both deployments and compare the answers side by side.</T>
            </p>
          )}
          {turns.map((turn, i) => (
            <div key={i} className="space-y-2">
              <div className="flex justify-end">
                <div className="max-w-[85%] whitespace-pre-wrap rounded-lg bg-accent px-3 py-2 text-sm text-accent-foreground">
                  {turn.prompt}
                </div>
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {turn.answers.map((a, side) => (
                  <div
                    key={side}
                    className="flex flex-col rounded-lg border border-border bg-card"
                  >
                    <div className="flex items-center gap-2 border-b border-border px-3 py-1.5">
                      <span className="truncate text-xs font-medium text-foreground">
                        {a.model}
                      </span>
                      {turn.historySide === side && <Badge tone="info">history</Badge>}
                      <span className="ml-auto shrink-0 text-xs tabular-nums text-muted-foreground">
                        {a.status === "loading" ? <Spinner /> : formatElapsed(a.elapsedMs)}
                      </span>
                    </div>
                    <div className="px-3 py-2 text-sm">
                      {a.status === "error" ? (
                        <p className="text-red-600 dark:text-red-400">{a.error}</p>
                      ) : a.status === "loading" ? (
                        <p className="flex items-center gap-2 text-xs italic text-muted-foreground">
                          <Spinner /> <T>Model is loading — retrying automatically…</T>
                        </p>
                      ) : (
                        <div className="whitespace-pre-wrap text-foreground/90">
                          {a.content || (a.status === "streaming" ? "…" : "")}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div className="flex items-center justify-end gap-2">
                <span className="text-xs text-muted-foreground">
                  <T>Continue from</T>
                </span>
                <Segmented
                  value={turn.historySide === 0 ? "left" : "right"}
                  onChange={(v) =>
                    setTurns((prev) =>
                      prev.map((tt, j) =>
                        j === i ? { ...tt, historySide: v === "left" ? 0 : 1 } : tt,
                      ),
                    )
                  }
                  options={[
                    { value: "left", label: "Left" },
                    { value: "right", label: "Right" },
                  ]}
                />
              </div>
            </div>
          ))}
          {busy && (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <Spinner /> <T>Waiting for the models…</T>
            </p>
          )}
          <div ref={bottomRef} />
        </div>

        {error && <Alert tone="err">{error}</Alert>}

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
            placeholder="Type a message for both models (Enter to send, Shift+Enter for a new line)…"
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
            disabled={turns.length === 0 || busy}
            onClick={() => {
              setTurns([]);
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
// Embed/Rerank mode — one panel, a sub-mode toggle. The deployment list is
// FILTERED per sub-mode (embedding-family vs reranker-family names) — a hard
// boundary via filtering, not just a label, so the picker can never offer a
// model the chosen operation can't run.
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

type EmbedRerankMode = "embed" | "rerank";

// Exported for tests: rendered by Playground with its polled deployment state.
export function EmbedRerankPanel({
  deployments,
  embeddingNames,
  rerankerNames,
  onNavigate,
}: {
  deployments: DeploymentRecord[];
  embeddingNames: Set<string>;
  rerankerNames: Set<string>;
  onNavigate?: NavigateToDeploy;
}) {
  const [subMode, setSubMode] = useState<EmbedRerankMode>("embed");
  const activeNames = subMode === "embed" ? embeddingNames : rerankerNames;

  // Same as every other picker: an evicted deployment reloads on the next
  // request, so include selectable (live + managed-evicted), not only live.
  const modeDeployments = useMemo(
    () =>
      selectableDeployments(deployments).filter(
        (d) => d.spec?.name && activeNames.has(d.spec.name),
      ),
    [deployments, activeNames],
  );
  const names = useMemo(
    () => modeDeployments.map((d) => d.spec?.name ?? "").filter(Boolean),
    [modeDeployments],
  );

  const [model, setModel] = useState("");
  useEffect(() => {
    if (names.length === 0) {
      if (model !== "") setModel("");
      return;
    }
    if (!names.includes(model)) setModel(names[0]);
  }, [names, model]);

  // Embed sub-state
  const [textA, setTextA] = useState("Facture 5 400 € TTC, échéance 30 jours.");
  const [textB, setTextB] = useState("Invoice total 5400 EUR, net 30 payment terms.");
  const [embedResult, setEmbedResult] = useState<{
    dims: number;
    sim: number;
    a: number[];
  } | null>(null);

  // Rerank sub-state
  const [query, setQuery] = useState("What is the invoice total?");
  const [docs, setDocs] = useState(
    "FACTURE total 5 400 € TTC, échéance 30 jours.\nWeather forecast for tomorrow: light rain.\nInvoice total 5400 EUR, net 30 payment terms.",
  );
  const [rerankResult, setRerankResult] = useState<{ doc: string; score: number }[] | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runEmbed() {
    if (!model) {
      setError("No live embedding deployment — deploy an embedding model (family: embedding).");
      return;
    }
    setError(null);
    setBusy(true);
    try {
      // Two INDEPENDENT calls, not one batched [textA, textB] request: a
      // batch input's response order isn't guaranteed to line up with the
      // request order on every backend, and pairing by array position alone
      // previously produced a silently-wrong (or empty) second vector.
      const [resA, resB] = await Promise.all([embed(model, textA), embed(model, textB)]);
      const a = resA.data?.[0]?.embedding ?? [];
      const b = resB.data?.[0]?.embedding ?? [];
      setEmbedResult({ dims: a.length, sim: cosine(a, b), a });
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

  async function runRerank() {
    if (!model) {
      setError(
        "No live reranker deployment — deploy a reranker model (family: reranker or multi_vector).",
      );
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
      setRerankResult(ranked);
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

  function switchMode(next: EmbedRerankMode) {
    setSubMode(next);
    setError(null);
  }

  return (
    <Card
      icon={<Fingerprint className="h-5 w-5" />}
      title="Embed / Rerank"
      subtitle="The retrieval primitives, computed on-node — nothing leaves the infra."
    >
      <div className="space-y-4">
        <Segmented
          value={subMode}
          onChange={switchMode}
          options={[
            { value: "embed", label: "Embed" },
            { value: "rerank", label: "Rerank" },
          ]}
        />

        <Field
          label={subMode === "embed" ? "Embedding deployment" : "Reranker deployment"}
          hint={
            subMode === "embed"
              ? "Deploy an embedding GGUF (e.g. LiquidAI/LFM2.5-Embedding-350M-GGUF) with family 'embedding'."
              : "Deploy a reranker: a GGUF (e.g. LiquidAI/LFM2.5-ColBERT-350M-GGUF, family 'reranker') or a safetensors ColBERT (e.g. mixedbread-ai/mxbai-edge-colbert-v0-32m, family 'multi_vector')."
          }
        >
          {names.length === 0 ? (
            <EmptyModelState
              noun={subMode === "embed" ? "embedding" : "reranker"}
              onNavigate={onNavigate}
            />
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

        {subMode === "embed" ? (
          <>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Text A">
                <TextArea rows={3} value={textA} onChange={(e) => setTextA(e.target.value)} />
              </Field>
              <Field label="Text B">
                <TextArea rows={3} value={textB} onChange={(e) => setTextB(e.target.value)} />
              </Field>
            </div>

            {error && <Alert tone="err">{error}</Alert>}

            <Button type="button" loading={busy} onClick={() => void runEmbed()} disabled={!model}>
              <Fingerprint className="h-4 w-4" />
              Embed & compare
            </Button>

            {embedResult && (
              <div className="space-y-3 rounded-md border border-border bg-muted/20 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="info">{embedResult.dims} dims</Badge>
                  <Badge
                    tone={
                      embedResult.sim > 0.7 ? "ok" : embedResult.sim > 0.4 ? "warn" : "neutral"
                    }
                  >
                    cosine similarity {embedResult.sim.toFixed(4)}
                  </Badge>
                </div>
                <div>
                  <p className="mb-1 text-xs font-medium text-muted-foreground">
                    <T>Text A vector (first 12 dims)</T>
                  </p>
                  <pre className="scroll-thin overflow-x-auto rounded-md border border-border bg-card p-3 text-xs text-foreground/90">
                    [{embedResult.a.slice(0, 12).map((v) => v.toFixed(4)).join(", ")}
                    {embedResult.a.length > 12 ? ", …" : ""}]
                  </pre>
                </div>
                <p className="text-xs text-muted-foreground">
                  <T>Both vectors were computed by the selected deployment.</T>
                </p>
              </div>
            )}
          </>
        ) : (
          <>
            <Field label="Query">
              <TextInput value={query} onChange={(e) => setQuery(e.target.value)} />
            </Field>
            <Field label="Documents" hint="One document per line.">
              <TextArea rows={5} value={docs} onChange={(e) => setDocs(e.target.value)} />
            </Field>

            {error && <Alert tone="err">{error}</Alert>}

            <Button type="button" loading={busy} onClick={() => void runRerank()} disabled={!model}>
              <ListOrdered className="h-4 w-4" />
              Rerank
            </Button>

            {rerankResult && (
              <div className="space-y-2 rounded-md border border-border bg-muted/20 p-4">
                {rerankResult.map((r, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 rounded-md border border-border bg-card p-2 text-sm"
                  >
                    <Badge tone={r.score > 0 ? "info" : "neutral"}>{r.score.toFixed(4)}</Badge>
                    <p className="text-foreground/90">{r.doc}</p>
                  </div>
                ))}
                <p className="text-xs text-muted-foreground">
                  <T>Sorted by relevance_score, highest first — the same order a retrieval pipeline would keep.</T>
                </p>
              </div>
            )}
          </>
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
  noun: "chat" | "embedding" | "reranker";
  onNavigate?: NavigateToDeploy;
}) {
  const { t } = useI18n();
  return (
    <div className="space-y-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-3 text-amber-700 dark:text-amber-400">
      <div className="flex items-start gap-2">
        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
        <div>
          <p className="text-sm font-medium">{t("No {noun} model deployed yet.", { noun: t(noun) })}</p>
          <p className="mt-0.5 text-xs text-amber-700/90 dark:text-amber-400/90">
            <T>Deploy one from Models → Search Hugging Face.</T>
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
  disabled = false,
}: {
  deployments: ReturnType<typeof usePolling<DeploymentRecord[]>>;
  selectable: DeploymentRecord[];
  value: string;
  onChange: (name: string) => void;
  emptyNoun?: "chat" | "embedding";
  onNavigate?: NavigateToDeploy;
  /** Locks the picker mid-request — e.g. Chat's cold-start retry chain keeps
   * calling the model it started with; switching mid-retry would silently
   * orphan the in-flight chain against the old selection. */
  disabled?: boolean;
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
      <Alert tone="warn">
        <T>Deployments unavailable — is the serving API up? The server default will be used.</T>
      </Alert>
    );
  }

  if (selectable.length === 0) {
    return <EmptyModelState noun={emptyNoun} onNavigate={onNavigate} />;
  }

  return (
    <Select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
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
