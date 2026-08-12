"use client";

// Agents tab — preconfigured agents over the served SLMs, exposed as
// OpenAI-compatible endpoints so an external agents platform can consume them
// like any model. Three nav-driven sub-views:
//   • catalog   — the preconfigured templates (Security Proxy, OCR Agent, …)
//   • instances — configured agents + their endpoints (copy base_url / curl)
//   • create    — instantiate a template or build a custom agent

import { useEffect, useMemo, useState } from "react";
import {
  Bot,
  Check,
  Copy,
  Pencil,
  Play,
  Plug,
  PlusCircle,
  Rocket,
  ScanText,
  ShieldCheck,
  Trash2,
  Wand2,
  Eye,
  Sparkles,
  FileText,
} from "lucide-react";
import {
  ApiError,
  agentBaseUrl,
  agentChat,
  createAgent,
  deleteAgent,
  deployModel,
  deploymentModelType,
  fileToBase64,
  getAgents,
  getAgentTemplates,
  getDeployments,
  getStore,
  isLiveDeployment,
  listSchemas,
  renderDocument,
  selectableDeployments,
  updateAgent,
  visionDeploymentNames,
  type AgentChatResponse,
  type AgentKind,
  type AgentTemplate,
  type AgentView,
  type StoreEntry,
} from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
import { toUserMessage } from "@/lib/errors";
import { useToast } from "./Toast";
import { Badge, Button, Card, ComingSoon, Field, Select, Skeleton, TextArea, TextInput } from "./ui";
import { PageHeader } from "./patterns/PageHeader";
import { Table, type Column } from "./patterns/Table";

// The three document-extraction stages an "ocr" agent can run. Persisted as
// options.mode; the UI is a picker over these.
const STAGE_MODES: {
  id: "ocr" | "ocr_extract" | "vision";
  label: string;
  desc: string;
  icon: React.ReactNode;
}[] = [
  {
    id: "ocr",
    label: "Plain OCR",
    desc: "Image → text. An OCR engine returns the document's text.",
    icon: <FileText className="h-4 w-4" />,
  },
  {
    id: "ocr_extract",
    label: "OCR → extract",
    desc: "OCR the image, then an LLM pulls structured JSON from the text.",
    icon: <ScanText className="h-4 w-4" />,
  },
  {
    id: "vision",
    label: "Vision → structured",
    desc: "Image straight to a vision model that generates JSON — no OCR step.",
    icon: <Eye className="h-4 w-4" />,
  },
];

// Fallback when the templates endpoint hasn't provided the entity list yet.
const PII_ENTITIES = [
  "EMAIL",
  "IBAN",
  "CREDIT_CARD",
  "NATIONAL_ID",
  "PHONE",
  "IP_ADDRESS",
];

// Moderation presets served by GLiNER2 guardrail checkpoints
// (fastino/GLiNER2-Guardrails-PII-Multi). Names mirror the encoder server's
// MODERATION_TASKS registry.
const GUARD_TASKS = ["prompt_safety", "prompt_toxicity", "jailbreak_detection"];

const KIND_META: Record<AgentKind, { label: string; icon: React.ReactNode }> = {
  proxy_security: { label: "Security proxy", icon: <ShieldCheck className="h-5 w-5" /> },
  ocr: { label: "Document extraction", icon: <ScanText className="h-5 w-5" /> },
  custom: { label: "Custom", icon: <Wand2 className="h-5 w-5" /> },
};

export function Agents({ view = "catalog" }: { view?: string }) {
  const [tab, setTab] = useState(view || "catalog");
  // Follow sidebar deep-links, but keep local switches (e.g. "Use template")
  // working between nav clicks.
  useEffect(() => setTab(view || "catalog"), [view]);

  const templates = useAsync("agent-templates", getAgentTemplates);
  const agents = useAsync("agents", getAgents);
  const [prefill, setPrefill] = useState<AgentTemplate | null>(null);
  const [editAgent, setEditAgent] = useState<AgentView | null>(null);

  function useTemplate(template: AgentTemplate) {
    setEditAgent(null);
    setPrefill(template);
    setTab("create");
  }

  function editExisting(agent: AgentView) {
    setPrefill(null);
    setEditAgent(agent);
    setTab("create");
  }

  const subtitle =
    tab === "instances"
      ? "Configured agents and their OpenAI-compatible endpoints."
      : tab === "create"
        ? editAgent
          ? `Editing ${editAgent.name} — toggle behaviors and save.`
          : "Instantiate a template or build your own agent over a served model."
        : "Preconfigured agents over the served SLMs — pick one to instantiate.";

  return (
    <div>
      <PageHeader
        title="Agents"
        subtitle={subtitle}
        actions={
          <Button
            size="sm"
            onClick={() => {
              setEditAgent(null);
              setPrefill(null);
              setTab("create");
            }}
          >
            <PlusCircle className="h-4 w-4" />
            New agent
          </Button>
        }
      />

      {tab === "instances" ? (
        <InstancesView agents={agents} onEdit={editExisting} />
      ) : tab === "create" ? (
        <CreateView
          templates={templates.data ?? []}
          prefill={prefill}
          editAgent={editAgent}
          onCreated={() => {
            agents.reload();
            setPrefill(null);
            setEditAgent(null);
            setTab("instances");
          }}
        />
      ) : (
        <TemplatesView templates={templates} onUse={useTemplate} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

function TemplatesView({
  templates,
  onUse,
}: {
  templates: { data: AgentTemplate[] | null; error: unknown; loading: boolean };
  onUse: (t: AgentTemplate) => void;
}) {
  if (templates.loading && !templates.data) {
    return (
      <div className="grid gap-4 lg:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-44 w-full" />
        ))}
      </div>
    );
  }
  if (templates.error && !templates.data) return <ComingSoon error={templates.error} />;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-3">
        {(templates.data ?? []).map((template) => (
          <Card key={template.id} className="flex flex-col" bodyClassName="flex flex-1 flex-col">
            <div className="flex items-start gap-3">
              <span className="grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-border bg-muted text-accent">
                {KIND_META[template.kind]?.icon ?? <Bot className="h-5 w-5" />}
              </span>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-foreground">
                  {template.display_name}
                </p>
                <Badge tone="info" className="mt-1">
                  {KIND_META[template.kind]?.label ?? template.kind}
                </Badge>
              </div>
            </div>
            <p className="mt-3 flex-1 text-xs leading-relaxed text-muted-foreground">
              {template.description}
            </p>
            <Button size="sm" className="mt-4 w-full" onClick={() => onUse(template)}>
              Use template
            </Button>
          </Card>
        ))}
      </div>

      <Card
        title="Platform endpoint"
        subtitle="Every enabled agent is an OpenAI model on one base_url — plug it into any agents platform."
        icon={<Plug className="h-4 w-4" />}
      >
        <div className="space-y-2">
          <CopyLine label="base_url" value={agentBaseUrl()} />
          <p className="text-xs text-muted-foreground">
            <code className="rounded bg-muted px-1">GET /models</code> lists the enabled
            agents; <code className="rounded bg-muted px-1">POST /chat/completions</code>{" "}
            routes by the <code className="rounded bg-muted px-1">model</code> field (the
            agent name). Auth: <code className="rounded bg-muted px-1">x-api-key</code> or a
            standard <code className="rounded bg-muted px-1">Authorization: Bearer</code>{" "}
            key.
          </p>
        </div>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Instances
// ---------------------------------------------------------------------------

function InstancesView({
  agents,
  onEdit,
}: {
  agents: { data: AgentView[] | null; error: unknown; loading: boolean; reload: () => void };
  onEdit: (agent: AgentView) => void;
}) {
  const { toast } = useToast();
  const [expanded, setExpanded] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  async function toggleEnabled(agent: AgentView) {
    setBusy(agent.name);
    try {
      await updateAgent(agent.name, { enabled: !(agent.enabled ?? true) });
      agents.reload();
    } catch (e) {
      toast({ title: "Update failed", description: errMessage(e), tone: "error" });
    } finally {
      setBusy(null);
    }
  }

  async function remove(agent: AgentView) {
    if (!window.confirm(`Delete agent "${agent.name}"? Its endpoint stops resolving.`)) return;
    setBusy(agent.name);
    try {
      await deleteAgent(agent.name);
      toast({ title: `Agent ${agent.name} deleted`, tone: "success" });
      agents.reload();
    } catch (e) {
      toast({ title: "Delete failed", description: errMessage(e), tone: "error" });
    } finally {
      setBusy(null);
    }
  }

  const columns: Column<AgentView>[] = [
    {
      key: "name",
      header: "Agent",
      sortAccessor: (a) => a.name,
      render: (a) => (
        <div className="min-w-0">
          <p className="truncate font-medium text-foreground">{a.display_name || a.name}</p>
          <p className="truncate text-xs text-muted-foreground">{a.name}</p>
        </div>
      ),
    },
    {
      key: "kind",
      header: "Kind",
      sortAccessor: (a) => a.kind,
      render: (a) => (
        <Badge tone={a.kind === "proxy_security" ? "info" : a.kind === "ocr" ? "warn" : "neutral"}>
          {KIND_META[a.kind]?.label ?? a.kind}
        </Badge>
      ),
    },
    {
      key: "model",
      header: "Backing model",
      render: (a) => (
        <span className="text-xs text-foreground/80">
          {a.model_profile || <span className="text-muted-foreground">studio default</span>}
        </span>
      ),
    },
    {
      key: "status",
      header: "Status",
      sortAccessor: (a) => String(a.enabled ?? true),
      render: (a) =>
        (a.enabled ?? true) ? <Badge tone="ok">enabled</Badge> : <Badge>disabled</Badge>,
    },
    {
      key: "endpoint",
      header: "Endpoint",
      render: (a) => <CopyButton value={agentBaseUrl(a.name)} label="Copy base_url" />,
    },
    {
      key: "actions",
      header: "",
      className: "text-right",
      render: (a) => (
        <div className="flex items-center justify-end gap-1.5">
          <Button
            variant="secondary"
            size="sm"
            disabled={busy === a.name}
            title="Edit this agent's behavior"
            onClick={(e) => {
              e.stopPropagation();
              onEdit(a);
            }}
          >
            <Pencil className="h-3.5 w-3.5" />
            Edit
          </Button>
          <Button
            variant="secondary"
            size="sm"
            disabled={busy === a.name}
            onClick={(e) => {
              e.stopPropagation();
              void toggleEnabled(a);
            }}
          >
            {(a.enabled ?? true) ? "Disable" : "Enable"}
          </Button>
          <Button
            variant="danger"
            size="sm"
            disabled={busy === a.name}
            onClick={(e) => {
              e.stopPropagation();
              void remove(a);
            }}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      ),
    },
  ];

  return (
    <Card title="My agents" subtitle="Click a row for connection details.">
      <Table
        columns={columns}
        rows={agents.data}
        loading={agents.loading}
        error={agents.error}
        getRowKey={(a) => a.name}
        emptyLabel="No agents yet"
        emptyDescription="Instantiate one from the catalog, or create your own."
        emptyIcon={<Bot className="h-5 w-5" />}
        expandedKey={expanded}
        onRowClick={(a) => setExpanded((k) => (k === a.name ? null : a.name))}
        renderExpanded={(a) => <AgentDetails agent={a} />}
      />
    </Card>
  );
}

// Prefilled sample so one click demonstrates masking (name, email, FR phone, IBAN).
const SAMPLE_PII_TEXT =
  "Report prepared by Jean Dupont. Contact: jean.dupont@acme.fr or " +
  "+33 6 12 34 56 78. Refund to IBAN DE89 3704 0044 0532 0130 00.";

function TryPanel({ agent }: { agent: AgentView }) {
  const isOcr = agent.kind === "ocr";
  // Vision-mode agents send the image straight to a vision model, so a PDF must
  // be rasterized to page images first (like the Playground). OCR modes keep
  // the raw PDF — the server-side OCR backend (e.g. pdf_text) reads it directly.
  const options = (agent.options ?? {}) as Record<string, unknown>;
  const ocrMode =
    typeof options.mode === "string"
      ? options.mode
      : options.extractor
        ? "ocr_extract"
        : "ocr";
  const isVisionAgent = isOcr && ocrMode === "vision";
  const [text, setText] = useState(SAMPLE_PII_TEXT);
  const [file, setFile] = useState<File | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AgentChatResponse | null>(null);

  async function run() {
    setError(null);
    setResult(null);
    let messages: { role: string; content: unknown }[];
    if (isOcr) {
      if (!file) {
        setError("Choose a document image or PDF first.");
        return;
      }
      const b64 = await fileToBase64(file);
      const isPdf =
        file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
      // Vision mode + PDF → rasterize to page images (a vision model can't read
      // a data:application/pdf URL); every page is sent. Otherwise the file goes
      // as-is (the OCR backend handles a raw PDF).
      const imageParts =
        isPdf && isVisionAgent
          ? (await renderDocument(b64, file.name, 200)).images.map((url) => ({
              type: "image_url",
              image_url: { url },
            }))
          : [
              {
                type: "image_url",
                image_url: { url: `data:${file.type || "image/png"};base64,${b64}` },
              },
            ];
      messages = [
        {
          role: "user",
          content: [{ type: "text", text: "OCR this document." }, ...imageParts],
        },
      ];
    } else {
      if (!text.trim()) {
        setError("Type some text first.");
        return;
      }
      messages = [{ role: "user", content: text }];
    }
    setRunning(true);
    try {
      setResult(await agentChat(agent.name, messages));
    } catch (e) {
      setError(errMessage(e));
    } finally {
      setRunning(false);
    }
  }

  const pii = result?.docie_agent?.pii;
  const content = result?.choices?.[0]?.message?.content;

  return (
    <div className="rounded-md border border-border bg-card p-3">
      <p className="mb-2 text-xs font-medium text-foreground">Try it</p>
      <div className="space-y-2">
        {isOcr ? (
          <input
            type="file"
            accept=".pdf,image/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="block w-full text-xs text-muted-foreground file:mr-3 file:rounded-md file:border file:border-border file:bg-muted file:px-3 file:py-1.5 file:text-xs file:text-foreground"
          />
        ) : (
          <TextArea rows={3} value={text} onChange={(e) => setText(e.target.value)} />
        )}
        <div className="flex items-center gap-2">
          <Button type="button" size="sm" loading={running} onClick={() => void run()}>
            <Play className="h-3.5 w-3.5" />
            Run
          </Button>
          {running && (
            <span className="text-xs text-muted-foreground">
              An evicted backing deployment auto-reloads on first request — that
              wait is the model load.
            </span>
          )}
        </div>

        {error && (
          <p className="rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-600 dark:text-rose-400">
            {error}
          </p>
        )}

        {result && (
          <div className="space-y-2">
            {pii && (
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge tone="info">analyzer: {pii.analyzer ?? "?"}</Badge>
                <Badge>mode: {pii.mode ?? "?"}</Badge>
                {(pii.entities ?? []).map((e) => (
                  <Badge key={e.type} tone={e.count > 0 ? "warn" : "neutral"}>
                    {e.type} ×{e.count}
                  </Badge>
                ))}
                {(pii.detected ?? 0) === 0 && <Badge tone="ok">no PII found</Badge>}
                {pii.degraded_to_regex && (
                  <Badge tone="err">guard down — degraded to regex</Badge>
                )}
              </div>
            )}
            {(pii?.placeholders?.length ?? 0) > 0 && (
              <p className="text-xs text-muted-foreground">
                Sent upstream as: {pii?.placeholders?.join(" ")}
              </p>
            )}
            <div>
              <p className="mb-1 text-xs font-medium text-muted-foreground">Response</p>
              <pre className="scroll-thin max-h-64 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-muted/40 p-3 text-xs leading-relaxed text-foreground/90">
                {typeof content === "string" ? content : JSON.stringify(result, null, 2)}
              </pre>
            </div>
            <details>
              <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
                Raw completion JSON
              </summary>
              <pre className="scroll-thin mt-1 max-h-64 overflow-auto rounded-md border border-border bg-muted/40 p-3 text-xs text-foreground/90">
                {JSON.stringify(result, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}

function AgentDetails({ agent }: { agent: AgentView }) {
  const curl = [
    `curl ${agentBaseUrl(agent.name)}/chat/completions \\`,
    `  -H "Authorization: Bearer $DOCIE_API_KEY" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '{"model":"${agent.name}","messages":[{"role":"user","content":"Hello"}]}'`,
  ].join("\n");
  return (
    <div className="space-y-3">
      <CopyLine label="base_url" value={agentBaseUrl(agent.name)} />
      <TryPanel agent={agent} />
      <div>
        <div className="mb-1 flex items-center justify-between">
          <p className="text-xs font-medium text-muted-foreground">From your platform</p>
          <CopyButton value={curl} label="Copy curl" />
        </div>
        <pre className="scroll-thin overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-xs leading-relaxed text-foreground/90">
          {curl}
        </pre>
      </div>
      {agent.system_prompt && (
        <div>
          <p className="mb-1 text-xs font-medium text-muted-foreground">System prompt</p>
          <p className="rounded-md border border-border bg-muted/40 p-3 text-xs text-foreground/90">
            {agent.system_prompt}
          </p>
        </div>
      )}
      {agent.options && Object.keys(agent.options).length > 0 && (
        <div>
          <p className="mb-1 text-xs font-medium text-muted-foreground">Options</p>
          <pre className="scroll-thin overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-xs text-foreground/90">
            {JSON.stringify(agent.options, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

function CreateView({
  templates,
  prefill,
  editAgent,
  onCreated,
}: {
  templates: AgentTemplate[];
  prefill: AgentTemplate | null;
  editAgent: AgentView | null;
  onCreated: () => void;
}) {
  const editing = editAgent !== null;
  const { toast } = useToast();
  const deployments = useAsync("deployments", getDeployments);
  // Managed deployments, segmented by semantic type — every model an agent
  // uses is a deployment the platform orchestrates, picked (not typed).
  const chatDeployments = useMemo(
    () =>
      selectableDeployments(deployments.data ?? [])
        .filter((d) => deploymentModelType(d) === "chat")
        .map((d) => d.spec?.name)
        .filter((n): n is string => !!n),
    [deployments.data],
  );
  const encoderDeployments = useMemo(
    () =>
      (deployments.data ?? [])
        .filter((d) => deploymentModelType(d) === "encoder" && !!d.spec?.name)
        .map((d) => ({
          name: d.spec?.name as string,
          live: isLiveDeployment(d),
        })),
    [deployments.data],
  );
  // Analyzer models SEEDED into the store but not yet deployed — offered as
  // one-click "Deploy" shortcuts (store path, no hardcoded repo, no
  // download-at-boot). Generic: whatever analyzer entries the store holds.
  const store = useAsync<StoreEntry[]>("store", getStore);
  const seededUndeployedEncoders = useMemo(() => {
    const deployed = new Set(encoderDeployments.map((d) => d.name));
    return (store.data ?? [])
      .filter((e) => e.analyzer && !deployed.has(e.name))
      .map((e) => e.name);
  }, [store.data, encoderDeployments]);

  const [templateId, setTemplateId] = useState(prefill?.id ?? "custom");
  const [name, setName] = useState("");
  const [modelProfile, setModelProfile] = useState("");
  const [backingCustom, setBackingCustom] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Kind-specific option state.
  const [mode, setMode] = useState("placeholder");
  const [entities, setEntities] = useState<string[]>(PII_ENTITIES);
  const [restorePii, setRestorePii] = useState(false);
  const [guardModel, setGuardModel] = useState("");
  const [guardFallback, setGuardFallback] = useState(false);
  const [guardTasks, setGuardTasks] = useState<string[]>([]);
  const [guardLabels, setGuardLabels] = useState("");
  const [deployingGuard, setDeployingGuard] = useState(false);

  // Deploy a SEEDED store encoder via the store path (Auto → serve_store_model
  // → encoder runtime): generic (any analyzer entry), no hardcoded repo, and
  // no download-at-boot (the snapshot is already in the store). Points the
  // guard field at it once fired.
  async function deploySeededEncoder(storeName: string) {
    setDeployingGuard(true);
    try {
      await deployModel({ model: storeName, name: storeName });
      setGuardModel(storeName);
      toast({
        title: "Encoder deploy started",
        description: `${storeName} — follow progress under Deployments.`,
        tone: "success",
      });
      deployments.reload();
    } catch (err) {
      toast({ title: "Encoder deploy failed", description: errMessage(err), tone: "error" });
    } finally {
      setDeployingGuard(false);
    }
  }
  const [ocrBackend, setOcrBackend] = useState("tesseract");
  const [ocrLanguage, setOcrLanguage] = useState("");
  const [ocrExtractor, setOcrExtractor] = useState("");
  // OCR→extract step-1 may be a deployed VISION model (VLM as OCR) instead of a
  // built-in backend. When set it wins; the built-in backend is ignored.
  const [ocrModel, setOcrModel] = useState("");
  // Suppress a reasoning model's think channel (enable_thinking=false) so it
  // emits the answer directly instead of rambling past the token budget.
  const [noThink, setNoThink] = useState(false);
  // Staged document-extraction mode: plain OCR | OCR→LLM | vision→structured.
  const [ocrMode, setOcrMode] = useState<"ocr" | "ocr_extract" | "vision">("ocr");
  const [visionModel, setVisionModel] = useState("");
  const [schemaName, setSchemaName] = useState("");

  // Vision deployments (store family flagged vision) — the model picker for the
  // vision→structured stage.
  const visionModels = useMemo(() => {
    const visionSet = visionDeploymentNames(store.data);
    return selectableDeployments(deployments.data ?? [])
      .map((d) => d.spec?.name)
      .filter((n): n is string => !!n && visionSet.has(n));
  }, [deployments.data, store.data]);
  // Extraction schemas available for structured output (GET /v1/schemas).
  const schemas = useAsync<string[]>("schemas", listSchemas);

  const template = templates.find((t) => t.id === templateId) ?? null;
  const kind: AgentKind = template?.kind ?? "custom";

  // Adopt catalog prefills whenever the user clicks "Use template".
  useEffect(() => {
    if (!prefill) return;
    setTemplateId(prefill.id);
    const options = prefill.defaults?.options ?? {};
    if (Array.isArray(options.entities)) setEntities(options.entities.map(String));
    if (typeof options.mode === "string") setMode(options.mode);
    if (typeof options.backend === "string") setOcrBackend(options.backend);
  }, [prefill]);

  // Load an existing agent's full config for editing (name + template locked).
  useEffect(() => {
    if (!editAgent) return;
    const templateForKind: Record<AgentKind, string> = {
      proxy_security: "proxy-security",
      ocr: "ocr-agent",
      custom: "custom",
    };
    setTemplateId(templateForKind[editAgent.kind] ?? "custom");
    setName(editAgent.name);
    setSystemPrompt(editAgent.system_prompt ?? "");
    const backing = editAgent.model_profile ?? "";
    setModelProfile(backing);
    const o = editAgent.options ?? {};
    if (Array.isArray(o.entities)) setEntities((o.entities as unknown[]).map(String));
    if (typeof o.mode === "string") setMode(o.mode);
    setRestorePii(o.restore_pii === true);
    setGuardModel(typeof o.guard_model === "string" ? o.guard_model : "");
    setGuardFallback(o.guard_fallback === "regex");
    setGuardTasks(Array.isArray(o.guard_tasks) ? (o.guard_tasks as unknown[]).map(String) : []);
    setGuardLabels(Array.isArray(o.guard_labels) ? (o.guard_labels as unknown[]).join(", ") : "");
    if (typeof o.backend === "string") setOcrBackend(o.backend);
    setOcrLanguage(typeof o.language === "string" ? o.language : "");
    setOcrExtractor(typeof o.extractor === "string" ? o.extractor : "");
    setOcrModel(typeof o.ocr_model === "string" ? o.ocr_model : "");
    setNoThink(o.no_think === true);
    setVisionModel(typeof o.vision_model === "string" ? o.vision_model : "");
    setSchemaName(typeof o.schema === "string" ? o.schema : "");
    // Back-compat: an agent saved before `mode` derives it — extractor → the
    // OCR→LLM pipeline, otherwise plain OCR.
    const savedMode = o.mode;
    setOcrMode(
      savedMode === "vision" || savedMode === "ocr_extract" || savedMode === "ocr"
        ? savedMode
        : typeof o.extractor === "string" && o.extractor
          ? "ocr_extract"
          : "ocr",
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editAgent]);

  function toggleEntity(entity: string) {
    setEntities((prev) =>
      prev.includes(entity) ? prev.filter((e) => e !== entity) : [...prev, entity],
    );
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    try {
      const options: Record<string, unknown> =
        kind === "proxy_security"
          ? {
              mode,
              entities,
              restore_pii: restorePii,
              guard_model: guardModel.trim() || null,
              guard_fallback: guardModel.trim() && guardFallback ? "regex" : null,
              guard_tasks:
                guardModel.trim() && guardTasks.length > 0 ? guardTasks : null,
              guard_labels:
                guardModel.trim() && guardLabels.trim()
                  ? guardLabels.split(",").map((l) => l.trim()).filter(Boolean)
                  : null,
            }
          : kind === "ocr"
            ? ocrMode === "vision"
              ? {
                  mode: "vision",
                  vision_model: visionModel || null,
                  schema: schemaName || null,
                  no_think: noThink,
                }
              : ocrMode === "ocr_extract"
                ? {
                    mode: "ocr_extract",
                    backend: ocrBackend,
                    // A VLM-as-OCR selection (a deployment name) wins over the
                    // built-in backend; null falls back to `backend`.
                    ocr_model: ocrModel || null,
                    language: ocrLanguage || null,
                    extractor: ocrExtractor || null,
                    schema: schemaName || null,
                    no_think: noThink,
                  }
                : {
                    mode: "ocr",
                    backend: ocrBackend,
                    language: ocrLanguage || null,
                  }
            : {};
      if (editing && editAgent) {
        await updateAgent(editAgent.name, {
          model_profile: kind === "ocr" ? null : modelProfile.trim() || null,
          system_prompt: systemPrompt.trim() || null,
          options,
        });
        toast({ title: `Agent ${editAgent.name} updated`, tone: "success" });
      } else {
        const created = await createAgent({
          name: name.trim(),
          template: templateId,
          model_profile: kind === "ocr" ? null : modelProfile.trim() || null,
          system_prompt: systemPrompt.trim() || null,
          options,
        });
        toast({
          title: `Agent ${created.name} created`,
          description: `OpenAI endpoint: ${agentBaseUrl(created.name)}`,
          tone: "success",
        });
      }
      onCreated();
    } catch (err) {
      toast({
        title: editing ? "Update failed" : "Create failed",
        description: errMessage(err),
        tone: "error",
      });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={submit}>
      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <Card title="Configuration" subtitle={template?.description}>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Template" htmlFor="agent-template" required>
              <Select
                id="agent-template"
                value={templateId}
                onChange={(e) => setTemplateId(e.target.value)}
                disabled={editing}
              >
                {templates.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.display_name}
                  </option>
                ))}
              </Select>
            </Field>
            <Field
              label="Name"
              htmlFor="agent-name"
              required
              hint={
                editing
                  ? "The agent id is immutable — edit its behavior, not its name."
                  : "Becomes the OpenAI model id — typed input is normalized to a lowercase slug."
              }
            >
              <TextInput
                id="agent-name"
                value={name}
                onChange={(e) => setName(slugifyAgentName(e.target.value))}
                placeholder="pii-proxy"
                pattern="[a-z0-9][a-z0-9._-]*"
                required
                disabled={editing}
              />
            </Field>
            {kind !== "ocr" && (
              <Field
                label="Backing model"
                htmlFor="agent-model"
                hint="A managed chat deployment — deploy models under Serving → Deployments. Advanced: a custom reference (profile / store:<name>)."
                className="sm:col-span-2"
              >
                <Select
                  id="agent-model"
                  value={
                    backingCustom
                      ? "__custom__"
                      : chatDeployments.includes(modelProfile)
                        ? modelProfile
                        : ""
                  }
                  onChange={(e) => {
                    if (e.target.value === "__custom__") {
                      setBackingCustom(true);
                      setModelProfile("");
                    } else {
                      setBackingCustom(false);
                      setModelProfile(e.target.value);
                    }
                  }}
                >
                  <option value="">Studio default</option>
                  {chatDeployments.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                  <option value="__custom__">Custom reference…</option>
                </Select>
                {backingCustom && (
                  <TextInput
                    className="mt-2"
                    value={modelProfile}
                    onChange={(e) => setModelProfile(e.target.value)}
                    placeholder="profile name or store:<name>"
                  />
                )}
                {chatDeployments.length === 0 && (
                  <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
                    No chat deployment is routable yet — deploy one first
                    (Serving → Models / Deployments).
                  </p>
                )}
              </Field>
            )}
            {kind !== "ocr" && (
              <Field
                label="System prompt"
                htmlFor="agent-prompt"
                hint="Prepended to every request the agent forwards."
                className="sm:col-span-2"
              >
                <TextArea
                  id="agent-prompt"
                  rows={4}
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="You are…"
                />
              </Field>
            )}
          </div>
        </Card>

        <div className="space-y-4">
          {kind === "proxy_security" && (
            <Card title="Security options" subtitle="What the proxy detects — and what it does about it.">
              <div className="space-y-4">
                <Field label="Mode" htmlFor="agent-pii-mode">
                  <Select id="agent-pii-mode" value={mode} onChange={(e) => setMode(e.target.value)}>
                    <option value="placeholder">Placeholder — anonymize before forwarding</option>
                    <option value="block">Block — refuse requests containing PII</option>
                    <option value="detect">Detect — forward untouched, report findings</option>
                  </Select>
                </Field>
                <div>
                  <p className="mb-1.5 text-xs font-medium text-foreground">Entities</p>
                  <div className="grid grid-cols-2 gap-1.5">
                    {PII_ENTITIES.map((entity) => (
                      <label
                        key={entity}
                        className="flex cursor-pointer items-center gap-2 text-xs text-foreground/90"
                      >
                        <input
                          type="checkbox"
                          checked={entities.includes(entity)}
                          onChange={() => toggleEntity(entity)}
                          className="h-3.5 w-3.5"
                        />
                        {entity.replaceAll("_", " ").toLowerCase()}
                      </label>
                    ))}
                  </div>
                </div>
                <label className="flex cursor-pointer items-center gap-2 text-xs text-foreground/90">
                  <input
                    type="checkbox"
                    checked={restorePii}
                    onChange={(e) => setRestorePii(e.target.checked)}
                    className="h-3.5 w-3.5"
                  />
                  Restore original values in the response
                </label>
                <Field
                  label="Guard model"
                  htmlFor="agent-guard-model"
                  hint="An encoder deployment (analyzer) that runs the checks. Leave empty to use the built-in regex analysis."
                >
                  <Select
                    id="agent-guard-model"
                    value={guardModel}
                    onChange={(e) => setGuardModel(e.target.value)}
                  >
                    <option value="">None — regex analyzer</option>
                    {encoderDeployments.map((d) => (
                      <option key={d.name} value={d.name}>
                        {d.name}
                        {d.live ? "" : " · not live — Load it in Deployments"}
                      </option>
                    ))}
                  </Select>
                  {/* One-click deploy of an analyzer already SEEDED in the store
                      (generic — no hardcoded model). */}
                  {seededUndeployedEncoders.length > 0 && (
                    <div className="mt-2 flex flex-wrap items-center gap-1.5">
                      <span className="text-xs text-muted-foreground">
                        Deploy a seeded encoder:
                      </span>
                      {seededUndeployedEncoders.map((n) => (
                        <Button
                          key={n}
                          type="button"
                          variant="secondary"
                          size="sm"
                          loading={deployingGuard}
                          onClick={() => void deploySeededEncoder(n)}
                          title={`Deploy the store analyzer "${n}" as a managed encoder deployment`}
                        >
                          <Rocket className="h-3.5 w-3.5" />
                          {n}
                        </Button>
                      ))}
                    </div>
                  )}
                  {encoderDeployments.length === 0 &&
                    seededUndeployedEncoders.length === 0 && (
                      <p className="mt-1 text-xs text-muted-foreground">
                        No analyzer model available — add one under Serving →
                        Models → Add model → Encoder, then it appears here to
                        deploy.
                      </p>
                    )}
                </Field>
                {guardModel.trim() && (
                  <>
                    <Field
                      label="Guard labels (advanced)"
                      htmlFor="agent-guard-labels"
                      hint="Comma-separated zero-shot labels sent to the encoder. Empty = derived from the entity checkboxes. Depends on the encoder — e.g. person, address, date_of_birth, password, api_key, secret."
                    >
                      <TextInput
                        id="agent-guard-labels"
                        value={guardLabels}
                        onChange={(e) => setGuardLabels(e.target.value)}
                        placeholder="person, email, phone_number, iban, payment_card, address, date_of_birth, password, api_key"
                      />
                    </Field>
                    <div>
                      <p className="mb-1.5 text-xs font-medium text-foreground">
                        Moderation tasks
                      </p>
                      <div className="grid gap-1.5">
                        {GUARD_TASKS.map((task) => (
                          <label
                            key={task}
                            className="flex cursor-pointer items-center gap-2 text-xs text-foreground/90"
                          >
                            <input
                              type="checkbox"
                              checked={guardTasks.includes(task)}
                              onChange={() =>
                                setGuardTasks((prev) =>
                                  prev.includes(task)
                                    ? prev.filter((t) => t !== task)
                                    : [...prev, task],
                                )
                              }
                              className="h-3.5 w-3.5"
                            />
                            {task.replaceAll("_", " ")}
                          </label>
                        ))}
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Needs a GLiNER2 guardrails checkpoint (e.g.
                        fastino/GLiNER2-Guardrails-PII-Multi). In Block mode a
                        non-benign verdict refuses the request before any PII
                        check.
                      </p>
                    </div>
                    <label className="flex cursor-pointer items-center gap-2 text-xs text-foreground/90">
                      <input
                        type="checkbox"
                        checked={guardFallback}
                        onChange={(e) => setGuardFallback(e.target.checked)}
                        className="h-3.5 w-3.5"
                      />
                      Degrade to regex analysis if the guard is unreachable
                      (default: fail closed)
                    </label>
                  </>
                )}
              </div>
            </Card>
          )}

          {kind === "ocr" && (
            <Card
              title="Extraction pipeline"
              subtitle="Choose how a document becomes output — plain text, OCR→LLM, or vision→structured."
            >
              <div className="space-y-4">
                {/* Stage selector */}
                <div className="grid gap-2 sm:grid-cols-3">
                  {STAGE_MODES.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => setOcrMode(m.id)}
                      className={cn(
                        "flex flex-col gap-1 rounded-lg border p-3 text-left transition",
                        ocrMode === m.id
                          ? "border-accent bg-accent/10 ring-1 ring-accent"
                          : "border-border hover:bg-muted",
                      )}
                    >
                      <span className="flex items-center gap-2 text-sm font-medium text-foreground">
                        {m.icon} {m.label}
                      </span>
                      <span className="text-xs text-muted-foreground">{m.desc}</span>
                    </button>
                  ))}
                </div>

                {/* OCR engine + language — the OCR-based stages. In OCR→extract
                    the engine may be a deployed vision model (VLM as OCR). */}
                {ocrMode !== "vision" && (
                  <>
                    <Field label="OCR engine" htmlFor="agent-ocr-backend">
                      <Select
                        id="agent-ocr-backend"
                        value={ocrMode === "ocr_extract" && ocrModel ? ocrModel : ocrBackend}
                        onChange={(e) => {
                          const v = e.target.value;
                          if (ocrMode === "ocr_extract" && visionModels.includes(v)) {
                            setOcrModel(v); // a vision deployment does the OCR
                          } else {
                            setOcrBackend(v);
                            setOcrModel("");
                          }
                        }}
                      >
                        <optgroup label="Built-in">
                          <option value="liteparse">
                            liteparse — PDF text + OCR fallback (light)
                          </option>
                          <option value="tesseract">tesseract</option>
                          <option value="paddleocr">paddleocr — needs the paddle extra</option>
                          {ocrBackend === "pdf_text" && !ocrModel && (
                            <option value="pdf_text">pdf_text — legacy alias of liteparse</option>
                          )}
                        </optgroup>
                        {ocrMode === "ocr_extract" && visionModels.length > 0 && (
                          <optgroup label="Vision model (VLM as OCR)">
                            {visionModels.map((n) => (
                              <option key={n} value={n}>
                                {n} — transcribe with this VLM
                              </option>
                            ))}
                          </optgroup>
                        )}
                      </Select>
                    </Field>
                    {!(ocrMode === "ocr_extract" && ocrModel) && (
                      <Field
                        label="Language"
                        htmlFor="agent-ocr-language"
                        hint="Engine-specific, e.g. en / fr. Empty = default."
                      >
                        <TextInput
                          id="agent-ocr-language"
                          value={ocrLanguage}
                          onChange={(e) => setOcrLanguage(e.target.value)}
                          placeholder="en"
                        />
                      </Field>
                    )}
                  </>
                )}

                {/* Extractor — the OCR→LLM stage */}
                {ocrMode === "ocr_extract" && (
                  <Field
                    label="Extractor model"
                    htmlFor="agent-ocr-extractor"
                    hint="A chat deployment (e.g. NuExtract) that turns the OCR text into JSON."
                  >
                    <Select
                      id="agent-ocr-extractor"
                      value={ocrExtractor}
                      onChange={(e) => setOcrExtractor(e.target.value)}
                    >
                      <option value="">Select a chat model…</option>
                      {chatDeployments.map((n) => (
                        <option key={n} value={n}>
                          {n}
                        </option>
                      ))}
                    </Select>
                  </Field>
                )}

                {/* Vision model — the direct vision→structured stage */}
                {ocrMode === "vision" && (
                  <Field
                    label="Vision model"
                    htmlFor="agent-vision-model"
                    hint="A deployed vision model (NuExtract, Gemma, LFM2-VL…). The image goes straight to it."
                  >
                    <Select
                      id="agent-vision-model"
                      value={visionModel}
                      onChange={(e) => setVisionModel(e.target.value)}
                    >
                      <option value="">Select a vision model…</option>
                      {visionModels.map((n) => (
                        <option key={n} value={n}>
                          {n}
                        </option>
                      ))}
                    </Select>
                    {visionModels.length === 0 && (
                      <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
                        No vision model deployed — deploy one from Models first.
                      </p>
                    )}
                  </Field>
                )}

                {/* Output schema — structured stages */}
                {ocrMode !== "ocr" && (
                  <Field
                    label="Output schema"
                    htmlFor="agent-schema"
                    hint={
                      ocrMode === "vision"
                        ? "Grammar-constrains the JSON (llama.cpp GBNF)."
                        : "Optional — constrains the LLM to this schema."
                    }
                  >
                    <Select
                      id="agent-schema"
                      value={schemaName}
                      onChange={(e) => setSchemaName(e.target.value)}
                    >
                      <option value="">
                        {ocrMode === "vision" ? "Free text (no schema)" : "None — model decides"}
                      </option>
                      {(schemas.data ?? []).map((s) => (
                        <option key={s} value={s}>
                          {s}
                        </option>
                      ))}
                    </Select>
                  </Field>
                )}

                {/* Reasoning models ramble past the token budget and never reach
                    the grammar answer — let the operator turn thinking off. */}
                {ocrMode !== "ocr" && (
                  <label className="flex cursor-pointer items-start gap-2 text-xs text-foreground/90">
                    <input
                      type="checkbox"
                      checked={noThink}
                      onChange={(e) => setNoThink(e.target.checked)}
                      className="mt-0.5 h-3.5 w-3.5"
                    />
                    <span>
                      Disable thinking (reasoning models)
                      <span className="block text-muted-foreground">
                        Sends enable_thinking=false so the model emits the answer
                        directly instead of a long think channel. No effect on models
                        without one.
                      </span>
                    </span>
                  </label>
                )}

                {ocrMode === "vision" && (
                  <p className="flex items-start gap-2 rounded-md border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
                    <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    Structured output needs a model whose runtime enforces a schema
                    (llama.cpp GBNF / vLLM). A NuExtract model here runs via generic
                    grammar, not its bespoke chat-template path.
                  </p>
                )}
              </div>
            </Card>
          )}

          <Card title="Endpoint" subtitle="Created agents are addressable immediately.">
            <CopyLine
              label="base_url"
              value={
                name.trim() ? agentBaseUrl(name.trim()) : `${agentBaseUrl()}/{name}`
              }
            />
          </Card>

          <Button type="submit" className="w-full" loading={submitting} disabled={!name.trim()}>
            <PlusCircle className="h-4 w-4" />
            {editing ? "Save changes" : "Create agent"}
          </Button>
        </div>
      </div>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function errMessage(e: unknown): string {
  return toUserMessage(e);
}

/** Agent names are OpenAI model ids: normalize typing into the slug the
 * backend accepts ("DLPProtection" -> "dlpprotection") instead of letting a
 * raw pydantic pattern error surface after submit. */
function slugifyAgentName(raw: string): string {
  return raw
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9._-]+/g, "")
    .replace(/^[^a-z0-9]+/, "")
    .slice(0, 63);
}

function CopyButton({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      type="button"
      variant="secondary"
      size="sm"
      onClick={(e) => {
        e.stopPropagation();
        void navigator.clipboard.writeText(value).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {label}
    </Button>
  );
}

function CopyLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="shrink-0 text-xs font-medium text-muted-foreground">{label}</span>
      <code className="scroll-thin min-w-0 flex-1 overflow-x-auto whitespace-nowrap rounded-md border border-border bg-muted/40 px-2.5 py-1.5 text-xs text-foreground/90">
        {value}
      </code>
      <CopyButton value={value} label="Copy" />
    </div>
  );
}
