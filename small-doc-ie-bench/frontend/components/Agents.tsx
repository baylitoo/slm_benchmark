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
  Plug,
  PlusCircle,
  ScanText,
  ShieldCheck,
  Trash2,
  Wand2,
} from "lucide-react";
import {
  ApiError,
  agentBaseUrl,
  createAgent,
  deleteAgent,
  getAgents,
  getAgentTemplates,
  getDeployments,
  selectableDeployments,
  updateAgent,
  type AgentKind,
  type AgentTemplate,
  type AgentView,
} from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { useToast } from "./Toast";
import { Badge, Button, Card, ComingSoon, Field, Select, Skeleton, TextArea, TextInput } from "./ui";
import { PageHeader } from "./patterns/PageHeader";
import { Table, type Column } from "./patterns/Table";

// Fallback when the templates endpoint hasn't provided the entity list yet.
const PII_ENTITIES = [
  "EMAIL",
  "IBAN",
  "CREDIT_CARD",
  "NATIONAL_ID",
  "PHONE",
  "IP_ADDRESS",
];

const KIND_META: Record<AgentKind, { label: string; icon: React.ReactNode }> = {
  proxy_security: { label: "Security proxy", icon: <ShieldCheck className="h-5 w-5" /> },
  ocr: { label: "OCR", icon: <ScanText className="h-5 w-5" /> },
  custom: { label: "Custom", icon: <Wand2 className="h-5 w-5" /> },
};

export function Agents({ view = "catalog" }: { view?: string }) {
  const [tab, setTab] = useState(view || "catalog");
  // Follow sidebar deep-links, but keep local switches (e.g. "Use template")
  // working between nav clicks.
  useEffect(() => setTab(view || "catalog"), [view]);

  const templates = useAsync(getAgentTemplates, []);
  const agents = useAsync(getAgents, []);
  const [prefill, setPrefill] = useState<AgentTemplate | null>(null);

  function useTemplate(template: AgentTemplate) {
    setPrefill(template);
    setTab("create");
  }

  const subtitle =
    tab === "instances"
      ? "Configured agents and their OpenAI-compatible endpoints."
      : tab === "create"
        ? "Instantiate a template or build your own agent over a served model."
        : "Preconfigured agents over the served SLMs — pick one to instantiate.";

  return (
    <div>
      <PageHeader
        title="Agents"
        subtitle={subtitle}
        actions={
          <Button size="sm" onClick={() => setTab("create")}>
            <PlusCircle className="h-4 w-4" />
            New agent
          </Button>
        }
      />

      {tab === "instances" ? (
        <InstancesView agents={agents} />
      ) : tab === "create" ? (
        <CreateView
          templates={templates.data ?? []}
          prefill={prefill}
          onCreated={() => {
            agents.reload();
            setPrefill(null);
            setTab("instances");
          }}
        />
      ) : (
        <CatalogView templates={templates} onUse={useTemplate} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

function CatalogView({
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
}: {
  agents: { data: AgentView[] | null; error: unknown; loading: boolean; reload: () => void };
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
      <div>
        <div className="mb-1 flex items-center justify-between">
          <p className="text-xs font-medium text-muted-foreground">Try it</p>
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
  onCreated,
}: {
  templates: AgentTemplate[];
  prefill: AgentTemplate | null;
  onCreated: () => void;
}) {
  const { toast } = useToast();
  const deployments = useAsync(getDeployments, []);
  const deploymentNames = useMemo(
    () =>
      selectableDeployments(deployments.data ?? [])
        .map((d) => d.spec?.name)
        .filter((n): n is string => !!n),
    [deployments.data],
  );

  const [templateId, setTemplateId] = useState(prefill?.id ?? "custom");
  const [name, setName] = useState("");
  const [modelProfile, setModelProfile] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Kind-specific option state.
  const [mode, setMode] = useState("placeholder");
  const [entities, setEntities] = useState<string[]>(PII_ENTITIES);
  const [restorePii, setRestorePii] = useState(false);
  const [guardModel, setGuardModel] = useState("");
  const [guardFallback, setGuardFallback] = useState(false);
  const [ocrBackend, setOcrBackend] = useState("tesseract");
  const [ocrLanguage, setOcrLanguage] = useState("");
  const [ocrExtractor, setOcrExtractor] = useState("");

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
            }
          : kind === "ocr"
            ? {
                backend: ocrBackend,
                language: ocrLanguage || null,
                extractor: ocrExtractor || null,
              }
            : {};
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
      onCreated();
    } catch (err) {
      toast({ title: "Create failed", description: errMessage(err), tone: "error" });
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
              hint="Lowercase slug — becomes the OpenAI model id."
            >
              <TextInput
                id="agent-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="pii-proxy"
                pattern="[a-z0-9][a-z0-9._-]*"
                required
              />
            </Field>
            {kind !== "ocr" && (
              <Field
                label="Backing model"
                htmlFor="agent-model"
                hint="A live deployment, models.yaml profile, or store:<name>. Empty = studio default."
                className="sm:col-span-2"
              >
                <TextInput
                  id="agent-model"
                  value={modelProfile}
                  onChange={(e) => setModelProfile(e.target.value)}
                  placeholder={deploymentNames[0] ?? "e.g. nuextract3"}
                  list="agent-deployments"
                />
                <datalist id="agent-deployments">
                  {deploymentNames.map((n) => (
                    <option key={n} value={n} />
                  ))}
                </datalist>
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
                  hint="Encoder analyzer endpoint (e.g. a `docie encoder` GLiNER deployment) — replaces the built-in regex analyzer for higher recall. Empty = regex."
                >
                  <TextInput
                    id="agent-guard-model"
                    value={guardModel}
                    onChange={(e) => setGuardModel(e.target.value)}
                    placeholder="gliner-pii"
                    list="agent-deployments"
                  />
                </Field>
                {guardModel.trim() && (
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
                )}
              </div>
            </Card>
          )}

          {kind === "ocr" && (
            <Card title="OCR options" subtitle="Engine, language, and optional structured extractor.">
              <div className="space-y-4">
                <Field label="OCR backend" htmlFor="agent-ocr-backend">
                  <Select
                    id="agent-ocr-backend"
                    value={ocrBackend}
                    onChange={(e) => setOcrBackend(e.target.value)}
                  >
                    <option value="tesseract">tesseract</option>
                    <option value="paddleocr">paddleocr</option>
                    <option value="pdf_text">pdf_text</option>
                  </Select>
                </Field>
                <Field label="Language" htmlFor="agent-ocr-language" hint="Backend-specific, e.g. en / fr. Empty = default.">
                  <TextInput
                    id="agent-ocr-language"
                    value={ocrLanguage}
                    onChange={(e) => setOcrLanguage(e.target.value)}
                    placeholder="en"
                  />
                </Field>
                <Field
                  label="Extractor"
                  htmlFor="agent-ocr-extractor"
                  hint="Optional SLM (e.g. a NuExtract deployment) — turns OCR into an OCR→SLM extraction pipeline."
                >
                  <TextInput
                    id="agent-ocr-extractor"
                    value={ocrExtractor}
                    onChange={(e) => setOcrExtractor(e.target.value)}
                    placeholder={deploymentNames[0] ?? "nuextract3"}
                    list="agent-deployments"
                  />
                </Field>
              </div>
            </Card>
          )}

          <Card title="Endpoint" subtitle="Created agents are addressable immediately.">
            <CopyLine
              label="base_url"
              value={`${agentBaseUrl(name.trim() || "<name>")}`}
            />
          </Card>

          <Button type="submit" className="w-full" loading={submitting} disabled={!name.trim()}>
            <PlusCircle className="h-4 w-4" />
            Create agent
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
  if (e instanceof ApiError || e instanceof Error) return e.message;
  return String(e);
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
