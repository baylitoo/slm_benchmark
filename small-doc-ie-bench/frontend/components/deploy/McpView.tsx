"use client";

import { useEffect, useState } from "react";
import { Plug, RefreshCw, Wrench } from "lucide-react";
import {
  disableMcpServer,
  enableMcpServer,
  getCodeInterpreterWorkers,
  listMcpCatalog,
  testMcpServer,
  type CodeInterpreterQueue,
  type McpCatalogEntry,
  type McpTool,
} from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { Alert, Badge, Button, Spinner, TextInput } from "../ui";
import { useToast } from "../Toast";
import { T } from "@/lib/i18n";

// MCP Tools view — the server catalog (GET /v1/mcp/catalog): enable a
// pre-configured tool server, test it (spawn + list tools), disable it.
// Enabled servers appear as chips in the Playground's Chat panel and are
// selectable per request via `mcp_servers`.

export function McpView({ active = true }: { active?: boolean }) {
  const catalog = useAsync<McpCatalogEntry[]>("mcp-catalog", listMcpCatalog);
  const { toast } = useToast();
  const [busy, setBusy] = useState<string | null>(null);
  const [params, setParams] = useState<Record<string, Record<string, string>>>({});
  const [tested, setTested] = useState<Record<string, McpTool[]>>({});

  if (!active) return null;

  async function enable(entry: McpCatalogEntry) {
    setBusy(entry.name);
    try {
      await enableMcpServer(entry.name, params[entry.name] ?? {});
      toast({ title: `${entry.title} enabled`, tone: "success" });
      catalog.reload();
    } catch (err) {
      toast({
        title: `Could not enable ${entry.title}`,
        description: err instanceof Error ? err.message : "Request failed.",
        tone: "error",
      });
    } finally {
      setBusy(null);
    }
  }

  async function disable(entry: McpCatalogEntry) {
    setBusy(entry.name);
    try {
      await disableMcpServer(entry.name);
      setTested((prev) => ({ ...prev, [entry.name]: undefined as unknown as McpTool[] }));
      toast({ title: `${entry.title} disabled`, tone: "success" });
      catalog.reload();
    } catch (err) {
      toast({
        title: `Could not disable ${entry.title}`,
        description: err instanceof Error ? err.message : "Request failed.",
        tone: "error",
      });
    } finally {
      setBusy(null);
    }
  }

  async function test(entry: McpCatalogEntry) {
    setBusy(entry.name);
    try {
      const res = await testMcpServer(entry.name);
      setTested((prev) => ({ ...prev, [entry.name]: res.tools }));
      toast({
        title: `${entry.title}: ${res.tools.length} tool${res.tools.length === 1 ? "" : "s"} live`,
        tone: "success",
      });
    } catch (err) {
      toast({
        title: `${entry.title} test failed`,
        description: err instanceof Error ? err.message : "Request failed.",
        tone: "error",
      });
    } finally {
      setBusy(null);
    }
  }

  if (catalog.error) {
    return <Alert tone="err">{String(catalog.error)}</Alert>;
  }
  if (catalog.loading && !catalog.data) {
    return (
      <p className="flex items-center gap-2 text-sm text-muted-foreground">
        <Spinner /> <T>Loading catalog…</T>
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        <T>{"Enable a tool server and chat models can use its tools — pick servers per conversation in the Playground's Chat panel. Enabled entries are written to the server registry and spawned per request."}</T>
      </p>
      <div className="grid gap-4 lg:grid-cols-3">
        {(catalog.data ?? []).map((entry) => (
          <div
            key={entry.name}
            className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Wrench className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">{entry.title}</span>
              </div>
              <Badge tone={entry.enabled ? "ok" : "neutral"}>
                {entry.enabled ? "enabled" : "off"}
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">{entry.description}</p>
            <div className="flex flex-wrap gap-1">
              {(tested[entry.name] ?? entry.tools.map((t) => ({ name: t }))).map((tool) => (
                <Badge key={typeof tool === "string" ? tool : tool.name} tone="info">
                  {typeof tool === "string" ? tool : tool.name}
                </Badge>
              ))}
            </div>
            {entry.name === "code-interpreter" && entry.enabled && <CodeInterpreterWorkers />}
            {entry.params.map((param) => (
              <TextInput
                key={param.name}
                type={param.secret ? "password" : "text"}
                aria-label={`${entry.name} ${param.name}`}
                placeholder={param.description}
                disabled={entry.enabled}
                value={params[entry.name]?.[param.name] ?? ""}
                onChange={(e) =>
                  setParams((prev) => ({
                    ...prev,
                    [entry.name]: { ...prev[entry.name], [param.name]: e.target.value },
                  }))
                }
              />
            ))}
            <div className="mt-auto flex gap-2">
              {entry.enabled ? (
                <>
                  <Button
                    size="sm"
                    variant="secondary"
                    loading={busy === entry.name}
                    onClick={() => void test(entry)}
                  >
                    <Plug className="h-3.5 w-3.5" />
                    Test
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={busy === entry.name}
                    onClick={() => void disable(entry)}
                  >
                    Disable
                  </Button>
                </>
              ) : (
                <Button
                  size="sm"
                  loading={busy === entry.name}
                  onClick={() => void enable(entry)}
                >
                  Enable
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Judge0's own worker-pool/queue status (#264), nested here rather than a
// new page: an operator enabling code-interpreter wants to know whether its
// sandbox is actually staffed, not just that the MCP handshake succeeded.
function CodeInterpreterWorkers() {
  const [queues, setQueues] = useState<CodeInterpreterQueue[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await getCodeInterpreterWorkers();
      setQueues(res.queues);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="rounded-md border border-border bg-muted/40 p-2 text-xs">
      <div className="mb-1 flex items-center justify-between">
        <span className="font-medium text-muted-foreground">
          <T>Sandbox worker pool</T>
        </span>
        <button
          type="button"
          aria-label="Refresh worker pool"
          onClick={() => void load()}
          className="text-muted-foreground hover:text-foreground"
        >
          <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>
      {error ? (
        <p className="text-destructive"><T>Couldn&apos;t reach the sandbox.</T> {error}</p>
      ) : queues === null ? (
        <p className="text-muted-foreground"><T>Loading…</T></p>
      ) : (
        <div className="space-y-1">
          {queues.map((q) => (
            <div key={q.queue} className="flex items-center gap-3">
              <span className="w-16 shrink-0 truncate">{q.queue}</span>
              <Badge tone={q.available > 0 ? "ok" : "warn"}>{q.available} workers</Badge>
              <span className="text-muted-foreground">
                {q.idle} idle · {q.working} working · {q.size} queued
              </span>
              {q.failed > 0 && <Badge tone="err">{q.failed} failed</Badge>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
