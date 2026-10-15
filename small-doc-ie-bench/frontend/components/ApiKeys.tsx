"use client";
// API keys — issue a key for a client of /v1, see what each one served, revoke.
//
// Was a card inside Observability; it is a management surface, not a dashboard
// widget, so it lives in its own section now. Same wiring throughout (the
// poller, the create/reveal/revoke handlers) — only the layout moved.
//
// The secret is shown ONCE, right after creation. "Reveal" exists for the
// operator who lost it (the backend stores it encrypted, not hashed, for
// exactly that) and is a POST, so a live credential never lands in browser
// history or a proxy log.

import { useCallback, useEffect, useMemo, useState } from "react";
import { Copy, Eye, KeyRound, Plus, Search, Trash2 } from "lucide-react";

import {
  apiKeyUsage,
  createApiKey,
  listApiKeys,
  revealApiKey,
  revokeApiKey,
  type ApiKeyUsageRow,
} from "@/lib/api/apiKeys";
import { usePolling } from "@/lib/usePolling";
import { Alert, Badge, Button, Field, Segmented, TextInput } from "./ui";
import { PageHeader } from "./patterns/PageHeader";
import { Toolbar } from "./patterns/Toolbar";
import { ResultLine } from "./patterns/ResultLine";
import { Table, type Column } from "./patterns/Table";

const WINDOWS = ["24h", "7d", "30d"] as const;
type UsageWindow = (typeof WINDOWS)[number];
const POLL_MS = 30_000;

function relative(iso: string | null): string {
  if (!iso) return "never";
  const seconds = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (seconds < 90) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86_400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86_400)}d ago`;
}

export function ApiKeys({ active = true }: { active?: boolean }) {
  const [usageWindow, setUsageWindow] = useState<UsageWindow>("24h");
  const [creating, setCreating] = useState(false);
  const [label, setLabel] = useState("");
  const [tenantId, setTenantId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The one-time reveal. Component state only — never persisted, and dropped
  // as soon as it is dismissed.
  const [secret, setSecret] = useState<{ id: string; key: string } | null>(null);
  const [encryptionReady, setEncryptionReady] = useState(true);
  // Non-null => restricted to that tenant (a company-signed-in user sees only
  // their own keys). Null => unscoped operator.
  const [scopedTo, setScopedTo] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  const fetchUsage = useCallback(() => apiKeyUsage(usageWindow), [usageWindow]);
  const usage = usePolling<{ window: string; keys: ApiKeyUsageRow[] }>(
    fetchUsage,
    POLL_MS,
    active,
  );
  const { refresh } = usage;
  // usePolling re-arms only on interval/enabled changes, so a window switch
  // needs an explicit refetch rather than a wait for the next tick.
  useEffect(() => {
    refresh();
  }, [usageWindow, refresh]);

  // Creation 503s when API_KEY_ENCRYPTION_KEY is unset; say so up front rather
  // than after the operator has typed a label.
  useEffect(() => {
    if (!active) return;
    void listApiKeys()
      .then((result) => {
        setEncryptionReady(result.encryption_configured);
        setScopedTo(result.scoped_to);
      })
      .catch(() => setEncryptionReady(true));
  }, [active]);

  const rows = usage.data?.keys ?? [];
  // Client-side over the rows already fetched — never a new request.
  const shown = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return rows;
    return rows.filter(
      (row) =>
        row.label.toLowerCase().includes(needle) ||
        row.prefix.toLowerCase().includes(needle) ||
        row.tenant_id.toLowerCase().includes(needle),
    );
  }, [rows, filter]);

  async function create(event: React.FormEvent) {
    event.preventDefault();
    if (busy || !label.trim() || (scopedTo === null && !tenantId.trim())) return;
    setBusy(true);
    setError(null);
    try {
      const created = await createApiKey(
        label.trim(),
        scopedTo === null ? tenantId.trim() : undefined,
      );
      setSecret({ id: created.id, key: created.key });
      setLabel("");
      setTenantId("");
      setCreating(false);
      refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function reveal(id: string) {
    setError(null);
    try {
      setSecret(await revealApiKey(id));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  async function revoke(id: string, keyLabel: string) {
    setError(null);
    // Revocation bites on the next request on every replica (no key cache
    // anywhere), so it is worth one confirmation.
    const warning = `Revoke "${keyLabel}"? Any client using it stops working immediately.`;
    if (!window.confirm(warning)) return;
    try {
      await revokeApiKey(id);
      refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  const columns: Column<ApiKeyUsageRow>[] = [
    {
      key: "label",
      header: "Key",
      sortAccessor: (row) => row.label,
      render: (row) => (
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="truncate font-medium text-foreground">{row.label}</span>
            {row.revoked && <Badge tone="neutral">revoked</Badge>}
          </div>
          <code className="font-mono text-[11px] text-muted-foreground">{row.prefix}…</code>
        </div>
      ),
    },
    {
      key: "tenant",
      header: "Owner",
      sortAccessor: (row) => row.tenant_id,
      render: (row) => <span className="text-muted-foreground">{row.tenant_id}</span>,
    },
    {
      key: "requests",
      header: "Requests",
      className: "text-right",
      sortAccessor: (row) => row.requests,
      render: (row) => <span className="tabular-nums">{row.requests}</span>,
    },
    {
      key: "tokens",
      header: "Tokens",
      className: "text-right",
      sortAccessor: (row) => row.prompt_tokens + row.completion_tokens,
      render: (row) => (
        <span className="tabular-nums">{row.prompt_tokens + row.completion_tokens}</span>
      ),
    },
    {
      key: "errors",
      header: "Errors",
      className: "text-right",
      sortAccessor: (row) => row.errors,
      render: (row) =>
        row.errors > 0 ? (
          <span className="tabular-nums text-destructive">{row.errors}</span>
        ) : (
          <span className="tabular-nums">0</span>
        ),
    },
    {
      key: "latency",
      header: "Avg latency",
      className: "text-right",
      sortAccessor: (row) => row.avg_latency_ms ?? -1,
      render: (row) => (
        <span className="tabular-nums text-muted-foreground">
          {row.avg_latency_ms === null ? "—" : `${row.avg_latency_ms} ms`}
        </span>
      ),
    },
    {
      key: "last_used",
      header: "Last used",
      sortAccessor: (row) => row.last_used_at ?? "",
      render: (row) => (
        <span className="text-muted-foreground">{relative(row.last_used_at)}</span>
      ),
    },
    {
      key: "actions",
      header: "",
      className: "text-right",
      render: (row) => (
        <div className="flex justify-end gap-1">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => void reveal(row.id)}
            aria-label={`Reveal ${row.label}`}
            title="Show the full key again"
          >
            <Eye className="h-3.5 w-3.5" />
          </Button>
          {!row.revoked && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => void revoke(row.id, row.label)}
              aria-label={`Revoke ${row.label}`}
              title="Revoke"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      ),
    },
  ];

  return (
    <div>
      <PageHeader
        title="API keys"
        subtitle={
          scopedTo === null
            ? "Keys issued on this deployment, with what each one has served. A key opens every model; usage is attributed to it."
            : "Your keys for clients of the /v1 API. A key opens every model; usage is attributed to it."
        }
        actions={
          <Button
            type="button"
            onClick={() => setCreating((v) => !v)}
            disabled={!encryptionReady}
          >
            <Plus className="h-4 w-4" /> Issue key
          </Button>
        }
      />

      <div className="space-y-4">
        {!encryptionReady && (
          <Alert tone="warn">
            API_KEY_ENCRYPTION_KEY is not set, so keys cannot be issued. Set it in the
            environment and restart the API.
          </Alert>
        )}
        {error && <Alert tone="err">{error}</Alert>}

        {secret && (
          <div className="space-y-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-5">
            <p className="text-sm font-medium text-amber-700 dark:text-amber-400">
              Copy this key now — it is not shown again in the list.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <code className="min-w-0 flex-1 break-all rounded-md bg-card px-2 py-1 font-mono text-xs">
                {secret.key}
              </code>
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={() => void navigator.clipboard?.writeText(secret.key)}
              >
                <Copy className="h-4 w-4" /> Copy
              </Button>
              <Button type="button" variant="ghost" size="sm" onClick={() => setSecret(null)}>
                Done
              </Button>
            </div>
          </div>
        )}

        {creating && (
          <form
            onSubmit={create}
            className="flex flex-wrap items-end gap-3 rounded-xl border border-border bg-card p-5"
          >
            <Field label="Label" hint="What this key is for.">
              <TextInput
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="CI pipeline"
                aria-label="Key label"
                autoFocus
              />
            </Field>
            {scopedTo === null ? (
              <Field label="Owner" hint="Tenant the usage is attributed to.">
                <TextInput
                  value={tenantId}
                  onChange={(e) => setTenantId(e.target.value)}
                  placeholder="acme"
                  aria-label="Tenant id"
                />
              </Field>
            ) : (
              <Field label="Owner" hint="Your account — keys are yours alone.">
                <TextInput value={scopedTo} readOnly aria-label="Owner" />
              </Field>
            )}
            <Button type="submit" loading={busy}>
              Create
            </Button>
            <Button type="button" variant="ghost" onClick={() => setCreating(false)}>
              Cancel
            </Button>
          </form>
        )}

        <Toolbar>
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <TextInput
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter by label, prefix or owner"
              aria-label="Filter keys"
              className="w-72 pl-8"
            />
          </div>
          <Segmented
            value={usageWindow}
            onChange={setUsageWindow}
            options={WINDOWS.map((value) => ({ value, label: value }))}
          />
          {filter && (
            <Button type="button" variant="secondary" size="sm" onClick={() => setFilter("")}>
              Reset
            </Button>
          )}
        </Toolbar>

        <div>
          <ResultLine
            shown={shown.length}
            total={rows.length}
            noun="keys"
            onFetch={refresh}
            fetching={usage.loading}
          />
          <Table
            columns={columns}
            rows={usage.data ? shown : null}
            getRowKey={(row) => row.id}
            loading={usage.loading}
            error={usage.error}
            emptyLabel={filter ? "No keys match this filter" : "No API keys yet"}
            emptyDescription={
              filter
                ? "Clear the filter to see every key."
                : "Issue one to let an application call the /v1 API. The secret is shown once."
            }
            emptyIcon={<KeyRound className="h-5 w-5" />}
          />
        </div>

        <p className="text-[11px] text-muted-foreground">
          Requests made with a key from <code className="font-mono">API_KEYS</code> are counted
          per tenant under Observability, not here — those have no key row to attribute to.
        </p>
      </div>
    </div>
  );
}
