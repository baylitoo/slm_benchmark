"use client";
// Studio-issued API keys: mint one for a client, see what each is doing, revoke.
//
// The secret is shown ONCE, right after creation, and never again in a
// listing. "Reveal" exists for the operator who lost it (the backend stores it
// encrypted, not hashed, for exactly that) and is a POST so a live credential
// never lands in browser history or a proxy log.
//
// Operator-only: every route 403s for a Studio-issued key, so a leaked client
// key cannot mint itself a successor that survives revocation.

import { useCallback, useEffect, useState } from "react";
import { Copy, KeyRound, Plus, Trash2, Eye } from "lucide-react";

import {
  apiKeyUsage,
  createApiKey,
  listApiKeys,
  revealApiKey,
  revokeApiKey,
  type ApiKeyUsageRow,
} from "@/lib/api/apiKeys";
import { usePolling } from "@/lib/usePolling";
import { Alert, Badge, Button, Card, Field, Segmented, TextInput } from "./ui";

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

export function ApiKeysCard({ active = true }: { active?: boolean }) {
  const [usageWindow, setUsageWindow] = useState<UsageWindow>("24h");
  const [label, setLabel] = useState("");
  const [tenantId, setTenantId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The one-time reveal. Held in component state only -- never persisted, and
  // dropped as soon as the operator dismisses it.
  const [secret, setSecret] = useState<{ id: string; key: string } | null>(null);
  const [encryptionReady, setEncryptionReady] = useState(true);
  // Non-null => this listing is restricted to that tenant (a company-signed-in
  // user sees only their own keys). Null => unscoped operator.
  const [scopedTo, setScopedTo] = useState<string | null>(null);

  const fetchUsage = useCallback(() => apiKeyUsage(usageWindow), [window]);
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

  // Creation can 503 when API_KEY_ENCRYPTION_KEY is unset; say so up front
  // instead of letting the operator type a label and then fail.
  useEffect(() => {
    if (!active) return;
    void listApiKeys()
      .then((result) => {
        setEncryptionReady(result.encryption_configured);
        setScopedTo(result.scoped_to);
      })
      .catch(() => setEncryptionReady(true));
  }, [active]);

  async function create(event: React.FormEvent) {
    event.preventDefault();
    // A scoped caller does not supply an owner: the backend uses their
    // verified login whatever is sent, so asking for it would be a lie.
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
    // Revocation is immediate on every replica (no key cache anywhere), so it
    // is worth one confirmation -- a live client stops working the moment
    // this lands.
    if (!window.confirm(`Revoke "${keyLabel}"? Any client using it stops working immediately.`))
      return;
    try {
      await revokeApiKey(id);
      refresh();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  const rows = usage.data?.keys ?? [];

  return (
    <Card
      icon={<KeyRound className="h-5 w-5" />}
      title="API keys"
      subtitle={
        scopedTo === null
          ? "Every key issued on this deployment, with what each one has served over the selected window. Shown once on creation, then masked."
          : "Your keys for clients of /v1, with what each one has served over the selected window. Shown once on creation, then masked."
      }
      actions={<Segmented
          value={usageWindow}
          onChange={setUsageWindow}
          options={WINDOWS.map((value) => ({ value, label: value }))}
        />}
    >
      <div className="space-y-4">
        {!encryptionReady && (
          <Alert tone="warn">
            API_KEY_ENCRYPTION_KEY is not set, so keys cannot be issued. Set it in the
            environment and restart the API.
          </Alert>
        )}
        {error && <Alert tone="err">{error}</Alert>}

        {secret && (
          <div className="space-y-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2">
            <p className="text-sm font-medium text-amber-700 dark:text-amber-400">
              Copy this key now — it is not shown again in the list.
            </p>
            <div className="flex items-center gap-2">
              <code className="flex-1 break-all rounded bg-card px-2 py-1 text-xs">
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

        <form onSubmit={create} className="flex flex-wrap items-end gap-2">
          <Field label="Label" hint="What this key is for.">
            <TextInput
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="CI pipeline"
              aria-label="Key label"
            />
          </Field>
          {scopedTo === null ? (
            <Field label="Tenant" hint="Owns the usage and quota.">
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
          <Button type="submit" loading={busy} disabled={!encryptionReady}>
            <Plus className="h-4 w-4" /> Issue key
          </Button>
        </form>

        {rows.length === 0 ? (
          <p className="text-xs text-muted-foreground">No keys issued yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-muted-foreground">
                <tr>
                  <th className="py-1 pr-3 font-medium">Key</th>
                  <th className="py-1 pr-3 font-medium">Tenant</th>
                  <th className="py-1 pr-3 text-right font-medium">Requests</th>
                  <th className="py-1 pr-3 text-right font-medium">Tokens</th>
                  <th className="py-1 pr-3 text-right font-medium">Errors</th>
                  <th className="py-1 pr-3 text-right font-medium">Avg latency</th>
                  <th className="py-1 pr-3 font-medium">Last used</th>
                  <th className="py-1 font-medium" />
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id} className="border-t border-border">
                    <td className="py-1.5 pr-3">
                      <div className="flex items-center gap-1.5">
                        <span className="font-medium text-foreground">{row.label}</span>
                        {row.revoked && <Badge tone="neutral">revoked</Badge>}
                      </div>
                      <code className="text-[11px] text-muted-foreground">
                        {row.prefix}…
                      </code>
                    </td>
                    <td className="py-1.5 pr-3 text-muted-foreground">{row.tenant_id}</td>
                    <td className="py-1.5 pr-3 text-right tabular-nums">{row.requests}</td>
                    <td className="py-1.5 pr-3 text-right tabular-nums">
                      {row.prompt_tokens + row.completion_tokens}
                    </td>
                    <td className="py-1.5 pr-3 text-right tabular-nums">
                      {row.errors > 0 ? (
                        <span className="text-destructive">{row.errors}</span>
                      ) : (
                        row.errors
                      )}
                    </td>
                    <td className="py-1.5 pr-3 text-right tabular-nums text-muted-foreground">
                      {row.avg_latency_ms === null ? "—" : `${row.avg_latency_ms} ms`}
                    </td>
                    <td className="py-1.5 pr-3 text-muted-foreground">
                      {relative(row.last_used_at)}
                    </td>
                    <td className="py-1.5 text-right">
                      <div className="flex justify-end gap-1">
                        <Button
                          type="button"
                          variant="ghost"
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
                            onClick={() => void revoke(row.id, row.label)}
                            aria-label={`Revoke ${row.label}`}
                            title="Revoke"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="text-[11px] text-muted-foreground">
          Requests made with a key from <code>API_KEYS</code> are counted per tenant under
          Usage, not here — those have no key row to attribute to.
        </p>
      </div>
    </Card>
  );
}
