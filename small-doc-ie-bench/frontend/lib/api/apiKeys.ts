// Studio-issued API keys (operator only -- see src/docie_bench/inngest/
// studio_api/api_keys.py). Distinct from lib/apiKey.ts, which holds the
// OPERATOR's own key in this browser so requests carry X-API-Key at all;
// these are keys the operator MINTS for clients.

import { request } from "./core";

export interface ApiKeyView {
  id: string;
  /** `docie_<prefix>` -- what is shown once the full value is masked. */
  prefix: string;
  label: string;
  tenant_id: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
  revoked: boolean;
}

/** Only the create and reveal responses ever carry the secret. */
export interface ApiKeyWithSecret extends ApiKeyView {
  key: string;
}

export interface ApiKeyUsageRow extends ApiKeyView {
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  errors: number;
  /** Null when the key saw no traffic in the window -- not 0 ms. */
  avg_latency_ms: number | null;
}

export async function listApiKeys(): Promise<{
  keys: ApiKeyView[];
  /** False when API_KEY_ENCRYPTION_KEY is unset: creation will 503. */
  encryption_configured: boolean;
  /** The tenant this listing is restricted to, or null for an unscoped
   * operator. A company-signed-in user only ever sees their own keys, so the
   * UI must not imply otherwise. */
  scoped_to: string | null;
}> {
  return request("/v1/studio/api-keys");
}

/** Issue a key. `tenantId` is honoured only for an unscoped operator: a
 * company-signed-in caller always gets their own verified login as the owner,
 * whatever is sent here. */
export async function createApiKey(
  label: string,
  tenantId?: string,
): Promise<ApiKeyWithSecret> {
  return request("/v1/studio/api-keys", {
    method: "POST",
    body: JSON.stringify(tenantId ? { label, tenant_id: tenantId } : { label }),
  });
}

/** POST, not GET: this returns a live credential, so it must not land in
 * browser history, a proxy log or a prefetch. */
export async function revealApiKey(id: string): Promise<{ id: string; key: string }> {
  return request(`/v1/studio/api-keys/${encodeURIComponent(id)}/reveal`, { method: "POST" });
}

export async function revokeApiKey(id: string): Promise<ApiKeyView> {
  return request(`/v1/studio/api-keys/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function apiKeyUsage(
  window: string,
): Promise<{ window: string; keys: ApiKeyUsageRow[] }> {
  return request(`/v1/studio/api-keys/usage?window=${encodeURIComponent(window)}`);
}
