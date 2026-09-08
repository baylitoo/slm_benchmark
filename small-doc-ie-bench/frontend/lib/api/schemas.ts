// Extraction schema registry: the static SCHEMA_REGISTRY listing, plus the
// two "define once, reuse by name" registries built on top of it -- named
// DynamicSchemaSpecs (the Studio's schema builder) and named RoutingPolicys
// (the discoverable/validated alternative to a raw server-side filesystem
// path).

import { request } from "./core";

/** Extraction schema names available for structured output (GET /v1/schemas). */
export function listSchemas(): Promise<string[]> {
  return request<{ schemas: string[] }>("/v1/schemas").then((r) => r.schemas ?? []);
}

/** Field names for one built-in schema (GET /v1/schemas/{name}), read off its
 * JSON schema `properties`. Excludes `document_type` (a fixed literal, not an
 * extracted field) and `extraction_notes` (metadata, not a field). */
export function getSchemaFields(name: string): Promise<string[]> {
  return request<{ properties?: Record<string, unknown> }>(
    `/v1/schemas/${encodeURIComponent(name)}`,
  ).then((r) =>
    Object.keys(r.properties ?? {}).filter(
      (k) => k !== "document_type" && k !== "extraction_notes",
    ),
  );
}

// ---------------------------------------------------------------------------
// Dynamic schemas -- named DynamicSchemaSpecs saved via the Studio's schema
// builder. The spec itself (schemas/dynamic.py) already validates and already
// compiles into a working extraction schema; these are the "define once,
// reuse by name" persistence routes, the missing counterpart to schema_name's
// static SCHEMA_REGISTRY above.
// ---------------------------------------------------------------------------

export type DynamicFieldType = "string" | "date" | "number" | "money" | "list";

export interface DynamicFieldSpec {
  name: string;
  type: DynamicFieldType;
  description?: string | null;
  /** Only meaningful when type === "list" -- one flat level of scalar
   * sub-fields (line-items style). Arbitrary nesting/object fields are not
   * exposed by this UI slice; still reachable via a hand-authored POST body. */
  fields?: DynamicFieldSpec[];
}

export interface DynamicSchemaSpec {
  document_type: string;
  fields: DynamicFieldSpec[];
}

export interface DynamicSchemaSummary {
  name: string;
  spec: DynamicSchemaSpec;
  created_at: string;
  updated_at: string;
}

export function listDynamicSchemas(): Promise<DynamicSchemaSummary[]> {
  return request<DynamicSchemaSummary[]>("/v1/studio/schemas/dynamic");
}

/** Save a DynamicSchemaSpec under its own document_type. Create-only; a 409
 * means that document_type is already taken. */
export function createDynamicSchema(spec: DynamicSchemaSpec): Promise<DynamicSchemaSummary> {
  return request<DynamicSchemaSummary>("/v1/studio/schemas/dynamic", {
    method: "POST",
    body: JSON.stringify(spec),
  });
}

export function deleteDynamicSchema(name: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(
    `/v1/studio/schemas/dynamic/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
}

// ---------------------------------------------------------------------------
// Routing policies -- named RoutingPolicys saved via the Studio, the
// discoverable/validated alternative to a raw server-side filesystem path
// (BenchmarkRequest.routing_policy). The policy shape itself
// (extract/routing.py) is rich (multi-stage rules/selectors/budgets); this
// slice doesn't mirror every field in TypeScript. The create route accepts
// `policy` as opaque JSON and validates it server-side against
// RoutingPolicy.model_validate, returning a 422 with details on a bad shape.
// ---------------------------------------------------------------------------

export interface RoutingPolicySummary {
  name: string;
  policy: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export function listRoutingPolicies(): Promise<RoutingPolicySummary[]> {
  return request<RoutingPolicySummary[]>("/v1/studio/routing-policies");
}

/** Save a RoutingPolicy under a registry `name`. Create-only; a 409 means
 * that name is already taken. `policy` is validated server-side before it's
 * stored -- pass the parsed YAML/JSON body, not a string. */
export function createRoutingPolicy(
  name: string,
  policy: Record<string, unknown>,
): Promise<RoutingPolicySummary> {
  return request<RoutingPolicySummary>("/v1/studio/routing-policies", {
    method: "POST",
    body: JSON.stringify({ name, policy }),
  });
}

export function deleteRoutingPolicy(name: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(
    `/v1/studio/routing-policies/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
}
