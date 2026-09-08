// MCP tool servers: catalog browse + enable/disable/test (Serving > MCP Tools)
// and the registered-server list the Playground's chat chips read. Selected
// names ride the chat request's `mcp_servers` field — the backend runs the
// tool exchange server-side and returns the final completion.

import { request } from "./core";

export interface McpCatalogParam {
  name: string;
  description: string;
  required: boolean;
}

export interface McpCatalogEntry {
  name: string;
  title: string;
  description: string;
  tools: string[];
  params: McpCatalogParam[];
  enabled: boolean;
}

export interface McpRegisteredServer {
  name: string;
  transport: string;
  url: string | null;
  command: string[] | null;
  /** Header/env NAMES only — values never leave the server. */
  headers: string[] | null;
  env: string[] | null;
}

export interface McpTool {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export function listMcpCatalog(): Promise<McpCatalogEntry[]> {
  return request<{ entries: McpCatalogEntry[] }>("/v1/mcp/catalog").then((r) => r.entries);
}

export function listMcpServers(): Promise<McpRegisteredServer[]> {
  return request<{ servers: McpRegisteredServer[] }>("/v1/mcp/servers").then((r) => r.servers);
}

export function enableMcpServer(
  catalog: string,
  params: Record<string, string> = {},
): Promise<{ name: string; registered: boolean }> {
  return request("/v1/mcp/servers", {
    method: "POST",
    body: JSON.stringify({ catalog, params }),
  });
}

export function disableMcpServer(name: string): Promise<{ name: string; registered: boolean }> {
  return request(`/v1/mcp/servers/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export function testMcpServer(
  name: string,
): Promise<{ name: string; ok: boolean; tools: McpTool[] }> {
  return request(`/v1/mcp/servers/${encodeURIComponent(name)}/test`, { method: "POST" });
}

/** One Judge0 queue's worker-pool state (docker-compose's judge0-server,
 * #264) — verified against Judge0's own GET /workers response shape. */
export interface CodeInterpreterQueue {
  queue: string;
  size: number;
  available: number;
  idle: number;
  working: number;
  paused: number;
  failed: number;
}

/** Only meaningful for the "code-interpreter" server (#264's sandboxed
 * execution) — a status card nested in the MCP tab, not a new page. */
export function getCodeInterpreterWorkers(): Promise<{ queues: CodeInterpreterQueue[] }> {
  return request("/v1/mcp/servers/code-interpreter/workers");
}
