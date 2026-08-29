import { Wrench } from "lucide-react";
import type { AgentToolCallTrace } from "@/lib/api";
import { T } from "@/lib/i18n";
import { Badge } from "./ui";

/** One tool call's card -- factored out so a chronological trace mixing tool
 * calls with reasoning steps (Playground's MCP chat mode) can render the
 * same card inline instead of duplicating its markup. */
export function ToolCallItem({ call, index }: { call: AgentToolCallTrace; index: number }) {
  return (
    <li className="rounded-md border border-border bg-muted/40 p-2 text-xs">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-muted-foreground">#{index + 1}</span>
        <span className="font-medium text-foreground">{call.tool}</span>
        <Badge tone={call.status === "ok" ? "ok" : "err"}>{call.status}</Badge>
        <span className="text-muted-foreground">{call.latency_ms}ms</span>
        {call.step_name && <Badge tone="info">step: {call.step_name}</Badge>}
      </div>
      {call.arguments && (
        <pre className="scroll-thin mt-1 max-h-32 overflow-auto whitespace-pre-wrap rounded bg-card p-1.5 text-[11px] text-foreground/80">
          {call.arguments}
        </pre>
      )}
      {call.result && (
        <pre className="scroll-thin mt-1 max-h-32 overflow-auto whitespace-pre-wrap rounded bg-card p-1.5 text-[11px] text-foreground/80">
          {call.result}
        </pre>
      )}
    </li>
  );
}

/** The "Try it" tool-call trace list: shared by the Agents surface and the
 * Playground's generic `mcp_servers` chat mode so a tool call renders
 * identically no matter which endpoint produced it. */
export function ToolCallTrace({ calls }: { calls: AgentToolCallTrace[] }) {
  if (calls.length === 0) return null;
  return (
    <div>
      <p className="mb-1 flex items-center gap-1 text-xs font-medium text-muted-foreground">
        <Wrench className="h-3.5 w-3.5" />
        <T>Tool calls</T>
      </p>
      <ol className="space-y-1.5">
        {calls.map((call, index) => (
          <ToolCallItem key={`${call.tool}-${index}`} call={call} index={index} />
        ))}
      </ol>
    </div>
  );
}
