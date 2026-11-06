"use client";

import { useState } from "react";
import { MemoryStick } from "lucide-react";
import {
  ApiError,
  formatBytes,
  getFleet,
  setNodeBudget,
  type FleetNode,
  type FleetView,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { toUserMessage } from "@/lib/errors";
import { Badge, Button, Card, TextInput } from "./ui";
import { useToast } from "./Toast";
import { Table, type Column } from "./patterns/Table";

const POLL_MS = 5000;
const GIB = 1024 ** 3;

const SOURCE_LABEL: Record<string, string> = {
  catalog: "set here",
  catalog_cache: "set here (last known)",
  setting: "node setting",
  none: "none",
};

/** Parse a GiB amount typed by an operator. Null for anything not positive. */
export function parseGib(text: string): number | null {
  const value = Number(text.trim().replace(",", "."));
  if (!Number.isFinite(value) || value <= 0) return null;
  return Math.round(value * GIB);
}

/**
 * Per-node RAM budget (#539): the kernel ceiling, the budget DocIE sizes
 * against, what the node is using, and an editor for the override.
 *
 * Everything shown is what each node published. A node that stopped reporting
 * shows no numbers rather than its last ones, and an override the node has
 * not picked up yet is marked as pending.
 */
export function NodeBudgets({ active = true }: { active?: boolean }) {
  const fleet = usePolling<FleetView>(getFleet, POLL_MS, active);
  const nodes = fleet.data?.nodes ?? [];

  const columns: Column<FleetNode>[] = [
    {
      key: "node",
      header: "Node",
      render: (n) => <span className="font-mono text-xs">{n.node_id}</span>,
    },
    {
      key: "ceiling",
      header: "Hard ceiling",
      render: (n) =>
        !n.reporting ? (
          "—"
        ) : n.ceiling_bytes != null ? (
          formatBytes(n.ceiling_bytes)
        ) : (
          <span title="No cgroup memory limit: nothing but DocIE enforces the budget on this host">
            <Badge tone="warn">none — budget only</Badge>
          </span>
        ),
    },
    {
      key: "budget",
      header: "Budget",
      render: (n) => <BudgetCell node={n} />,
    },
    {
      key: "used",
      header: "Used",
      render: (n) => (n.reporting ? formatBytes(n.working_set_bytes ?? null) : "—"),
    },
    {
      key: "headroom",
      header: "Headroom",
      render: (n) => <HeadroomCell node={n} />,
    },
    {
      key: "edit",
      header: "",
      className: "text-right",
      render: (n) => <BudgetEditor node={n} onSaved={fleet.refresh} />,
    },
  ];

  return (
    <Card
      icon={<MemoryStick className="h-5 w-5" />}
      title="Node budgets"
      subtitle="DocIE's share of each host. Admission sizes against min(budget, ceiling); lowering a budget refuses new loads and unloads nothing."
    >
      {fleet.error && !fleet.data ? (
        <p className="text-sm text-muted-foreground">{toUserMessage(fleet.error)}</p>
      ) : fleet.data && !fleet.data.observed_available ? (
        <p className="text-sm text-muted-foreground">
          {fleet.data.detail ?? "Per-node budgets need the model catalog."}
        </p>
      ) : nodes.length === 0 ? (
        <p className="text-sm text-muted-foreground">No serving node has reported yet.</p>
      ) : (
        <Table columns={columns} rows={nodes} getRowKey={(n) => n.node_id} />
      )}
    </Card>
  );
}

function BudgetCell({ node }: { node: FleetNode }) {
  const requested = node.mem_budget_bytes ?? null;
  const effective = node.effective_budget_bytes ?? null;
  const pending = node.reporting && requested != null && requested !== effective;
  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      <span>{effective != null ? formatBytes(effective) : node.reporting ? "none" : "—"}</span>
      {node.reporting && node.budget_source && (
        <span className="text-xs text-muted-foreground">
          {SOURCE_LABEL[node.budget_source] ?? node.budget_source}
        </span>
      )}
      {node.budget_stale && (
        <span title="The node could not read the catalog and is using the last budget it was given">
          <Badge tone="warn">catalog unreachable</Badge>
        </span>
      )}
      {pending && (
        <span title={`Set to ${formatBytes(requested)}; the node applies it on its next cycle`}>
          <Badge tone="info">applying {formatBytes(requested)}</Badge>
        </span>
      )}
      {!node.reporting && <Badge tone="err">unreachable</Badge>}
    </span>
  );
}

function HeadroomCell({ node }: { node: FleetNode }) {
  if (!node.reporting) return <>—</>;
  const budget = node.effective_budget_bytes ?? null;
  const used = node.working_set_bytes ?? null;
  if (budget != null && used != null && used > budget) {
    return (
      <span title="New loads are refused until usage falls below the budget; nothing is unloaded">
        <Badge tone="err">over by {formatBytes(used - budget)}</Badge>
      </span>
    );
  }
  return <>{formatBytes(node.free_bytes)}</>;
}

function BudgetEditor({ node, onSaved }: { node: FleetNode; onSaved: () => void }) {
  const { toast } = useToast();
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const parsed = parseGib(draft);

  async function save(budgetBytes: number | null) {
    setBusy(true);
    try {
      await setNodeBudget(node.node_id, budgetBytes, node.mem_budget_version ?? null);
      setDraft("");
      toast({
        tone: "success",
        title:
          budgetBytes == null
            ? `${node.node_id}: override cleared`
            : `${node.node_id}: budget set to ${formatBytes(budgetBytes)}`,
      });
      onSaved();
    } catch (error) {
      const message =
        error instanceof ApiError && error.status === 403
          ? "Setting a budget needs an operator key (API_KEYS), not a Studio sign-in or issued key."
          : error instanceof ApiError && error.status === 409
            ? "Someone changed this budget first. The table has been refreshed; check it and try again."
            : toUserMessage(error);
      toast({ tone: "error", title: "Budget not saved", description: message });
      onSaved();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex items-center justify-end gap-2">
      <TextInput
        aria-label={`New budget for ${node.node_id} in GiB`}
        placeholder="GiB"
        inputMode="decimal"
        className="h-8 w-20"
        value={draft}
        disabled={busy}
        onChange={(e) => setDraft(e.target.value)}
      />
      <Button
        size="sm"
        disabled={busy || parsed == null}
        loading={busy}
        onClick={() => parsed != null && save(parsed)}
      >
        Set
      </Button>
      {node.mem_budget_bytes != null && (
        <Button
          size="sm"
          variant="ghost"
          disabled={busy}
          title="Remove the override: the node falls back to its own setting, then its ceiling"
          onClick={() => save(null)}
        >
          Clear
        </Button>
      )}
    </div>
  );
}
