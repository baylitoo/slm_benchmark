"use client";

import { useMemo, useState } from "react";
import { Server, HardDrive, Boxes, Network, PauseCircle, PlayCircle } from "lucide-react";
import {
  ApiError,
  drainNode,
  getFleet,
  formatBytes,
  getStore,
  resumeNode,
  type FleetNode,
  type FleetView,
  type StoreEntry,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { toUserMessage } from "@/lib/errors";
import { useToast } from "./Toast";
import { Alert, Badge, Button, Card } from "./ui";
import { LiveIndicator } from "./LiveIndicator";
import { Table, type Column } from "./patterns/Table";
import { Meter, StatRow, StatTile } from "./patterns/StatTile";

const POLL_MS = 5000;

/** A generated node id is 6 bytes of hex; an operator-chosen one is a name. */
function looksGenerated(nodeId: string): boolean {
  return /^[0-9a-f]{12}$/.test(nodeId);
}

/**
 * Fleet tab (#536): the serving nodes, their capacity, and which of them holds
 * each model's blob.
 *
 * Everything is the OBSERVED surface each node's reconciler publishes —
 * `GET /v1/serving/nodes` and `GET /v1/serving/store`. Nothing is derived
 * client-side, in particular not liveness: `reporting` comes from the same
 * staleness rule the router uses to demote a node's placements, so this page
 * and the routing layer cannot disagree about which nodes are up.
 *
 * SINGLE NODE IS THE DEFAULT, not a degenerate case. A stock deployment has
 * one node, an auto-generated id and no fleet env vars set, and a table with
 * one row called `7f3a2b…` beside two empty columns reads as something broken.
 * So one node gets a plain summary instead, and the fleet tables appear when
 * there is actually a fleet.
 *
 * Two distinctions the multi-node view exists to make visible, both of which
 * mislead if flattened:
 *
 *  - a node that has NEVER published is new, not dead. It shows as "staged",
 *    because the action is to start it rather than investigate a failure.
 *  - a model `missing` from a node is not `unreported`. The first means
 *    re-seed that node; the second means its store list has not arrived yet.
 *    Showing both as missing would send an operator to re-seed a healthy node.
 */
export function Fleet({ active = true }: { active?: boolean }) {
  const fleet = usePolling<FleetView>(getFleet, POLL_MS, active);
  const store = usePolling<StoreEntry[]>(getStore, POLL_MS, active);

  const nodes = fleet.data?.nodes ?? null;
  const staleAfter = fleet.data?.stale_after_s ?? null;
  const refresh = () => {
    fleet.refresh();
    store.refresh();
  };

  const silent = (nodes ?? []).filter((n) => !n.reporting && !n.never_reported);
  const isSingle = nodes != null && nodes.length === 1;
  const draining = (nodes ?? []).filter((n) => n.drain_state === "draining");

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end">
        <LiveIndicator
          live={fleet.live}
          refreshing={fleet.refreshing}
          lastUpdated={fleet.lastUpdated}
          onRefresh={refresh}
        />
      </div>

      {fleet.data && !fleet.data.observed_available && (
        <Alert tone="warn">
          Fleet state unavailable — {fleet.data.detail ?? "the catalog is unreachable"}.
          Running models keep serving; this view is what needs the catalog.
        </Alert>
      )}

      {silent.length > 0 && (
        <Alert tone="err">
          {silent.length === 1
            ? `Node ${silent[0].node_id} has stopped reporting.`
            : `${silent.length} nodes have stopped reporting.`}{" "}
          Their placements are demoted, so traffic goes to the nodes that are live.
          Logs are node-local — reach the node itself to diagnose.
        </Alert>
      )}

      {draining.length > 0 && (
        <Alert tone="warn">
          {draining.length === 1
            ? `Node ${draining[0].node_id} is draining.`
            : `${draining.length} nodes are draining.`}{" "}
          No new requests route to them; their runtimes stop once in-flight
          requests have had their grace period, and they report empty when done.
        </Alert>
      )}

      {isSingle ? (
        <SingleNode node={nodes[0]} store={store.data} />
      ) : (
        <>
          <NodeTable
            nodes={nodes}
            staleAfter={staleAfter}
            loading={fleet.loading}
            error={fleet.error}
            onChanged={refresh}
          />
          <BlobPresence
            store={store.data}
            loading={store.loading}
            error={store.error}
          />
        </>
      )}
    </div>
  );
}

/**
 * One node: the stock deployment.
 *
 * Rebuilt around what the page actually knows. The first version was three
 * stacked cards whose subtitles explained concepts, and whose content was four
 * readings, one sentence and a prose runbook -- roughly nine parts chrome to
 * one part data. Two specific corrections:
 *
 * * **Free/total was a string.** It is the one genuinely visual reading here,
 *   so it gets a meter; the numbers stay beside it because a bar is not a
 *   quantity and colour alone is not a reading.
 * * **"Adding a node" was a four-step runbook inside a dashboard**, duplicating
 *   docs/operations/fleet-runbook.md. Demoted to a disclosure: still one click
 *   away for a first-time operator, no longer the largest thing on screen.
 */
function SingleNode({ node, store }: { node: FleetNode; store: StoreEntry[] | null }) {
  const held = node.store_models?.length ?? null;
  const total = store?.length ?? null;
  const total_bytes = node.total_bytes ?? 0;
  const free_bytes = node.free_bytes ?? 0;
  const usedFraction = total_bytes > 0 ? (total_bytes - free_bytes) / total_bytes : 0;

  return (
    <div className="space-y-4">
      {/* Identity as a strip, not a card. The node id is the least actionable
          thing here -- it had the most prominent slot and now sits inline. */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <NodeState node={node} staleAfterS={null} />
        <span className="font-mono text-xs text-muted-foreground" title={node.node_id}>
          {node.node_id}
        </span>
        <span className="text-xs text-muted-foreground">· single-node deployment</span>
      </div>

      <StatRow>
        <StatTile
          label="Memory"
          value={
            node.never_reported ? (
              <span className="text-muted-foreground">—</span>
            ) : (
              `${formatBytes(free_bytes)} free`
            )
          }
          meter={
            node.never_reported ? undefined : (
              <Meter fraction={usedFraction} label="memory used on this node" />
            )
          }
          detail={
            node.never_reported
              ? "No measurement published yet."
              : `${formatBytes(total_bytes - free_bytes)} used of ${formatBytes(total_bytes)}`
          }
        />
        <StatTile
          label="Deployments"
          value={node.deployments}
          detail={node.deployments === 1 ? "running here" : "running on this node"}
        />
        <StatTile
          label="Models held"
          value={held ?? "—"}
          detail={
            held == null
              ? "Store not reported yet — it publishes next cycle."
              : total != null && total !== held
                ? `${total - held} in the catalog are seeded nowhere yet`
                : "Everything the catalog lists is here"
          }
        />
      </StatRow>

      {held != null && held > 0 && node.store_models && (
        <Card title="Models on this node" icon={<HardDrive className="h-4 w-4" />}>
          <div className="flex flex-wrap gap-1.5">
            {node.store_models.map((name) => (
              <span
                key={name}
                className="rounded-md border border-border bg-muted/40 px-2 py-1 font-mono text-xs text-foreground/90"
              >
                {name}
              </span>
            ))}
          </div>
        </Card>
      )}

      {looksGenerated(node.node_id) && (
        <p className="text-xs text-muted-foreground">
          The id was generated on first publish and persists in the serving volume.
          Set <code>DOCIE_SERVING_NODE_ID</code> to give this machine a readable name —
          optional, and only worth doing when there is more than one.
        </p>
      )}

      <details className="group rounded-lg border border-border bg-card">
        <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-3 text-sm font-medium text-foreground">
          <Network className="h-4 w-4 text-muted-foreground" />
          Adding a second node
          <span className="ml-auto text-xs font-normal text-muted-foreground">
            nothing changes on this one
          </span>
        </summary>
        <ol className="list-decimal space-y-2 border-t border-border px-4 py-3 pl-9 text-sm text-muted-foreground">
          <li>
            Set <code>POSTGRES_BIND_HOST</code> and <code>INNGEST_BIND_HOST</code> to this
            host&apos;s private address — both are loopback-bound by default, so a node
            cannot reach them until you do.
          </li>
          <li>
            Bring the node up with <code>docker-compose.serving-node.yml</code>, then run{" "}
            <code>docie-serving doctor --probe-ports</code> on it before it takes traffic.
          </li>
          <li>Seed the model on that node — blob stores are not shared.</li>
          <li>
            Deploy with a target node, then check the recorded endpoint from this host
            before routing anything through it.
          </li>
        </ol>
      </details>
    </div>
  );
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-muted">{label}</dt>
      <dd className="mt-1 text-sm">{children}</dd>
    </div>
  );
}

function NodeTable({
  nodes,
  staleAfter,
  loading,
  error,
  onChanged,
}: {
  nodes: FleetNode[] | null;
  staleAfter: number | null;
  loading?: boolean;
  error?: unknown;
  onChanged: () => void;
}) {
  const { toast } = useToast();
  const [busy, setBusy] = useState<string | null>(null);

  async function act(node: FleetNode, drain: boolean) {
    setBusy(node.node_id);
    try {
      if (!drain) {
        await resumeNode(node.node_id);
      } else {
        try {
          await drainNode(node.node_id);
        } catch (err) {
          // The server refuses a drain that would take a model offline, naming
          // which ones. That sentence is the whole decision, so it is what the
          // operator is asked to confirm — not a generic "are you sure".
          if (err instanceof ApiError && err.status === 409) {
            if (!window.confirm(`${err.message}\n\nDrain anyway?`)) {
              return;
            }
            await drainNode(node.node_id, { acknowledgeOutage: true });
          } else {
            throw err;
          }
        }
      }
      toast({
        title: drain ? "Node draining" : "Node admitting",
        description: node.node_id,
        tone: "success",
      });
      onChanged();
    } catch (err) {
      const msg = toUserMessage(err, {
        unavailable: "Draining isn't available on this server.",
        fallback: drain ? "Drain failed." : "Resume failed.",
      });
      toast({ title: drain ? "Drain failed" : "Resume failed", description: msg, tone: "error" });
    } finally {
      setBusy(null);
    }
  }

  const columns = useMemo<Column<FleetNode>[]>(
    () => [
      {
        key: "node",
        header: "Node",
        sortAccessor: (n) => n.node_id,
        render: (n) => (
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 shrink-0 text-muted" />
            <span className="font-mono text-xs">{n.node_id}</span>
          </div>
        ),
      },
      {
        key: "state",
        header: "State",
        sortAccessor: (n) => (n.reporting ? 0 : n.never_reported ? 1 : 2),
        render: (n) => <NodeState node={n} staleAfterS={staleAfter} />,
      },
      {
        key: "capacity",
        header: "Free / total",
        sortAccessor: (n) => n.free_bytes ?? 0,
        render: (n) =>
          n.never_reported ? (
            // Zeroes, not a measurement. Rendering them as numbers would read
            // as "no room" rather than "nothing measured yet".
            <span className="text-muted">—</span>
          ) : (
            <span className="tabular-nums">
              {formatBytes(n.free_bytes)} / {formatBytes(n.total_bytes)}
            </span>
          ),
      },
      {
        key: "source",
        header: "Measured",
        render: (n) =>
          n.never_reported ? (
            <span className="text-muted">—</span>
          ) : (
            <Badge tone={n.source === "cgroup" ? "ok" : "warn"}>
              {n.source === "cgroup" ? "cgroup" : "vm (soft)"}
            </Badge>
          ),
      },
      {
        key: "deployments",
        header: "Deployments",
        sortAccessor: (n) => n.deployments,
        render: (n) => <span className="tabular-nums">{n.deployments}</span>,
      },
      {
        key: "store",
        header: "Models held",
        render: (n) =>
          n.store_models == null ? (
            <span className="text-muted" title="This node has not reported its store yet">
              not reported
            </span>
          ) : (
            <span className="tabular-nums">{n.store_models.length}</span>
          ),
      },
      {
        key: "actions",
        header: "",
        className: "text-right",
        render: (n) => {
          const drained = Boolean(n.drained_at);
          return (
            <Button
              size="sm"
              variant={drained ? "secondary" : "ghost"}
              loading={busy === n.node_id}
              disabled={busy !== null}
              title={
                drained
                  ? "Resume: admit this node again. Its reconciler restores the deployments the drain stopped."
                  : "Drain: stop routing to this node, then let it stop its runtimes once in-flight requests have finished."
              }
              onClick={() => act(n, !drained)}
            >
              {drained ? (
                <>
                  <PlayCircle className="h-3.5 w-3.5" /> Resume
                </>
              ) : (
                <>
                  <PauseCircle className="h-3.5 w-3.5" /> Drain
                </>
              )}
            </Button>
          );
        },
      },
    ],
    // `act` is redefined each render, so it is referenced rather than listed:
    // including it would rebuild the columns on every poll tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [staleAfter, busy],
  );

  return (
    <Card
      title="Serving nodes"
      icon={<Server className="h-4 w-4" />}
      subtitle={
        staleAfter
          ? `A node counts as reporting while its last publish is under ${Math.round(staleAfter)}s old — the same rule the router uses to demote a node's placements.`
          : undefined
      }
    >
      <Table
        columns={columns}
        rows={nodes}
        getRowKey={(n) => n.node_id}
        loading={loading}
        error={error}
        emptyLabel="No serving nodes have reported"
        emptyDescription="A node appears here once its reconciler publishes a capacity snapshot."
        emptyIcon={<Server className="h-6 w-6" />}
      />
    </Card>
  );
}

function NodeState({
  node,
  staleAfterS,
}: {
  node: FleetNode;
  staleAfterS: number | null;
}) {
  // Drain is shown ahead of liveness on purpose. A drained node is out of the
  // rotation by an operator's declaration, which holds whether or not it is
  // still reporting -- showing "reporting" there would describe the machine
  // correctly and the routing wrongly.
  if (node.drain_state === "draining" || node.drain_state === "drained") {
    const done = node.drain_state === "drained";
    return (
      <span className="flex items-center gap-2">
        <Badge tone="warn">{done ? "drained" : "draining"}</Badge>
        <span className="text-xs text-muted">
          {done
            ? "no placements left"
            : `${node.deployments} still finishing`}
        </span>
      </span>
    );
  }
  if (node.never_reported) {
    return (
      <span className="flex items-center gap-2">
        <Badge tone="info">staged</Badge>
        <span className="text-xs text-muted">no measurement yet</span>
      </span>
    );
  }
  if (!node.reporting) {
    const age = node.age_s == null ? null : Math.round(node.age_s);
    return (
      <span className="flex items-center gap-2">
        <Badge tone="err">not reporting</Badge>
        {age != null && (
          <span className="text-xs text-muted">
            last publish {age}s ago
            {staleAfterS ? ` (threshold ${Math.round(staleAfterS)}s)` : ""}
          </span>
        )}
      </span>
    );
  }
  return <Badge tone="ok">reporting</Badge>;
}

/**
 * Which nodes hold each model's blob. The catalog is global and the blob store
 * is node-local, so a model being listed in Models says nothing about which
 * nodes can actually serve it — deploying to a node without the blob fails for
 * a reason nothing else here would show.
 */
function BlobPresence({
  store,
  loading,
  error,
}: {
  store: StoreEntry[] | null;
  loading?: boolean;
  error?: unknown;
}) {
  const rows = useMemo(() => (store ?? []).filter((m) => m.nodes != null), [store]);

  const columns = useMemo<Column<StoreEntry>[]>(
    () => [
      {
        key: "name",
        header: "Model",
        sortAccessor: (m) => m.name,
        render: (m) => (
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs">{m.name}</span>
            {m.store_view === "catalog" && (
              <span title="Known to the catalog; this process has no blob for it, so size is unknown">
                <Badge tone="neutral">catalog only</Badge>
              </span>
            )}
          </div>
        ),
      },
      {
        key: "present",
        header: "Held by",
        render: (m) =>
          m.nodes && m.nodes.present.length > 0 ? (
            <NodeChips ids={m.nodes.present} tone="ok" />
          ) : (
            <span className="text-muted">nowhere</span>
          ),
      },
      {
        key: "missing",
        header: "Missing from",
        render: (m) =>
          m.nodes && m.nodes.missing.length > 0 ? (
            <div className="space-y-1">
              <NodeChips ids={m.nodes.missing} tone="warn" />
              <p className="text-xs text-muted">
                Seed it there: <code>docie-serving store seed-ollama …</code>
              </p>
            </div>
          ) : (
            <span className="text-muted">—</span>
          ),
      },
      {
        key: "unreported",
        header: "Unknown on",
        render: (m) =>
          m.nodes && m.nodes.unreported.length > 0 ? (
            <span
              className="flex flex-wrap gap-1"
              title="These nodes have not reported their store yet — not the same as missing"
            >
              <NodeChips ids={m.nodes.unreported} tone="neutral" />
            </span>
          ) : (
            <span className="text-muted">—</span>
          ),
      },
    ],
    [],
  );

  return (
    <Card
      title="Where each model's blob is"
      icon={<HardDrive className="h-4 w-4" />}
      subtitle="The model catalog is global; blob stores are per node. A deploy only succeeds on a node that holds the blob."
    >
      <Table
        columns={columns}
        rows={rows}
        getRowKey={(m) => m.name}
        loading={loading}
        error={error}
        emptyLabel="No per-node blob information yet"
        emptyDescription="Reported once a node publishes the contents of its store."
        emptyIcon={<Boxes className="h-6 w-6" />}
      />
    </Card>
  );
}

function NodeChips({ ids, tone }: { ids: string[]; tone: "ok" | "warn" | "neutral" }) {
  return (
    <span className="flex flex-wrap gap-1">
      {ids.map((id) => (
        <Badge key={id} tone={tone}>
          <span className="font-mono text-[11px]">{id}</span>
        </Badge>
      ))}
    </span>
  );
}
