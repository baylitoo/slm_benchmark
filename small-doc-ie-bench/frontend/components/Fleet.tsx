"use client";

import { useMemo } from "react";
import { Server, HardDrive, Boxes, Network } from "lucide-react";
import {
  getFleet,
  formatBytes,
  getStore,
  type FleetNode,
  type FleetView,
  type StoreEntry,
} from "@/lib/api";
import { usePolling } from "@/lib/usePolling";
import { Alert, Badge, Card } from "./ui";
import { LiveIndicator } from "./LiveIndicator";
import { Table, type Column } from "./patterns/Table";

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

      {isSingle ? (
        <SingleNode node={nodes[0]} store={store.data} />
      ) : (
        <>
          <NodeTable
            nodes={nodes}
            staleAfter={staleAfter}
            loading={fleet.loading}
            error={fleet.error}
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
 * One node: the stock deployment. No tables, no empty per-node columns, and no
 * implication that something is missing — because nothing is.
 */
function SingleNode({ node, store }: { node: FleetNode; store: StoreEntry[] | null }) {
  const held = node.store_models?.length ?? null;
  const total = store?.length ?? null;

  return (
    <div className="space-y-4">
      <Card
        title="Single-node deployment"
        icon={<Server className="h-4 w-4" />}
        subtitle="One serving node, which is the default. The per-node views below appear once a second node joins."
      >
        <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Node">
            <span className="font-mono text-xs">{node.node_id}</span>
          </Stat>
          <Stat label="State">
            {node.reporting ? (
              <Badge tone="ok">reporting</Badge>
            ) : (
              <Badge tone="err">not reporting</Badge>
            )}
          </Stat>
          <Stat label="Free / total">
            {node.never_reported ? (
              <span className="text-muted">—</span>
            ) : (
              <span className="tabular-nums">
                {formatBytes(node.free_bytes)} / {formatBytes(node.total_bytes)}
              </span>
            )}
          </Stat>
          <Stat label="Deployments">
            <span className="tabular-nums">{node.deployments}</span>
          </Stat>
        </dl>

        {looksGenerated(node.node_id) && (
          <p className="mt-4 text-xs text-muted">
            The id was generated on first publish and persists in the serving volume.
            Set <code>DOCIE_SERVING_NODE_ID</code> to give this machine a readable name —
            optional, and only worth doing when there is more than one.
          </p>
        )}
      </Card>

      <Card
        title="Models on this node"
        icon={<HardDrive className="h-4 w-4" />}
        subtitle="Blob stores are per node, so with one node everything the catalog lists is either here or not seeded at all."
      >
        {held == null ? (
          <p className="text-sm text-muted">
            This node has not reported its store yet — it publishes on the next
            reconcile cycle.
          </p>
        ) : (
          <p className="text-sm">
            <span className="tabular-nums font-medium">{held}</span> model
            {held === 1 ? "" : "s"} held
            {total != null && total !== held && (
              <>
                {" "}
                of <span className="tabular-nums">{total}</span> in the catalog — the
                difference is seeded nowhere yet.
              </>
            )}
            .
          </p>
        )}
      </Card>

      <Card
        title="Adding a node"
        icon={<Network className="h-4 w-4" />}
        subtitle="What a second machine needs. No change is required on this one beyond opening Postgres and Inngest to the private network."
      >
        <ol className="list-decimal space-y-2 pl-5 text-sm text-muted">
          <li>
            Set <code>POSTGRES_BIND_HOST</code> and <code>INNGEST_BIND_HOST</code> to this
            host&apos;s private address — both are loopback-bound by default, so a node
            cannot reach them until you do.
          </li>
          <li>
            Bring the node up with <code>docker-compose.serving-node.yml</code>. Four
            values have no default and it refuses to start without them.
          </li>
          <li>Seed the model on that node — blob stores are not shared.</li>
          <li>
            Deploy with a target node, then check the recorded endpoint from this host
            before routing anything through it.
          </li>
        </ol>
      </Card>
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
}: {
  nodes: FleetNode[] | null;
  staleAfter: number | null;
  loading?: boolean;
  error?: unknown;
}) {
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
    ],
    [staleAfter],
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
