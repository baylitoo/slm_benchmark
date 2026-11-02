# Fleet cookbook: from one machine to several

The ordered sequence for putting DocIE on more than one machine, and for
turning an existing single-machine install into the first node of a fleet
without interrupting what it serves.

This page is the *sequence*. Two neighbours cover the other halves and are
referenced rather than repeated:

- {doc}`fleet-runbook` — the per-task reference: add, check, drain, remove,
  diagnose one node.
- {doc}`../architecture/serving` — why the pieces behave as they do: desired
  versus observed state, how a request finds a model, what is fleet-aware and
  what is not.
- {doc}`getting-started` — bringing the stack up on one machine, which is the
  starting point for Track B below.

## What a second machine does and does not buy

Adding a machine adds **placement and replication options**. It does not pool
RAM or CPU: a model still has to fit, in full, on whichever node runs it.

| | One machine | Several |
|---|---|---|
| Where a model can run | here | any node that fits it |
| Replicas of one model | one per port | spread across machines |
| A node going down | the service is down | its placements leave the rotation |
| Largest servable model | this machine's RAM | **the largest single node's RAM** |

That last row is the one that disappoints if unstated. Three 32 GiB nodes do
not serve a model that needs 48 GiB.

## Before you start

Both tracks assume these. None of them is created by the fleet code.

:::{list-table}
:header-rows: 1
:widths: 30 70

* - Requirement
  - Why
* - A private network between the machines
  - The catalog, the lifecycle events and every inference call travel over it.
    There is no supported arrangement where a node reaches the central stack
    over the public internet.
* - A decided address range
  - Checked against every machine's existing routes, not just one. Docker
    bridges, other VPNs and provider private networks all consume ranges.
* - An explicit RAM budget per node
  - These are usually shared hosts. Free RAM at the moment you looked is not
    DocIE's to take.
* - The same application settings on every machine
  - A node reads the same configuration as the central stack and overrides
    only a handful of values.
:::

### Choosing the network

Two shapes work. Measure before committing, because the answer depends on
your machines rather than on the design.

**Direct peering** — the central box and each node share a tunnel, or a
provider-private network such as a vRack. One hop, no third party in the
inference path. Prefer this where the machines can reach each other directly.

**Through an existing hub** — the fleet joins a VPN you already run. Less to
build, at two costs: every inference request is decrypted and re-encrypted on
a machine that is not part of the fleet, and the fleet shares a network with
whatever else is on that VPN.

:::{warning}
A shared administrative VPN is **not** an authentication boundary for model
ports. A deployed runtime carries no credentials of its own — the API gateway
in front of it is what checks the key. Publishing a model port on a shared
VPN address offers unmetered inference to everyone already on that network.
Give DocIE its own interface and range, or restrict the port window to the
central box's address with host firewall rules.
:::

### What to measure

Three numbers, in order. Only the third decides anything.

1. Round-trip time over the path you already have, both directions.
2. Round-trip time through the candidate tunnel. The difference is the
   encryption and encapsulation cost.
3. **Time to first token for your fastest model, over the tunnel versus
   locally.**

The third is the one that matters because it is the worst case. A long
generation hides network latency — at 30 tokens per second a 400-token answer
takes thirteen seconds, and twenty milliseconds of tunnel is noise. A small
model answering in milliseconds does not hide anything: if time to first token
is single-digit milliseconds locally, a tunnel can multiply it.

This is also a placement argument, not only a network one. Models whose value
is low latency belong where the gateway reaches them without a tunnel; models
whose value is throughput are indifferent to it.

## Track A — a new fleet

Start here when nothing is deployed yet.

1. **Bring up the central stack** on the machine that will keep the
   application, following {doc}`getting-started`. It keeps the API, the
   Studio, PostgreSQL, Inngest, Redis, the workers, and document storage.
2. **Bring up the private network** and confirm a handshake in both
   directions.
3. **Open the central services onto the private address**, below.
4. **Add each serving node**, following *Add a node* in
   {doc}`fleet-runbook`.
5. **Verify**, below.

A node runs `docker-compose.serving-node.yml`: one container, the serving role
only. It is not a second copy of the application, and it mounts no shared
storage — everything it needs from the central stack travels over the network.

## Track B — an existing single machine

This is the common case: one machine already serves models and must keep doing
so while a second joins.

The existing machine becomes **central and serving at once**. That is a
supported arrangement, and it is what makes the transition additive: nothing
moves, nothing is re-seeded, and no model is interrupted.

:::{important}
Everything in this track is additive. Model weights stay exactly where they
are, the existing deployments keep their ports and their placements, and the
second node holds nothing until you deploy to it. If a step appears to require
stopping the existing node, it is the wrong step.
:::

### 1. Update the deployed stack first

The fleet fixes have to be live on the **central** machine before a second
node exists, because two of them are about reading another node's state:

- a worker resolving a deployment that lives on another node,
- a node identity that must not be re-minted on a transient read error.

Deploying a second node against an older central stack reproduces both. Check
what is actually deployed rather than what is merged.

### 2. Give the machine a private address

The existing machine joins the network like any other. Its public hostname is
unaffected: that name fronts the API, and is never how the gateway reaches a
node.

### 3. Open PostgreSQL and Inngest onto that address

Both are published on loopback by default, so no other machine can reach them.
In the **central** `.env`:

```
POSTGRES_BIND_HOST=<central private address>
INNGEST_BIND_HOST=<central private address>
```

That one address. Never `0.0.0.0`: PostgreSQL at least demands credentials,
and the Inngest dashboard has no authentication at all.

If the published host ports were changed from their defaults — a deployment
platform commonly does this to avoid collisions — the node has to use the
**host-published** values, not the container-internal ones:

:::{list-table}
:header-rows: 1
:widths: 40 60

* - Central knob
  - What the node sets
* - `POSTGRES_PORT`
  - `DOCIE_DB_PORT`
* - `INNGEST_HOST_PORT`
  - the port in `INNGEST_BASE_URL`
* - `INNGEST_CONNECT_HOST_PORT`
  - the port in `INNGEST_CONNECT_GATEWAY_URL`
:::

### 4. Confirm reachability from the second machine

Before configuring anything on it:

```
nc -z <central private address> <postgres host port>
nc -z <central private address> <inngest host port>
```

Both have to succeed. Everything after this step assumes they do.

### 5. Add the second node

Follow *Add a node* in {doc}`fleet-runbook`. The values that are genuinely
per-node, and what each one breaks when wrong:

:::{list-table}
:header-rows: 1
:widths: 34 66

* - Variable
  - Consequence of getting it wrong
* - `DOCIE_SERVING_ADVERTISE_HOST`
  - Every deploy records its endpoint as `http://{this}:{port}/v1`, resolved
    in the central API's own network. A wrong value routes this node's models
    to another machine, which answers **HTTP 200 with the wrong model**.
* - `DOCIE_SERVING_PUBLISH_IP`
  - Which interface the model ports are published on. A public interface
    exposes unauthenticated inference.
* - `DOCIE_SERVING_NODE_ID`
  - Must be unique. Set it explicitly wherever something other than you owns
    the state volume's lifecycle — a replaced volume otherwise becomes a new
    identity and orphans this node's catalog rows.
* - `DOCIE_DB_HOST`
  - The central machine's private address. Not a container name, and not the
    address of a VPN hub, which routes but holds no DocIE service.
* - `DOCIE_SERVING_MEM_LIMIT`
  - The node's RAM ceiling, and the denominator the sizing engine prices
    against. Set what the machine can spare.
* - `DOCIE_SERVING_PORT_RANGE_END`
  - How many runtimes this node can hold at once. The default window is
    twenty ports; replicas consume one each.
:::

### 6. Run the doctor before it takes traffic

```
docker compose -f docker-compose.serving-node.yml run --rm serving \
  docie-serving doctor --probe-ports
```

Most of what can be wrong with a serving node **fails as a wrong answer rather
than as an error**: it starts, publishes capacity, accepts deploys, and then
answers for models that live elsewhere. The doctor exits non-zero on a
configuration that cannot work and warns about the ones that otherwise present
as a wrong answer.

## Verifying the fleet

A node appearing on the Studio's Fleet page proves it can reach the catalog.
It does not prove routing, failover or isolation. Those need the checks in
*Add a node* and, at minimum, these:

1. Deploy one model to a chosen node. It lands there, not on the central box.
2. Scale it across both machines. Two placements, distinct identities,
   distinct owning nodes.
3. Send repeated requests. They reach both machines — verified **at the
   node**, not from the response.
4. Stop one node's model process. Requests continue elsewhere; its row leaves
   the rotation within one reconcile interval.
5. Stop the node entirely. Requests continue; its rows leave the rotation
   within the staleness window.
6. Restart it. It rejoins with no operator action.
7. Stop PostgreSQL. A previously resolved model still routes and completes,
   and already-loaded models keep serving.

Check 7 is the one that verifies the rule the whole design is shaped around —
the central control plane must never become a dependency for an already-running
model to keep serving — rather than assuming it.

## Model weights, today

Each node holds a local copy of what it serves. This is not an implementation
gap: `llama-server` memory-maps a local file and the snapshot families load a
local directory, so no runtime streams weights from object storage. A
distribution tier changes *how a node gets* a blob, never *where it serves it
from*.

Until that tier exists, a new node is seeded by hand — `docie-serving store
seed-ollama` or `store add-gguf` on the machine itself — which is one operation
per node per model. Plan the first fleet around a small number of models for
that reason, not because more would not fit.

## Going back to one machine

The transition is reversible, which is worth knowing before starting rather
than after.

1. Drain the second node from the Fleet page. It leaves the rotation
   immediately and releases its runtimes once in-flight requests finish.
2. Delete its deployments, so no placement rows point at it. Rows owned by a
   node that no longer exists are the one case nothing cleans up
   automatically — they cannot be told apart from a node that is temporarily
   off.
3. Stop and remove its container.
4. Optionally return the central services to loopback.

The central machine keeps serving throughout. Its own models were never moved.
