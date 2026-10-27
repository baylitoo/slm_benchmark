# Serving control plane

This is the most intricate part of the system. It decides what models should be
running, makes that true, watches whether it stayed true, and answers the
question "where do I send this request".

## Desired and observed

Two kinds of state, deliberately separated.

**Desired state** is what should be running: a `DeploymentSpec` with a launch
spec and a `desired_state` of `RUNNING` or `STOPPED`. It lives in
`deployments.json` under `DOCIE_SERVING_HOME`, written by `PersistentSupervisor`
(`serving/supervisor.py`).

**Observed state** is what actually is: the lifecycle state of the record, and a
row in the Postgres `model_placement` table carrying phase, pid, RSS, health,
last error, endpoint, slot count and measured throughput.

`ServingReconciler` (`serving/reconciler.py`) is the loop that compares them,
repairs the difference, and publishes what it saw. It is the sole writer of
observed state.

Postgres is optional. With no `DATABASE_URL` the cycle still repairs and still
saves `deployments.json`; it just skips the publish. That is why the
activation flag lives in the file rather than the database — the routing
contract must not acquire a database dependency.

```{important}
A non-live placement stores `endpoint=""`, not `NULL` — the column is NOT NULL.
Every reader must treat the empty string as "no live endpoint". Forgetting this
is how a request gets routed to a model that is not running.
```

## Components

| Module | Owns |
|---|---|
| `control_plane.py` | The facade: deploy, stop, load, remove, replicas, port allocation. A thin layer over small protocols, rebuilt per request |
| `supervisor.py` | Desired state, and spawning processes through runtime adapters |
| `reconciler.py` | The observe → repair → publish loop |
| `catalog.py` | The Postgres layer: store entries, placements, the node row, activity |
| `profile_resolver.py` | Routing by deployment name or config profile |
| `placement_resolver.py` | Routing by `store:<name>`, including replica choice |
| `model_store.py` | The on-disk model store and the family contracts |
| `resources.py` | Node memory, footprint prediction and calibration |
| `lifecycle.py` | Fit checks, load coordination, LRU victim selection |

## How a request finds a model

Everything routes through one seam, `resolve_extraction_profile`
(`serving/profile_resolver.py`), which tries four things in order:

1. **`deployment=<name>`** — must be a *live* deployment or it raises. Never
   falls back to anything.
2. **`store:<name>`** — goes to the placement resolver and likewise never falls
   through to config or environment.
3. **A bare name** — a profile from `models.yaml`, an upstream model id, or a
   live deployment name.
4. **Nothing** — the configured default profile.

An explicit selector never silently degrades into something else. That is the
property that makes a misconfiguration visible instead of mysterious.

### The two real paths

They differ in where truth lives, and it matters.

| | `deployment=` / bare name | `store:<name>` |
|---|---|---|
| Source of truth | `deployments.json`, read fresh per call | Postgres placements |
| Needs a database | no | yes |
| Replicas | no | yes, with load balancing |
| Works from another machine | yes, via the placement row | **yes** |

Both paths now work across machines, but for different reasons. The `store:`
path always did: a placement row carries a full endpoint URL and the resolver
picks among live replicas. The `deployment=` and bare-name paths read
`deployments.json`, which is one node's file, so they used to refuse a
deployment running anywhere else; they now fall back to the deployment's
placement row when the local file has no answer. The local file still wins when
it does — it is fresher and carries launch detail a placement row lacks.

Replica choice is `round_robin_choice` by default, or `session_affinity_choice`
when a `session_id` is supplied, which keeps a conversation on one replica so
llama.cpp's prefix KV cache stays warm.

Affinity is **computed, not remembered**: highest-random-weight (rendezvous)
selection over the live replicas, hashed with `hashlib` rather than the builtin
`hash`, which is salted per process. The assignment is therefore a pure
function of the session id and the live set, so every API and worker process
agrees without coordinating — which matters because extraction resolves
placements on the `worker` service, the one built to scale. Removing a replica
reassigns only the sessions that were on it.

```{warning}
The round-robin counter is still **process-local**: a per-process dict that
resets on restart. With several API or worker replicas the distribution is
uneven rather than wrong. Expect that when measuring spread across nodes —
both nodes receiving traffic is the property, not an even split.
```

A `store:` reference that resolves to nothing live does not simply fail: the
API triggers a load and answers `202` with an estimate. A name absent from the
catalog entirely is still a `404`.

## Supervision

A runtime adapter builds a command line and spawns the process. The supervisor
keeps the handle, the pid, and the pid's creation time — the last of these so a
recycled pid cannot impersonate a dead runtime.

Then the loop takes over, and its rules are where the hard-won behaviour lives:

- **Liveness is health, not pid.** A phase of `hot` is published only after a
  real health probe passes. A process that exists but does not answer is not
  ready.
- **Fast death for ready, patience for loading.** A runtime that was ready and
  stops answering is declared failed after a couple of misses. A runtime that
  has never been ready is still loading its weights and is left alone.
- **Restarts are gated.** A dead deployment is respawned only if restart budget
  remains *and* a live fit check passes. Without that gate, a model that OOMs
  on load would crash, respawn, and OOM again forever.
- **Port collisions queue a repair** — a redeploy on a reallocated port, run
  outside the supervisor lock.

An observation pass runs under the supervisor lock, off the event loop. The
Postgres publish and the throughput probe run *outside* it, because a slow
database must never stall the handlers.

Throughput (tokens per second, time to first token) is **measured**, never
predicted. A number guessed from a CPU model name would be a confident
fabrication — memory bandwidth, CPU quota, co-resident runtimes and
quantization all move it by multiples — so an un-probed deployment publishes
nothing at all.

## One reconciler per node

```{important}
The serving role is single-replica **by design**. It is the only process that
holds the subprocess handles and the only writer of `deployments.json`. Two of
them would clobber that file and mis-probe the advertise host.
```

Compose's replica count is advisory — `--scale serving=2` bypasses it — so
there is a second guard: `ReconcilerLease`, an advisory heartbeat file in the
serving home. A fresh lease belonging to a different instance makes the claim
fail loudly rather than letting two reconcilers fight. It is explicitly *not* a
distributed lock; it turns a misconfiguration into a visible refusal.

The reconciler must also use the **shared** control plane, not a freshly
constructed supervisor, because only the shared instance holds the adapters'
process handles.

The node resource snapshot is one row **per node**, upserted, written only by
that node's own reconciler. It is keyed by a node id generated once and
persisted beside `deployments.json`, so it survives a restart on the same
volume and is lost exactly when the deployments it describes are.

So "single replica" is a statement about one node, not about the fleet: each
node runs exactly one serving process, and several nodes publish side by side.

## Memory

The whole admission story rests on one question: how much memory is actually
available, and to whom.

Readings are **cgroup-first**. A real `memory.max` limit is authoritative and
free is computed from it. When the cgroup reports no limit, it falls back to the
host view — and says so, tagging the reading `vm` rather than `cgroup`, so a
soft number is never mistaken for an enforced one. On a shared host that
distinction is the difference between sizing against your own slice and sizing
against memory other applications are about to want.

A deployment's footprint starts as a **prediction** — weights, KV cache for the
configured context and slots, overhead. Once the runtime has been hot and
stable for several cycles, the measured resident size **replaces** the
prediction. Measurement beats arithmetic as soon as measurement exists.

Admission compares the larger of the two against free memory minus a safety
margin, and fails open on anything unmeasurable.

Eviction is LRU over hot deployments only, with guards that matter: pinned
deployments are never chosen, nothing hot for less than the minimum is chosen,
at most a couple go per cycle, and **nothing is evicted unless the allowed
evictions would actually free enough** — no sacrificing a model for a load that
will fail anyway. An evicted deployment keeps its record and its port, so the
next request reloads it.

Idle unload is separate and simpler: hot for long enough, unused for longer than
the TTL, not pinned.

### Ceilings on Docker Desktop

The total the tracker reports is the **lowest ceiling that applies**, not the
host's RAM. On Windows there are three stacked layers:

```text
host RAM (e.g. 16 GB)
  └── WSL2 VM — defaults to 50% of host; raise in %UserProfile%\.wslconfig
        └── serving container mem_limit — DOCIE_SERVING_MEM_LIMIT, default 8g
              └── what the tracker reports (source=cgroup)
```

A 16 GB machine on defaults shows 8 GB total, and that is correct: it is what
the serving node can actually use. The default is deliberately conservative. A
`mem_limit` larger than the VM never binds, so the tracker would publish an
authoritative-looking ceiling the node cannot deliver, and runtimes sized to it
are killed by the VM instead of refused by the fit check. Raise the VM first
(`[wsl2] memory=24GB`, then `wsl --shutdown`), then the limit.

On a Linux host only the container limit applies, and cgroup numbers are exact.

### Tuning

| Variable | Default | |
|---|---|---|
| `DOCIE_SERVING_RECONCILE_INTERVAL` | 10 | seconds between cycles |
| `DOCIE_SERVING_IDLE_TTL_SECONDS` | 900 | 0 disables idle unload |
| `DOCIE_SERVING_MIN_HOT_SECONDS` | 120 | anti-thrash floor |
| `DOCIE_SERVING_MAX_EVICTIONS_PER_CYCLE` | 2 | bounds churn |
| `DOCIE_SIZING_MARGIN_FRACTION` | 0.10 | safety margin on node total |
| `DOCIE_SERVING_MEM_LIMIT` | — | the cgroup limit; makes readings authoritative |
| `DOCIE_SERVING_PORT_RANGE_START` / `_END` | 8088 / 8188 | allocation range |

## Running a fleet

Several serving nodes publish into one catalog. Each runs exactly one
reconciler over its own `deployments.json`; nothing is shared between them but
Postgres and Inngest, both of which live with the central stack.

A separate manager service was considered and rejected: the placement table is
already the registry such a service would have had to invent.

### What a second node runs

`docker-compose.serving-node.yml` deploys one container and nothing else: the
serving role, its reconciler, and the runtimes its deploys launch. Postgres,
Inngest, Redis, the api, the web UI and the scaled `worker` all stay central
and are reached over the network, so a node mounts no shared storage — `data`,
`runs`, `annotations` and `artifact-store` are worker surfaces the serving role
never writes.

Start from `.env.serving-node.example`, which is a delta to append to a copy of
the central `.env`. Four values have no default and compose refuses to start
without them, because each one fails silently rather than loudly:

| Value | What goes wrong when it is guessed |
|---|---|
| `DOCIE_SERVING_ADVERTISE_HOST` | every deploy records `http://{this}:{port}/v1`, and the central api resolves that string in **its own** network. A compose service name resolves there to the central container, so a request for this node's model is answered by whatever holds the same port — HTTP 200, wrong model. |
| `DOCIE_SERVING_PUBLISH_IP` | a deployed runtime carries no authentication of its own; the api gateway in front of it holds the key. Published on a public interface, the window is free inference for anyone who can reach the host. |
| `DOCIE_DB_HOST` | `postgres` resolves only inside the central compose network. A node left at that default writes placements nowhere the api can read them. |
| `POSTGRES_PASSWORD` | the node connects to the central database; it does not run one. |

The port window (`DOCIE_SERVING_PORT_RANGE_START`/`_END`) is both what deploys
allocate from and what the compose file publishes, driven by the same two
variables so they cannot disagree — a runtime on an unpublished port would
record an endpoint nothing can reach. Nodes do **not** need disjoint windows:
the recorded endpoint carries the host, so each node may use the same ports on
its own address.

Inngest Connect is outbound, so a node needs no inbound reachability for job
delivery — only the two Inngest endpoints and Postgres. The only inbound
traffic is the central api calling the runtime ports.

Each node's capacity row is keyed by its own node id, generated on first
publish and persisted in `DOCIE_SERVING_HOME` beside `deployments.json`. That
volume is node-local by construction: two machines sharing it would fight over
one `deployments.json` and adopt each other's identity.

### What is fleet-aware, and what is not

With several nodes publishing, these behave as a fleet:

| Surface | Behaviour |
|---|---|
| `serving_node` | one row per node, each written by its own reconciler |
| `model_placement` | carries `node_id`, set by the node that launched the runtime and rewritten each cycle under the staleness guard, so a stale publish cannot claim a deployment away from the node that holds it |
| `store:` routing | load-balances across replicas on any node, and **demotes** replicas whose node stopped reporting |
| `deployment=` / bare name | falls back to the placement row, so a deployment on any node resolves |
| Deployment listing | unions the local records with catalog rows for deployments this process has no file for, each tagged `node_id`, `record_source` and `node_reporting` |
| Model store listing | annotates every model with `nodes: {present, missing, unreported}` — the blob store is node-local, so the catalog knowing a model says nothing about which nodes hold it |

Demotion rather than exclusion is deliberate. A stale snapshot proves the
*reconciler* stopped publishing, not that the runtime died — a crashed
reconciler leaves `llama-server` answering. So staleness only ever prefers a
better replica; if every candidate is on a node that stopped reporting, all are
kept and the request is attempted anyway.

A node that has *never* published is not stale, it is new: a deploy records its
placement immediately while the first publish is up to one cycle away.
`missing` and `unreported` are separate for the same reason — one means
re-seed, the other means wait.

### Aiming an operation at a node

A deploy can now be addressed. `POST /v1/studio/deploy` takes an optional
`node_id`, and a targeted deploy is recorded as **desired state owned by that
node** rather than fired as an event. Only that node's reconciler reads it.

It is validated before anything is written, because each mistake is otherwise
silent: an unknown node produces a row nothing will ever read, a node that
stopped reporting will not converge, and a node without the model's blob fails
the deploy for a reason the Studio does not show — the catalog knows the model
globally while the blob store is node-local.

```{note}
A targeted deploy has no realtime progress stream: the node converges on its
own schedule, so the response carries `channel: null`. A caller must treat that
as "no progress to follow" rather than waiting on it. Untargeted deploys still
return a channel.
```

Omitting `node_id` keeps the previous behaviour exactly — an event on the
shared `docie-serving` app, delivered to whichever node Inngest picks. That is
correct for a single node, which is why targeting is opt-in.

### Letting the platform choose

`auto_place: true` ranks the nodes and picks, instead of naming one. Naming a
node is fine when there are two; by the time there are six, "which one has room
for this" is arithmetic the sizing engine already does per node.

The ranking is four hard gates and one preference. The gates are the same
conditions the targeted path refuses on — never reported, stopped reporting,
drained, or missing the model's blob — repeated here because an auto-placed
deploy has no operator reading a `409`. The one that bites is a node with a row
and **no measurement**: its capacity fields are zeroes, which price as a
machine with nothing on it, so a ranking that trusted the numbers would prefer
a node that has never started over one serving perfectly.

Among the nodes that pass, the emptiest wins first and headroom breaks ties.
The fit itself is not recomputed: `sizing.compute_sizing` is called once per
node and `fits_now` is read off it, because a second opinion about what fits
would be a second set of bugs.

**Replicas go to distinct nodes.** With `replicas: N` and `auto_place`, the
deploy is placed on N different machines — two replicas on one node share its
memory, its cgroup limit and its failure, so that is not what asking for two
replicas means. A partial spread is **refused** rather than delivered: "I asked
for 3 and got 2" is a capacity decision only the operator can make. The refusal
names which node failed which gate, since that is what says whether to seed a
blob, resume a drained node or add a machine.

A node whose fit cannot be priced at all stays eligible but ranks below every
node whose fit is known — unknown is neither full nor roomy, and preferring it
would be preferring ignorance.

**Seeding still cannot be aimed.** `serving/seed.requested` is an ordinary
event on that same shared app, and the blob store is node-local, so a model
must be seeded on each node that will serve it:

```
docie-serving store seed-ollama <ref> --name <name> --family <family>
docie-serving store list
```

`/v1/serving/store` reports where each blob actually is, so the result is
checkable rather than guessed. `missing` means re-seed that node;
`unreported` means it has not published its store yet.

The CLI deploy path remains available and is unchanged:

```
docie-serving up <name> --deployment-name <name>-<node>
```

`--deployment-name` matters there: the record name defaults to the store name,
which is not unique across nodes, and `model_placement.name` is the primary
key — two nodes using the same name means the second publish overwrites the
first and one replica goes unroutable while still holding its RAM. The
targeted and scale paths mint names to avoid this; the direct path keeps the
store name so single-node use is unchanged, and offers the override.

**Logs are still read from the local serving home**, so diagnosing a remote
node means reaching that node.

**Unload is the one lifecycle action that works remotely.** A stop is the
single intent `plan_convergence` already derives correctly for a deployment
present on its node, so an unload aimed at a deployment another node owns is
recorded as desired state for that owner — and the response carries no channel,
because that node converges on its own cycle. The row's existing spec is
preserved: a stop does not read it, but a later start does, so replacing it
would quietly drop the context length and port an operator chose.

Load, delete, pin and repair still answer `409`. Each needs something desired
state cannot currently express — a plan state for "present but not running", an
explicitly declared removal, or configuration the desired spec does not
carry — which is a design decision rather than an edit (#590).

### Taking a node out

`POST /v1/serving/nodes/{id}/drain` removes a node from the rotation, and
`/resume` puts it back. The Fleet page offers both per row.

A drain has **two effects with different timings**, and conflating them is
what makes a naive implementation drop requests:

| | When | Needs the node? |
|---|---|---|
| Stops admitting | Immediately | No — it is a central routing decision |
| Stops the runtimes | After a grace period | Yes — its reconciler does it |

The grace period is the longest `timeout_seconds` among the node's own
deployments, clamped to `[30s, 600s]`. That is exactly how long a request
admitted an instant before the drain could still legitimately be running, so
unloading sooner would kill work a client is waiting on. The two phases are
reported separately as `draining` and `drained`, because an operator waiting
on a machine needs to know which one they are in.

**A drain excludes, where staleness only demotes.** The two rules look alike
and are deliberately opposite:

- Staleness is *inferred*. A node that stopped publishing may still be
  serving perfectly, so routing prefers a live replica but will still use the
  stale one rather than refuse the only one.
- A drain is *declared*. An operator said to take the machine out, so routing
  excludes it even when it holds the last replica — "unless it is the only
  one" is not something they asked for, and a drain that kept serving would
  never let the node report empty.

Because of that, draining the last node serving a model takes the model
offline. The API refuses with a `409` naming the affected models, and the
refusal is overridable (`acknowledge_outage: true`) — taking a machine out
even at that cost is a legitimate thing to want, and only the operator knows
whether it is.

The flag rides the node's **desired generation**, which is the reason it lives
beside desired state rather than on its own. A node reading `drained`
unconditionally would unload its whole working set the moment it was pointed
at the wrong database — precisely the failure the convergence gate below
exists to prevent. Riding the generation means only a node that has proven the
catalog is describing *it* ever acts on a drain. The unload therefore also
needs `DOCIE_SERVING_DESIRED_APPLY` (see *Applying is opt-in*); without it the
node leaves the rotation but keeps its memory, and says so in its log.

During a catalog outage the router falls back to last-known-good placements,
which cannot carry drain state — so a drained node keeps receiving traffic
until the catalog is readable again.

## Desired state

Postgres holds the authoritative global desired state; each node keeps
`deployments.json` as its local snapshot. The split is deliberate — the local
file is what lets a node keep repairing and serving with no database, which is
the requirement that the control plane must never become a dependency for an
already-running model.

A desired row is keyed `(node_id, name)`, so it belongs to exactly one node and
moving a deployment between nodes is a delete here and an insert there rather
than a mutation. **A removal is a DELETE, not a state**: "should not be
running" and "no longer desired at all" call for different actions, so they are
represented differently.

### Two generation counters

They answer different questions and are bumped in the same transaction, so they
cannot disagree.

| | Where | For |
|---|---|---|
| per row | `serving_desired_deployment.generation` | concurrency between *writers* — every write carries the generation it read, and a mismatch loses, so two API calls racing on one deployment cannot interleave |
| per node | `serving_node.desired_generation` | staleness for the *reader* — it lets a node decide in one read whether the catalog is provably current for it |

### Why "the catalog is authoritative" is not enough

Three situations look identical to a node: the catalog is readable and lists
fewer deployments than the node is running.

1. An operator genuinely removed one while this node was partitioned.
2. The node is pointed at the wrong database, or a fresh one — a
   `DOCIE_DB_HOST` typo reaching another stack's Postgres.
3. The rows were lost to a restore or a migration.

Only the first should converge. A node that treats *readable* as
*authoritative* tears down every model it is serving in the other two, with no
error anywhere, because it believes it is converging correctly.

So the test is not readability but whether the catalog is provably **current
for this node** — a generation strictly newer than the one the node last
applied. A wrong database cannot produce that, because it has never seen this
node's id. Everything else holds: keep serving, change nothing, say why.

```{important}
`HOLD` is a third outcome, not an error and not a no-op. Teardown is
unreachable from it, and no teardown is ever derived from a row's *absence* —
only from a complete snapshot at a newer generation.
```

A node that has never published is **new**, not gone; a node with no desired
state yet is **ordinary**, not suspicious. Both are distinguished from the
dangerous cases by the node's own record of what it last applied, persisted in
`DOCIE_SERVING_HOME` beside `deployments.json` so it shares that file's
lifecycle.

### What a node removes

Only deployments it previously created **from desired state**. A node also
holds deployments the catalog was never told about — `docie-serving up` creates
one directly — so "local but not desired" would tear those down on the first
pass. The allow-list is remembered rather than inferred.

### Applying is opt-in

```
DOCIE_SERVING_DESIRED_APPLY=1
```

Unset, the reconciler still reads its desired state, decides, and **logs the
plan it would apply** without carrying it out. This is the only path in the
serving role that can stop a model nobody asked it to stop, so desired rows can
be written and their consequences read off the logs before anything acts.

Inert in a second sense too: with nothing written, every node sees no
generation against an applied zero, which reads as "nothing written for this
node yet". An existing single-node stack is unaffected either way.

Starts are converged before stops and removals, so a generation that replaces
one deployment with another never leaves the model unserved in between. A
generation is recorded as applied only when the whole plan succeeded —
otherwise the next cycle re-plans and retries, because a reconciler that gives
up after one attempt is not reconciling.

## Where this is going

What the fleet still cannot do, in the order it bites:

- **Seeding and logs are not node-addressable.** A model must be seeded on each
  node that will serve it, and a lifecycle action on another node's deployment
  answers `409` naming the owner rather than acting. Both want the same thing:
  an operation addressed to a node, which desired state now provides for
  deploys and not yet for anything else.
- **A long start blocks its own cycle.** `await_ready` is synchronous, so a
  multi-gigabyte load holds the reconcile loop for its duration. The lease and
  the node snapshot are refreshed between actions to keep that from looking
  like a dead node, but a single long load still delays everything else that
  cycle. Performing starts off-cycle is the fix.
- **No placement policy.** A deploy is aimed at a node by name; nothing chooses
  one for you, so `replicas > 1` with an explicit node is refused rather than
  guessed.
- **Round-robin is process-local.** Spread across nodes is approximately even
  rather than coordinated, and uneven counts across a scaled `worker` are
  expected rather than a fault.
