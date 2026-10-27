# Fleet runbook: adding, checking and removing a serving node

Operating a multi-node fleet (#538). For *how* the pieces work, read
{doc}`../architecture/serving`; this page is the sequence of things you do.

The single most useful thing on this page is
[`docie-serving doctor`](#check-the-node-before-it-takes-traffic). Most of what
can be wrong with a serving node **fails as a wrong answer rather than as an
error** — the node starts, publishes capacity, accepts deploys, and then serves
requests for its models from a different machine. Run the doctor before a node
takes traffic and after every `.env` change.

## Before the first node

The central stack publishes Postgres and Inngest on loopback, so no other
machine can reach them. In the **central** `.env`:

```
POSTGRES_BIND_HOST=<central host's private IP>
INNGEST_BIND_HOST=<central host's private IP>
```

That one address, never `0.0.0.0`: Postgres answers to real credentials, and
the Inngest dashboard has no authentication at all. Put the fleet on a private
network (WireGuard is what this was designed against) and use those addresses
throughout.

## Add a node

1. **Copy the central `.env`** to the new machine, so every application setting
   matches, then append `.env.serving-node.example` and fill in what it marks
   REQUIRED. Four values have no default and the stack refuses to start
   without them, because each one silently misroutes or over-exposes if
   guessed: `DOCIE_DB_HOST`, `DOCIE_SERVING_ADVERTISE_HOST`,
   `DOCIE_SERVING_PUBLISH_IP`, `POSTGRES_PASSWORD`.

2. **Check it before starting:**

   ```
   docker compose -f docker-compose.serving-node.yml run --rm serving \
     docie-serving doctor --probe-ports
   ```

   `--probe-ports` is worth passing here and only here: binding the
   deploy window means something in the serving container's own network
   namespace, and nothing on the host.

3. **Start it:**

   ```
   docker compose -f docker-compose.serving-node.yml up -d
   ```

4. **Confirm it joined.** The Studio's **Fleet** page lists it within one
   reconcile interval (10s by default), or:

   ```
   curl -s -H "Authorization: Bearer $KEY" https://<central>/v1/serving/nodes
   ```

   A node that appears with `never_reported: true` has a row but has never
   published a measurement — desired state was staged for it and it has not
   started, or it cannot reach Postgres.

5. **Seed the models it should serve.** The catalog is global; the blob store
   is **node-local**. A model listed in the Studio says nothing about which
   nodes hold its weights.

   Seeding is **not node-addressable**: the Studio's "Add model" path is an
   Inngest event on the shared serving app, so it lands on whichever node
   Inngest picks. To put a model on a *chosen* node, seed it on that node:

   ```
   # from a model Ollama already pulled on this machine
   docker compose -f docker-compose.serving-node.yml exec serving \
     docie-serving store seed-ollama lfm2.5:350m --name lfm2.5-350m --family openai_chat

   # or from a GGUF already on this node's disk
   docker compose -f docker-compose.serving-node.yml exec serving \
     docie-serving store add-gguf /models/model.gguf --name lfm2.5-350m --family openai_chat
   ```

   There is no node-local "pull from Hugging Face" command. For a model that
   is not already on the machine, download the GGUF onto the node by whatever
   means (`huggingface-cli download`, `curl`) and then `store add-gguf` it.

   Use the **same `--name` on every node**. That name is what the catalog and
   every deploy refer to, so the same weights registered under two names are
   two different models as far as routing is concerned.

   `docie-serving doctor` lists which catalogued models this node is missing.

### What a node actually needs

Three things, and nothing else:

- the central **Postgres**, for the catalog it publishes into and reads
  desired state from;
- the central **Inngest**, for lifecycle events (outbound only — a node
  needs no inbound reachability for job delivery);
- its own **`DOCIE_SERVING_HOME`** volume, which must never be shared
  between machines.

Those three are durable. The **model weights are not** — they can be seeded
or pulled again.

By default the weights live at `$DOCIE_SERVING_HOME/models`, a **child** of
the state directory. Removing that directory does *not* remove `node_id`
beside it. What the default costs is **placement**: the weights share one
volume with the durable state, so they cannot be put on a different disk,
their growth consumes that volume, and destroying or resizing it takes both.

### Moving the model cache to its own volume

Optional. Worth doing when the weights outgrow the state volume, or when you
want to reclaim that space without touching anything durable.

**The weights do not move by themselves.** Setting the variable without
copying them makes the node report an empty blob store and re-seed
everything, because the old files are still sitting in the state volume
where nothing is looking for them.

1. Stop the node:

   ```
   docker compose -f docker-compose.serving-node.yml stop serving
   ```

2. Uncomment the `model-cache` mount in `docker-compose.serving-node.yml`
   and set, in the node `.env`:

   ```
   DOCIE_MODEL_STORE_DIR=/app/.cache/models
   ```

3. Copy the existing weights across, with both volumes mounted:

   ```
   docker compose -f docker-compose.serving-node.yml run --rm \
     --entrypoint sh serving -c \
     'mkdir -p /app/.cache/models && cp -a /app/.serving/models/. /app/.cache/models/'
   ```

4. Start it and confirm the move before deleting anything:

   ```
   docker compose -f docker-compose.serving-node.yml up -d serving
   docker compose -f docker-compose.serving-node.yml exec serving \
     docie-serving store list
   ```

   The same models should be listed as before. `docie-serving doctor` also
   reports what this node holds against the catalog.

5. Only then reclaim the old copy:

   ```
   docker compose -f docker-compose.serving-node.yml exec serving \
     rm -rf /app/.serving/models
   ```

A cache root that **is, or contains, `$DOCIE_SERVING_HOME`** is refused at
startup — a directory you intend to clear must not hold `node_id`,
`deployments.json` or the applied-state file.

Leaving the variable unset keeps the current layout exactly. Nothing has to
move to upgrade.

In particular it does **not** need a synchronised copy of `configs/`.
That file is the routing table, read by the central api and the worker
role; the serving role takes its launch arguments from the family contract
in code plus the store entry. The node compose mounts it only so
`docie-serving gateway`/`plan` work if you exec them in the container, and a
stale copy there cannot change what the node serves (#593).

## Check the node before it takes traffic

`docie-serving doctor` exits non-zero when a check FAILS, so a bringup script
can gate on it. Each check maps to a failure that is otherwise invisible:

| Check | What a bad result means |
|---|---|
| **worker role** | `DOCIE_WORKER_ROLE` without `serving`: the process registers no serving functions and will never deploy, reconcile or publish. |
| **serving home** | Not writable: the node cannot remember what it is running, or what it is called. |
| **node id** | Duplicated across nodes. `ModelPlacement.name` is a primary key, so two nodes sharing an id overwrite each other's rows — one node's replica goes unroutable while still holding its RAM. |
| **advertise host** | Resolves to loopback or to *another machine*. Deploys record `http://{this}:{port}/v1`, and the central API resolves that in **its** network. This is the failure that returns a 200 with the wrong model. |
| **model catalog** | Unreachable Postgres. The node keeps serving what is loaded, but it is invisible to the fleet and converges on nothing. |
| **inngest** | Unreachable: no lifecycle events arrive. Desired state still converges, since that is read from the catalog. |
| **desired-state apply** | Off: the node joins, reports healthy, and ignores every deploy aimed at it, every auto-placed deploy that picks it, and the unload half of a drain. Routing exclusion and capacity publishing work regardless. |
| **port window** | Inverted, or ports held by something else. A runtime on an unpublished port records an endpoint nothing can reach. |
| **blob store** | Lists the catalogued models this node lacks. Normal on a fleet; a deploy of one of them aimed here is refused until it is seeded. |
| **clocks** | The database clock against this node's. Liveness and the drain grace period compare timestamps written here against the **central API's** clock, so a large number is a reason to check NTP on both. |

`--json` gives the same verdicts as a list of objects.

## Deploy to a specific node

Deploys are addressable; seeding is not. In the Studio's Deploy form pick a
**Target node**, or let the platform rank the nodes by what actually fits.

A targeted deploy is recorded as **desired state owned by that node**, not fired
as an event, so the response carries no progress stream — the node converges on
its own cycle. If nothing happens, check `desired-state apply` in the doctor
output: that is the setting which makes the difference between a recorded
intent and a running model.

## Take a node out

To stop a node receiving traffic, use the **drain** control on the Fleet page
(or `POST /v1/serving/nodes/{id}/drain`), which excludes it from routing
immediately and lets it stop its runtimes once in-flight requests have
finished. See *Taking a node out* in {doc}`../architecture/serving` for the two
phases and why a drain excludes even the last replica.

If drain is not available on your build, stopping the container is the blunt
version:

```
docker compose -f docker-compose.serving-node.yml stop serving
```

Its placements stay in the catalog and are **demoted** rather than removed —
routing prefers replicas on live nodes but will still use a stale node's row
rather than refuse the only one, because a reconciler that stopped publishing
does not prove the runtime died. So stopping the only node serving a model
leaves that model's requests going to a machine that is off.

## Remove a node permanently

1. Move its models elsewhere first: deploy them on another node, and confirm
   from the Fleet page that each has a live placement that is not on this node.
2. Stop and remove the container.
3. Delete its deployments so no placement rows point at it. Rows owned by a
   node that no longer exists are the one case nothing cleans up
   automatically — they are indistinguishable from a node that is temporarily
   off, which is exactly why they are not reaped.
4. Its `serving_node` row stays as a tombstone showing when it last reported.

## Diagnose

| Symptom | Likely cause |
|---|---|
| Requests for a model return a 200 with the **wrong** model's output | `DOCIE_SERVING_ADVERTISE_HOST` resolves to another machine. Run the doctor. |
| A node appears in the Fleet page as `staged` / `never reported` | It has a row but has never published: not started, or cannot reach Postgres. |
| A targeted or auto-placed deploy records and nothing starts | `DOCIE_SERVING_DESIRED_APPLY` is off on that node. |
| A model deploys fine centrally but fails on one node | Its blob is not seeded there. The doctor lists what is missing. |
| A node's replica holds RAM but takes no traffic | Duplicate node id: another node's publish overwrote its placement row. |
| Every deploy on a node fails to allocate a port | Something outside the container holds the published window. |
| A node flaps between reporting and not | Clock skew between it and the central API, or a reconcile interval longer than the staleness window. |

Logs are node-local. `docker compose -f docker-compose.serving-node.yml logs -f
serving` on the machine itself; there is no central log aggregation, and a
lifecycle action aimed at another node's deployment is refused rather than
proxied.
