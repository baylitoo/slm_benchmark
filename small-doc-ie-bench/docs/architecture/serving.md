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
| Works from another machine | no | **yes** |

The `store:` path is already network-transparent: a placement row carries a
full endpoint URL, and the resolver picks among live replicas. This is why the
platform is closer to multi-node than it looks — the data plane already
distributes; it is the control plane that is single-node.

Replica choice is `round_robin_choice` by default, or `session_affinity_choice`
when a `session_id` is supplied, which pins a conversation to one replica to
keep llama.cpp's prefix KV cache warm.

```{warning}
Both choosers keep **process-local** state. The round-robin counter is a
per-process dict and resets on restart; the affinity map is never evicted. With
several API replicas the distribution is uneven rather than wrong — but it is
not coordinated, and that is a known limitation, not an oversight.
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

## Singletons, and why

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

The node resource snapshot is likewise exactly one row, upserted, written only
by the serving reconciler.

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

## EmbeddingGemma 2

Both artifact formats serve **text** on `/v1/embeddings`, not chat or extraction:

| Hub repository | Family | Runtime |
|---|---|---|
| `google/embeddinggemma-2` | `embeddinggemma2` | `docie dense-embedding` / SentenceTransformers |
| `ggml-org/embeddinggemma-2-GGUF` | `embedding` | llama-server with `--embedding --pooling mean` |

Select the repository in Hub onboarding, seed its snapshot or GGUF, then deploy
the detected family. For GGUF text embeddings no `mmproj` is required. The
serving image pins llama.cpp to the v0.7.0 / b11669 commit
`620d2225cc436ab0b001e3200e590e3b286ef9ab`; older binaries may not recognize
`gemma-embedding2`. Rebuild the serving image to pick up the pin.

The native checkpoint requires SentenceTransformers >=6.1 and Transformers
>=5.19. These conflict with the analyzer stack's Transformers constraint.
The serving image therefore installs the `embeddings` extra into
`/opt/docie-embeddings`, separate from `encoders`, and selects its CLI via
`DOCIE_DENSE_EMBEDDING_EXECUTABLE`. Do not install both extras in one environment.
The extra includes `torchvision`: the checkpoint's shared processor requires it
even when only the text encoder is loaded.
For standalone native serving in a separate virtual environment:

```sh
pip install 'small-doc-ie-bench[embeddings]'
docie dense-embedding --model google/embeddinggemma-2 --port 8093
```

Native inference disables the unused vision/audio encoders and uses bfloat16
on supported CUDA hardware, otherwise float32. It never uses float16.
The API accepts a string or a batch of strings; image/audio/video objects and
token-ID inputs are not supported by this text endpoint.

### Retrieval prefixes and output dimensions

For either runtime, format query and document inputs explicitly:

```json
{
  "model": "store:your-embedding-deployment",
  "input": [
    "task: search result | query: What causes the northern lights?",
    "title: none | text: The northern lights are caused by charged particles from the sun."
  ]
}
```

Use `title: {title} | text: {content}` when a document has a title. The native
runtime also accepts `prompt_name: "SearchQuery"` or `"Document"` for an entire
batch of unformatted strings. Omit `prompt_name` for already formatted inputs;
no task prefix is chosen implicitly. For GGUF use the explicit text prefixes,
not the native runtime's `prompt_name` extension.

Native output defaults to 768 dimensions. `dimensions` accepts 128, 256, 512,
or 768 and the result is L2-normalized **after** truncation. Use the same
dimension for queries and indexed documents. `encoding_format` accepts
`"float"` or `"base64"` (little-endian float32). Native inputs exceeding the
8,192-token context, including prefixes and special tokens, are rejected rather
than silently truncated.

Sources: [Google model card](https://huggingface.co/google/embeddinggemma-2),
[GGUF conversion](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF).

## Where this is going

The control plane is single-node; the `store:` data plane is not. Extending to
several serving nodes is tracked in the distributed-serving issue, which also
records why a separate manager service was rejected — the placement table is
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
