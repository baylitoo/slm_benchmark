# The job layer

Everything that takes longer than a request runs as a durable Inngest function:
extraction, benchmarks, model deployment, seeding, garbage collection. The API
container runs none of them — it sends events and reads results.

Understanding two things explains most of this layer: **why there are two
roles**, and **why there are two apps**.

## Roles

`DOCIE_WORKER_ROLE` decides which functions a process registers. Both the
`serving` and `worker` containers run the same entry point; the variable is the
only difference.

| Role | Registers | Scalable | Runs the reconciler |
|---|---|---|---|
| `serving` | model lifecycle: deploy, delete, load, unload, repair, reconfigure, pin, seed | **no** | yes |
| `worker` | extraction, benchmark, GC | yes | no |
| `all` | both, over one connection | dev only | only if explicitly enabled |

The split exists because **the side that spawns processes must be a single
writer**. The serving role holds the subprocess handles and owns
`deployments.json`; two of them would clobber it. Extraction and benchmarking
hold no processes — they call runtimes over HTTP — so they scale freely.

`all` is the local-development default. Note that the reconciler is *off* in
that mode unless explicitly enabled, which stops an accidentally scaled process
from running several reconcilers.

```{warning}
Never run more than one `serving` replica. The compose replica count is
advisory — `--scale serving=2` bypasses it — so there is a second guard: an
advisory lease file in the serving home. A second reconciler finding a live
lease refuses to start rather than competing. It retries every 30 seconds,
because a `compose up -d` recreate legitimately overlaps the old and new
containers for a moment.

The lease is not a distributed lock. It converts a misconfiguration into a
visible refusal, which is all it is meant to do.
```

## Two apps

There are two Inngest app ids, `docie-studio` and `docie-serving`, and they must
stay distinct.

A Connect handshake sends the app's function list as the **authoritative** set
for that app. If two fleets registered disjoint lists under one id, each
connection would overwrite the other's: functions would flap between registered
and archived, and an event could arrive while its handler was deregistered.

Hence the invariant stated in `apps_for_role`: **each app id always carries its
full function set, never a subset.** The `all` role registers both apps over one
connection rather than merging them.

Events route by **name** within the environment, not by app. That is why the API
can send a `serving/*` event through the studio client and still have it land on
the serving app.

## Connect, not serve

The transport is Connect — an outbound WebSocket from the worker to the Inngest
server. There is no HTTP `serve()` endpoint, because a worker never needs to be
publicly reachable.

| Variable | Purpose |
|---|---|
| `INNGEST_BASE_URL` | the API and event host |
| `INNGEST_CONNECT_GATEWAY_URL` | rewrites the gateway URL the server advertises |
| `INNGEST_DEV` | anything other than empty/`0`/`false`/`no` means dev mode |
| `INNGEST_EVENT_KEY`, `INNGEST_SIGNING_KEY` | required in production |

The gateway rewrite matters in Docker: the server advertises a loopback address
that is correct locally and unreachable from another container.

```{note}
Two traps worth knowing. Leaving `INNGEST_BASE_URL` unset in production mode
makes the SDK default to Inngest Cloud and 401. And the Inngest service must
**not** be given the project's `.env` — the binary reads its own `INNGEST_*`
variables and would bind the wrong port or start in the wrong mode.
```

## The functions

**Worker role**

| Function | Trigger |
|---|---|
| `doc-extract` | `doc/extract.requested` |
| `benchmark-run` | `benchmark/run.requested` |
| `studio-runs-gc` | cron, daily |

**Serving role** — deploy, delete, load, unload, repair, reconfigure, pin, and
two seeders (Ollama and Hugging Face). Deploy carries a global concurrency limit
of one, so two deploys cannot jointly overcommit the node's memory.

```{warning}
`batch-extract` is defined with trigger `doc/batch.requested` but is registered
by **no role**, so batch extraction events are silently dropped. Tracked as a
bug; do not assume batch works until it is fixed.
```

## One request end to end

A Studio extraction, from click to result:

```{mermaid}
sequenceDiagram
    participant B as Browser
    participant A as API
    participant I as Inngest
    participant W as Worker
    participant S as Serving
    B->>A: POST /v1/studio/extract
    A->>A: validate, mint channel, record owner
    A->>I: doc/extract.requested
    A-->>B: event_ids, channel, topics
    I->>W: invoke doc-extract
    W->>W: publish status
    opt model not live
        W->>S: serving/load.requested
        W->>W: wait in <=60s steps
    end
    W->>W: step "extract"
    W->>I: publish result
    W->>W: persist to extraction_results
    I-->>B: realtime (or browser polls /runs/{id})
```

The handler validates fast and synchronously — bad input is a `422` before any
event is sent — then mints a channel id, records which tenant owns the event
ids, and returns immediately.

The function publishes progress as it goes, waits for a cold model in bounded
steps, runs the extraction inside a memoized step, and writes the result to
Postgres before publishing it.

## How the browser gets the answer

Two paths, and the second is not a fallback bolted on afterwards — it is the
one that works when realtime does not.

**Realtime** is primary: the browser mints a subscription token and subscribes
to the channel's topics.

**Polling** takes over when realtime is unavailable or silent: the frontend
falls back after 15 seconds without a message, or immediately if the token mint
fails. It polls the run endpoint, which reads the **durable result row first**
and only proxies Inngest's own run API when there is nothing stored.

This is why publishing never raises. A realtime publish failure logs a warning
and the work continues, because the result is in Postgres either way. Realtime
is an accelerator over a durable store, not the delivery mechanism.

## Durability

Each kind of job has somewhere its outcome survives a restart:

| Job | Stored as |
|---|---|
| Extraction | `extraction_results`, upserted by event id |
| Benchmark | `studio_runs` plus artifacts in the shared volume |
| Batch | `batch_runs` / `batch_items` |
| Seed | `seed_runs`, plus a progress sidecar on the shared volume |
| Serving | `deployments.json` and the lease, on the serving volume |

**Idempotency** is handled per job kind. Benchmarks use platform idempotency on
a key derived from the run-defining fields. Batch and seed claim their row
before working, so a redelivery short-circuits. Loads converge through a
per-deployment lock, so a retried load cannot double-spawn. Seed downloads hold
a per-name lock so a retry resumes the partial file instead of corrupting it.

**Steps are the retry boundary.** Work inside `step.run` is memoized: a retry
after a later failure does not redo it.

When Inngest itself is unreachable, every trigger route answers `503` with an
explicit message rather than a raw exception, and reads keep working.

## Things that will catch you

- **No step may depend on the server's per-step time limit**, which is not
  pinned here. Cold-model waits are chunked into 60-second steps for exactly
  this reason; a model load can take far longer and converges through retries.
- **The reconciler must use the shared control plane**, not a fresh supervisor —
  only the shared instance holds the process handles.
- **The advertise host must not be a scaled service name.** A legacy value
  pointing at `worker` would advertise a round-robin endpoint, so compose pins
  it and a deploy-time guard rejects it.
- **App ids are read at import time.** Changing them needs a process restart.
- **A run can 404 from polling** if recording the event owner failed silently,
  since the run route is tenant-scoped.
