# Overview

DocIE extracts structured data from documents using small language models it
runs itself. It is three things stacked:

- a **serving control plane** that deploys, supervises and routes to model
  runtimes on its own hardware,
- an **extraction pipeline** that turns a document into validated,
  evidence-linked fields,
- a **Studio** to operate both, plus an OpenAI-compatible API for applications.

The reason it is a platform rather than a script is the first item. Running
small models yourself means someone has to decide what is loaded, notice when it
dies, fit it in memory, and send requests to the right place. Most of this
codebase is that problem.

## The containers

```{mermaid}
flowchart TB
    subgraph edge[Request path]
        API[api<br/>FastAPI]
        WEB[web<br/>Studio]
    end
    subgraph jobs[Durable jobs]
        ING[inngest<br/>server]
        WRK[worker<br/>scalable]
        SRV[serving<br/>single replica]
    end
    subgraph state[State]
        PG[(postgres)]
        RD[(redis)]
        VOL[serving volume<br/>+ artifacts]
    end
    WEB --> API
    API -->|events| ING
    ING --> WRK
    ING --> SRV
    WRK -->|HTTP| SRV
    SRV --> VOL
    API --> PG
    WRK --> PG
    SRV --> PG
    ING --> PG
    ING --> RD
```

`api` serves requests and sends events; it runs no jobs and spawns no models.
`worker` runs extraction and benchmarks and scales horizontally. `serving` is
**single-replica by design** — it holds the model subprocess handles and is the
only writer of the desired-state file.

That asymmetry is the single most important structural fact about the system.
Nearly every invariant elsewhere follows from it.

## How a request flows

Two shapes, depending on whether the caller waits.

**Synchronous** — `POST /v1/extract/text` and friends. The API resolves a model
profile, runs the extraction in-process, and returns the result. Used by
applications integrating against `/v1`.

**Durable** — the Studio's path. The API validates, emits an event, and returns
immediately with ids. A worker picks it up, loads the model if needed, extracts,
persists the result, and publishes progress. The browser watches over realtime
and falls back to polling a durable row.

Both converge on the same resolution step: a string naming a model becomes a
`ModelProfile` saying where to send the request and how to shape it. See
{doc}`serving`.

## Where state lives

| State | Where | Notes |
|---|---|---|
| Desired deployments | `deployments.json` on the serving volume | single writer, node-local |
| Observed placements | Postgres | published by the reconciler |
| Model weights | the serving volume | metadata in Postgres |
| Job outcomes | Postgres | extraction results, runs, batches, seeds |
| Artifacts | shared volume | served by id |
| Queues, job state | Inngest, backed by Postgres and Redis | |

Postgres is optional for the serving loop. Without it the reconciler still
repairs and still writes the desired-state file; it just skips publishing. That
property is deliberate — it keeps the routing contract free of a database
dependency — and it is why the activation flag lives in the file rather than the
table.

## The ideas worth absorbing early

**Measured, not predicted.** Throughput is probed, never inferred from a CPU
name. A deployment is healthy only after a real health check, never because its
process exists. Memory footprint starts as arithmetic and is replaced by
observation as soon as observation exists.

**Explicit selectors never silently degrade.** Asking for a specific deployment
or store model either gets it or fails. Nothing falls back to a default and
quietly answers with a different model.

**Failures are classified, not lumped.** A read timeout is not retried while a
connection timeout is. A truncated answer raises immediately while an
unparseable one walks a fallback ladder. Each distinction exists because the
undifferentiated version caused a real problem.

**Degradation is designed.** Realtime publishing never raises, because the
result is in Postgres anyway. Missing configuration produces a loud `503`
rather than a silent downgrade. The system is built to lose pieces without
lying about it.

## Reading order

{doc}`inngest` for how work actually runs, {doc}`serving` for the hardest part,
{doc}`extraction` for what comes out, {doc}`auth` for who may ask.

Design decisions, with the alternatives and why they lost, live in GitHub
issues rather than here.
