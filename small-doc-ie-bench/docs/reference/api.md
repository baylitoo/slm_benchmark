# HTTP API

Everything the platform does is reachable over HTTP on the `api` service
(port 8080 in the default compose file). This page describes the surface by
group, the conventions shared across it, and the contract for a backend service
that depends on extraction.

The exhaustive, always-current list is the API's own OpenAPI document, at
`/openapi.json` and rendered at `/docs`. This page does not repeat it.

## Authentication

Every route except the health probes and `POST /v1/auth/session` takes a
credential in the `X-API-Key` header. Which credential to use, and what each
one may do, is in {doc}`../architecture/auth`.

For an application, the answer is a **Studio-issued key**, one per application.
Do not give an application an `API_KEYS` entry: that is the operator tier and
can manage every other key.

A request for a resource that belongs to another tenant answers `404`, never
`403`, so the existence of another tenant's run is not disclosed either.

## The surface

| Group | Prefix | For |
|---|---|---|
| Probes | `/healthz`, `/readyz`, `/metrics` | orchestrators and Prometheus; unauthenticated |
| Synchronous extraction | `/v1/extract/text`, `/file`, `/stream` | a caller that waits for the answer |
| Schemas | `/v1/schemas` | the built-in schemas |
| OpenAI-compatible | `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/models` | any OpenAI client pointed at the platform |
| Agents | `/v1/agents` | named agents, each with its own `/{name}/chat/completions` |
| MCP | `/v1/mcp` | registered tool servers |
| Studio | `/v1/studio` | durable jobs: extraction, batches, benchmarks, comparisons, schemas, routing policies, seeding, keys and usage |
| Serving | `/v1/serving` | deployments, nodes, store, sizing, resources |
| Review | `/v1/reviews` | the human review queue |
| Orchestrator | `/v1/experiments`, `/v1/workers` | distributed benchmark runs |
| Session | `/v1/auth/session` | exchanges a company refresh token for a Studio session; unauthenticated by design |

The Studio group is the one an application integrates against for documents.
The synchronous group predates it and remains for callers that cannot poll.

## Conventions

**Explicit selectors fail rather than substitute.** Naming a deployment, a store
model or a routing policy either uses that one or returns an error. A request is
never answered by a different model than the one it asked for. Resolution order
is in {doc}`../architecture/serving`.

**Validation happens at the edge.** A malformed schema, a missing routing
policy, or a conflicting pair of selectors is a `4xx` on the request, not a
failed job discovered later by polling.

**Missing infrastructure is a `503`.** No Inngest, no database, no session
secret: the route says so instead of quietly degrading.

**Asking for a cold model is a `202`.** A `store:` reference with nothing live
triggers a load and answers `202` with an estimate. A name absent from the
catalog is a `404`.

## Extracting a document from another service

This is the contract for a backend that depends on extraction at runtime.

### Trigger

`POST /v1/studio/extract` starts a durable job and returns at once:

```json
{"event_ids": ["01J..."], "channel": "extract:3f9a...", "topics": ["status", "progress", "result", "error"]}
```

| Field | |
|---|---|
| `content_b64` + `filename`, or `text` | the document; one of the two is required |
| `dynamic_schema_name` | a schema saved under `/v1/studio/schemas/dynamic` |
| `dynamic_schema` | a full schema spec, used once and not saved; wins over the name |
| `schema_name` | a built-in schema; default `invoice` |
| `deployment` / `model_profile` | which model; `deployment` wins |
| `routing_policy` | a saved policy instead of one model; exclusive with the two above |
| `ocr_backend`, `language` | ingestion hints |

### Wait

Poll `GET /v1/studio/runs/{event_id}` until the first entry's `status` is
terminal, then read `output`, which has the same shape as every other
extraction result.

A finished extraction is answered from a durable row the job writes itself, so
`status` is exactly `Completed` or `Failed`. While it is still running the
answer is proxied from Inngest, whose statuses include `Cancelled`. The
self-hosted Inngest server does not reliably carry `output` on its own run
endpoint, which is why the durable row exists.

```python
import base64
import time

import requests

BASE = "https://docie.internal/v1/studio"
HEADERS = {"X-API-Key": "REPLACE_ME"}


def extract(content: bytes, filename: str, schema_name: str) -> dict:
    payload = {
        "content_b64": base64.b64encode(content).decode(),
        "filename": filename,
        "dynamic_schema_name": schema_name,
    }
    trigger = requests.post(f"{BASE}/extract", json=payload, headers=HEADERS, timeout=30)
    trigger.raise_for_status()
    event_id = trigger.json()["event_ids"][0]

    deadline = time.monotonic() + 120  # size to your largest document
    while time.monotonic() < deadline:
        response = requests.get(f"{BASE}/runs/{event_id}", headers=HEADERS, timeout=15)
        response.raise_for_status()
        runs = response.json()
        if runs and runs[0].get("status") in {"Completed", "Failed", "Cancelled"}:
            if runs[0]["status"] != "Completed":
                raise RuntimeError(f"extraction {runs[0]['status']}: {runs[0]}")
            return runs[0]["output"]
        time.sleep(2)
    raise TimeoutError(f"extraction {event_id} did not finish in time")
```

Push delivery exists — `GET /v1/studio/realtime-token` mints a subscription
token for the returned `channel` — but needs an Inngest realtime client.
Polling needs nothing beyond the key.

### Schemas

A dynamic schema is a `document_type` and a list of fields typed `string`,
`date`, `number`, `money`, `object` or `list`. `object` nests a fixed group;
`list` nests a repeated one, and nesting recurses. Each field list holds at
most 40 entries, counted per level rather than across the tree.

```json
{
  "document_type": "resume",
  "fields": [
    {"name": "name", "type": "string"},
    {"name": "experience", "type": "list", "fields": [
      {"name": "company", "type": "string"},
      {"name": "start_date", "type": "date"}
    ]},
    {"name": "skills", "type": "list", "fields": [
      {"name": "skill", "type": "string"}
    ]}
  ]
}
```

Save it with `POST /v1/studio/schemas/dynamic`, then pass its `document_type`
as `dynamic_schema_name`.

There is no list of bare strings: `list` always nests an object, which is why
`skills` above is a list of one-field objects. Every leaf carries
`{value, confidence, evidence_ids}`, and a bare scalar would have nowhere to put
its evidence.

### Sharing one instance between applications

Several cooperating applications can share one instance, each with its own key.
Two limits apply:

- **No compute isolation between tenants.** The gateway's concurrency limit and
  circuit breaker are per deployment, shared by everyone calling it. A heavy
  batch from one application queues another's requests, and past the queue
  limit they get `429`. The lever is a dedicated deployment for the heavy
  caller, not a second instance.
- **Quotas are one global setting,** applied to every tenant and enforced per
  API replica. See {doc}`../architecture/auth`.

## Synchronous extraction

`POST /v1/extract/text` and `/v1/extract/file` run the extraction inside the
request and return:

```json
{
  "request_id": "...",
  "schema_name": "invoice",
  "model_profile": "...",
  "document_hash": "sha256:...",
  "result": {},
  "validation": {},
  "usage": {},
  "latency_ms": 1234
}
```

`routing` and `dynamic_schema` are added when a routing policy or dynamic
schema was used. `/v1/extract/stream` is the same extraction as server-sent events.

## Review queue

With persistence enabled, invalid, low-confidence and weakly grounded
extractions are admitted to the review queue automatically; `POST /v1/reviews`
adds tasks directly.

Reviewers claim a task under an expiring lease and send the current
`expected_version` with every change. A stale version, an expired claim, or a
write from another reviewer is a `409`, never an overwrite. Corrections address
fields by dotted path (`invoice_number.value`), and every revision is kept.

`POST /v1/reviews/exports` writes approved corrections once under
`ANNOTATION_EXPORT_DIR`. Only the `train` split is accepted, so reviewed labels
cannot leak into evaluation data.
