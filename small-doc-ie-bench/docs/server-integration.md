# Integrating a server-to-server caller against DocIE

This is for a backend service that wants to call `POST /v1/studio/extract`
as a hard runtime dependency — not a human using the Studio UI or the CLI.
Written against the ADBI platform's four in-house apps (CV parser, contract
generator, CV dossier generator, document vault), which today each embed
their own duplicated OCR/parsing stack (Docling in one, tesseract.js/pdf-parse
in two others) and want to retire all of it in favor of one call to DocIE per
document type, selected via `dynamic_schema_name`. The contract below applies
to any similar caller.

## 1. Auth and tenant scoping

Every route (including `/v1/studio/extract`) sits behind `TenantDependency`
(`src/docie_bench/security.py`): an `X-API-Key` header is checked with
`hmac.compare_digest` against `DOCIE_API_KEYS`, a comma- or JSON-mapped
`key:tenant_id` list. Give **each of the four apps its own key and its own
`tenant_id`** — do not share one key across apps. That buys you, for free:

- Every extraction/deploy/benchmark event is tagged with the caller's
  `tenant_id` and recorded as that event's owner.
- `GET /v1/studio/runs/{event_id}` (see below) is tenant-scoped: a request
  for another tenant's run id is a 404, never a 403 — run *existence* is
  never leaked cross-tenant, not just its contents.
- Per-tenant rate limiting and concurrency quotas apply automatically
  (`DOCIE_RATE_LIMIT_REQUESTS`, `DOCIE_TENANT_MAX_CONCURRENT_REQUESTS`, plus
  a separate, larger `DOCIE_TENANT_READ_RATE_LIMIT_REQUESTS` budget for
  GET/polling so a chatty status-poller can't starve the app's own writes).

**Caveat to plan around:** these quotas are a single global value applied to
*every* authenticated tenant — there is no per-API-key override today. If one
app's traffic pattern (e.g. a batch contract run) is heavy enough to need a
different budget than another's (e.g. a low-latency CV upload), that is not
expressible yet. Fine for four cooperating internal apps under one team; would
need a config change (`api_keys` → a per-tenant limits map) before this scales
to less-cooperative or truly adversarial callers.

## 2. Triggering an extraction and getting the result back

`POST /v1/studio/extract` is asynchronous by design: it fires an Inngest
event and returns immediately.

```json
{"event_ids": ["01J..."], "channel": "extract:3f9a...", "topics": ["status","progress","result","error"]}
```

**Decision: no new synchronous endpoint is needed.** A non-Inngest-native
caller (a plain Flask/Node service with no Inngest client) can already
consume this end-to-end over plain HTTP, because a polling fallback already
exists and requires nothing beyond the API key each app already has:

```
GET /v1/studio/runs/{event_id}
```

For an extraction event (no durable DB row) this proxies Inngest's own
`GET /v1/events/{id}/runs` (`src/docie_bench/inngest/studio_api/runs.py`),
tenant-scoped by the ownership check above. Poll it until the run's status is
terminal; the function's return value — the full extraction result, same
shape as every other extraction response — is under `output` on the
completed run entry.

Reference client (plain `requests`, no Inngest SDK):

```python
import base64
import time

import requests

BASE = "https://docie.internal/v1/studio"
HEADERS = {"X-API-Key": "REPLACE_ME"}

def extract(content_bytes: bytes, filename: str, schema_name: str) -> dict:
    payload = {
        "content_b64": base64.b64encode(content_bytes).decode(),
        "filename": filename,
        "dynamic_schema_name": schema_name,  # e.g. "resume" or "contract"
    }
    trigger = requests.post(f"{BASE}/extract", json=payload, headers=HEADERS, timeout=30)
    trigger.raise_for_status()
    event_id = trigger.json()["event_ids"][0]

    deadline = time.monotonic() + 120  # size this to your largest document
    while time.monotonic() < deadline:
        status = requests.get(f"{BASE}/runs/{event_id}", headers=HEADERS, timeout=15)
        status.raise_for_status()
        runs = status.json()
        if runs and runs[0].get("status") in {"Completed", "Failed", "Cancelled"}:
            if runs[0]["status"] != "Completed":
                raise RuntimeError(f"extraction {runs[0]['status']}: {runs[0]}")
            return runs[0]["output"]
        time.sleep(2)
    raise TimeoutError(f"extraction {event_id} did not complete in time")
```

Equivalent curl sequence:

```bash
EVENT_ID=$(curl -sS -X POST "$BASE/extract" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"content_b64":"'"$(base64 -w0 sample.pdf)"'","filename":"sample.pdf","dynamic_schema_name":"resume"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["event_ids"][0])')

curl -sS "$BASE/runs/$EVENT_ID" -H "X-API-Key: $API_KEY"
# repeat until status is terminal, then read .[0].output
```

**Not independently verified this session:** the exact `status` string casing
returned by the Inngest proxy (`Completed`/`Failed`/... above is Inngest's
documented convention, read from code, not exercised against a live stack —
no docker/live-model calls were run in this session). Confirm the exact
values once against a running stack before hardening the poll-loop check.

If a future need for push-style delivery (no polling) emerges, `GET
/v1/studio/realtime-token` mints an Inngest realtime subscription JWT for the
`channel` from the trigger response — but that requires an Inngest realtime
client, which none of the four apps have today. Polling is the right choice
for this integration; realtime stays available as a later upgrade for a
caller that already has an Inngest SDK.

## 3. Authoring `resume` and `contract` schemas

`POST /schemas/dynamic` takes a `DynamicSchemaSpec`
(`src/docie_bench/schemas/dynamic.py`): a `document_type` name plus a flat
list of fields, each `string | date | number | money | object | list`. An
`object`/`list` field nests its own field list one level deep (list = list of
that nested shape).

Both target schemas were authored and round-tripped in-process (spec →
validated pydantic model → NuExtract template) as part of this audit — this
is not a paper design, it compiled and produced the templates below:

```python
resume = {
    "document_type": "resume",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "email", "type": "string"},
        {"name": "experience", "type": "list", "fields": [
            {"name": "company", "type": "string"},
            {"name": "title", "type": "string"},
            {"name": "start_date", "type": "date"},
            {"name": "end_date", "type": "date"},
        ]},
        {"name": "skills", "type": "list", "fields": [
            {"name": "skill", "type": "string"},
        ]},
    ],
}

contract = {
    "document_type": "contract",
    "fields": [
        {"name": "parties", "type": "list", "fields": [
            {"name": "party_name", "type": "string"},
            {"name": "role", "type": "string"},
        ]},
        {"name": "effective_date", "type": "date"},
        {"name": "termination_date", "type": "date"},
        {"name": "total_value", "type": "money"},
        {"name": "clauses", "type": "list", "fields": [
            {"name": "heading", "type": "string"},
            {"name": "text", "type": "string"},
        ]},
    ],
}
```

`POST` each to `/schemas/dynamic`, then pass `dynamic_schema_name: "resume"`
/ `"contract"` on `/v1/studio/extract` — no other DocIE-internals knowledge
required.

**Rough edge found while dogfooding, not fixed (a design choice, not a bug):**
there is no bare "list of plain strings" field type. `type: "list"` always
nests an object, even for something as simple as `skills: ["Python", "Go"]`
— hence `skills` above is a list of one-field objects
(`skills[i].skill.value`), not a list of bare strings. This is consistent
with the platform's evidence-grounding invariant: *every* leaf value, however
nested, carries `{value, confidence, evidence_ids}` (`schemas/common.py`), so
a truly bare scalar list would have nowhere to attach per-item evidence
anyway. Worth a deliberate design call (a genuine bare-scalar-list type,
traded against losing per-item evidence) if ADBI's consumers end up wanting
flatter JSON — that decision is intentionally left open here rather than
built unilaterally.

**Not run this session:** live extraction against a real sample CV/contract
through a deployed model (no docker/live-model calls in this session, per
standing project constraint). The spec/model/template round-trip above is
verified; the model's actual field-level accuracy on real resumes/contracts
is not.

## 4. Is one DocIE instance fit to serve all four apps concurrently?

**Yes, as designed today, for this specific case** — four cooperating
internal apps under one team, not adversarial multi-tenant SaaS. Real
mechanisms already exist and were verified in code: timing-safe API-key auth,
per-tenant rate limit + concurrency quotas on every route including
`/extract`, tenant-scoped run ownership with no cross-tenant existence
leakage, and (as of this audit's PR) a fixed circuit-breaker wedge and
exported deployment health metrics.

Two caveats to accept knowingly rather than discover in production:

- **No per-tenant compute isolation.** The model gateway's concurrency
  semaphore and circuit breaker are keyed per `(base_url, model)`, shared
  across every tenant hitting the same deployment — a heavy batch job from
  one app queues (and, past `queue_limit`, gets a 429 for) another app's
  request rather than being isolated from it, and a bad-schema flood from one
  app that returns invalid JSON can trip the shared breaker for all of them.
  Bounded, not silent (429s and metrics, not hangs) — but not fair-shared
  either.
- **Quotas are global, not per-tenant** (see §1) — a config gap, not an
  architecture one.

Given DocIE serves CPU-only (no GPU contention to reason about), the
practical scaling lever if one app's load grows disproportionately is a
dedicated deployment (a second llama-server instance of the same model) for
that app rather than a second DocIE instance — the auth/quota/observability
layer already generalizes to more deployments, it does not need to be
re-built per product.

**Recommendation:** ship on one shared instance with per-app API keys. Do not
split into per-product instances now — that reintroduces exactly the
maintenance duplication this migration exists to remove, and the isolation
gaps above are config/allocation follow-ups, not fundamental blockers.
