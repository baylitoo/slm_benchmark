# The Studio

The Studio is the web console at `http://localhost:3000`. Everything an
operator does day to day happens there, and each section is a real URL, so a
link to `/deploy/sizing` opens exactly that view.

This page walks the sidebar in order. Bringing the stack up is in
{doc}`getting-started`.

## Playground

Run one request against a live deployment and watch it complete.

- **Extract** — paste text or upload a PDF or image, pick a schema and a
  deployment, run. If the runtime cannot compile the strict grammar, the
  extraction negotiates down a ladder of output styles rather than returning
  garbage; the run's status shows which one was used.
- **Vision** — a PDF is rasterised to page images and sent to a vision model,
  with no OCR step.
- **Embeddings** — two texts in, cosine similarity out.
- **Chat** — plain conversation with any live deployment.

## Serving

**Catalog** browses the Hugging Face Hub, segmented by serving family and
support tier. **Models → Add model** onboards from a Hub repo, a curated
collection, an encoder checkpoint, or a local Ollama pull. Before anything
downloads, an inspection reads the repo's metadata and reports the detected
architecture, the family it maps to, whether a vision projector is present,
the recommended quant, and whether it fits this node. Unsupported and "needs a
family" are separate answers.

Choosing the family is the step that matters: its contract decides the chat
template, generation defaults and launch arguments every deployment of the
model inherits. Download progress streams live.

**Deployments** is observed state, not a record of what was clicked. Each row
shows phase, port, resident memory and the last probe, refreshed by the
reconciler roughly every ten seconds. From a row:

| Action | |
|---|---|
| **Logs** | the runtime's own output, including load and per-request timings |
| **Scale** | more instances behind one model id, load-balanced |
| **Unload** / **Load** | an unloaded deployment keeps its record and port, reads `evicted`, and reloads on the next request |
| **Pin** | never evicted, never idle-unloaded |
| **Repair** | redeploy on a fresh port with the restart budget reset |

**Sizing** shows node memory with the safety margin held back as an explicit
number, how many *more* instances of each model fit right now, and a what-if
planner for a hypothetical mix. A model whose size cannot be determined says
so instead of guessing. The figures use the same arithmetic as the admission
check, so the page cannot say something fits that a deploy would then refuse.

**Downloads** lists seeding jobs. **MCP Tools** registers the tool servers
agents may call. **Fleet** lists serving nodes and lets you drain or resume
one; see {doc}`fleet-runbook`.

## Agents

An agent is a named, reusable configuration over a deployed model, exposed as
its own OpenAI-compatible endpoint at `/v1/agents/<name>/chat/completions`.
Anything that speaks the OpenAI API can use one with no adapter.

**Templates** are the starting points:

- **OCR** — plain OCR, OCR then a model for structured output, or vision
  straight to JSON.
- **Security proxy** — screens prompts and responses for PII with a local
  encoder model before anything reaches the backing model. *Block* refuses a
  request containing PII; *Placeholder* masks it before forwarding. If the
  guard model is unreachable the proxy refuses the request rather than
  forwarding it, unless the agent is configured to fall back to regex.

**My Agents** lists each agent with its `base_url` and a **Try** panel that
runs a request and shows what the guard found. **Create** builds one from
scratch.

### Schemas

**Agents → Schemas** manages the schemas extraction runs against. Built-in
schemas are read-only; saved ones can be edited, duplicated and deleted.

A list field defines the shape of **one item**, not how many to extract.
Define `company` and `title` once and the model returns as many experiences as
the document has. Never add numbered fields such as `experience_1`. Use
**Object** for a single nested group and **List of objects** for a repeated
one; either can contain further objects and lists. **Use resume starter**
gives a complete worked example.

An edit keeps the name and applies to every later extraction; past results are
not rewritten, and there is no versioning, so duplicate a schema before
experimenting with one that agents depend on. Saving needs the Studio
database.

## Benchmark

**Run** benchmarks a registered dataset against a model and schema chosen from
real lists. **Results** shows past runs with metrics and their artifacts —
`report.html`, `predictions.jsonl`, `metrics.json`. **Batch** extracts a set of
documents and exports the results.

Artifacts are written by whichever worker ran the job and served back by id,
so any API replica can return them. Runs older than
`STUDIO_RUN_RETENTION_DAYS` (30) or beyond the newest
`STUDIO_RUN_RETENTION_MAX` (500) are pruned nightly, along with blobs no
remaining run references.

## Review, Settings, Observability

**Review → Queue** is the human review queue over `/v1/reviews`: reviewers
claim a task under a lease, and a stale or conflicting write is refused rather
than overwriting someone else's work. **Settings → API keys** issues and revokes keys for
applications. **Observability → Dashboards** embeds the project's Grafana
dashboard — agent outcomes, PII detections by type, latency, extraction and
gateway outcomes — with links to Grafana, the Inngest console and raw
Prometheus metrics.

## Sample material

`data/sample_dataset/files/` holds seven invoices and two identity cards in
French, English, German and Spanish, each with ground truth in
`manifest.jsonl`. `invoice-001.txt` is a good first extraction; the identity
cards are the strongest test of a PII guard. `lfm2.5-350m` handles an invoice
in about ten seconds on CPU.
