# DocIE Studio — step-by-step demo

Everything below is driven **through the Studio UI** at `http://localhost:3000`.
No terminal, no curl — if a step needs an affordance the UI doesn't have, that's
a gap to fix, not a step to fake.

Verified end-to-end on 2026-08-09 against a live compose stack. Steps marked
**[known issue]** were tested and do **not** work today; don't demo them.

---

## The story you are telling

> "You have documents to turn into structured data. You cannot send them to a
> cloud API. Which small model do you run, how many fit on this machine, and how
> do you know it works? This framework answers those questions on your own
> hardware — and refuses to invent any number it cannot measure."

Four beats: **onboard any model → use it immediately → know your capacity
honestly → expose it as a service.**

---

## Act 0 — Preflight (do this before the audience arrives, ~5 min)

1. Stack is up: `docker compose ps` shows `api`, `serving`, `worker`, `web`,
   `inngest`, `postgres`, `redis` healthy.
2. Open `http://localhost:3000` — the header status dot must be green.
3. **Serving → Deployments**: click **Load** on `lfm2.5-350m` and wait for the
   phase chip to read **hot** (~15 s). This is your demo model.
4. Leave one model *evicted* on purpose (e.g. `lfm2.5-vl-450m`) — you'll use it
   for the auto-reload beat in Act 5.
5. Note: idle deployments unload themselves after the idle TTL. If your demo
   model goes cold mid-talk, **don't apologise — that's Act 5**, show the
   reload instead.

---

## Act 1 — "What can this machine serve?" (2 min)

**Serving → Models.**

- 12 models in the store, and they're not all the same kind: chat
  (`lfm2.5-350m`, `lfm2.5-2.6b`, `gemma-2-2b-it`, `qwen2.5-1.5b`), vision
  (`lfm2.5-vl-1.6b`, `unlimited-ocr`), embeddings (`nomic-embed-text-v1.5`),
  and an analyzer (`guardrails-pii`).

**Say:** one control plane, four different runtime kinds — llama.cpp for GGUF
chat and vision, a transformers fallback for models with no GGUF at all, an
encoder runtime for analyzers, and an embeddings surface. The framework is
runtime-agnostic; the operator picks a model, not a serving stack.

---

## Act 2 — Onboard a model from anywhere (4 min)

**Serving → Models → Add model → "Search Hugging Face".**

1. Search e.g. `lfm2` or `gliner`.
2. Click a result. The panel shows a **support verdict** before anything is
   downloaded: detected architecture, the family it maps to, whether a vision
   projector is present.
3. Deploy a supported one (or show the verdict and stop — the verdict is the
   point).

**Say:** the platform tells you up front what it can serve, by reading the
repo's architecture metadata — instead of downloading several GB and failing at
load time. Unsupported and "needs a family" are distinct answers, and both are
honest. The other tabs onboard the same store from a repo id, a curated
collection, an encoder checkpoint, or a local Ollama model — source-agnostic.

---

## Act 3 — It is actually serving (3 min)

**Serving → Deployments.**

- Point at your hot model: phase **hot**, its port, memory in use, last probe.
- Click the **logs** action on the row — the real `llama-server` output,
  including the load and per-request timings.

**Say:** this table is *observed* state, not what someone clicked ten minutes
ago. A background reconciler probes every deployment on a cycle and publishes
what it finds; if a runtime dies, the row says so within seconds and requests
stop being routed to it. It also self-heals: when a runtime keeps dying because
another process holds its port, the control plane redeploys it on a fresh port
by itself. (That fired for real on this machine today — it's in the serving
log.)

---

## Act 4 — Use it, right now (5 min) ⭐ the core beat

**Playground → Extract.**

1. Paste an invoice — use `data/sample_dataset/files/invoice-001.txt` (French
   invoice, ~30 lines) or drop a PDF.
2. Schema: **invoice**. Deployment: your hot model.
3. Run.

**Expected:** structured JSON in ~10 s on a 0.2 GB model — invoice number,
customer, issue/due dates, currency, line items.

**Say:** this is a 350-megabyte model doing schema-constrained extraction on a
laptop CPU. Two things made that work: the schema is enforced at decode time,
and when a runtime can't compile the strict grammar the framework **negotiates
down** — `json_schema → json_object → repair` — instead of returning garbage.
You can see that negotiation in the run's status.

Then show the other panels quickly:
- **Vision** — send a PDF; it's rasterized to page images and sent to a vision
  model (no OCR step).
- **Embeddings** — two texts in, cosine similarity out, computed by a deployed
  embedding model.
- **Chat** — plain conversation with any live deployment.

---

## Act 5 — Capacity, honestly (4 min) ⭐ the differentiator

**Serving → Sizing.**

1. **Capacity bar** — node total/free, and an explicit safety margin held back
   (10% of total by default, shown as a number, not hidden).
2. **Fit table** — per model, "how many MORE instances fit right now". Models
   whose size cannot be determined say **size unknown** — never a guess.
3. **What-if** — stage a mix (e.g. 2 × `lfm2.5-350m` + 1 × `lfm2.5-2.6b`) and
   get fits / an explicit deficit.

**Say:** every number here is measured inside the serving container's cgroup,
not the host VM, and the same footprint math gates real deploys — so the tab
can't tell you something fits and then have the deploy OOM. When a model has
never run, it's priced from its weights plus KV cache; once it has run, the
measured steady-state memory wins.

Then the elasticity beat, back on **Deployments**:
- **Scale** a model to 2–3 instances — each is addressable and traffic is
  load-balanced behind one model id.
- **Unload** one: the record and its port survive, the phase reads **evicted**.
- Go to **Playground** and send a request to the evicted one — it **auto-reloads
  on demand** and answers.

**Say:** you can keep more models configured than fit in RAM at once; the node
time-shares them, evicts the least recently used under pressure, and never
evicts to a state that still doesn't fit.

---

## Act 6 — Turn a model into a service (4 min)

**Agents → Catalog.**

1. Show the templates: **Security proxy** (PII / guardrails) and **Document
   extraction** (OCR / OCR→LLM / vision→structured staged pipelines).
2. **Use template → Security proxy**, pick your hot model as the backing
   deployment, enable PII entities, save.
3. **My Agents** — copy the `base_url`. Show the `GET /models` and
   `POST /chat/completions` note.

**Say:** every agent is an OpenAI-compatible model id on one base URL, so any
agents platform, SDK or IDE plugin can consume it with no adapter. The security
proxy screens prompts and responses for PII with a local encoder model — the
document never leaves the machine, and if the guard is unreachable the proxy
fails closed by default rather than silently forwarding.

---

## Act 7 — Operations (2 min)

**Observability.** Grafana dashboards, Prometheus metrics, the Inngest job
console. Mention that every deploy, seed and extraction is a durable job with a
run id and a live progress stream (with a polling fallback when the realtime
channel is unavailable).

Optional closer — **deep links**: paste `http://localhost:3000/deploy/sizing`
in a fresh tab. It opens exactly there. Every section is a real URL you can
send to a colleague, and the back button works.

---

## Act 8 — Where it's going (1 min)

Say what's next, honestly:

- **Model comparison benchmarking** — the dataset registry, ground truth,
  per-field accuracy and cost accounting all exist; the runner now targets live
  deployments (fixed today), but sustained multi-document runs still destabilize
  the runtime and need work before they're demoable. **[known issue — do not
  demo]**
- **Measured throughput per deployment** — tokens/sec and time-to-first-token,
  measured on load and folded from real traffic, so "how many fit" (already
  shipped) becomes "how many documents per hour on this node". In progress.

---

## If something goes wrong on stage

| Symptom | What to do / say |
|---|---|
| Model shows **evicted** | "That's the idle unload." Click **Load**, or just send a request and let it auto-reload — it's a feature. |
| Deployment shows **failed** | Open its **logs** from the row — the real reason is there. Then click **Repair** (redeploys on a fresh port, resets the restart budget). |
| A page says "not available on this server" | That endpoint isn't enabled on this build; skip it — the UI degrades instead of crashing. |
| Extraction is slow (>30 s) | You're on CPU with a cold cache. Say so; show the logs with tokens/sec. Don't wait in silence. |
| Everything looks stale | The "Live · Xs ago" chip shows the last refresh; hit the refresh button next to it. |

---

## Demo assets

- **Documents:** `data/sample_dataset/files/` — 7 invoices + 2 ID cards, in
  French, English, German and Spanish, each with ground truth in
  `manifest.jsonl`.
- **Schemas:** `invoice`, `identity_card`.
- **Best demo model:** `lfm2.5-350m` (0.2 GB, ~10 s per invoice on CPU).
- **Vision demo model:** `lfm2.5-vl-1.6b` or `unlimited-ocr`.
