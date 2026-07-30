# Demo — Agents end to end, entirely from the Studio

One continuous story, told from the browser only: pick a model, attach it to a
family, deploy it, wrap it in agents (OCR, security gate, anonymizer), exercise
them on real documents, and finish on live metrics. No terminal after the
initial `docker compose up`.

**Cast** (all CPU):

| Role | Model / engine | Where it comes from |
|---|---|---|
| Chat SLM (agent backing) | `LiquidAI/LFM2.5-350M-GGUF` (LFM2.5 family) | downloaded straight from the HF Hub, in-UI |
| Guard encoder (PII analyzer) | `fastino/GLiNER2-Guardrails-PII-Multi` | one click in the Agents form |
| OCR engine | tesseract (eng+fra, in the image) | nothing to fetch |
| Optional OCR extractor | a NuExtract3 deployment | if already in your store |

**Demo documents** (in the repo):

- `data/sample_dataset/files/invoice-001.txt` — French invoice with an IBAN,
  SIRET, VAT ids (paste into Try panels).
- `data/sample_dataset/files/id-card-001.txt` — French ID card: name, birth
  date, MRZ (the aggressive PII sample).
- Any invoice JPG/PNG/PDF on your machine for the OCR agent (the Try panel
  uploads it straight from the browser).

Prerequisite once: `docker compose up -d api serving worker web`
(+ `--profile observability` for Grafana). No Ollama needed — models come
straight from the Hugging Face Hub.

---

## Act 1 — Model → family → deployment (Deploy section)

1. **Models → Add model → Hugging Face tab**: repo
   `LiquidAI/LFM2.5-350M-GGUF` → **Inspect** (lists every quant with
   its size) → pick `Q4_K_M`, store name prefilled `lfm2.5-350m`,
   **Family: `lfm2`** — this is the "integrate into a family" step: the family
   contract carries the template style, generation defaults and launch args
   the deploy will inherit. **Download & seed** → a live progress bar streams
   the download (percent + bytes) over the same realtime backbone as every
   other job. *(The "Collection" tab seeds a provider's whole curated
   collection at once; "Ollama (legacy)" keeps the old path.)*
2. **Deployments → Deploy a model**: pick `lfm2.5-350m`, runtime
   Auto, deploy.
   Watch the row: phase `loading` → `hot`, PID, endpoint, observed RSS filled
   by the reconciler (~10s cadence).
3. **Sizing tab**: the new process is counted in the node bar; point at the
   fit table ("how many more instances fit right now").

## Act 2 — The three agents (Agents section)

### 2a. OCR agent
Catalog → **OCR Agent** → Use template → name `ocr-reader`, backend
`tesseract`, language `eng` (or `fra`) → Create.
*(Have NuExtract3 deployed? Set Extractor to its deployment name — the same
agent becomes an OCR→SLM structured-extraction pipeline.)*

### 2b. Security gate (filtering / blocking proxy)
Catalog → **Security Proxy Agent** → Use template → name `gate`:
- Backing model: `lfm2.5-350m` (datalist offers the live deployment)
- **Mode: Block — refuse requests containing PII**
- Guard model: click **Deploy** (spawns the `guardrails-pii` managed deployment —
  show it appear under Deployments, phase `loading` → `hot` like any SLM),
  field is prefilled with `guardrails-pii`.
- Create.

### 2c. Anonymizer
Same template → name `anonymizer`:
- Backing model `lfm2.5-350m`, guard `guardrails-pii`
- **Mode: Placeholder — anonymize before forwarding** (leave "restore" off so
  the masking stays visible in the response)
- Create.

My Agents now lists three rows, each with its own OpenAI-compatible
`base_url` — the platform-integration hook (one copy button per agent, plus
the platform-wide endpoint card in Catalog).

## Act 3 — Exercise them (Try panel, in each agent's row)

1. **`anonymizer`** → expand → Try. Replace the sample with the full text of
   `invoice-001.txt`, prepend *"Repeat this text exactly:"* → Run.
   Show: entity badges (`analyzer: guard:guardrails-pii`, `IBAN ×1`, `PERSON …`),
   the placeholder list that went upstream, and the model's reply containing
   `[IBAN_1]`-style tokens instead of real values. Raw JSON collapsible for
   the `docie_agent` report.
2. **`anonymizer`** again with `id-card-001.txt` — names, birth date, MRZ.
3. **`gate`** → same ID-card text → Run → red error
   `pii_blocked: request blocked …` — nothing reached the model (that's the
   gate).
4. **`ocr-reader`** → upload an invoice image → Run → OCR text back (with an
   extractor configured: structured JSON instead).
5. **Fail-closed** (the security money-shot): Deployments → Unload
   `guardrails-pii` → Try `anonymizer` → `guard_unavailable` 502, request refused.
   Load it back → works again. (Optional: an agent with
   `guard_fallback: "regex"` degrades instead and its report shows
   `degraded to regex`.)

## Act 4 — Metrics (Observability section)

Open **Observability → Dashboards** (Grafana, dashboard *Small Document IE
Benchmark*). Everything Act 3 generated is there within seconds:

- **Agent requests by outcome** — `ok` vs `pii_blocked` vs `guard_unavailable`
  per agent (the unload experiment is visible as a red series).
- **PII detected by type** — EMAIL / IBAN / PERSON… counts from the guard.
- **Agent latency p95** — encoder+SLM end-to-end per agent.
- Stat tiles: blocked by the gate, entities caught, guard failures, OK count.
- Below: extraction requests + model-gateway outcomes for the classic
  benchmark traffic.

Raw series: Observability → Links → Prometheus metrics
(`docie_agent_requests_total`, `docie_agent_pii_detected_total`,
`docie_agent_latency_seconds_*`).

---

## 30-second pitch to close on

Model chosen → family attached → deployed with live RAM/health → wrapped in
agents → consumed over OpenAI-compatible endpoints any platform can call →
audited in Grafana. The security proxy is the piece that makes remote/pooled
serving acceptable: PII never leaves the node unmasked, and when the analyzer
dies the system fails closed, visibly.
