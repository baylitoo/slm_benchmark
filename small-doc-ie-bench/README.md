# Small Document IE Benchmark

Enterprise-grade CPU-only benchmark harness for structured information extraction from invoices and identity documents using small local/open models behind an OpenAI-compatible API.

The project is designed for a Ryzen-class CPU server with 64 GB RAM and no GPU. It separates OCR/layout, model serving, constrained JSON extraction, validation, persistence, and benchmark reporting.

## Model serving factory

The serving control plane turns local and remote inference runtimes into a
consistent operational workflow. It supports vLLM, llama.cpp, Ollama, and
OpenAI-compatible remote endpoints, with a content-addressed model registry,
resource planning, and a persistent local deployment supervisor.

```bash
docie runtime list
docie model pull ./path/to/model-manifest.json
docie plan my-model
docie serve my-model --runtime llamacpp
docie list
docie status my-model
```

Every command supports deterministic automation output through `--json`.
See [docs/serving-factory.md](docs/serving-factory.md) for architecture,
runtime requirements, model manifests, and operational examples.

### Quickstart — serve a model, then benchmark it

Seed a GGUF into the canonical store once (see
[src/docie_bench/serving/README.md](src/docie_bench/serving/README.md)), then it's
three commands — no separate `llama-server` window:

```powershell
docie up nuextract3        # serve in the background with the right family flags (--jinja/--mmproj, :8088)
docie-bench benchmark run --dataset data\voxel51_invoices\manifest.jsonl --model-profile nuextract3
docie stop nuextract3
```

`docie up <name>` launches the model **detached** via the supervisor and pins the
port to the model profile's `base_url`, so the benchmark consumes it unchanged.
To put *every* configured profile behind one OpenAI-compatible `/v1` endpoint,
run `docie gateway`. Full flow: [serving README](src/docie_bench/serving/README.md).

## What this project gives you

- **OpenAI-compatible LLM abstraction**: call local `llama.cpp`, vLLM, Ollama-compatible gateways, or a remote OpenAI-compatible endpoint through one client.
- **Schema-constrained extraction**: JSON Schema / Pydantic first; no free-form JSON guessing.
- **OCR modularity**: `pdf_text`, `tesseract`, and `paddleocr` backends behind one interface.
- **OCR laboratory**: content-addressed OCR artifacts, persistent cache, and no-LLM OCR reports.
- **Production API**: FastAPI service with health checks, metrics, file-size limits, structured logs, and optional Postgres audit persistence.
- **Benchmark runner**: run many model profiles over the same dataset and produce JSONL predictions, metrics, and an HTML report.
- **Docker Compose stack**: API, benchmark container, local llama.cpp-compatible server, Postgres, Prometheus, and Grafana.
- **Model-agnostic**: benchmark Qwen/Gemma/Granite/SmolLM/Llama/Phi/Ministral GGUFs or any HTTP endpoint exposing `/v1/chat/completions`.

## Architecture

```text
PDF/image
  │
  ├── OCR/layout backend
  │     ├── pdf_text: text layer extraction for digital PDFs
  │     ├── tesseract: CPU OCR fallback
  │     └── paddleocr: richer OCR/layout backend
  │
  ├── OCR blocks with evidence ids + bounding boxes
  │
  ├── extraction prompt builder
  │
  ├── OpenAI-compatible LLM client
  │     ├── response_format: OpenAI json_schema
  │     ├── response_format: llama.cpp schema mode
  │     ├── guided_json / structured_outputs for vLLM-like servers
  │     └── plain JSON fallback with strict validation
  │
  ├── Pydantic validation + normalizers
  │
  └── metrics / predictions / audit store
```

## Fast start

```bash
cp .env.example .env
make build
make up-infra
```

Place a GGUF model at `./models/model.gguf`, then run:

```bash
MODEL_PATH=/models/model.gguf docker compose --profile local-llm up -d llm-llamacpp
make up-api
```

Smoke-test with OCR text instead of a file:

```bash
curl -s http://localhost:8080/v1/extract/text \
  -H 'Content-Type: application/json' \
  -d @examples/invoice_ocr_request.json | jq .
```

Run benchmark:

```bash
make bench
```

## Distributed orchestration

Set `DATABASE_URL` to enable the persistent experiment and worker APIs. Submit a run with
deterministic task keys through `POST /v1/experiments`, then workers use:

- `POST /v1/workers/tasks/claim`
- `POST /v1/workers/tasks/{task_id}/heartbeat`
- `POST /v1/workers/tasks/{task_id}/complete`
- `POST /v1/workers/tasks/{task_id}/fail`

Lease tokens prevent expired workers from publishing duplicate final results. Expired tasks are
recovered during claims or through `POST /v1/workers/recover`. Runs can be inspected, queried,
cancelled, and resumed under `/v1/experiments`. The in-process `BenchmarkWorker` supports sync or
async executors and stores artifacts atomically through the configurable artifact-store protocol;
`LocalArtifactStore` uses content-addressed paths to keep competing attempts isolated.

Evaluate with an LLM judge by selecting a judge profile separately from extraction models:

```yaml
judge:
  profile: remote_openai_compatible
```

```bash
docie-bench benchmark run \
  --dataset sample@1.0.0 \
  --eval-mode both
```

Run only one split with `--split test`. Each prediction row records its source split.

An unlabeled document does not need a manifest:

```bash
docie-bench benchmark run \
  --document data/sample_dataset/files/invoice-001.txt \
  --schema-name invoice \
  --model-profile local_llamacpp \
  --eval-mode llm_judge
```

Judge results are stored per prediction and aggregated as `judge_faithfulness` and
`judge_completeness` in `metrics.json`. In `both` mode, `judge_field_accuracy_delta`
compares aggregate judge faithfulness with labeled field accuracy. Judge failures are
recorded as `judge_error` without changing extraction success.

#### Judge calibration gate

The judge is uncalibrated by default, so its scores must not silently block a release.
`benchmark compare` only lets `judge_faithfulness` / `judge_completeness` budgets **fail**
when a judge<->human calibration set proves the judge agrees with human labels; otherwise
those budgets are downgraded to a non-blocking `warn`. Populate a calibration file (see
`configs/judge_calibration.example.json`) with the judge's recorded scores paired with
human scores on the same documents, then:

```bash
docie-bench benchmark judge-calibration configs/judge_calibration.json
docie-bench benchmark compare baseline candidate \
  --budgets configs/regression-budgets.yaml \
  --calibration configs/judge_calibration.json
```

The gate certifies the judge only when **every** scored dimension
(`faithfulness` and `completeness`) independently clears three bars: at least 30 **real
labeled pairs** for that dimension (padded rows without both `judge_`/`human_` scores do
not count), a mean absolute error within `--max-judge-mae` (default 0.15), and a
judge<->human correlation of at least 0.3. Too few pairs is reported as
`insufficient_calibration_samples`; a near-constant judge with undefined/zero-variance
correlation is reported as `correlation_below_threshold`. Either way it can only warn,
never block — the gate fails closed so an untrustworthy judge can never block a release.

#### Reading `hallucination_rate` by ingestion path

`hallucination_rate` is derived from evidence grounding against OCR text, so each profile
summary is labelled with how to read it:

- `hallucination_basis: consumed_ocr_text` (`hallucination_reflects_model: true`) — OCR
  paths ground against the same text the model consumed, so an ungrounded field is a
  plausible model signal.
- `hallucination_basis: no_consumed_text` (`hallucination_reflects_model: false`) — **vision**
  profiles consume page images and grounding runs with no consumed text, so the value is a
  grounding artifact (near 1.0 by construction), **not** model hallucination. Segment by
  `ingestion_path` before comparing across profiles.

### Reproducible and resumable runs

Each benchmark run writes an immutable `manifest.json` with the git state, sanitized selected
model profiles, model config and dataset hashes, document hashes, dependency versions, system
resources, invocation arguments, and stable task IDs. Predictions and lifecycle events are
durably appended, while final prediction, metric, and report artifacts are replaced atomically.

Resume an interrupted run by reusing its output directory:

```bash
docie-bench benchmark run \
  --dataset data/sample_dataset/manifest.jsonl \
  --output-dir runs/my-run \
  --resume
```

Resume skips completed and failed terminal tasks, repairs a truncated final JSONL record, and
refuses to proceed when code, model config, selected profiles, dataset contents, or task inputs
have drifted. Concurrency may change when resuming because it does not affect task identity.

## Model profiles

Model profiles live in `configs/models.yaml`. A profile can point to:

- the local llama.cpp server in Docker Compose;
- a vLLM OpenAI-compatible server;
- a remote OpenAI-compatible endpoint;
- a different port if you run multiple servers manually.

Example profile:

```yaml
profiles:
  qwen3_0_6b_cpu:
    model: qwen3-0.6b-instruct-q4_k_m
    base_url: http://llm-llamacpp:8000/v1
    api_key_env: LOCAL_LLM_API_KEY
    response_format_style: openai_json_schema
    temperature: 0.0
    max_tokens: 900
    capability_discovery: optional
    retry_max_attempts: 3
    circuit_breaker_failure_threshold: 5
    circuit_breaker_reset_seconds: 30
    max_concurrency: 2
    queue_limit: 16
    queue_timeout_seconds: 30
```

The model gateway shares scheduling and circuit-breaker state across clients targeting
the same endpoint/model. It retries only transient, rate-limited, and invalid-response
failures; permanent 4xx responses fail immediately. `capability_discovery` can be
`disabled`, `optional`, or `required`. Discovery uses `GET /models`, verifies the
configured model id, and validates advertised `vision` and `response_format_styles`
metadata when the endpoint provides it.

## Dataset format

`manifest.jsonl`:

```json
{"doc_id":"invoice-001","file_path":"files/invoice-001.txt","schema_name":"invoice","language":"fr","split":"test","ground_truth":{"invoice_number":"INV-2026-0001","vendor_name":"ACME SAS","total_ttc.amount":"1245.30","total_ttc.currency":"EUR","issue_date":"2026-05-21"}}
```

### Versioned dataset registry

`data/datasets.yaml` maps stable references such as `sample@1.0.0` to manifests. A version
pins a dataset SHA-256 and stored statistics. Benchmark runs verify the pinned hash before
execution and include the resolved reference, version, manifest, and hash in `metrics.json`.
Omitting `@version` resolves the registry's `latest` version. Direct manifest paths remain
supported for ad hoc and legacy runs.

Register an immutable semantic version after validation:

```bash
docie-bench dataset version invoices 1.0.0 \
  --manifest data/invoices/v1/manifest.jsonl
```

Inspect statistics and hash, validate integrity and split leakage, or run leakage detection
on its own:

```bash
docie-bench dataset inspect invoices@1.0.0
docie-bench dataset validate invoices@1.0.0
docie-bench dataset leakage invoices@1.0.0 --near-duplicate-threshold 0.92
```

Validation checks JSONL rows, unique document IDs, supported and existing files, non-empty
splits, pinned hashes, and exact/near-duplicate cross-split leakage. Exact detection uses
document bytes. Near-duplicate detection currently compares normalized text documents and
is intentionally conservative; OCR must be materialized as `.txt` to near-match PDF/image
content.

Legacy manifests load with the `unspecified` split. Migrate one to an explicit default split,
or provide a JSON object mapping document IDs to splits:

```bash
docie-bench dataset migrate data/legacy/manifest.jsonl data/invoices/v1/manifest.jsonl \
  --default-split test

docie-bench dataset migrate data/legacy/manifest.jsonl data/invoices/v1/manifest.jsonl \
  --split-map data/invoices/splits.json
```

Dataset hashes are stable across manifest relocation and row ordering. They cover each
semantic manifest row plus the referenced document's SHA-256, so any label, split, metadata,
schema, or document-content change creates a new dataset identity.

Supported files:

- `.txt`: already OCR'd text;
- `.pdf`: text layer via `pdfplumber` or OCR fallback;
- `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`: OCR backend required.

### Dynamic schemas

Set `schema_mode` to `dynamic` to infer a schema for an unknown OCR-text document. The response
includes `dynamic_schema`; persist that JSON and pass it as `dynamic_schema` on later requests (or
manifest rows) to reuse it without another proposal call. A NuExtract profile needs an
instruction-following `schema_proposer_profile` for first-time inference, but can extract directly
from a reused dynamic schema.

Dynamic fields support scalar `string`, `date`, `number`, and `money` types plus recursive `object`
and repeated-row `list` types. Container fields define their reusable children in `fields`.

Invoice extraction includes typed `line_items` with description, SKU, quantity, unit price, line
total, and tax rate. To evaluate a table, put a list under the matching ground-truth key. Rows are
aligned by maximum cell similarity before cell accuracy and row precision/recall/F1 are calculated:

```json
{"ground_truth":{"line_items":[{"description":"Keyboard","quantity":"2","line_total.amount":"150.00"}]}}
```

Validation checks each `quantity * unit_price` against `line_total`, the sum of line totals against
the invoice subtotal, and reports mismatches as warnings.

Vision-capable model profiles can bypass OCR for PDFs and images:

```yaml
profiles:
  ollama_qwen25_vl_7b:
    model: qwen2.5vl:7b
    base_url: http://localhost:11434/v1
    api_key: local-not-used
    response_format_style: json_object
    vision: true
    vision_max_pages: 8
    vision_pdf_dpi: 150
```

With `vision: true`, image files are normalized to PNG and PDF pages are rasterized
with PyMuPDF, then sent as OpenAI-compatible `image_url` content blocks. `.txt` input
is intentionally rejected for vision profiles. Benchmark artifacts label each result
as `vision` or `ocr:<backend>` so the same manifest can compare both paths side by side.

## Multi-stage routing

`ExtractionRouter` wraps existing `ExtractionService` instances without changing their
single-stage API. Policies are Pydantic models, so they can be loaded directly from YAML
or JSON:

```python
from docie_bench.extract.routing import (
    ExtractionRouter, ExtractionServiceStage, RoutingPolicy,
)

policy = RoutingPolicy.model_validate({
    "version": "2026-06",
    "stages": [
        {
            "name": "fast",
            "rules": [{
                "when": {"status": "success", "validation_valid": True, "min_confidence": 0.85},
                "decision": "accept",
                "reason": "fast model passed quality gate",
            }],
        },
        {
            "name": "accurate",
            "rules": [{
                "when": {"status": "success", "validation_valid": True},
                "decision": "accept",
                "reason": "fallback model returned a valid extraction",
            }],
        },
    ],
    "budget": {"max_stages": 2, "max_total_tokens": 4096, "max_latency_ms": 30000},
})
router = ExtractionRouter(
    stages=[
        ExtractionServiceStage("fast", fast_service),
        ExtractionServiceStage("accurate", accurate_service),
    ],
    policy=policy,
)
```

Rules are evaluated in order. A stage can be accepted, sent to the next fallback, escalated,
or failed. Stage selectors can branch on request context such as file suffix, language, OCR
confidence, page/block counts, or caller-supplied complexity/capability metadata, plus prior-stage
validation and metadata. The routed response contains a `routing` audit with every stage output,
attempt, decision, skipped stage, budget total, and terminal outcome. Benchmark summaries and
HTML reports aggregate routing acceptance, fallback, escalation, stage failure, latency, tokens,
cost, budget exhaustion, and average-attempt metrics when this audit is present.

### Benchmarking a routed pipeline

Run a whole dataset through a policy and compare it against single-model baselines with
`--routing-policy`. Each policy stage's `name` must match a profile in your models config
(name convention), so adding a model to a cascade is just adding the profile and referencing
its name as a stage. See `configs/routing-policy.example.yaml` for a runnable CPU cascade.

```bash
docie-bench benchmark run --dataset sample@1.0.0 --routing-policy configs/routing-policy.example.yaml
```

Routed runs collapse the per-profile sweep into one pass per document (the router selects
profiles internally), label the result row `routed:<policy version>`, and record the policy in
the run manifest for reproducibility. `--routing-policy` and `--model-profile` are mutually
exclusive.

## Choosing a CPU model

Recommended benchmark order:

1. Qwen3/Qwen3.5 tiny profile if your runtime supports it.
2. Gemma 3/4 small profile if available as GGUF and runtime-compatible.
3. Granite 4.x 1B/3B for enterprise/compliance-friendly experiments.
4. SmolLM3-3B for open small text baseline.
5. Llama 3.2 3B Instruct as stable deployment baseline.
6. Phi-4-mini or similar if you can accept higher CPU latency.

Do not select by leaderboard alone. Select by field-level accuracy under constrained decoding.

## Production notes

- Keep OCR evidence ids. The extractor must cite evidence ids for each field.
- Store document hashes, not raw documents, unless your policy allows raw storage.
- Use deterministic sampling: `temperature=0`, `top_p=1`.
- Enforce `max_tokens` and request timeouts.
- Run a field-level benchmark before every model upgrade.
- Do not accept raw model output unless it passes Pydantic validation.

## API

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /v1/schemas`
- `POST /v1/extract/text`
- `POST /v1/extract/file`
- `POST /v1/benchmarks/run`
- `POST /v1/reviews`
- `GET /v1/reviews`
- `GET /v1/reviews/metrics`
- `GET /v1/reviews/{task_id}`
- `POST /v1/reviews/{task_id}/claim`
- `POST /v1/reviews/{task_id}/release`
- `POST /v1/reviews/{task_id}/correct`
- `POST /v1/reviews/{task_id}/approve`
- `POST /v1/reviews/{task_id}/reject`
- `POST /v1/reviews/exports`

The extraction endpoints return:

```json
{
  "request_id": "...",
  "schema_name": "invoice",
  "model_profile": "qwen3_0_6b_cpu",
  "document_hash": "sha256:...",
  "result": {...},
  "validation": {...},
  "usage": {...},
  "latency_ms": 1234
}
```

### Human review workflow

When database persistence is enabled, invalid, low-confidence, and weakly grounded
extractions are automatically admitted to the review queue. Tasks submitted through
`POST /v1/reviews` can additionally carry model-disagreement and expected-learning-value
scores. Every task exposes an explainable priority breakdown.

Reviewers claim tasks with an expiring lease and must send the current `expected_version`
with every mutation. Stale versions, expired claims, and writes from another reviewer
return HTTP 409 instead of overwriting work. Corrections use dotted field paths such as
`invoice_number.value`; every correction revision and lifecycle event is immutable and
included in the task history.

Approved corrections can be exported with `POST /v1/reviews/exports`. Export versions are
write-once under `ANNOTATION_EXPORT_DIR`, and only the `train` split is accepted to prevent
reviewed labels from leaking into evaluation data. Review metrics report queue depth,
correction rate, reviewer agreement, queue latency, and per-reviewer workload.

## Security boundaries

This project does not claim PII compliance out of the box. See
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) for deployment guidance, residual risks, and
security configuration. The API provides:

- bounded upload, text, and OCR-block limits;
- MIME allowlisting with content validation;
- optional API-key authentication and per-tenant in-memory quotas;
- prompt-injection boundaries for adversarial document content;
- hash-based audit storage;
- configurable audit and response field redaction;
- document-content logging disabled by default;
- per-run artifacts separated from service logs.

## Development

```bash
make lint
make test
make format
```

## Why not one monolithic “document AI” model?

Because CPU-only production needs predictable failure modes. OCR + schema-constrained extraction + validation is easier to audit, benchmark, and improve than a black-box raw-image VLM pipeline. VLMs should be benchmarked as a fallback path, not blindly deployed as the default.
