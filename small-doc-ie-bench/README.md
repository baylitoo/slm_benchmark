# Small Document IE Benchmark

Enterprise-grade CPU-only benchmark harness for structured information extraction from invoices and identity documents using small local/open models behind an OpenAI-compatible API.

The project is designed for a Ryzen-class CPU server with 64 GB RAM and no GPU. It separates OCR/layout, model serving, constrained JSON extraction, validation, persistence, and benchmark reporting.

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
make bench DATASET=data/sample_dataset/manifest.jsonl
```

Run an OCR-only comparison without invoking an LLM:

```bash
docie-bench benchmark ocr run \
  --dataset data/ocr_dataset/manifest.jsonl \
  --backend pdf_text \
  --backend tesseract
```

OCR manifests use the normal dataset fields plus one of `ocr_reference_text`,
`ocr_reference_path`, or `ocr_reference_blocks`. The report includes character error
rate, word error rate, layout preservation, latency, cache hit rate, and low-quality
OCR rate. Pass extraction benchmark output with `--extraction-metrics
runs/<run>/metrics.json` to correlate OCR character accuracy with downstream field
accuracy.

OCR artifacts are versioned JSON containing blocks, bounding boxes, confidence,
optional embedded page images, backend metadata, and quality signals. Non-vision
extraction and judge evaluation share the persistent cache at `OCR_CACHE_DIR`.
Cache keys include document content, backend, language, backend/runtime version, and
canonical backend configuration. Entries are checksum-validated, atomically replaced,
rebuilt after corruption, and evicted least-recently-used when `OCR_CACHE_MAX_MB` is
exceeded.

Evaluate with an LLM judge by selecting a judge profile separately from extraction models:

```yaml
judge:
  profile: remote_openai_compatible
```

```bash
docie-bench benchmark run \
  --dataset data/sample_dataset/manifest.jsonl \
  --eval-mode both
```

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
```

## Dataset format

`manifest.jsonl`:

```json
{"doc_id":"invoice-001","file_path":"data/sample_dataset/files/invoice-001.txt","schema_name":"invoice","language":"fr","ground_truth":{"invoice_number":"INV-2026-0001","vendor_name":"ACME SAS","total_ttc.amount":"1245.30","total_ttc.currency":"EUR","issue_date":"2026-05-21"}}
```

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

## Security boundaries

This project does not claim PII compliance out of the box. It provides the hooks you need:

- request size limits;
- content-type allowlist;
- hash-based audit storage;
- configurable raw document storage policy;
- redaction-friendly logging;
- per-run artifacts separated from service logs.

## Development

```bash
make lint
make test
make format
```

## Why not one monolithic “document AI” model?

Because CPU-only production needs predictable failure modes. OCR + schema-constrained extraction + validation is easier to audit, benchmark, and improve than a black-box raw-image VLM pipeline. VLMs should be benchmarked as a fallback path, not blindly deployed as the default.
