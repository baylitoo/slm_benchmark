# Small Document IE Benchmark

Enterprise-grade CPU-only benchmark harness for structured information extraction from invoices and identity documents using small local/open models behind an OpenAI-compatible API.

The project is designed for a Ryzen-class CPU server with 64 GB RAM and no GPU. It separates OCR/layout, model serving, constrained JSON extraction, validation, persistence, and benchmark reporting.

## What this project gives you

- **OpenAI-compatible LLM abstraction**: call local `llama.cpp`, vLLM, Ollama-compatible gateways, or a remote OpenAI-compatible endpoint through one client.
- **Schema-constrained extraction**: JSON Schema / Pydantic first; no free-form JSON guessing.
- **OCR modularity**: `pdf_text`, `tesseract`, and `paddleocr` backends behind one interface.
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
