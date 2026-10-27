# DocIE

A document information-extraction platform built on small language models it
runs itself. It deploys and supervises model runtimes on CPU hardware, exposes
them behind one OpenAI-compatible API, runs extraction as durable jobs that
return validated, evidence-linked fields, and ships a benchmark harness for
deciding which model to trust with which document.

- **DocIE Studio** — a web console to deploy models from Hugging Face, check
  what fits in memory, build extraction and PII-guard agents, run benchmarks,
  and watch it all in Grafana.
- **`docie-bench`** — headless benchmark runs, versioned datasets, regression
  gates.
- **`docie-serving`** — node-level operations: the local model store, preflight
  checks, standalone runtimes.

**Documentation: <https://baylitoo.github.io/slm_benchmark/>**

## Quickstart

```bash
cp .env.example .env
# set POSTGRES_PASSWORD, and INNGEST_EVENT_KEY / INNGEST_SIGNING_KEY (openssl rand -hex 32)
docker compose up -d --build
```

Open the Studio at <http://localhost:3000>, then follow
[Getting started](https://baylitoo.github.io/slm_benchmark/operations/getting-started.html)
to serve a first model and run an extraction.

### API only, no Studio

```bash
cp .env.example .env
make build && make up-infra
MODEL_PATH=/models/model.gguf make up-local-llm   # a GGUF at ./models/model.gguf
make up-api
curl -s http://localhost:8080/v1/extract/text \
  -H 'Content-Type: application/json' \
  -d @examples/invoice_ocr_request.json | jq .
```

### Benchmark a model

```bash
docie-bench benchmark run --dataset sample@1.0.0 --model-profile local_llamacpp
```

See [Running a benchmark](https://baylitoo.github.io/slm_benchmark/benchmarking/running.html).

## Where to read next

| | |
|---|---|
| Running it | [Getting started](https://baylitoo.github.io/slm_benchmark/operations/getting-started.html), [the Studio](https://baylitoo.github.io/slm_benchmark/operations/studio.html), [security](https://baylitoo.github.io/slm_benchmark/operations/security.html), [adding serving nodes](https://baylitoo.github.io/slm_benchmark/operations/fleet-runbook.html) |
| Integrating an application | [HTTP API](https://baylitoo.github.io/slm_benchmark/reference/api.html) |
| Qualifying a model | [benchmarks](https://baylitoo.github.io/slm_benchmark/benchmarking/running.html), [datasets](https://baylitoo.github.io/slm_benchmark/benchmarking/datasets.html), [model profiles](https://baylitoo.github.io/slm_benchmark/benchmarking/model-profiles.html) |
| Changing the code | [architecture overview](https://baylitoo.github.io/slm_benchmark/architecture/overview.html), [contributing](https://baylitoo.github.io/slm_benchmark/development/contributing.html) |

The sources are in [`docs/`](docs/index.md). The frontend has its own
[README](frontend/README.md).

## Development

```bash
make lint
make test
make format
```
