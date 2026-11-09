# DocIE

[![Documentation](https://img.shields.io/badge/docs-baylitoo.github.io-blue)](https://baylitoo.github.io/slm_benchmark/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

A document information-extraction platform built on small language models that
it runs itself. DocIE deploys and supervises model runtimes on CPU hardware,
serves them behind one OpenAI-compatible API, and runs extraction as durable
jobs that return validated, evidence-linked fields.

**Documentation: <https://baylitoo.github.io/slm_benchmark/>**

## What it does

- **Serves models.** Seeds GGUF, encoder and transformers models from Hugging
  Face, sizes them against the memory a node may use, and supervises the
  runtimes. → [Onboarding a model](https://baylitoo.github.io/slm_benchmark/models/onboarding.html)
- **Extracts.** Turns documents into schema-validated JSON with OCR, layout and
  evidence grounding, through the API or as batch jobs. → [Extraction](https://baylitoo.github.io/slm_benchmark/architecture/extraction.html)
- **Qualifies models.** Benchmarks models against versioned datasets, with
  regression gates for CI. → [Running a benchmark](https://baylitoo.github.io/slm_benchmark/benchmarking/running.html)
- **Scales out.** Adds serving nodes behind the same entrypoint, with
  node-targeted deploys, drain and per-node memory budgets. → [Fleet cookbook](https://baylitoo.github.io/slm_benchmark/operations/fleet-cookbook.html)

## Interfaces

| | |
|---|---|
| **DocIE Studio** | Web console: deploy and size models, build extraction agents, run benchmarks, manage the fleet. |
| **HTTP API** | OpenAI-compatible `/v1` plus extraction, serving and fleet endpoints. → [API reference](https://baylitoo.github.io/slm_benchmark/reference/api.html) |
| **`docie-bench`** | Headless benchmark runs, datasets and regression gates. |
| **`docie-serving`** | Node-level operations: local model store, preflight checks, drain. → [Serving CLI](https://baylitoo.github.io/slm_benchmark/reference/serving-cli.html) |

## Requirements

- Linux with Docker and Docker Compose v2
- CPU only; no GPU is needed
- About 8 GB of RAM for the serving container by default (`DOCIE_SERVING_MEM_LIMIT`)

## Quickstart

```bash
cp .env.example .env
# Set POSTGRES_PASSWORD, INNGEST_EVENT_KEY and INNGEST_SIGNING_KEY
# (for example with: openssl rand -hex 32)
docker compose up -d --build
```

| Service | Address |
|---|---|
| Studio | <http://localhost:3000> |
| API | <http://localhost:8080> |

Then follow [Getting started](https://baylitoo.github.io/slm_benchmark/operations/getting-started.html)
to serve a first model and run an extraction. Before exposing the stack on a
network, read [Security](https://baylitoo.github.io/slm_benchmark/operations/security.html).

## Documentation

| Topic | Read |
|---|---|
| Running it | [Getting started](https://baylitoo.github.io/slm_benchmark/operations/getting-started.html) · [Studio](https://baylitoo.github.io/slm_benchmark/operations/studio.html) · [Security](https://baylitoo.github.io/slm_benchmark/operations/security.html) |
| Several machines | [Fleet cookbook](https://baylitoo.github.io/slm_benchmark/operations/fleet-cookbook.html) · [Fleet runbook](https://baylitoo.github.io/slm_benchmark/operations/fleet-runbook.html) |
| Models | [Onboarding](https://baylitoo.github.io/slm_benchmark/models/onboarding.html) · [Reasoning models](https://baylitoo.github.io/slm_benchmark/models/reasoning.html) |
| Benchmarking | [Running](https://baylitoo.github.io/slm_benchmark/benchmarking/running.html) · [Datasets](https://baylitoo.github.io/slm_benchmark/benchmarking/datasets.html) · [Model profiles](https://baylitoo.github.io/slm_benchmark/benchmarking/model-profiles.html) |
| Architecture | [Overview](https://baylitoo.github.io/slm_benchmark/architecture/overview.html) · [Serving](https://baylitoo.github.io/slm_benchmark/architecture/serving.html) · [Jobs](https://baylitoo.github.io/slm_benchmark/architecture/inngest.html) · [Auth](https://baylitoo.github.io/slm_benchmark/architecture/auth.html) |
| Reference | [HTTP API](https://baylitoo.github.io/slm_benchmark/reference/api.html) · [Serving CLI](https://baylitoo.github.io/slm_benchmark/reference/serving-cli.html) · [Glossary](https://baylitoo.github.io/slm_benchmark/reference/glossary.html) · [Decisions](https://baylitoo.github.io/slm_benchmark/decisions/index.html) |

The documentation sources are in [`docs/`](docs/index.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/docie_bench/` | API, jobs, serving control plane, extraction, benchmarks |
| `frontend/` | DocIE Studio (Next.js) — see its [README](frontend/README.md) |
| `configs/` | Model profiles, benchmark and routing configuration |
| `infra/` | Dockerfiles, Postgres, Prometheus and Grafana provisioning |
| `docs/` | Documentation sources |
| `tests/` | Test suite |

## Development

```bash
make lint
make test
```

See [Contributing](https://baylitoo.github.io/slm_benchmark/development/contributing.html)
and [Testing](https://baylitoo.github.io/slm_benchmark/development/testing.html).

## License

[Apache License 2.0](LICENSE).
