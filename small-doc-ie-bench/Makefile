SHELL := /usr/bin/env bash
DATASET ?= sample@1.0.0
MODEL_PROFILE ?= local_llamacpp

.PHONY: build up-infra up-api up-local-llm down logs test lint format bench smoke \
	studio studio-down studio-logs studio-smoke worker-dev

build:
	docker compose build

up-infra:
	docker compose up -d postgres

up-api:
	docker compose up -d api

up-local-llm:
	docker compose --profile local-llm up -d llm-llamacpp

down:
	docker compose --profile local-llm --profile tools --profile observability down

logs:
	docker compose logs -f --tail=200

test:
	python -m pytest -q

lint:
	ruff check .
	mypy src

format:
	ruff check --fix .
	ruff format .

bench:
	docker compose --profile tools run --rm bench benchmark run \
		--dataset $(DATASET) \
		--models-config configs/models.yaml \
		--model-profile $(MODEL_PROFILE) \
		--output-dir runs/$$(date +%Y%m%d-%H%M%S)

smoke:
	curl -s http://localhost:8080/healthz | jq .
	curl -s http://localhost:8080/v1/schemas | jq .

# ---- DocIE Studio (Inngest-backed async studio + web UI) --------------------

# One command: bring up the full studio stack (db, redis, inngest, worker, api, web).
studio:
	docker compose up -d --build postgres redis inngest worker api web

studio-down:
	docker compose down

studio-logs:
	docker compose logs -f --tail=200 inngest worker api web

# Trigger an extraction through the studio API and print the trigger response.
# Needs the stack up and your OPENAI_COMPAT_* LLM reachable from the worker.
studio-smoke:
	curl -s -X POST http://localhost:8080/v1/studio/extract \
		-H "Content-Type: application/json" \
		-d '{"text":"INVOICE\nTotal: 120.00 EUR","schema_name":"invoice"}' | jq .

# Run the Connect worker locally against `npx inngest-cli@latest dev` (no Docker).
worker-dev:
	INNGEST_DEV=1 docie-worker
