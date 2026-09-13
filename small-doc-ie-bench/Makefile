SHELL := /usr/bin/env bash
DATASET ?= sample@1.0.0
MODEL_PROFILE ?= local_llamacpp

.PHONY: build up-infra up-api up-local-llm down logs test lint format bench smoke \
	studio studio-down studio-logs studio-smoke worker-dev \
	frontend-test frontend-typecheck frontend-check

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

test: frontend-test
	python -m pytest -q

lint: frontend-typecheck
	ruff check .
	mypy src

# ---- Frontend ---------------------------------------------------------------
# vitest and tsc ship with the web app and pass; until now nothing in the repo
# ran them, so 179 tests and the whole typecheck were only ever exercised by
# hand. Both skip with a message rather than failing when node or the install
# is missing, so a Python-only checkout still runs `make test`.

frontend-check: frontend-typecheck frontend-test

frontend-typecheck:
	@if ! command -v npm >/dev/null 2>&1; then \
		echo "frontend-typecheck: skipped (npm not found)"; \
	elif [ ! -d frontend/node_modules ]; then \
		echo "frontend-typecheck: skipped (run 'npm ci' in frontend/ first)"; \
	else \
		cd frontend && npm run --silent typecheck; \
	fi

frontend-test:
	@if ! command -v npm >/dev/null 2>&1; then \
		echo "frontend-test: skipped (npm not found)"; \
	elif [ ! -d frontend/node_modules ]; then \
		echo "frontend-test: skipped (run 'npm ci' in frontend/ first)"; \
	else \
		cd frontend && npm run --silent test; \
	fi

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
