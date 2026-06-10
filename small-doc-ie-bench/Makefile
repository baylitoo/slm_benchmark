SHELL := /usr/bin/env bash
DATASET ?= data/sample_dataset/manifest.jsonl
MODEL_PROFILE ?= local_llamacpp

.PHONY: build up-infra up-api up-local-llm down logs test lint format bench smoke

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
