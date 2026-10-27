# Running a benchmark

A benchmark runs one or more model profiles over a dataset and writes
predictions, metrics and an HTML report. The Studio's **Benchmark → Run** does
this as a durable job; `docie-bench` does it from a shell, which is what the
rest of this page uses.

```bash
docie-bench benchmark run --dataset sample@1.0.0 --model-profile local_llamacpp
```

`--dataset` takes a registry reference or a manifest path — see
{doc}`datasets`. Without `--model-profile`, every profile in
`configs/models.yaml` runs.

## What a run writes

| File | |
|---|---|
| `predictions.jsonl` | one row per document and profile, with its split and ingestion path |
| `metrics.json` | per-profile summaries, plus the resolved dataset reference and hash |
| `report.html` | the same, readable |
| `manifest.json` | everything needed to reproduce the run; written once, never changed |

The manifest records git state, the selected profiles with secrets removed,
config and dataset hashes, each document's hash, dependency versions, machine
resources, the invocation, and stable task ids.

### Resuming

Re-run with the same `--output-dir` and `--resume`. Finished and failed tasks
are skipped and a truncated final JSONL row is repaired. The run **refuses** to
continue if code, model config, selected profiles, dataset contents or task
inputs have changed since it started. `--concurrency` may change, because it is
not part of a task's identity.

## Evaluation modes

`--eval-mode` chooses how predictions are scored.

| Mode | Needs | |
|---|---|---|
| `ground_truth` (default) | ground truth in the manifest | field accuracy, validity, table metrics |
| `llm_judge` | a judge profile | faithfulness and completeness from a judge model |
| `both` | both | adds `judge_field_accuracy_delta`, the gap between the two |

The judge is configured separately from the models under test:

```yaml
judge:
  profile: remote_openai_compatible
```

A document without ground truth needs no manifest:

```bash
docie-bench benchmark run \
  --document data/sample_dataset/files/invoice-001.txt \
  --schema-name invoice --model-profile local_llamacpp --eval-mode llm_judge
```

A judge failure is recorded as `judge_error` on the prediction and does not
change whether the extraction succeeded. `--split test` restricts a run to one
split.

### Reading `hallucination_rate`

The rate comes from grounding each field against OCR text, so its meaning
depends on the ingestion path, and each summary says which applies:

- **`consumed_ocr_text`** — the model read the same text it is grounded
  against, so an ungrounded field is a real signal.
- **`no_consumed_text`** — a vision profile read page images, there was no text
  to ground against, and the rate is close to 1.0 by construction. It says
  nothing about the model.

Compare hallucination only within one ingestion path.

## Comparing against a baseline

```bash
docie-bench benchmark baseline promote runs/2026-10-01 main
docie-bench benchmark compare main runs/2026-10-27 --budgets configs/regression-budgets.yaml
```

A budget names a metric, the dimension it is measured on (`aggregate`, or a
slice such as `schema_name`), the largest regression allowed, and the minimum
number of paired documents before it may fail. `configs/regression-budgets.yaml`
is a working example.

### The judge calibration gate

A judge is uncalibrated until shown otherwise, so judge-score budgets only
**warn** by default. They can fail a comparison only when `--calibration` points
at a set of judge scores paired with human scores on the same documents
(format: `configs/judge_calibration.example.json`), and every scored dimension
independently clears three bars:

- at least 30 real pairs — rows missing either score do not count,
- mean absolute error within `--max-judge-mae` (0.15),
- correlation of at least 0.3.

`docie-bench benchmark judge-calibration <file>` reports the verdict alone.
Failing any bar keeps the gate in warn mode: an untrustworthy judge can never
block a release.

## Benchmarking a routed pipeline

`--routing-policy` runs a dataset through a cascade of models instead of one,
so it can be compared against single-model baselines. It is exclusive with
`--model-profile`.

```bash
docie-bench benchmark run --dataset sample@1.0.0 --routing-policy configs/routing-policy.example.yaml
```

Each stage's `name` must match a profile in the models config. A routed run
makes one pass per document, labels its row `routed:<policy version>`, and
records the policy in the manifest. Summaries add acceptance, fallback,
escalation, stage failure, latency, tokens, cost and budget exhaustion. How
policies decide is in {ref}`architecture/extraction:routing policies`.

## Distributed runs

With `DATABASE_URL` set, `POST /v1/experiments` submits a run with deterministic
task keys, and any number of workers pull from it:

| | |
|---|---|
| `POST /v1/workers/tasks/claim` | take a task under a lease |
| `POST /v1/workers/tasks/{id}/heartbeat` | keep the lease |
| `POST /v1/workers/tasks/{id}/complete`, `/fail` | finish it |
| `POST /v1/workers/recover` | return expired tasks to the queue; claims do this too |

The lease token stops a worker that lost its lease from publishing a duplicate
result. Runs can be listed, cancelled and resumed under `/v1/experiments`.
Artifacts are written to content-addressed paths, so competing attempts at the
same task never overwrite each other.
