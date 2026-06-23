# docie-bench — CLI Tutorial

A guided tour of the command line for **serving models**, **adding model profiles**, and
**benchmarking document information-extraction (IE)** on small / local models.

This is the script behind the demo. Everything in the **Core walkthrough** has been run
end-to-end on a CPU-only Windows box with Ollama; the **Serving control plane** section is
the scaling architecture and is marked where you should verify on your own machine.

---

## 0. The mental model (30 seconds)

There are two CLIs, with two different jobs:

| Entrypoint | Job | Status |
|---|---|---|
| `docie-bench` | Run the benchmark: schemas, datasets, model profiles, extraction, scoring, comparison | **The product.** Fully working. |
| `docie` / `docie-serving` | Control plane: inspect runtimes, plan & operate local model deployments | **Architecture / scaling layer.** Inspect commands work today; full lifecycle is roadmap. |

The model itself is served by a **runtime**. We integrate **four** runtime adapters
(`serving/runtime.py`), and **two of them run locally today**:

- **Ollama** — OpenAI-compatible endpoint at `http://localhost:11434/v1`. Easiest to drive
  (`ollama pull` does acquisition for you).
- **llama.cpp** — our adapter launches the real `llama-server` binary against a **GGUF** file,
  exposing an OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`. This is the path for
  any GGUF we build/quantize ourselves (e.g. the fable-5 distilled models), without Ollama.
- **vLLM** and **remote** — also wired; vLLM needs the `vllm` executable (GPU/Linux), remote
  points at an existing HTTP endpoint.

`docie-bench` is runtime-agnostic: it just needs an OpenAI-compatible `base_url`, whether that
is Ollama's `:11434`, llama.cpp's `:8000`, or anything else. The `docie` control plane is the
single operations API over all four runtimes — it probes them, plans deployments against host
resources, and supervises the launched process (start / health-check / restart / stop).

```
            ┌─────────────────────────────────────────────┐
            │  docie-bench  (benchmark)                     │
            │  schema → dataset → profile → extract → score │
            └───────────────┬─────────────────────────────┘
                            │ OpenAI-compatible HTTP (any base_url)
            ┌───────────────┼───────────────┬───────────────┐
            ▼               ▼               ▼               ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  Ollama    │  │ llama.cpp  │  │   vLLM     │  │  remote    │
     │ :11434/v1  │  │  :8000/v1  │  │ (gpu/linux)│  │  http(s)   │
     │ (working)  │  │ (working)  │  │ (roadmap)  │  │ (config)   │
     └────────────┘  └────────────┘  └────────────┘  └────────────┘
            ▲               ▲               ▲               ▲
            └───────────────┴───────┬───────┴───────────────┘
                                    │  managed by
                          ┌─────────────────────────┐
                          │ docie  (control plane)   │
                          │ runtime · plan · serve   │
                          └─────────────────────────┘
```

---

## 1. Setup

```powershell
# From the repo root, with the venv active:
docie-bench --help
docie --help          # alias: docie-serving --help
```

Make sure Ollama is up and has at least one model pulled:

```powershell
ollama list
# if empty, pull a small one (fast on CPU):
ollama pull qwen2.5:1.5b
```

---

## 2. Core walkthrough — the benchmark

This is the spine of the demo. Each step produces real, show-able output.

### 2.1 See what schemas exist

A *schema* is the structured target — the fields we want extracted from a document.

```powershell
docie-bench schema list
docie-bench schema show invoice
# show several at once:
docie-bench schema show invoice identity_card
```

`schema show` prints the JSON Schema (field names, types, evidence requirements). This is the
contract every model is graded against.

### 2.2 Validate a dataset

A *dataset* is a manifest (`.jsonl`) where each row is one document + its ground-truth fields.
Validate before you trust it — this catches missing files, bad hashes, and near-duplicate
leakage between splits.

```powershell
docie-bench dataset validate data\sample_dataset\manifest.jsonl
docie-bench dataset inspect  data\sample_dataset\manifest.jsonl
```

`validate` exits non-zero if the dataset is broken — good for CI. `inspect` adds the resolved
path and per-document statistics. There's also `dataset leakage` (train/test contamination
check) and `dataset version` (register an immutable, content-hashed version in the registry).

### 2.3 Add a model profile  ←  "how to add a model"

A *profile* in `configs/models.yaml` tells the benchmark how to call a model: endpoint,
response format, prompt style, vision on/off. You don't hand-write these — `models add`
generates a working one by auto-detecting the model's capabilities from Ollama.

```powershell
# Auto-detect everything (vision, format) from an already-pulled Ollama model:
docie-bench models add qwen2.5:1.5b --name demo_qwen

# A vision model — capability is auto-detected via Ollama's /api/show:
docie-bench models add gemma4:e2b

# NuExtract is special-cased by name (no JSON wrapper, its own stop token, its prompt):
docie-bench models add hf.co/numind/NuExtract:Q4_K_M
```

What it does under the hood (`llm/model_catalog.py`):
- Calls Ollama `/api/show` to detect **vision** capability and model family.
- Picks a **response format**: `json_object` universal default (works across Ollama models);
  `none` for NuExtract (it emits structure natively). Override with `--response-format`.
- Picks a **prompt profile**: `strict_extraction_v1`, or `nuextract_v1` for NuExtract.
- Appends the block to `models.yaml` **preserving your comments**, then reloads to verify it
  parses — if the append would corrupt the file, it reverts and errors.

If Ollama is unreachable it warns, assumes text-only, and tells you to `ollama pull` first.

Confirm what's configured:

```powershell
docie-bench models list
# demo_qwen  [json_object]  qwen2.5:1.5b
# ollama_gemma4_e2b  [vision, json_object]  gemma4:e2b
# ...
```

### 2.4 Run the benchmark

Point it at a dataset + a profile. Output is an immutable, content-hashed run directory.

```powershell
docie-bench benchmark run `
  --dataset data\sample_dataset\manifest.jsonl `
  --model-profile demo_qwen
```

It prints four artifacts:
- **Predictions** — per-document model output, with evidence IDs.
- **Metrics** — field accuracy, evidence coverage, latency.
- **Report** — human-readable summary.
- **Manifest** — the immutable, secret-sanitized record of exactly what ran (inputs, hashes,
  concurrency) so any run is reproducible.

Useful flags:
- `--concurrency N` — parallel documents (1–32).
- `--repeat N` — replay the dataset N times for stress / latency stability.
- `--document <file> --eval-mode llm_judge` — one-off extraction on a single file, scored by a judge model instead of ground truth.
- `--routing-policy <yaml>` — run each doc through the **multi-stage router** (e.g. cheap model first, escalate on low confidence) instead of a single profile. Stage names map to profiles in `models.yaml`. Mutually exclusive with `--model-profile`.
- `--resume` (with `--output-dir`) — continue an interrupted run; it repairs a partial tail and only executes the missing tasks, refusing to proceed if inputs drifted.

### 2.5 Compare two runs

The benchmark is only useful if you can say "B is better than A." `compare` does that with a
verdict and a budget gate.

```powershell
# Promote a trusted run to a named baseline:
docie-bench benchmark baseline promote <run_dir> demo_baseline

# Compare a candidate against it (exits non-zero if it regresses past budget):
docie-bench benchmark compare demo_baseline <candidate_run_dir> `
  --budgets configs\regression-budgets.yaml
```

`compare` prints **PASS/FAIL**, a verdict file, and a diff report. This is the CI gate: a new
model or prompt has to beat the baseline within the regression budget to land.

### 2.6 (Optional) OCR backend comparison

For scanned/PDF inputs you can benchmark OCR backends independently of extraction:

```powershell
docie-bench benchmark ocr run --dataset <dataset> --backend pdf_text --backend tesseract
```

---

## 3. Serving — the unified GGUF store

This is the serving headline, and it's **on master** (PR #45, tested). The problem it solves:
Ollama and llama.cpp can run the *same* GGUF weights, but they cache them differently and they
do **not** honour the same template/vision mechanisms. So we keep **one canonical store** of
real `*.gguf` files and serve from it with either runtime — correctly.

```
<store-root>/<name>/model.gguf      # weights (one copy, ever)
<store-root>/<name>/mmproj.gguf     # optional vision projector
<store-root>/index.json             # name -> {family, paths, source}
```

Why one store mattered — the two things the runtimes disagree on (this is *why* we unified):

| | llama.cpp `llama-server` | Ollama |
|---|---|---|
| **GGUF path** | any `*.gguf` file | opaque `blobs/sha256-…` (no extension) |
| **Structured output** (`chat_template_kwargs`) | ✅ with `--jinja` | ❌ silently dropped |
| **Vision** projector | explicit `--mmproj proj.gguf` | bundled in the model only |

A **`FamilyContract`** records how each model family must be served and prompted, so a model is
always served *correctly*, not just loaded:

| family | template delivery | llama-server flags | Ollama-faithful? |
|---|---|---|---|
| `nuextract3` | `chat_template_kwargs` | `--jinja` (+ `--mmproj` for vision) | **no** — Ollama drops the kwargs |
| `nuextract_v1` | baked into the prompt | – | yes |
| `openai_chat` | OpenAI `response_format` | – | yes |

`ollama_modelfile()` **refuses** families Ollama can't serve faithfully — so you physically
can't deploy a NuExtract3 that ignores its template.

### 3.1 Seed once, serve either way  (the NuExtract3 path — verified)

```powershell
# 1. Pull the GGUF ONCE via Ollama (downloads weights + vision projector)
ollama pull hf.co/numind/NuExtract3-GGUF:Q4_K_M

# 2. Seed the canonical store from that Ollama model — hard-links the blobs, no re-download
$env:PYTHONPATH = "src"
python - <<'PY'
from pathlib import Path
from docie_bench.serving.model_store import ModelStore
store = ModelStore(Path.home() / ".local/share/docie-bench/serving/models")
entry = store.seed_from_ollama(
    "hf.co/numind/NuExtract3-GGUF:Q4_K_M", name="nuextract3", family="nuextract3"
)
print("model :", entry.model_path)
print("mmproj:", entry.mmproj_path)
print("serve :", " ".join(store.llama_server_command("nuextract3", port=8088)))
PY

# 3. Serve it with llama-server (the printed command adds --jinja + --mmproj from the contract).
#    Pick a FREE port. Ollama serves the SAME weights with no second download.
llama-server --model "...\models\nuextract3\model.gguf" --mmproj "...\models\nuextract3\mmproj.gguf" `
             --alias nuextract3 --jinja --host 127.0.0.1 --port 8088 -c 8192

# 4. Confirm it's really llama-server (JSON, not HTML):
Invoke-RestMethod "http://127.0.0.1:8088/health"     # -> status : ok
```

`configs/models.yaml` ships `nuextract3` (reasoning off) and `nuextract3_think` (reasoning on)
pointing at `http://localhost:8088/v1`, so you go straight to:

```powershell
docie-bench benchmark run --dataset data\sample_dataset\manifest.jsonl --model-profile nuextract3
```

> Need `llama-server`? The serving README ships a one-liner to grab the latest CPU build:
> `src/docie_bench/serving/README.md`.

### 3.2 The control plane (`docie`) — operate runtimes

Above the store sits one operations API over all four runtimes — **vLLM, llama.cpp, Ollama,
remote**. `docie-bench` stays runtime-agnostic; it only needs an OpenAI-compatible `base_url`.

#### Inspect runtimes (works today)

```powershell
docie runtime list                 # all adapters + whether each is available on this host
docie runtime probe ollama         # detailed probe of one runtime
docie --json runtime list          # stable JSON for automation
```

`runtime list` shows the multi-runtime design concretely: it detects which engines are
installed on the machine (e.g. `ollama` and `llama-server` present, `vllm` not). This is the
strongest serving visual that needs no registry state.

#### Plan / serve / supervise

```powershell
docie serve <gguf-path> --runtime llamacpp --name inv-extractor   # explicit runtime: real launch
docie list                         # managed deployments + state
docie status inv-extractor         # health / endpoint / restarts
docie stop inv-extractor
docie plan  <model> --replicas 1   # resource planner (needs the model in the local registry)
```

Two distinct paths, important for the demo:
- **`docie serve <model> --runtime llamacpp|ollama` (explicit runtime)** *bypasses the
  registry* and launches the process directly via the supervisor (start → health-check →
  restart-on-failure → stop, state persisted to JSON). This is a **real** launch path.
- **`docie serve <model>` / `docie plan <model>` (no `--runtime`)** ask the resource planner to
  choose, which resolves the model through the local registry. On a fresh box that registry is
  empty, so the *auto-plan* path is roadmap until a model is registered. `docie model pull`
  also expects a manifest-JSON path, not an Ollama identity.

#### Verification block — run this before the demo

Hand this to whoever drives the terminal; it tells us definitively which serving commands are
"live demo" vs "slide":

```powershell
docie runtime list
docie runtime probe ollama
docie runtime probe llamacpp
docie list
docie plan qwen2.5:1.5b          # may error (empty registry) — note the message
docie model pull qwen2.5:1.5b    # expected to error (manifest-path backend) — confirm it
```

Whatever errors here stays in the *architecture* part of the demo; the unified-store path in
§3.1 and `runtime list/probe` are the *live* part.

---

## 4. The 5-minute demo arc (draft — we'll tighten this next)

1. **Frame (30s):** "Benchmarking IE for small/local models. Two pieces: the benchmark, and the serving layer that scales it."
2. **Schema (30s):** `schema show invoice` — "this is the contract."
3. **Add a model (60s):** `ollama list` → `models add qwen2.5:1.5b` → `models list`. "One command turns any pulled model into a graded competitor."
4. **Run (90s):** `benchmark run` on the sample dataset — open the report. "Field accuracy, evidence coverage, latency, fully reproducible manifest."
5. **Compare (60s):** `benchmark compare` against a baseline — show the PASS/FAIL gate. "This is how a new model earns its way in."
6. **Serve & scale (90s):** `docie runtime list` (multi-runtime), then the unified-store
   punchline — "we keep **one** GGUF and serve it via Ollama *or* llama.cpp; family contracts
   guarantee vision (`--mmproj`) and structured output (`--jinja chat_template_kwargs`) are
   delivered correctly. NuExtract3 runs here." Show `nuextract3` extracting against the
   `:8088` llama-server endpoint if it's already up, else show the `seed_from_ollama` → serve
   flow from §3.1.

---

## Appendix — command index

| Area | Command |
|---|---|
| Schemas | `docie-bench schema list` · `schema show <names...>` |
| Datasets | `dataset validate` · `inspect` · `leakage` · `version` · `migrate` |
| Models | `docie-bench models add <model>` · `models list` |
| Benchmark | `benchmark run` · `benchmark compare` · `benchmark baseline promote/list` · `benchmark ocr run` |
| Serving (inspect) | `docie runtime list` · `runtime probe <name>` · `docie list` |
| Serving (lifecycle) | `docie plan` · `serve` · `start` · `stop` · `status` · `model pull/show/remove` |
