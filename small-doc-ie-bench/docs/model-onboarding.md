# Onboarding a model — process, philosophy, and the road to auto-detection

## The unification philosophy

The framework exposes **one OpenAI-compatible surface** over every model, whatever
the runtime underneath (llama.cpp / Ollama / vLLM / the encoder shim / a remote
endpoint). A client points `base_url` at one place and stops caring how a model
is served. The abstraction that makes this work is the **family**.

A **family** (`docie_bench.serving.model_store.FAMILIES`) is the *serving
contract* for a class of models — everything the runtime and the extraction
client need to serve and prompt that class faithfully:

- `template_delivery` — how the schema/prompt reaches the model (OpenAI
  `response_format`, `chat_template_kwargs`, in-prompt, …).
- `response_format_style` — how a response is constrained (`openai_json_schema`,
  `nuextract3`, `none` for free text…).
- `prompt_profile` — how the prompt is built.
- `llama_server_args` — launch flags (`--jinja`, `--mmproj`, `--embedding`…).
- capability flags — `vision`, `needs_mmproj`, `embedding`, `analyzer`.
- generation defaults — temperature / max_tokens / timeout.

A family is the single source of truth: "drop the weights + pick a family" is the
whole onboarding when a suitable family already exists. Adding a model is **not**
editing a per-model config; it is seeding weights and tagging them with a family.

## When a new model "just works" vs needs a new family

A model reuses an existing family when its **serving contract matches** — same
template delivery, same response shape, same launch flags, same modality. It
needs a **new family** when the contract differs even if the plumbing overlaps.

### Case study: NuExtract3 vs Unlimited-OCR (both GGUF + vision + mmproj)

Both are GGUF vision models served by llama-server with a `--mmproj` projector.
They share the *plumbing*. They do **not** share the *contract*:

| Aspect | NuExtract3 | Unlimited-OCR (`deepseek2-ocr`) |
|---|---|---|
| Base arch | Qwen2.5-VL | DeepSeek-OCR |
| Template | rich jinja, consumes `chat_template_kwargs` | trivial (`{% for m in messages %}{{m.content}}{% endfor %}`) |
| Task | schema-driven structured extraction | free-text OCR |
| `template_delivery` | `chat_template_kwargs` | must NOT be — the template ignores kwargs |
| `response_format_style` | `nuextract3` (JSON) | `none` (plain text; a JSON grammar would corrupt OCR) |

Putting Unlimited-OCR on the `nuextract3` family would silently drop the schema
(the trivial template ignores the kwargs) and force JSON on a free-text OCR
output. So it needs its own family — sharing the vision/mmproj plumbing
(`vision=True`, `needs_mmproj=True`, `--jinja`) but with `response_format_style
= none`. Roughly:

```python
"deepseek_ocr": FamilyContract(
    template_delivery=TemplateDelivery.OPENAI_JSON_SCHEMA,  # unused for free text
    response_format_style="none",
    prompt_profile="strict_extraction_v1",
    llama_server_args=("--jinja",),
    needs_mmproj=True, vision=True,
    default_max_tokens=4096, default_timeout_seconds=600,
    ollama_faithful=False,
),
```

**The overhead this documents:** most models drop onto an existing family with
zero code; a genuinely new architecture/template contract costs ~10 lines + a
test to add a family. That overhead is the price of faithful serving — the
alternative (guessing per model) is exactly the silent-wrong-output trap the
family contract exists to prevent.

### How the mmproj (vision projector) is handled — no per-model work

Vision families need a projector; the framework finds and wires it by filename:

1. **Detect** — `hf_hub._gguf_from_sibling`: `is_mmproj = "mmproj" in filename.lower()`.
   `pick_mmproj(files)` returns the projector (largest on ties).
2. **Auto-download at seed** — `_run_seed_hf`: for a `needs_mmproj` family the
   projector is downloaded beside the model and stored as `mmproj.gguf`; a
   `needs_mmproj` family with no projector in the repo is refused loudly.
3. **At launch** — `family_launch_args`: appends `--mmproj <path>` when the
   family needs it and a projector is present.

So for any `needs_mmproj` family, the projector is detected, downloaded and
passed automatically — nothing model-specific to write.

### The real gate: does the runtime support the architecture?

Independent of our code: `llama-server` must support the model's architecture
(and its clip/mmproj). A brand-new arch (`deepseek2-ocr`) requires a recent
llama.cpp build. If unsupported, the GGUF refuses to load — visible in the
deployment's runtime log. Adding a family does not add runtime support.

## The serving ladder — and the last-resort transformers runtime

Onboarding is a ladder, tried top-down. Each rung is lighter and more faithful
than the one below; you drop to the next only when the one above can't serve the
model **from this repo**:

1. **GGUF + llama.cpp** — the preferred path. A quantized GGUF served by
   `llama-server` (`--jinja`, `--mmproj`, `--embedding` as the family needs).
   Light, fast, portable. Covers a model when a GGUF exists *and* llama.cpp
   supports the arch.
2. **Add a family** (`~10 lines + a test`) — llama.cpp already supports the
   arch, but its serving *contract* (template/response shape) has no family yet.
   The NuExtract3-vs-Unlimited-OCR case study above is exactly this rung.
3. **transformers / AutoModel** — the **last resort**. A model with **no GGUF in
   its repo** (or an arch llama.cpp cannot load) is served directly from
   unquantized `transformers` weights by the `docie transformers` shim
   (`transformers_server/server.py`, `RuntimeKind.TRANSFORMERS`), the exact
   mirror of the encoder shim. `AutoProcessor` + `AutoModelForImageTextToText` /
   `AutoModelForCausalLM` load the checkpoint, its chat template and its
   (multimodal) processor from the repo — so onboarding needs **no per-model
   family contract at all**. Chat and vision both work generically.

### Why rung 3 is deliberately last

This is the tradeoff [LocalAI](https://localai.io/blog/why-we-write-our-own-engines/)
frames precisely. Their objection to wrapping heavyweight Python engines is
*deployment friction* — "a multi-gigabyte Python install we could not ship."
**That objection does not bite us**: the serving image already carries
torch + transformers for the GLiNER encoders (`PIP_EXTRAS=ocr,encoders`), so the
transformers runtime costs ~zero marginal image size. What *does* bite us is
their second axis — **runtime memory and speed**: unquantized safetensors use
**~2-3x the RAM** of a GGUF Q4 and CPU inference is markedly slower. So the
transformers runtime is a safety net, never the strategy: the real fix for the
onboarding bottleneck is widening rungs 1-2 (GGUF discovery + arch-map breadth),
not leaning on the escape hatch.

The last-resort posture is enforced, not just advised:

- **The "no servable GGUF" gate** (`arch_registry.resolve_family`): a repo with a
  GGUF is *never* routed to transformers. Only a safetensors-only repo falls to
  the `transformers` family — and even then the verdict carries a `runtime_note`
  memory disclaimer ("prefer a GGUF repo of this model if one exists") that the
  Studio renders in the same amber caveat as a runtime gate.
- **`trust_remote_code` is a separate, explicit family.** Native-arch checkpoints
  load with zero custom code. A custom-code checkpoint (`config.json` `auto_map`
  — e.g. UnlimitedOCR/DeepSeek-OCR) needs `trust_remote_code=True`, which
  executes the repo's Python **on the serving node**. That trust is a *distinct*
  family (`transformers_trust_remote_code`), so it is an auditable choice at
  deploy, never a default.

Storage/lifecycle-wise, a transformers model is a **safetensors snapshot** in the
store — the same directory-entry path the encoders already use — so seed,
deploy, sizing and the fit gate all treat it like any other store model.

### Case study: OvisOCR2 — a rung-1-shaped model that is actually rung-3-only

`ATH-MaaS/OvisOCR2` (0.9B, image-text-to-text OCR/document-parsing) looks like
a rung-1 (GGUF + llama.cpp) candidate: it's small, its base is
`Qwen/Qwen3.5-0.8B`, and its own Hub page lists "Quantizations (21 models)".
It is not. This is the failure mode the ladder doesn't automatically catch —
worth naming so it doesn't repeat.

**Why the GGUF path doesn't work.** Qwen3.5 is a *hybrid* architecture (same
family as Qwen3-Next): most layers are Gated DeltaNet (linear attention — a
fixed-size recurrent state, not a growing KV cache) with a minority of full
softmax-attention layers for global retrieval. llama.cpp's Qwen3.5 support
(`LLM_ARCH_QWEN35`/`QWEN35MOE`, [ggml-org/llama.cpp#19435](https://github.com/ggml-org/llama.cpp/pull/19435),
merged/reverted/re-merged as #19453/#19468) is, per the author, **text only —
no multimodal capability**, and is new enough to still have an open GPU
correctness bug ([#19683](https://github.com/ggml-org/llama.cpp/issues/19683):
degenerate output on CUDA, correct on CPU). OvisOCR2's actual point — the
Ovis vision stack (an AIMv2 vision transformer + a custom Visual Embedding
Table bolted onto the Qwen3.5 backbone) — has **no llama.cpp/mtmd
implementation at all**
([#19917 "Qwen3.5 image input is not supported"](https://github.com/ggml-org/llama.cpp/issues/19917),
corroborated by [ollama#15898](https://github.com/ollama/ollama/issues/15898)).
So every one of those 21 community GGUF quantizations, however it loads in
llama-server, can only be the **text backbone** — a chat model, never an OCR
model. `arch_registry`'s `RUNTIME_NOTES` pattern (a known-arch caveat keyed by
GGUF `general.architecture`) is exactly the mechanism for surfacing this once
a real GGUF of this repo is inspected; nothing has been added preemptively
for an arch string ("ovis"/"ovis2_5"/whatever this converts to) nobody has
confirmed yet — that's a `RUNTIME_NOTES` entry to add the day a real one shows
up, not now on inference.

**Why "add a family" (rung 2) doesn't apply.** Rung 2 assumes the runtime can
already load the architecture and only the serving *contract* is missing. No
family contract fixes a missing ggml kernel — the vision stack literally
cannot be loaded by any llama-server build today.

**Why rung 3 (transformers, `trust_remote_code`) is the answer — and costs
zero new family code.** Ovis models bolt on custom modeling code (the AIMv2
tower + Visual Embedding Table) that isn't upstreamed into `transformers`
(tracked, still open, as
[huggingface/transformers#36824](https://github.com/huggingface/transformers/issues/36824)).
That is precisely what the existing `transformers_trust_remote_code` family
exists for — no per-model contract, the pre-flight inspector already detects
`config.json`'s `auto_map` and gates deploy behind an explicit
`remote_code_approval_required` blocker (see "Pre-flight support detection"
below). Onboarding this model is "pick `transformers_trust_remote_code` at
deploy," not writing a family.

**The real gap this surfaced, and what got fixed.** The generic transformers
shim (`transformers_server/server.py`) inferred vision capability from
*which* AutoModel constructor happened to succeed — `AutoModelForImageTextToText`
first, `AutoModelForCausalLM` as the fallback. Ovis's known registration
pattern (predating the image-text-to-text auto class) registers its
`auto_map` only under `AutoModelForCausalLM`, so a load would silently land
in the "text-only" branch despite being genuinely multimodal. Fixed by
deriving `vision` from the loaded **processor's own shape**
(`_processor_is_multimodal`: does it carry an `image_processor`, the
standard `ProcessorMixin` composition every VLM processor uses regardless of
which AutoModel class loaded it) instead of from load-path success — a fix
that generalizes to any future custom-code VLM in this exact shape, not just
Ovis. Separately, the shim's chat-completions route caught only `ValueError`
from a backend failure; anything else (an `AttributeError`/`TypeError` from a
custom model whose calling convention diverges from the standard
`apply_chat_template(images=...)` contract this shim assumes generically —
a real risk specifically for last-resort onboarding) reached the caller as an
unhandled 500 with no actionable message. Now any backend exception comes
back OpenAI-shaped, with the real error text.

**What is NOT verified here.** No GPU/large-download environment was
available to actually load `ATH-MaaS/OvisOCR2`'s real weights end to end.
Whether Ovis's specific custom code implements the standard
`apply_chat_template(images=...)` contract at all (rather than a bespoke
`.chat()`/`preprocess_inputs()` entry point, a real and recurring pattern for
older custom-code VLM repos) is unconfirmed — if it doesn't, the two fixes
above turn that failure into a clear error instead of a crash, but do not by
themselves make generation work; that would need a per-model adapter, which
is out of scope until someone confirms it's actually needed against real
weights.

## Pre-flight support detection

Studio provides **HuggingFace-like browsing with a deploy decision that already
knows whether we support the model**. `GET /v1/studio/hf/inspect?repo=…` reads
Hub metadata without downloading weights and returns both the architecture
verdict and an actionable deployment plan.

### What we can read WITHOUT downloading the model

- **GGUF repos**: the GGUF header carries `general.architecture` and the chat
  template as KV metadata at the START of the file. The HF model-info API
  already parses and exposes this (the `gguf` block: architecture, chat
  template, context length) — so a single API call yields the architecture. A
  self-hosted fallback is an HTTP Range read of the first ~1 MB of the `.gguf`
  to parse the header locally.
- **Transformers/safetensors repos** (encoders, base models): `config.json`
  carries `model_type` and `architectures` — one raw-file fetch.
- Plus what we already parse: quant availability, mmproj presence, multipart.

### The architecture registry (replaces guessing, not models.yaml)

`models.yaml` is *deployment routing*, a different concern. What this needs is a
mapping **architecture → family** — a supported-architecture library:

```
qwen2 / llama / mistral         → openai_chat
qwen2_vl (+ mmproj)             → lfm2_vl-class vision-extraction
gliner / gliner2                → encoder_gliner / encoder_gliner2
bert-family embeddings          → embedding
deepseek2-ocr (+ mmproj)        → deepseek_ocr        (once the family exists)
<unknown arch>                  → unsupported (needs a new family, or the
                                   runtime can't serve it)
```

A model card / config-reading layer is only needed for the long tail; the
architecture string from GGUF metadata or `config.json` is the primary signal
and is machine-readable.

### The support verdict + Deploy UX

A pre-flight inspect returns:

- the detected architecture + modality (vision? embedding? encoder?),
- the matched family (or "no family — needs onboarding"),
- the concrete runtime (`llama.cpp`, `transformers`, or `encoder`),
- a verdict: **supported** (deploy now), **needs-family** (arch known to
  llama.cpp but no contract yet — a small PR), **unsupported** (runtime can't
  serve it),
- the preferred quant plus every selectable single-file artifact,
- the exact known download set, including a required vision projector or all
  safetensors snapshot files,
- predicted RAM at the requested deployment context (8192 by default),
- fit against the serving node's live deployable budget,
- structured blockers, warnings, and recommendations.

Each quant carries its own download, memory, and fit estimate. If the preferred
quant does not fit but a smaller one does, selecting it immediately clears the
memory blocker. Multipart-only repos, missing projectors, unknown families, and
unsupported repositories are stopped before a multi-gigabyte download begins.
Capacity stays explicitly unknown when the serving reconciler has not published
a node snapshot; preflight never turns a missing measurement into a false fit.
