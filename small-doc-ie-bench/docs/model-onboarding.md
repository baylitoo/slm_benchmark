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

### Case study: OvisOCR2 — a rung-1 model, two wrong turns before getting there

`ATH-MaaS/OvisOCR2` (0.8B, image-text-to-text OCR/document-parsing) turned
out to be exactly what onboarding should be: seed the weights, tag the right
family, done — rung 1, zero new code. Getting there took two wrong turns
first, both worth recording because the traps generalize.

**Wrong turn 1: the name "Ovis" is not the architecture.** AIDC-AI ships an
unrelated model family also branded "Ovis" (Ovis2/Ovis2.5) that genuinely is
custom-code (a bespoke vision tower, needing `trust_remote_code=True`,
tracked as still-not-upstreamed at
[huggingface/transformers#36824](https://github.com/huggingface/transformers/issues/36824)).
Reasoning from the product name alone wrongly concluded OvisOCR2 shared that
lineage. It doesn't: `ATH-MaaS/OvisOCR2`'s actual `config.json` (fetched
directly) reads `"architectures": ["Qwen3_5ForConditionalGeneration"]`,
`"model_type": "qwen3_5"`, with a native `vision_config` and no `auto_map` —
this is ATH-MaaS's own fine-tune of Qwen's own native Qwen3.5-VL, unrelated
to AIDC-AI's Ovis. Lesson: **the repo's `config.json` is ground truth; a
model's marketing name is not.**

**Wrong turn 2: trusting a research summary over the primary source.** Having
corrected wrong turn 1, a research pass concluded llama.cpp had *no* vision
support for Qwen3.5 at all — citing PR #19435 (an early, reverted, text-only
attempt) and issue #19917 as an open "image input not supported" gap. Both
citations were stale or misread. The actual PR that landed,
[ggml-org/llama.cpp#19468](https://github.com/ggml-org/llama.cpp/pull/19468)
("[MODEL] support qwen3.5 series", merged 2026-02-10), explicitly **includes
and tests vision** ("This pr includes the vision part and I test it too!
qwen3.5 uses the same vit as qwen3vl"). Issue
[#19917](https://github.com/ggml-org/llama.cpp/issues/19917) is **closed**
and reports exactly the standard hint text llama-server prints when
`--mmproj` is missing — a usage error, not a missing feature. The one real
correctness bug in this arch's history,
[#19683](https://github.com/ggml-org/llama.cpp/issues/19683), is CUDA-only
and MoE-only (closed; CPU inference and the dense variant were never
affected) — irrelevant to a CPU-serving dense 0.8B model. Lesson: **verify
against the PR/issue text itself, not a paraphrase of it** — a summary can
carry forward a stale or wrong conclusion just as easily as a name can.

**What's actually true, verified against the repo and the family contracts
directly:**

- `bartowski/ATH-MaaS_OvisOCR2-GGUF` ships a matching `mmproj-*.gguf`
  alongside its quantizations — a real, working GGUF+projector pair for this
  exact 0.8B checkpoint (not a differently-sized variant; the projector's
  output dimension must match the exact checkpoint it was converted from,
  which same-repo pairing guarantees and cross-repo mixing does not).
- That GGUF's `general.architecture` is `"qwen35"` — a string already in
  `ARCH_TO_FAMILY` (as `openai_chat`, text). The existing mmproj-upgrade path
  (`TEXT_ARCH_TO_VISION_FAMILY`) already promotes it to a vision family the
  moment a projector is present, with **zero new detection code** — this is
  the exact mechanism the doc's opening philosophy describes.
- The mmproj-upgrade default, `lfm2_vl`, forces
  `response_format_style="openai_json_schema"` — right for a schema-driven
  extractor, wrong for a free-text Markdown OCR model. OvisOCR2 wants
  `vision_ocr` (`response_format_style="none"`) instead, the exact same
  extraction-vs-OCR choice the NuExtract3/Unlimited-OCR case study above
  describes — an operator's family pick at deploy, not a new family to
  write.

**What shipped in the PR that came out of this investigation:**

- `RUNTIME_NOTES["qwen35"]` — the arch is new enough (PR #19468, 2026-02-10)
  that a `supported` verdict deserves the same "rebuild if it won't load"
  honesty every other recently-landed arch in this table gets, plus the
  same-checkpoint-projector caveat above.
- Two general fixes to the (for THIS model, unnecessary, but still real)
  transformers last-resort path, found while chasing wrong turn 1 down
  before it was corrected: `HfTransformersBackend` inferred vision
  capability from *which* AutoModel constructor happened to succeed rather
  than from the loaded processor's own shape (`_processor_is_multimodal`:
  does it carry an `image_processor`) — a real misclassification risk for
  any checkpoint (e.g. AIDC-AI's actual Ovis) whose `auto_map` registers
  only under `AutoModelForCausalLM`. And the chat-completions route only
  caught `ValueError` from a backend failure; any other exception now comes
  back OpenAI-shaped instead of an unhandled 500.

**What is NOT verified here.** No environment with the real ~1.5GB+ GGUF and
mmproj pair was available to actually run llama-server against them and
confirm real OCR output quality end to end. Everything above is verified
against primary sources (the PR/issue text themselves, the repo's actual
`config.json`, the actual GGUF metadata, and this codebase's own family
contracts) — which is a materially stronger basis than the two wrong turns
that preceded it, but "the metadata lines up" is still not the same claim as
"confirmed working," and this doc will not conflate them.

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
