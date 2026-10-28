# Onboarding a model

Every model is served behind the same OpenAI-compatible surface, whatever runs
it. What makes that possible for a model the code has never seen is the
**family**. Onboarding a model is seeding its weights and tagging them with a
family; it is not writing per-model configuration.

## Families

A family (`FAMILIES` in `serving/model_store.py`) is the serving contract for a
class of models — everything the runtime and the extraction client need to
serve and prompt it faithfully:

| Part | Decides |
|---|---|
| `template_delivery` | how the schema reaches the model: `response_format`, `chat_template_kwargs`, in the prompt |
| `response_format_style` | how output is constrained: `openai_json_schema`, `nuextract3`, `none` for free text |
| `prompt_profile` | how the prompt is built |
| `llama_server_args` | launch flags such as `--jinja`, `--mmproj`, `--embedding` |
| capability flags | `vision`, `needs_mmproj`, `embedding`, `analyzer` |
| generation defaults | temperature, `max_tokens`, timeout |

The families today: `openai_chat`, `lfm2`, `lfm2_vl`, `lfm3`, `lfm3_vl`,
`spark2_5`, `nuextract_v1`, `nuextract3`, `vision_ocr`, `embedding`, `embeddinggemma2`,
`reranker`, `multi_vector`, `encoder_gliner`, `encoder_gliner2`,
`encoder_gliformer`, `transformers`, `transformers_trust_remote_code`.

### Same plumbing is not the same contract

A model reuses a family when its contract matches: same template delivery,
response shape, launch flags and modality. It needs a different one when the
contract differs, even if the plumbing is identical.

NuExtract3 and Unlimited-OCR are both GGUF vision models served by llama-server
with a projector. NuExtract3's template consumes `chat_template_kwargs` and
returns schema-shaped JSON. Unlimited-OCR's template ignores kwargs and returns
free text. On the `nuextract3` family, Unlimited-OCR would silently drop the
schema and have a JSON grammar forced onto OCR output; it belongs on
`vision_ocr`, whose `response_format_style` is `none`.

The same choice comes up at deploy time for any OCR-style vision model: a
schema-driven extractor wants `lfm2_vl`, a Markdown transcriber wants
`vision_ocr`. That is a family pick, not new code.

A genuinely new contract costs about ten lines and a test. That is the price of
faithful serving; guessing per model is the silent wrong output the contract
exists to prevent.

### Vision projectors

Vision families need a projector, and nothing model-specific is written for it:

1. A file is a projector when its name contains `mmproj` (`hf_hub.pick_mmproj`,
   largest on a tie).
2. Seeding a `needs_mmproj` family downloads it beside the model as
   `mmproj.gguf`, and refuses a repo that has none.
3. Launch appends `--mmproj <path>` (`ModelStore.family_launch_args`).

Use the projector from the same repo as the weights. Its output dimension must
match the exact checkpoint it was converted from, which same-repo pairing
guarantees and mixing repos does not.

## The serving ladder

Tried from the top; each rung is lighter and more faithful than the one below.

1. **GGUF on llama.cpp.** The default: a quantised GGUF on `llama-server`.
   Covers a model when a GGUF exists and llama.cpp supports its architecture.
2. **A new family.** llama.cpp supports the architecture, but no family matches
   its contract yet.
3. **transformers.** For a repo with no GGUF, or an architecture llama.cpp
   cannot load, `docie-serving transformers` serves the unquantised weights
   through `AutoProcessor` and `AutoModelForImageTextToText` or
   `AutoModelForCausalLM`. It needs no family contract; chat and vision both
   work generically.

### Why transformers is last

The image already carries torch and transformers for the encoders, so the
runtime costs almost nothing to ship. What it costs is memory and speed:
unquantised safetensors take roughly two to three times the RAM of a Q4 GGUF,
and CPU inference is markedly slower. It is a safety net, not a strategy, and
the code enforces that:

- `arch_registry.resolve_family` never sends a repo **with** a GGUF to
  transformers. Only a safetensors-only repo falls through, and its verdict
  carries a note recommending a GGUF repo of the same model if one exists.
- A checkpoint whose `config.json` has an `auto_map` needs
  `trust_remote_code`, which runs the repository's Python on the serving node.
  That is its own family, `transformers_trust_remote_code`, so it is an
  explicit choice at deploy and never a default.

A transformers model is stored as a safetensors snapshot, the same way encoders
are, so seeding, deploying, sizing and the fit check treat it like any other
store model.

### Runtime support is the real gate

Adding a family does not add runtime support. If llama-server cannot load the
architecture, the GGUF fails to load and the deployment's log says so. Newly
supported architectures carry a note in `RUNTIME_NOTES` naming the llama.cpp
change they need, because "supported" for those means "after rebuilding the
serving image".

### One architecture, several modalities

Every LFM3 GGUF reports the architecture `lfm3`: text, VL, Audio and Omni
alike. The arch string cannot tell them apart, so the projector and the repo's
Hub tags do:

| Repo | Family |
|---|---|
| `LiquidAI/LFM3-1.7B-GGUF` (no projector) | `lfm3` |
| `LiquidAI/LFM3-VL-2B-GGUF`, `LiquidAI/LFM3-Omni-2B-GGUF` | `lfm3_vl` |
| `LiquidAI/LFM3-Audio-2B-GGUF` | refused: its projector is an audio encoder |

A projector tagged `audio` with no image signal is refused rather than
promoted to a vision family. LFM3 needs a llama-server built after
ggml-org/llama.cpp#31252.

## Preflight

`GET /v1/studio/hf/inspect?repo=…` — the Studio's **Add model** inspection —
decides support before anything downloads.

It reads only metadata. A GGUF's header carries `general.architecture` and the
chat template, which the Hub's model-info API already exposes. A safetensors
repo's `config.json` carries `model_type` and `architectures`. Quant
availability, projector presence and multipart files come from the file list.

The architecture is mapped to a family through `ARCH_TO_FAMILY` in
`serving/arch_registry.py`. A text architecture with a projector present is
promoted to a vision family through `TEXT_ARCH_TO_VISION_FAMILY`, defaulting to
`lfm2_vl`.

The result:

- the architecture, modality, matched family and runtime;
- a verdict — `supported`, `needs_family` (llama.cpp knows the architecture but
  no contract exists), or `unsupported`;
- the preferred quant and every alternative, each with its own download size,
  predicted memory at the requested context (8192 by default) and fit against
  the node's live budget;
- the exact files that will be downloaded, projector included;
- blockers, warnings and recommendations.

Multipart-only repos, missing projectors, unknown families and unsupported
repos are stopped before the download. With no node snapshot published yet,
capacity is reported as unknown, never as a fit.

## Pitfalls

**The name is not the architecture.** `ATH-MaaS/OvisOCR2` shares a brand with
AIDC-AI's custom-code Ovis models, but its `config.json` says
`Qwen3_5ForConditionalGeneration` with no `auto_map`: a native Qwen3.5-VL
fine-tune. Its GGUF reports `qwen35`, which the registry already promotes to a
vision family when a projector is present. It needed no new code. Read
`config.json` or the GGUF header, not the model card's title.

**Check the primary source.** Whether llama.cpp supports an architecture is
answered by the pull request that added it and its own tests, not by a summary
of an issue thread. An open-looking issue can turn out to be a usage error, and
an early reverted attempt can look like the final word.
