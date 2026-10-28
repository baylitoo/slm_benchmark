# Model profiles

A profile in `configs/models.yaml` is a named answer to "how do I call this
model": an endpoint, a model id, and how to shape requests to it. Benchmarks
select models by profile name. A Studio deployment produces an equivalent
profile from its family contract, so this file is for endpoints the serving
layer does not manage — a standalone llama.cpp, vLLM, Ollama, or a remote API.

```yaml
profiles:
  lfm25_350m_cpu:
    model: lfm2.5-350m
    base_url: http://llm-llamacpp:8000/v1
    api_key_env: OPENAI_COMPAT_API_KEY
    response_format_style: openai_json_schema
    prompt_profile: strict_extraction_v1
    temperature: 0.0
    top_p: 1.0
    max_tokens: 900
    timeout_seconds: 180
    capability_discovery: optional
    max_concurrency: 2
    queue_limit: 16
    queue_timeout_seconds: 30
    retry_max_attempts: 3
    circuit_breaker_failure_threshold: 5
    circuit_breaker_reset_seconds: 30
```

`docie-bench models add <ollama-name or hf.co/repo:quant>` appends a profile
for a model Ollama can serve; `docie-bench models list` shows them.

## How calls behave

Clients aimed at the same endpoint and model share one gateway: one concurrency
limit, one queue, one circuit breaker. Only transient failures, rate limits and
invalid responses are retried; any other `4xx` fails at once.

`capability_discovery` is `disabled`, `optional` or `required`. Discovery calls
`GET /models`, checks the configured model id is there, and checks any
advertised vision and response-format support against the profile. `optional`
uses what it learns without ever blocking a run.

`response_format_style` is the strongest structured-output style to try first.
If the runtime cannot honour it, extraction steps down the ladder described in
{doc}`../architecture/extraction`.

## Vision profiles

```yaml
  lfm25_vl_1_6b:
    model: lfm2.5-vl-1.6b
    base_url: http://localhost:8088/v1
    api_key: local-not-used
    response_format_style: json_object
    vision: true
    vision_max_pages: 8
    vision_pdf_dpi: 150
```

With `vision: true`, images are normalised to PNG and PDF pages rasterised,
then sent as `image_url` content blocks, with no OCR step.

## Choosing a CPU model

Select by field-level accuracy under constrained decoding on your own
documents, not by leaderboard. A reasonable order to benchmark in:

1. **LFM2.5**, 350M to 2.6B — the fastest at each size here, and the native
   family.
2. **NuExtract** — built for document-to-JSON; usually the sharpest field
   precision at its size, but extraction-only.
3. **Qwen3 / Qwen3.5**, smallest size the runtime supports.
4. **Gemma 3 / 4**, small, where a compatible GGUF exists.
5. **Granite 4.x** 1B or 3B.
6. **SmolLM3-3B**, an open small baseline.
7. **Llama 3.2 3B Instruct**, a stable baseline.
8. **Phi-4-mini**, if higher latency is acceptable.

A reasoning model can spend its whole token budget in `reasoning_content` and
never produce the answer. Extraction turns thinking off and prefills the JSON
object's opening brace so the model continues the object instead of narrating
first. A new reasoning model that returns empty extractions is most likely
ignoring both.

## Why not one large document model

On CPU, predictable failure modes matter more than peak capability. OCR, then
schema-constrained extraction, then validation is easier to audit, benchmark
and improve than an opaque image-to-JSON model, which is why vision models are
benchmarked as an alternative path rather than assumed as the default.
