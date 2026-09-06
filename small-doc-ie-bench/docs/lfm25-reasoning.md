# LFM2.5-2.6B reasoning

LFM2.5-2.6B is an always-thinking checkpoint. Its official Jinja template
unconditionally appends `<think>` to a new assistant turn; it does not read
`enable_thinking` or `reasoning_effort`. This differs from the smaller LFM2
instruct checkpoints. See the [model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B#chat-template)
and [template](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/chat_template.jinja).

DocIE recognizes LFM2.5-2.6B model/profile names, including repository IDs and
quantized filenames. Structured extraction preserves its native thinking:

- It enables backend thinking and omits reasoning effort values.
- It does not append the assistant `{` prefill that bypasses the thinking prompt.
- It does not retry with reasoning disabled when generation exhausts its budget.
- This policy takes precedence over an agent or pipeline's `no_think` option.

For an opaque served alias, set `options: {native_reasoning: true}` on the model
profile. Set it to `false` to explicitly opt out, for example for a custom
fine-tune with different behavior. Base checkpoints are not auto-enabled.
This policy applies to DocIE structured extraction and pipeline extraction;
raw chat requests continue to use the controls supplied by their caller.

## Effort versus budget in llama.cpp

Current [llama-server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
distinguishes three controls:

- `reasoning_effort`: `none` disables thinking; other values are passed to Jinja.
  Since this checkpoint's template does not use the value, low/medium/high
  do not provide native effort levels for LFM2.5-2.6B. DocIE does not expose
  graded effort through its structured extraction profile API.
- `--reasoning-budget N`: caps thinking tokens. `-1` is unrestricted, `0`
  ends thinking immediately, and positive values reserve a finite thinking budget.
  This is a runtime token limit, not a learned effort level.
- `--reasoning-format`: controls parsing of thoughts, not their generation.

Use the GGUF's embedded template with `--jinja`. A 4096-token `max_tokens`
budget covers reasoning plus the final JSON; it does not reserve 4096 tokens
for the answer alone. The 4096 LFM2 default is proposed separately in
[PR #435](https://github.com/baylitoo/slm_benchmark/pull/435).

On the runtime launch spec, `extra_args: ["--reasoning-budget", "1024"]` is an
example of a finite thinking cap, leaving room in the total output budget for
JSON. Choose the budget by testing the actual schema and document; 1024 is
illustrative, not a validated default. Check the deployed llama-server build's
`--help` for support before setting this flag. Existing explicit output-budget
overrides and runtime flags are not migrated by this change.
