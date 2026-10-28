# Reasoning models

A reasoning model can spend its whole output budget thinking and never reach
the answer. For extraction that is a failure, so **extraction never thinks**.
Chat is left to the caller.

## Always-thinking checkpoints

Most chat templates honour `enable_thinking`. LFM2.5-2.6B's does not: its
template appends `<think>` to every new assistant turn unconditionally, and
ignores both `enable_thinking` and `reasoning_effort`
([model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B#chat-template)). The
smaller LFM2 instruct checkpoints do not share this.

LFM3 thinks by default too, but its template honours `enable_thinking: false`
by closing the block (`<think></think>`), so it is not an always-thinking
checkpoint. Extraction already sends the flag.

`llm/reasoning.py` recognises these checkpoints by model or profile name,
including repository ids and quantised filenames, and excludes base
checkpoints. For a served alias whose name says nothing, set
`options: {native_reasoning: true}` on the profile, or `false` to opt out — for
a fine-tune that behaves differently, say.

## What extraction does

Every structured extraction sends `enable_thinking: false`. For a checkpoint
that ignores it, the request also **prefills the assistant turn with `{`**:
continuing a JSON object is the one thing that skips the template's `<think>`.
The prefill applies with the `strict_extraction_v1` prompt profile, whenever
thinking is disabled or the model is recognised as always-thinking.

## What chat does

A chat request is passed the controls its caller sent. For a recognised
always-thinking checkpoint with thinking not explicitly disabled, the client
turns backend thinking on, removes any `reasoning_effort` (the template does
not read it), and drops a prefill that would bypass the trained thinking
prompt.

## llama.cpp's controls

`llama-server` has three, and only one is an effort level:

| Control | |
|---|---|
| `reasoning_effort` | `none` disables thinking; other values are passed to the template, which for LFM2.5-2.6B ignores them |
| `--reasoning-budget N` | caps thinking tokens: `-1` unlimited, `0` ends thinking immediately |
| `--reasoning-format` | how thoughts are parsed, not whether they are generated |

`max_tokens` covers thinking and answer together. To give a thinking chat model
a finite cap and leave room for the answer, add
`extra_args: ["--reasoning-budget", "1024"]` to the deployment's launch spec —
1024 is an example, not a tested default — after checking the deployed build's
`--help` supports it.
