# Hybrid agent execution

Hybrid workflows use the existing `AgentSpec` workflow runner and OpenAI-compatible model profiles. They do not add another orchestration layer.

## Existing paths and profile behavior

| Path | Upstream route | Profile settings and current behavior |
|---|---|---|
| Custom agent | `agents.runtime._post_chat` | Uses timeout, concurrency/queue, bounded retry, and defaults for temperature/top-p/token limit/stop sequences through `ModelGateway`; tool rounds remain one model call at a time. |
| OCR structured or policy extraction | `ExtractionService` / `ExtractionRouter` → `OpenAICompatibleClient` → `ModelGateway` | Already uses profile timeout, queue, retries, circuit breaker, response-format negotiation, and vision preflight. Usage can be absent. |
| OCR pipeline and VLM transcription | `serving.solutions` raw response adapter | Now uses profile scheduling, retries, timeout, and generation defaults; specialized OCR/JSON processing remains in the adapter. |
| Security proxy and guard analyzer | proxy → shared transport; encoder guard → shared transport | Uses profile timeout, scheduling, and generation defaults. The proxy buffers and redacts the initial messages before its backing model call. Masking does not remove arbitrary confidential content. |
| Inline workflow | `_run_workflow_steps` → custom/MCP call path | First step sees original messages; later steps see the prior substantive answer. Routing labels choose a named step and are not chained as content. Every model round is budgeted and traced. |
| Saved-agent reference | `AgentRegistry` lookup at request time | `custom` references execute the saved prompt/tools; `workflow` references execute nested steps and share policy/budgets. `ocr` and `proxy_security` references fail clearly because their document/redaction processing is not a chat-only step. |
| Agent streaming | `open_chat_stream` or the buffered MCP/workflow event path | Profile queue capacity is held until stream closure. Retry is limited to connection/status failures before output; cancellation closes the upstream. Tool/workflow events preserve their existing response processing. |

## Capability and retry contract

Profiles may declare `capabilities` for `tools`, `tool_choice`, `structured_output`, `vision`, `streaming`, and `usage`. Each value is `true`, `false`, or `null`; omitted/null means unknown. When the configured `capability_discovery` mode returns endpoint metadata, observations are reported separately from the operator expectation. A known false capability fails before the request. Unknown support is sent using the OpenAI-compatible request shape; provider rejection is returned as a sanitized capability error. Native provider protocols are not implied.

The shared transport uses `ModelGateway` for profile concurrency, bounded queues, retry limits, exponential backoff, `Retry-After`, and circuit state. It retries only an individual buffered model call. It never retries a tool loop or workflow as a whole. A stream may retry before it yields output; after the first byte, a failure ends that stream. Read timeouts are not retried because the provider may still be generating. Provider response bodies and transport exception strings are not returned or logged as errors.

For workflow kinds, options are:

- `allow_remote_profiles` (default `true` for existing workflow compatibility). `false` blocks remote and unclassified profiles before each model call, including nested workflows and later MCP rounds. A profile must set `execution_location: local` to be accepted by a local-only workflow; `remote` and `unknown` are treated as remote.
- `deadline_seconds`: one wall-clock deadline shared across nested workflows and tool rounds.
- `max_model_calls`: a total model call limit shared across nested workflows and tool rounds.

Workflow traces include resolved profile/model, local-vs-remote classification, attempts, queue wait, elapsed time, outcome, and step identity on each model call. Usage totals remain null if any step lacks complete provider usage; they are not context-window occupancy estimates.

## Local example

The runnable example is in `examples/hybrid_execution/README.md`; it includes local/remote profiles, an MCP-enabled agent, and two mixed workflows. The remote endpoint is opt-in and is never called by the test suite.
