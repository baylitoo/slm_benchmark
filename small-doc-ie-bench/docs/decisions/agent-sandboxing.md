# Sandboxing model-generated code

**Decided:** Judge0, as a separate pair of services. Issues
[#263](https://github.com/baylitoo/slm_benchmark/issues/263) and
[#264](https://github.com/baylitoo/slm_benchmark/issues/264).

## Context

An agent's model can call tools over MCP. Which servers exist is an operator
decision made at registration, and each runs with its own privileges — fine for
the first-party servers (`calculator`, `dates`, `web-fetch`, `docs-search`),
each small, audited and narrowly scoped.

That stops being fine for a tool that runs **arbitrary model-supplied code**. It
needs a real isolation boundary between the snippet and the host the serving
process runs on, and safely running untrusted code is a problem with mature
prior art.

The deployment this is scoped to: self-hosted through plain `docker compose`
under Coolify — no Kubernetes, no guaranteed KVM, CPU-only, one operator.

## Options

| Option | Isolation | Why not |
|---|---|---|
| Hand-rolled bubblewrap | real namespaces | every hardening decision and all lifecycle and pooling is ours; an early attempt exposed the whole container filesystem read-only |
| gVisor / Firecracker | syscall boundary / VM | need host daemon configuration or KVM that this deployment does not guarantee |
| Managed sandbox API | strong | an external dependency and account, breaking the local-first model every other tool keeps |
| Piston | `isolate` | needs `--privileged` like Judge0, with no documented worker pool |
| **Judge0** | `isolate` | chosen |

`--privileged` is required by `isolate` itself, so it does not distinguish the
two self-hosted engines. What does: Judge0 ships a persistent, Redis-queued
worker pool with a `GET /workers` endpoint, so submissions go to running
workers instead of cold processes.

## Decision

Judge0 runs as `judge0-server` and `judge0-worker` behind the `sandbox` compose
profile, with its own Postgres and Redis, on the internal network only — no
published port. On top of its defaults:

- `AUTHN_TOKEN` is set. Upstream defaults to no authentication, which would let
  any container on the network execute code. The `code_interpreter` tool
  refuses to run without a token.
- `ENABLE_CALLBACKS=false`. Callbacks are the feature behind Judge0's one
  disclosed SSRF advisory, and the integration does not need them.
- `ENABLE_WAIT_RESULT=true`, so the tool makes one synchronous request.
- Judge0's own per-submission CPU, wall-time and memory limits — no separate
  limiting layer.

The `code_interpreter` MCP server (`mcp_servers/code_interpreter.py`) is a thin
client: it submits Python, waits, and maps `stdout`, `stderr`, status and time
into what the model reads. An agent opts in with
`options.mcp_servers: ["code-interpreter"]`. The worker pool's state appears as
a card in the Studio's MCP tab.

## Revisit when

Judge0's advisory class or its `--privileged` requirement becomes
unacceptable, or the requirement moves from untrusted code to actively hostile
code. gVisor, Firecracker and Piston are not rejected outright, only for this
deployment shape.
