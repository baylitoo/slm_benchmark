# ADR: Sandboxing for arbitrary tool/code execution

**Status:** decided and implemented (#263, #264 shipped together — a
docs-only design PR wasn't acceptable on its own; this ADR ships alongside
the real `code_interpreter` tool it decides for).

## Context

Milestone "Agents v2" (#259) already lets an agent's model call arbitrary
tools over MCP — a config-time trust decision by whoever registers a server
(`configs/mcp-servers.json` / the catalog, see `docie_bench/mcp_catalog.py`).
llama-server's own MCP docs are explicit about the model this inherits: *"the
child process runs with the server's own privileges — only declare commands
you trust."* That's fine for the first-party servers shipped so far
(`calculator`, `dates`, `web-fetch`, `docs-search`, #275) — each is a small,
audited, read-only or narrowly-scoped stdio process.

It stops being fine the moment a tool kind runs **arbitrary** model-supplied
code. That needs a real isolation boundary between "the model's generated
snippet" and "the host the serving process runs on" — not an admin's
judgment call per registered server, and not something we hand-roll and
hope is airtight: this specific problem (safely running untrusted code,
with lifecycle/pooling already solved) has mature, widely-deployed prior
art. Reinventing it was the wrong instinct — the first draft of this ADR
did, and was corrected before anything shipped.

**Deployment shape this decision is scoped to**: self-hosted via Coolify,
which orchestrates plain `docker compose` — no Kubernetes, no guaranteed
KVM, CPU-only, single operator (not a multi-tenant judge service).

## Options surveyed

| Option | Isolation | Cost | Fits here? |
| --- | --- | --- | --- |
| **Hand-rolled bubblewrap subprocess** | Real namespaces, but *we* own every hardening decision | We own 100% of the security review and 100% of lifecycle/pooling, from scratch. A first pass of this already leaked the whole container filesystem read-only before anyone caught it. | No — the exact "hard to get right, easy to get wrong" trap this ADR exists to avoid |
| **gVisor** / **Firecracker** | Stronger (syscall boundary / real VM) | Needs host-level Docker daemon config or KVM Coolify doesn't manage | No — assumes infra this deployment shape doesn't guarantee |
| **Managed sandbox service** (cloud code-exec API) | Strong, but it's someone else's infrastructure | External network dependency, third-party account/cost, breaks the local-first story every other tool kind here keeps | No |
| **Piston** (self-hosted code-execution engine) | `isolate` (namespaces + cgroups) | Needs `--privileged`; no documented worker-pool/concurrency model (verified against its own README, not assumed) | No — same privilege cost as Judge0 below, with a *less* solved lifecycle story |
| **Judge0** (self-hosted code-execution engine) | Same `isolate` primitive | Needs `--privileged` (same cost as Piston — unavoidable, it's `isolate`'s own requirement, not a Judge0 choice); one disclosed SSRF advisory (fixed upstream, and specifically triggered by a feature — result callbacks — we don't need) | **Yes** |

`--privileged` is a real, non-negotiable cost of `isolate`-based execution
(both mature options need it) — it's not something either self-hosted
option dodges, so it isn't a differentiator between them. What *is* a
differentiator: Judge0 ships an actual persistent worker pool (Redis-queued
Sidekiq-style workers, `COUNT` configurable, default `2 * nproc`) with a
`GET /workers` introspection endpoint — submissions queue and get picked up
by already-running workers, not spawned cold per call. Piston documents
none of that.

## Decision

**Judge0**, run as its own pair of services (`server` + `worker`, both
`privileged: true` — required by `isolate`, not by Judge0's choice) in
`docker-compose.yml`, with its own dedicated Postgres/Redis (not shared with
the app's), on the internal compose network only (no host port published).

Hardening applied on top of Judge0's defaults (verified against its own
`judge0.conf`, not assumed):

- `AUTHN_TOKEN` set (auth is **off** by default upstream — an
  unauthenticated code-execution service reachable by any other container
  on the network is a lateral-movement risk even without a published port).
- `ENABLE_CALLBACKS=false` — result callbacks are the exact feature the
  disclosed SSRF advisory exploited, and this integration doesn't need them
  (the MCP tool gets its result from the response, not a webhook).
- `enable_wait_result=true` (Judge0's own default) — lets the v1 integration
  do a plain synchronous request instead of a polling loop. Judge0's own
  docs note this "doesn't scale well" for a public multi-tenant judge
  service; irrelevant here, one operator's agent tool calls at a time.
- Judge0's own `cpu_time_limit`/`wall_time_limit`/`memory_limit` per
  submission — no separate limiting layer to build or get wrong.

The `code_interpreter` MCP tool (#264, `docie_bench/mcp_servers/
code_interpreter.py`) is a thin HTTP client: `POST /submissions?wait=true`
(`language_id=71`, Python) against the Judge0 server, mapping its
`stdout`/`stderr`/`status`/`time` response to a completion the model reads.
Wired the same way every other MCP tool is (#259) — an agent opts in via
`options.mcp_servers: ["code-interpreter"]`, no new plumbing.

Worker-pool visibility answers a real operational ask (not just a nice-to-
have): Judge0's `GET /workers` (queue size, idle/working/paused counts) is
surfaced as a status card nested in the Studio's existing MCP tab, not a
new page.

Piston, gVisor, and Firecracker aren't rejected forever — if Judge0's
disclosed advisory class or its `--privileged` requirement becomes
unacceptable, or genuinely hostile (not just untrusted) code execution
becomes a requirement, this should be revisited. They're just the wrong
fit for what this milestone actually needs.
