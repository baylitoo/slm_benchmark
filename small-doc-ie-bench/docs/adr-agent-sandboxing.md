# ADR: Sandboxing for arbitrary tool/code execution

**Status:** decided (design only — see #263, #264). No sandbox code ships in
this PR; #264 implements the `code_interpreter` tool kind on top of it.

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
code (#264's `code_interpreter`) or shells out to a system command chosen by
the model rather than an operator. That needs a real isolation boundary
between "the model's generated snippet" and "the host the serving process
runs on" — not an admin's judgment call per registered server.

**Deployment shape this decision is scoped to** (see `docs/docie-studio.md`,
`project_coolify_deployment` context): self-hosted via Coolify, which
orchestrates plain `docker compose` — no Kubernetes, no guaranteed access to
reconfigure the Docker daemon's runtime list, no KVM guarantee, CPU-only. The
`api`/`worker` images are `python:3.11-slim-bookworm` (Debian). Any option
that assumes a container orchestrator beyond compose, or host-level config
an operator has to hand-tune outside their normal Coolify deploy, is not
realistic here — it would work in the demo environment and silently not work
(or need undocumented host surgery) for anyone who actually self-hosts this.

## Options surveyed

| Option | Isolation strength | Fits this deployment shape? |
| --- | --- | --- |
| **Namespaced subprocess** (bubblewrap: mount/PID/net namespaces) + POSIX rlimits | Real kernel-enforced isolation (separate mount namespace, no network, no visibility of other processes); weaker than a VM (shares the host kernel) | **Yes** — one `apt-get install bubblewrap` line in the existing Debian image, runs as a plain subprocess inside the current container, no host config, no orchestrator dependency |
| **gVisor** (`runsc`) — user-space kernel intercepting syscalls | Strong (a real syscall boundary, not just namespaces) | No for v1 — needs the *host's* Docker daemon configured with an extra runtime (`/etc/docker/daemon.json`), which Coolify doesn't manage; running it from inside an already-containerized worker needs docker-in-docker or a host socket mount, a materially bigger operational ask than this feature justifies |
| **Firecracker** microVMs | Strongest (real VM boundary) | No for v1 — needs KVM and a VMM control plane; not guaranteed on a generic Coolify host, heaviest to operate of any option here |
| **Managed sandbox service** (e.g. a cloud code-execution API) | Strong, but the boundary is someone else's infrastructure | No for v1 — adds an external network dependency, a third-party account/cost, and cuts against this project's local-first/air-gapped serving story; every other tool kind so far runs entirely on the operator's own infrastructure |

## Decision

**Bubblewrap (`bwrap`), wrapping a plain subprocess, with POSIX rlimits as a
second layer.** Concretely, for #264 to implement:

- Mount namespace: bind-mount only a fresh, per-call scratch directory
  (read-write) plus the Python interpreter/stdlib (read-only) — nothing else
  of the container's filesystem is visible.
- Network namespace: unshared, unconditionally, for v1 — no network access
  at all. (This also makes "no package installation" in #264's scope a
  property of the sandbox, not a rule the tool has to remember to enforce.)
- PID namespace: unshared — the snippet can't see or signal any other
  process, including its own supposed children surviving past the call.
- `resource.setrlimit` inside the sandboxed process as defense in depth
  (namespaces alone don't cap CPU/memory): `RLIMIT_CPU` (a few seconds),
  `RLIMIT_AS` (memory), `RLIMIT_NPROC` (effectively 1 — no fork bombs).
- A hard wall-clock timeout on the subprocess itself (belt-and-suspenders
  against a CPU-limit that doesn't trip for an I/O-bound infinite loop).
- stdout/stderr captured with a byte cap (mirrors the tool-call trace
  truncation already shipped in #262's `_TRACE_TEXT_LIMIT`) — a runaway print
  loop must not blow up the model's context either.
- **Fail closed, not silently unsandboxed**: if `bwrap` isn't on `PATH`, the
  `code_interpreter` tool kind refuses to run with a clear error, the same
  shape as `mcp_tools.MCPUnavailableError`/`_require_mcp()` for the optional
  `mcp` SDK dependency — never falls back to running the snippet bare.

gVisor/Firecracker are not rejected forever — if a stronger guarantee becomes
necessary (e.g. genuinely hostile multi-tenant code execution, not an
operator's own agents), they're the natural next step and this decision
should be revisited. They're just the wrong first cut for this deployment
shape.

## Consequences for #264

- New optional runtime dependency: `bubblewrap` in the `api`/`worker`
  Dockerfiles (Debian package, one line).
- New settings, mirroring the `mcp_tool_timeout_seconds`/
  `mcp_max_tool_iterations` naming convention (`docie_bench/settings.py`):
  a CPU-seconds cap, a memory cap, an output-byte cap — operator-tunable,
  not hardcoded.
- The `code_interpreter` tool's "unavailable" error is a distinct, clearly
  labeled error type (like `mcp_unavailable`) so an operator sees exactly
  why a tool call refused, not a generic 500.
