# Serving CLI

`docie-serving` (also installed as `docie`) operates the serving layer of the
machine it runs on. The one exception is the `fleet` group, which talks to the
central API about *another* node — see below. The Studio and the `/v1/serving` API are the normal way to
deploy; the CLI exists for what they cannot reach — a specific node's own store,
preflight checks before a node joins, and standalone runtimes for development.

```bash
docie-serving --help
docie-serving --json store list
```

`--json` goes before the command. Output is one line of compact JSON with
sorted keys; errors are `{"error":{"message":"...","type":"..."}}` with exit
status 1.

## Node checks

| Command | |
|---|---|
| `doctor` | Checks this machine is correctly configured as a fleet node. Exits non-zero on any failing check, so a bringup script can gate on it. `--probe-ports` only means something inside the serving container |

See {doc}`../operations/fleet-runbook` for when to run it.

## The fleet, through the central API

The only group here that is not about the machine running it. Draining is a
central decision — it has to refuse a drain that would take a model's last live
replica offline, which only the catalog can see — so these are HTTP calls, and
they need two variables that no other command reads:

| Variable | |
|---|---|
| `DOCIE_API_URL` | The central API's base url, as whatever runs the command reaches it |
| `DOCIE_API_KEY` | A key the central API accepts |

A serving node needs neither. These belong to whatever **replaces a container**,
which already holds a credential.

| Command | |
|---|---|
| `fleet status [NODE]` | Each node's drain readiness, alongside what the API reports |
| `fleet drain NODE [--wait] [--timeout N]` | Take a node out of the rotation. With `--wait`, block until it holds no live runtime |
| `fleet resume NODE` | Admit it again |

`fleet drain --wait` is the whole of a pre-stop hook:

```bash
docie-serving fleet drain ovh-cpu-02 --wait --timeout 900
```

Exit 0 means the node holds no live runtime. Exit non-zero means a timeout, a
node that stopped reporting while still holding runtimes, or a node that is not
draining at all — and in every one of those cases the container must **not** be
replaced.

:::{important}
**What exit 0 claims, and what it does not.** The node waited out the longest
request timeout its deployments declared — capped at `DRAIN_GRACE_CEILING_S`
(600s) — and then stopped them. That is a *bounded* guarantee, not proof that
every in-flight request finished:

- a deployment declaring a timeout longer than the cap is cut short by design;
- a caller whose own timeout exceeds the deployment's is outside the bound;
- queued work that never reached a runtime is not counted.

Raise the deployments' `timeout_seconds` (#475) if the bound is too tight for
what runs on that node. `--acknowledge-outage` is never added automatically by
anything: a refusal names the models about to go offline, which is a decision.
:::

Readiness is reported as one of:

| Readiness | |
|---|---|
| `admitting` | No drain in effect. Returns immediately rather than waiting — either the drain never reached the catalog or something resumed the node |
| `routing_excluded` | Drained, still holding live runtimes. Nothing new is routed here |
| `runtimes_stopped` | Drained with no live placement. The one state that is safe to replace |
| `stale` | Drained, holding live runtimes, and no longer reporting. Not converging, so waiting may never finish |
| `unknown` | No readable row for that node |

`resume` has no `--wait` on purpose: there is nothing to block on that would
mean more than "the flag is cleared". The node restores the deployments the
drain stopped on its own next cycle, and `fleet status` is how to watch it.

## The node-local store

The blob store is per node. Seeding through the Studio emits an event that
lands on whichever serving node picks it up, so getting a model onto a
*particular* node means running these there. They never emit an event.

| Command | |
|---|---|
| `store list` | What this node holds |
| `store seed-ollama <ref> --name --family` | Register a model Ollama already pulled here. Idempotent |
| `store add-gguf <path> --name --family` | Register a GGUF already on this disk |

Both seeding commands hard-link by default; `--copy` copies instead. `--mmproj`
supplies a vision projector explicitly.

## Deployments

| Command | |
|---|---|
| `up <store-name>` | Serve a store model with its family's llama-server flags. Port auto-allocated from `DOCIE_SERVING_PORT_RANGE_*` unless `--port`; `--ctx-size` defaults to 8192 |
| `serve <model> --name --runtime --replicas` | Create and start a deployment from the model registry |
| `list` / `status <name>` | Managed deployments and their health |
| `start <name>` / `stop <name>` | Stop keeps the specification |
| `plan <model> --runtime --replicas` | Compatibility and resource assessment without launching |

```{warning}
`up` names the record after the store model unless `--deployment-name` is
given. Store names are not unique across nodes, so in a fleet pass a distinct
name per node.
```

## Registry and runtimes

| Command | |
|---|---|
| `model list` / `model show <model>` | Registered artifacts |
| `model pull <model> --runtime --revision` | Fetch an artifact; pin `--revision` for anything real |
| `model remove <model>` | |
| `runtime list` / `runtime probe <name>` | Available inference runtimes |

`model pull --trust-remote-code` and `transformers --trust-remote-code` execute
the repository's own Python on the node. Both are off by default; do not enable
them for a source you have not reviewed.

## Standalone endpoints

Each of these serves one OpenAI-compatible surface in the foreground, outside
the supervisor. They are for development and for runtimes a deployment does not
cover.

| Command | Serves |
|---|---|
| `gateway` | One endpoint over every profile in `models.yaml` (port 8080) |
| `encoder` | A PII/NER token classifier (port 8090) |
| `transformers` | A Hugging Face model via transformers, the last-resort runtime (port 8091) |
| `multi-vector` | A ColBERT-style retriever on `/v1/rerank` |
| `dense-embedding` | Text `/v1/embeddings`; needs the isolated embeddings extra |
