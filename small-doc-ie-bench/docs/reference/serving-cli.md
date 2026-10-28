# Serving CLI

`docie-serving` (also installed as `docie`) operates the serving layer of the
machine it runs on. The Studio and the `/v1/serving` API are the normal way to
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
