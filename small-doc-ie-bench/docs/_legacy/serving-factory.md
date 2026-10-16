# Serving Factory Control Plane

The serving control plane provides one operations surface over the model registry,
runtime adapters, deployment supervisor, and resource planner. It is intentionally
separate from the benchmark CLI and can be called as a Python module:

```bash
python -m docie_bench.serving.cli --help
```

Use `--json` before the command for compact, deterministic JSON suitable for scripts:

```bash
python -m docie_bench.serving.cli --json model list
python -m docie_bench.serving.cli --json status invoice-fast
```

JSON object keys are sorted, output is one line, and backend list order is preserved.
Errors use `{"error":{"message":"...","type":"..."}}` and exit with status 1.

## Operations

```bash
# Artifacts
python -m docie_bench.serving.cli model list
python -m docie_bench.serving.cli model show Qwen/Qwen3-4B
python -m docie_bench.serving.cli model pull Qwen/Qwen3-4B \
  --runtime vllm --revision pinned-sha
python -m docie_bench.serving.cli model remove Qwen/Qwen3-4B

# Runtime discovery
python -m docie_bench.serving.cli runtime list
python -m docie_bench.serving.cli runtime probe llamacpp

# Plan before launch, then operate the deployment
python -m docie_bench.serving.cli plan Qwen/Qwen3-4B --runtime llamacpp --replicas 2
python -m docie_bench.serving.cli serve Qwen/Qwen3-4B --name invoice-fast \
  --runtime llamacpp --replicas 2
python -m docie_bench.serving.cli list
python -m docie_bench.serving.cli status invoice-fast
python -m docie_bench.serving.cli stop invoice-fast
python -m docie_bench.serving.cli start invoice-fast
```

`model pull --trust-remote-code` is explicit by design. Do not enable it for an
unreviewed source. Production workflows should always pin a revision.

## Backend Contract

`ControlPlane` accepts four injected collaborators. Methods may be synchronous or
asynchronous and may return dataclasses, Pydantic models, mappings, or sequences.

| Collaborator | Required methods |
|---|---|
| `Registry` | `list_models`, `get_model`, `pull_model`, `remove_model` |
| `RuntimeCatalog` | `list_runtimes`, `probe_runtime` |
| `Supervisor` | `list_deployments`, `deployment_status`, `serve`, `start`, `stop` |
| `Planner` | `plan` |

The default factory adapts the concrete `ModelRegistry`, runtime adapter map,
`PersistentSupervisor`, and `ResourcePlanner` primitives from their respective
`docie_bench.serving` modules. State defaults to
`~/.local/share/docie-bench/serving`; set `DOCIE_SERVING_HOME` to override it.
The local process supervisor currently supports one replica. A provider-backed
registry can pull by model identity; the default local registry accepts a model
manifest JSON path whose artifacts each include a source.

For tests and remote control-plane clients, inject the facade directly:

```python
from docie_bench.serving.cli import create_app
from docie_bench.serving.control_plane import ControlPlane

plane = ControlPlane(registry, runtimes, supervisor, planner)
app = create_app(plane)
```

The CLI is therefore usable locally while retaining a narrow boundary for a future
multi-node or HTTP-backed control plane.
