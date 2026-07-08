# `docie_bench.serving` — model serving & GGUF unification

This package turns local inference runtimes into one operational workflow and
keeps a **single canonical GGUF store** that both **llama.cpp (`llama-server`)**
and **Ollama** can serve from. Each model **family** declares how its prompt
template must be delivered, so a model is always served *correctly* — not just
loaded.

**TL;DR** — seed a model once, then `docie up <name>` serves it in the background
with the right family flags and `docie-bench benchmark run --model-profile <name>`
tests it. Full flow in [Serve + benchmark](#serve--benchmark--the-one-command-path).

## Why a canonical `models/` store

Ollama and llama.cpp can run the *same* GGUF, but they cache it differently and
they do **not** honour the same template mechanisms:

| | llama.cpp `llama-server` | Ollama |
| --- | --- | --- |
| GGUF path | any `*.gguf` file | opaque `blobs/sha256-…` (no extension) |
| `chat_template_kwargs` | ✅ honoured **with `--jinja`** | ❌ silently dropped |
| Vision projector | `--mmproj proj.gguf` | bundled in the model only |

So we keep one directory of real `*.gguf` files:

```
<store-root>/<name>/model.gguf      # weights
<store-root>/<name>/mmproj.gguf     # optional vision projector
<store-root>/index.json             # name -> {family, paths, source}
```

A store entry can be **seeded from a model Ollama already pulled** by
hard-linking its blobs in — the GGUF is never downloaded twice.

## Model families (`model_store.py`)

A `FamilyContract` records how to serve and prompt a family:

| family | template delivery | llama-server flags | Ollama-faithful? |
| --- | --- | --- | --- |
| `nuextract3` | `chat_template_kwargs` | `--jinja` (+`--mmproj`, vision) | **no** (Ollama drops the kwargs) |
| `nuextract_v1` | baked into the prompt | – | yes |
| `openai_chat` | OpenAI `response_format` | – | yes |

`ollama_modelfile()` **refuses** families that Ollama can't serve faithfully, so
you can't accidentally deploy a NuExtract3 that ignores its template.

---

## Serve + benchmark — the one-command path

Once a model is in the store (seed it once — see below), serve it in the
**background** with its family's launch flags applied automatically. No separate
terminal window, no hand-typed `--jinja/--mmproj`:

```powershell
docie up nuextract3                                 # llama-server, detached, on :8088
Invoke-RestMethod "http://127.0.0.1:8088/health"    # wait for status: ok (4B VLM loads slowly on CPU)
docie-bench benchmark run --dataset data\voxel51_invoices\manifest.jsonl --model-profile nuextract3
docie stop nuextract3                               # when done
```

`docie up <name> [--port <n>] [--ctx-size 8192]` resolves the GGUF and the family
contract from the store and deploys it through the supervisor — tracked in
`deployments.json`, managed with `docie status` / `docie stop` / `docie list`, and
visible in `docie-serve-dash`. `docie up` returns immediately; the model keeps
loading in the background, so wait for `/health` before benchmarking.

**Port allocation.** Omit `--port` and the deploy auto-assigns the first free port
in `DOCIE_SERVING_PORT_RANGE_START`–`DOCIE_SERVING_PORT_RANGE_END` (default
**8088–8188**) that is neither held by an existing deployment record nor bound by a
live socket — so two concurrent deploys land on distinct ports with no manual
guessing. **8088 stays the first pick**, so a single deploy still matches the
`nuextract3` / `nuextract3_think` profiles' `base_url` (`http://localhost:8088/v1`)
unchanged. Pass an explicit `--port <n>` to pin one; it is honored verbatim (no
probing, no silent reallocation), and if that port is already bound the deploy
fails on bind — the real `llama-server` stderr is surfaced onto the deployment's
`last_error`, and an *auto-allocated* deploy reallocates to a free port (bounded).
This is best-effort, not race-free: a probed-free port can be grabbed before the
runtime binds it, and a worker cannot observe a concurrent host-native `docie up`,
so the bind is the authoritative arbiter. `GET /v1/serving/ports` returns the
window, the deployment→port map, used/free ports, and a `recommended_next` hint
(the Studio Deploy tab renders this live).

> `docie up` launches `llama-server`, so it must be on your **PATH** (see the
> acquisition note below). A bind collision no longer freezes at a bare "runtime
> process exited": the actual bind error is read back from the runtime log.

### One-time: seed the store (Windows / PowerShell)

NuExtract3 is a 4B multimodal VLM; on a CPU-only box use a small quant
(`Q4_K_M`, ~2.7 GB). The GGUF is hard-linked from the Ollama blob — never
downloaded twice.

```powershell
# Pull the GGUF once via Ollama (weights + vision projector)
ollama pull hf.co/numind/NuExtract3-GGUF:Q4_K_M

# Seed the canonical store from it (hard-links, no re-download)
$env:PYTHONPATH = "src"
@'
from pathlib import Path
from docie_bench.serving.model_store import ModelStore
store = ModelStore(Path.home() / ".local/share/docie-bench/serving/models")
e = store.seed_from_ollama("hf.co/numind/NuExtract3-GGUF:Q4_K_M", name="nuextract3", family="nuextract3")
print("model :", e.model_path)
print("mmproj:", e.mmproj_path)   # must NOT be None for a vision family
'@ | python -
```

> Need `llama-server` on PATH (so `docie up` can launch it)? Grab a CPU build:
> ```powershell
> $rel = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
> $asset = $rel.assets | Where-Object { $_.name -match 'bin-win-cpu-x64' } | Select-Object -First 1
> Invoke-WebRequest $asset.browser_download_url -OutFile "$env:TEMP\llamacpp.zip"
> Expand-Archive "$env:TEMP\llamacpp.zip" -DestinationPath "$env:USERPROFILE\llama-cpp" -Force
> $env:PATH = "$env:USERPROFILE\llama-cpp;$env:PATH"   # add it to PATH for this session
> ```

### Manual / under the hood

`docie up nuextract3` runs exactly the invocation below (`--jinja` + `--mmproj`
come from the family contract). Run it directly only to debug or pass a custom
flag — note it occupies the terminal:

```powershell
llama-server --model "$env:USERPROFILE\.local\share\docie-bench\serving\models\nuextract3\model.gguf" `
             --mmproj "$env:USERPROFILE\.local\share\docie-bench\serving\models\nuextract3\mmproj.gguf" `
             --alias nuextract3 --jinja --host 127.0.0.1 --port 8088 -c 8192
```

## Setup — server (Linux / bash)

`docie up nuextract3` works the same on Linux (with `llama-server` on PATH) and
is the easy path. Seed once, then:

```bash
ollama pull hf.co/numind/NuExtract3-GGUF:Q4_K_M
python - <<'PY'
from pathlib import Path
from docie_bench.serving.model_store import ModelStore
store = ModelStore(Path.home() / ".local/share/docie-bench/serving/models")
store.seed_from_ollama("hf.co/numind/NuExtract3-GGUF:Q4_K_M", name="nuextract3", family="nuextract3")
PY
docie up nuextract3        # detached; manage with docie status / stop / list
```

For an explicit, externally-supervised process (e.g. a **systemd** unit bound to
`0.0.0.0`), run `llama-server` directly with the command
`store.llama_server_command("nuextract3", host="0.0.0.0", port=8088)` prints:

```bash
llama-server --model ~/.local/share/docie-bench/serving/models/nuextract3/model.gguf \
             --mmproj ~/.local/share/docie-bench/serving/models/nuextract3/mmproj.gguf \
             --alias nuextract3 --jinja --host 0.0.0.0 --port 8088 -c 8192
```

If you don't have a model in Ollama, register a GGUF you downloaded directly:

```python
store.add_gguf(
    name="nuextract3", family="nuextract3",
    model_gguf="NuExtract3-Q4_K_M.gguf", mmproj="mmproj-NuExtract3-BF16.gguf",
)
```

## Serving an Ollama-faithful family via Ollama

For families Ollama *can* serve faithfully (e.g. `nuextract_v1`, `openai_chat`),
generate a Modelfile from the same canonical store and register it:

```bash
python - <<'PY'
from docie_bench.serving.model_store import ModelStore
store = ModelStore("…/models")
print(store.ollama_modelfile("my-legacy-model"))
PY
# > Modelfile, then:  ollama create my-legacy-model -f Modelfile
```

`ollama_modelfile("nuextract3")` deliberately raises — NuExtract3 must use
`llama-server` (see the table above).

## Configuration

- `DOCIE_SERVING_HOME` — base dir for the registry/deployments (default
  `~/.local/share/docie-bench/serving`). The canonical model store lives under
  `…/models` in the examples above.
- `OLLAMA_MODELS` — where `seed_from_ollama` looks for Ollama blobs (default
  `~/.ollama/models`).

See [docs/serving-factory.md](../../../docs/serving-factory.md) for the broader
control-plane (`runtime`/`registry`/`planner`/`supervisor`) architecture.
