# `docie_bench.serving` — model serving & GGUF unification

This package turns local inference runtimes into one operational workflow and
keeps a **single canonical GGUF store** that both **llama.cpp (`llama-server`)**
and **Ollama** can serve from. Each model **family** declares how its prompt
template must be delivered, so a model is always served *correctly* — not just
loaded.

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

## Setup — local machine (Windows / PowerShell)

NuExtract3 end-to-end (the verified path). The model is a 4B multimodal VLM; on a
CPU-only box use a small quant (`Q4_K_M`, ~2.7 GB).

```powershell
# 1. Pull the GGUF once via Ollama (downloads weights + vision projector)
ollama pull hf.co/numind/NuExtract3-GGUF:Q4_K_M

# 2. Seed the canonical store from that Ollama model (hard-links, no re-download)
$env:PYTHONPATH = "src"
python - <<'PY'
from pathlib import Path
from docie_bench.serving.model_store import ModelStore
store = ModelStore(Path.home() / ".local/share/docie-bench/serving/models")
entry = store.seed_from_ollama(
    "hf.co/numind/NuExtract3-GGUF:Q4_K_M", name="nuextract3", family="nuextract3"
)
print("model :", entry.model_path)
print("mmproj:", entry.mmproj_path)
print("serve :", " ".join(store.llama_server_command("nuextract3", port=8088)))
PY

# 3. Serve it (use the exact command printed above; --jinja + --mmproj are added
#    automatically from the family contract). Pick a FREE port (8080 may be taken).
llama-server --model "$env:USERPROFILE\.local\share\docie-bench\serving\models\nuextract3\model.gguf" `
             --mmproj "$env:USERPROFILE\.local\share\docie-bench\serving\models\nuextract3\mmproj.gguf" `
             --alias nuextract3 --jinja --host 127.0.0.1 --port 8088 -c 8192

# 4. In another window: confirm it's really llama-server (JSON, not HTML)
Invoke-RestMethod "http://127.0.0.1:8088/health"     # -> status : ok
```

> Need `llama-server`? Grab the latest CPU build:
> ```powershell
> $rel = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
> $asset = $rel.assets | Where-Object { $_.name -match 'bin-win-cpu-x64' } | Select-Object -First 1
> Invoke-WebRequest $asset.browser_download_url -OutFile "$env:TEMP\llamacpp.zip"
> Expand-Archive "$env:TEMP\llamacpp.zip" -DestinationPath "$env:USERPROFILE\llama-cpp" -Force
> ```

Then run the benchmark against it — `configs/models.yaml` ships `nuextract3`
(reasoning off) and `nuextract3_think` (reasoning on) pointing at
`http://localhost:8088/v1`.

## Setup — server (Linux / bash)

```bash
# 1. Pull once via Ollama (or download the GGUF + mmproj from Hugging Face)
ollama pull hf.co/numind/NuExtract3-GGUF:Q4_K_M

# 2. Seed the canonical store
python - <<'PY'
from pathlib import Path
from docie_bench.serving.model_store import ModelStore
store = ModelStore(Path.home() / ".local/share/docie-bench/serving/models")
store.seed_from_ollama("hf.co/numind/NuExtract3-GGUF:Q4_K_M", name="nuextract3", family="nuextract3")
print(" ".join(store.llama_server_command("nuextract3", host="0.0.0.0", port=8088)))
PY

# 3. Serve with the printed command (systemd unit recommended for persistence)
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
