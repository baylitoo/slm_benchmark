# Hybrid local/frontier example

This example configures a local OpenAI-compatible model, an opt-in frontier endpoint, a remote MCP-enabled agent with a read-only tool allowlist, a local routing workflow, and a local-transform/remote-reasoning workflow.

## Start the local model and API

Install the MCP extra and start a local OpenAI-compatible server on port 8088. For example, with llama-server:

```powershell
pip install -e ".[mcp]"
llama-server --model C:\models\local.gguf --alias local-chat --port 8088
```

Set the remote endpoint and config paths before starting the API. The remote profile is explicit opt-in; this example never calls it during CI or implementation checks.

```powershell
$env:HYBRID_FRONTIER_BASE_URL = "https://provider.example/v1"
$env:HYBRID_FRONTIER_API_KEY = "<your server-side key>"
$env:DOCIE_MODELS_CONFIG = "examples/hybrid_execution/models.yaml"
$env:DOCIE_AGENTS_PATH = "examples/hybrid_execution/agents.json"
$env:DOCIE_MCP_SERVERS_CONFIG = "examples/hybrid_execution/mcp-servers.json"
uvicorn docie_bench.api:app --host 127.0.0.1 --port 8000
```

The API process must run from the repository root so the example paths resolve. The MCP filesystem server is launched through `npx` and is restricted to `examples/hybrid_execution/docs`; the agent allowlist exposes only `list_directory` and `read_file`. Ensure Node.js/npx is installed. Set `HYBRID_FRONTIER_BASE_URL` to an OpenAI-compatible `/v1` endpoint and update `model: frontier-model` to the model ID your endpoint accepts.

Try the remote tool agent:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/agents/frontier-files/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"messages":[{"role":"user","content":"Read quarterly-summary.txt and report the number of open requests."}]}'
```

Try the local router (use `local` or `frontier` in the request to steer its classifier):

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/agents/hybrid-router/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"messages":[{"role":"user","content":"local: summarize the support counts"}]}'
```

The workflow trace in `docie_agent` shows which profile was selected, the input chained to each step, call attempts/timing, and local-versus-remote classification. The `transform-then-reason` agent demonstrates the separate local transformation followed by remote reasoning path.

## Optional real-provider smoke test

This command is the only real-provider test and may incur provider charges. It is intentionally skipped unless the opt-in flag is set; do not add that flag to ordinary CI.

```powershell
$env:DOCIE_RUN_PROVIDER_SMOKE = "1"
pytest -q tests/test_hybrid_provider_smoke.py
```
