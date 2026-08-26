# Speech-to-text (ASR)

DocIE's first audio capability is offline transcription: local audio becomes
text through a backend-neutral contract, an OpenAI-compatible HTTP endpoint,
or a CLI. The initial backend is
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper), configured for
CPU/int8 by default. It is optional and lazy: importing `docie_bench`, starting
the API, and running the regular test suite do not import CTranslate2, download
weights, or require the ASR extra.

## Install and configure

For a host-native API or CLI:

```bash
pip install -e ".[asr]"
```

The Compose `serving` image enables the `asr` extra. ASR weights are seeded into
the canonical model store and loaded by one managed runtime on that serving
node. API replicas only validate and proxy uploads, so scaling the public API
does not duplicate model memory. Canonical snapshots live in the shared serving
state, while the serving node's `hf-cache` volume persists Hub caches.

| Setting | Default | Purpose |
|---|---:|---|
| `ASR_MODEL` | `small` | Default model for the host-native CLI |
| `ASR_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `ASR_COMPUTE_TYPE` | `int8` | CTranslate2 compute type |
| `ASR_CPU_THREADS` | `0` | Backend thread count; zero lets CTranslate2 choose |
| `ASR_NUM_WORKERS` | `1` | faster-whisper worker count |
| `ASR_BEAM_SIZE` | `5` | Beam-search width |
| `ASR_VAD_FILTER` | `true` | Filter non-speech with the backend VAD |
| `ASR_TIMEOUT_SECONDS` | `600` | Public API timeout for the managed runtime |
| `ASR_MAX_UPLOAD_MB` | `25` | Per-audio decoded multipart limit |
| `ASR_ALLOWED_UPLOAD_MIME_TYPES` | common audio types | Canonical MIME allowlist |

`ASR_MAX_UPLOAD_MB` must fit below the global `MAX_REQUEST_BODY_MB` ceiling
(26 MiB by default, including multipart overhead). Raise both deliberately for
long recordings. Accepted containers are WAV, MP3, FLAC, M4A/MP4, Ogg, and
WebM. The API checks the suffix, declared MIME type, and container magic bytes;
renaming an arbitrary binary is not sufficient.

## OpenAI-compatible API

`POST /v1/audio/transcriptions` uses the familiar multipart fields `file`,
`model`, `language`, `prompt`, `response_format`, and `temperature`:

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -H "X-API-Key: $DOCIE_API_KEY" \
  -F "file=@meeting.wav" \
  -F "model=asr-default" \
  -F "language=en" \
  -F "response_format=verbose_json" \
  -F "temperature=0"
```

Response formats are `json`, `verbose_json`, `text`, `srt`, and `vtt`.
`verbose_json` includes segment timestamps, duration, processing time,
real-time factor, and the actual backend/model. The compact `json` response is
exactly `{"text": "..."}`.

The route shares the platform's API-key, tenant rate-limit, concurrency, and
global request-size policy. Its `model` field resolves only to the name or
served alias of an already-created, ready ASR deployment. It can never become a
Hub download instruction. Uploads stream into a bounded temporary file and are
deleted after success or failure. The managed runtime runs CPU inference in a
worker thread so it does not block FastAPI's event loop.

Stable failure classes are intentionally visible:

- `404`: the requested model is not a managed ASR deployment/alias;
- `413`: the global request or ASR upload limit was exceeded;
- `415`: suffix, MIME, magic bytes, or allowlist validation failed;
- `422`: invalid fields or audio decoding/transcription failed;
- `503`: the deployment is cold/unready/unreachable, or model loading failed.

## Managed model lifecycle

Use the normal model-store workflow and choose the `asr_whisper` family for a
faster-whisper CTranslate2 repository (it must contain `model.bin`):

```bash
# In Studio: Models -> search/select a faster-whisper model -> Seed
# Then deploy the canonical store entry (example name):
docie up whisper-small --name asr-default
```

The deployment appears in the same lifecycle views as LLM and encoder models.
Startup remains `starting` while weights load; only a successful `/healthz`
probe promotes it to `ready`. Stop, unload, automatic idle eviction, restart
budgeting, observed failure state, port allocation, RSS calibration, and
recency accounting are inherited from the serving control plane. The public
request may use the deployment name (`asr-default`) or its unique served alias
(the canonical store name).

## CLI transcription

The local CLI is trusted operator input, so `--model` may override the configured
model (unlike the public API):

```bash
docie-bench asr transcribe meeting.wav --language en
docie-bench asr transcribe meeting.m4a --format verbose_json --output result.json
docie-bench asr transcribe meeting.wav --format srt --output meeting.srt
```

## WER/CER benchmark

Create a UTF-8 JSONL manifest. Audio paths resolve relative to the manifest:

```jsonl
{"id":"call-001","audio":"audio/call-001.wav","reference":"Thank you for calling.","language":"en"}
{"id":"call-002","audio":"audio/call-002.flac","reference":"Votre commande est prête.","language":"fr","prompt":"Acme product names"}
```

Then run:

```bash
docie-bench asr benchmark data/asr/manifest.jsonl --output-dir runs/asr/baseline-small
```

The run writes immutable/reproducible inputs and four artifacts:

- `manifest.json`: dataset/audio hashes, explicit model/backend/configuration,
  Git state, dependency versions, and machine provenance;
- `predictions.jsonl`: references, hypotheses, language, duration, timing,
  per-item WER/CER, and normalized forms;
- `metrics.json`: corpus WER/CER and aggregate real-time factor;
- `report.html`: a portable human-readable summary and comparison table.

Scoring applies Unicode NFKC, case-folding, punctuation/symbol removal, and
whitespace collapse. WER is Levenshtein distance over words. CER is distance
over normalized non-whitespace characters. Both headline rates are calculated
from corpus error/reference totals—not an unweighted mean of per-file rates.
Real-time factor is total processing seconds divided by total audio seconds;
values below 1 mean faster-than-real-time processing.

## Current boundaries

This milestone currently provides batch transcription only. It does not yet
provide streaming, speaker diarization, speech translation, or browser
microphone capture. Durable transcription jobs and the Studio workspace remain
separate milestone slices rather than hidden partial support.
