# Model storage backends

`ModelStore` (`serving/model_store.py`) has always kept weights under a single
local-disk root (`<root>/<name>/model.gguf`, `index.json`). That's fine for
one serving box; it stops being fine the moment models need to be reachable
from more than one host, survive a disk loss, or be shared across a team
without re-downloading from Hugging Face every time.

`serving/storage_backend.py` introduces the seam for that: a `StorageBackend`
interface with two implementations today, `LocalDiskBackend` and
`S3CompatibleBackend`.

## Why this is two-tier, not a drop-in swap

`llama-server` and `AutoModel.from_pretrained` both need a real filesystem
path — neither can load a model directly out of a bucket. So a bucket-backed
store is necessarily **durable storage + a local serving cache**, not a
transparent replacement for `<root>/`:

- `write_verified` / `write_tree` push bytes to wherever they durably live
  (local disk, or a bucket).
- `resolve_local_path` / `resolve_local_dir` return an actual filesystem path,
  downloading into a local cache first if the backend isn't already
  disk-backed. This is what gets handed to `llama-server --model` or
  `from_pretrained(...)`.

For `LocalDiskBackend` these collapse to the same thing (the "cache" is the
canonical copy — no download). For `S3CompatibleBackend` they're genuinely
different steps, and `resolve_local_path` skips the download when a
same-named cached file already verifies against the stored digest.

## What's implemented and tested

- `LocalDiskBackend` is the hard-link/copy, atomic sha256-verified transfer
  that used to live directly in `model_store.py` (`_transfer`,
  `_transfer_verified`), extracted verbatim — not rewritten. `ModelStore` now
  delegates to it for every blob/tree write. This is a pure refactor: the
  full existing test suite (`test_serving_model_store.py`,
  `test_seed_integrity.py`, `test_hf_seed.py`, etc.) passes unchanged against
  it, which is the evidence this didn't alter local-disk behavior.
- `S3CompatibleBackend` (boto3, `endpoint_url` configurable — so it's the
  same class for AWS S3, MinIO, Cloudflare R2, or Backblaze B2) is
  implemented and unit-tested against a mocked S3 (`moto`). It has **not**
  been run against a live bucket of any kind.
- `resolve_storage_backend(root)` is an env-driven factory
  (`DOCIE_STORAGE_BACKEND=local|s3`, `DOCIE_S3_BUCKET`, `DOCIE_S3_PREFIX`,
  `DOCIE_S3_ENDPOINT_URL`, `DOCIE_STORAGE_CACHE_DIR`), not wired into
  anything yet — see below.

## What's deliberately NOT done

- **`ModelStore` does not accept a backend choice yet.** It hardcodes
  `LocalDiskBackend` in `__init__`. Wiring `ModelStore` up to actually take
  `S3CompatibleBackend` — including what "seed" and "deploy" mean when the
  canonical copy isn't on the serving box (cache eviction, concurrent-deploy
  cache races, `index.json` becoming a shared/bucket-backed manifest instead
  of a single host's file) — is real design work, deferred rather than
  guessed at here.
- **No self-hosted MinIO story.** The standard self-hosted S3-compatible
  server is normally run via Docker, which is off-limits for this pass
  (agreed: come back to it explicitly, on request).
- **DVC (dataset/model versioning) is a separate, parked discussion** — a
  different problem (git-tracked version history of which model/dataset was
  used for which run) from this (where the bytes durably live). It can sit
  on top of either backend as its remote; not designed here.
- **No live bucket has been exercised** — AWS S3, R2, B2, or a self-hosted
  MinIO. `S3CompatibleBackend` is correct against moto's S3 emulation, which
  is a high-fidelity mock but not a substitute for a real endpoint (auth,
  network failure modes, multipart thresholds).

## Extra

`pyproject.toml` gained a `storage` optional-dependency group (`boto3`) and
`moto[s3]` in `dev`, for `S3CompatibleBackend` and its tests respectively.
`boto3` is imported lazily inside `_new_s3_client`, so nothing outside a test
or an explicit `DOCIE_STORAGE_BACKEND=s3` pulls it in.
