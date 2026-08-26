# ASR rollout and rollback

This runbook is the release gate for the ASR milestone. It applies to the final
`dev-asr` candidate and to production recovery. `master` remains production;
feature PRs must continue to target `dev-asr` until the milestone exit audit is
complete.

## Release-gate smoke

Prerequisites are Docker Engine with Compose, Git, outbound access to Hugging
Face, at least 4 GiB available to the serving container, and free local ports.
The gate generates its own non-copyrighted WAV and temporary credentials. It
uses `Systran/faster-whisper-tiny.en` on CPU/int8, records the immutable Hub
revision observed before and after download, and fails if the revision changes
during the seed.

Run from the repository root on the exact candidate commit:

```bash
make asr-compose-smoke
```

The command creates a unique `docie-asr-smoke-...` Compose project. It starts
only Postgres, Redis, Inngest, the singleton serving control plane, and the
public API. It then proves this sequence:

1. API health becomes ready.
2. The Studio API seeds the converted Whisper snapshot into the canonical
   store as `asr_whisper`.
3. The Studio deploy event produces a ready managed ASR deployment whose
   endpoint is private `http://serving:<port>/v1` routing.
4. An arbitrary Hub-like model selector is rejected with `404` before it can
   become a download instruction.
5. A generated WAV is transcribed through the public authenticated API and the
   verbose result contains model, backend, duration, processing time, and text.
6. The deployment is deleted through its lifecycle API.
7. The exact smoke project and its smoke-only volumes are removed.

Machine-readable evidence is written to
`artifacts/asr-compose-smoke/evidence.json`; the last 300 lines of container
logs are written alongside it. Evidence includes the Git commit/tree and dirty
flag, model repo/revision, audio SHA-256, triggers, observed deployment,
negative routing check, transcription, timings, image listing, and cleanup
status. Neither file contains the generated API key or database/Inngest
secrets. Preserve both outside the worktree for the release record.

The release gate refuses a dirty worktree. `--allow-dirty` exists only for an
early diagnostic run; evidence from such a run is not releasable.

For a retry without rebuilding images:

```bash
python -m docie_bench.asr.compose_smoke --no-build \
  --evidence artifacts/asr-compose-smoke/retry.json
```

If the gate fails, read the JSON `result.error` first, then the log file. Check
`docker info`, available Docker VM memory, Hugging Face reachability, Inngest
worker registration, the seed run error, and the deployment `last_error`. The
harness still attempts exact-project cleanup after every failure. If that
cleanup reports a nonzero exit, use only the project name recorded in evidence:

```bash
docker compose --project-name <recorded-docie-asr-smoke-project> down \
  --volumes --remove-orphans
```

Never substitute the production Compose project name in that command.

## Rollout checklist

Before opening or merging `dev-asr -> master`:

- all milestone issues are closed by reviewed PRs into `dev-asr`;
- the candidate commit and tree match the reviewed integration branch;
- CI is green on the synthetic production merge, including Python and Studio;
- the release-gate smoke passes on that exact candidate and its evidence is
  retained;
- the current production commit, tree, Compose project name, environment file,
  image digests, database backup/restore command, and volume names are recorded;
- no caller-controlled model id can bypass the managed ASR deployment allowlist;
- an operator explicitly authorizes the production merge and rollout.

Roll out immutable images by digest where the deployment platform permits it.
Start infrastructure first, then the singleton `serving` service, worker, API,
and web. Do not scale `serving` above one. After rollout, verify `/healthz`,
Inngest registration for both apps, `/v1/serving/resources`, the ASR store and
deployment views, one authenticated transcription, and normal non-ASR traffic.

## Rollback procedure

Rollback is a service recovery operation, not a Git-history rewrite.

1. Stop new ASR traffic at the ingress/client layer. Let in-flight synchronous
   requests finish or time out. Pause submission of durable ASR jobs.
2. Record affected job ids and the current deployment record. Use
   `POST /v1/serving/deployments/{name}/unload` to stop model memory while
   retaining a reloadable record, or `DELETE /v1/serving/deployments/{name}` to
   remove the runtime, port reservation, and placement. Confirm through the
   deployment list; do not rely on a UI badge alone.
3. Redeploy the previously recorded production image digests/commit. Restore
   all services to one mutually compatible release; do not leave an older API
   paired indefinitely with a newer serving/worker protocol.
4. Keep Postgres and the `serving-state`, `artifact-store`, and `hf-cache`
   volumes. The ASR tables are additive and created idempotently; older code
   ignores them. Do not drop ASR tables during an application rollback.
5. Keep queued/settled job rows and output artifacts for diagnosis. Raw inputs
   follow their requested retention (`delete_after_completion`, `retain_7d`, or
   `retain_30d`). Do not manually delete content-addressed blobs: shared GC is
   reference-aware across Studio, batch extraction, and ASR.
6. Keep seeded model snapshots and Hub cache unless corruption is proven. They
   are inert without a live deployment and make a corrected roll-forward
   faster. If corruption is proven, quarantine the exact store entry/cache path
   after resolving it; never remove the broad serving or cache volume.
7. Verify recovery by Git tree/image digest, container health, Inngest app
   registration, non-ASR production smoke, and absence (or intentional cold
   state) of the ASR deployment. Re-enable traffic only after these checks.

If a schema or data restore is genuinely required, stop all application writers
first and use the environment's tested database backup/restore procedure. A
routine ASR application rollback does not require a database restore.

## Roll forward after rollback

Fix the fault on a new issue branch targeting `dev-asr`, rerun CI and this
Compose gate on the new integration tree, and open a fresh reviewed production
PR. Do not reuse stale evidence from the rolled-back candidate.
