# DocIE Studio — Serving Control Plane (design)

**Status:** design, for owner review. Preconditions in §0.1 must land before PR-1 code.
**Goal:** turn "click-deploy-and-forget" into a live, memory-aware serving control plane: truth that stays synced with reality, a resource tracker, a Sizing tab, and dynamic load/unload driven by memory pressure (cold-start TTFT accepted).

A grounding note that shapes every section: **only the process-owning container can observe or signal the runtimes.** It spawns `llama-server` via `subprocess.Popen`, keeps the handles in `RuntimeAdapter._processes` (`runtime.py:189`), and holds one long-lived, `@lru_cache`d control plane across deploy jobs (`functions.py:310-314`). The `api` container rebuilds a fresh `ControlPlane` per request (`serving_api.py:28-36`), lives in a different PID namespace, and can neither see nor kill those processes; the `web` container mounts no serving volume at all. So the reconciler and every process mutation must live in the one replica that owns the Popen handles — and §0.1 makes that replica singular and named.

---

## 0. Why records go stale today (the diagnosis, cited)

| Symptom | Root cause in code |
|---|---|
| A dead `llama-server` still shows `ready`. | `reconcile()` (`supervisor.py:147`) runs **only** when explicitly called — `deploy` (`:113/:130`), `stop` (`:132-135`), `reconcile_all` (`:144`), `await_ready` (`:230-264`). Nothing runs it periodically. Once a deployment reaches `READY`, a later crash is never re-observed; the record is frozen. |
| The DB "only reflects what Studio touched." | Observed state (`state`, `endpoint`) is written to Postgres `model_placement` **once, at deploy time** (`control_plane.py:809`, `functions.py:325-358`) and cleared on stop/remove (`control_plane.py:818-826`). Nothing updates it afterward. |
| Deletes don't delete. | `ControlPlane` exposes `up/start/stop/serve/plan` — **no `remove`** (`control_plane.py:326-387`). `_DefaultSupervisor.remove` exists (`:823`) but is unreachable from the facade or any API/Inngest surface. `stop()` keeps the record and its port reservation (`supervisor.py:132-135`). Records and ports accumulate forever. |
| Liveness read from the api is wrong. | `is_running` uses the in-process `_processes` handle when present, else falls back to `psutil.pid_exists(pid)` (`runtime.py:281-287`). From the api container that pid is meaningless (different namespace); after a restart the stale pid may belong to an unrelated process. |
| Sizing/plan can't see reality. | `HostResources` is a one-shot `psutil.virtual_memory().available` snapshot taken at `from_defaults()` (`control_plane.py:245,262-271`); the planner prices **one** model against it (`planner.py:274-282`) and never subtracts running deployments. |

Everything below is built to kill these, in the order they hurt.

### 0.1 Preconditions (must land before any reconciler code)

Two facts falsify the original "singleton by construction" and "columns just appear" assumptions. Both are first-class design decisions, not PR-5 polish — the reconciler is unsafe and non-functional without them.

**P1 — a dedicated single-replica `serving` compose service owns deploys + the reconciler. `--scale worker=1` is a comment, not a pin.** `docker-compose.yml:86-93` is prose; there is no `deploy.replicas`/scale constraint on `worker`, and the same file *invites* scale-out (line 79: "scale freely: `docker compose up -d --scale worker=3`") because extraction/benchmark are replica-safe. Under a realistic "deploy at 1, scale to 3 for a benchmark" sequence, N reconcilers would each mount the shared `serving-state` volume and clobber `deployments.json` (atomic `os.replace` prevents torn reads, not lost writes), and `adapter.health()` — which probes the **advertise** endpoint `worker:port` (`runtime.py:239-240` via `reachable_launch`) — would round-robin onto a replica that never ran the deploy, reporting false-dead and triggering a gated restart of a model another replica owns. `_guard_deterministic_advertise` blocks *deploys* under scale>1, but the reconciler is not a deploy; it runs unconditionally on every replica.

Decision: **split the current `worker` into two compose services registered against the same Inngest app:**
- **`serving`** — `deploy.replicas: 1`. Registers the lifecycle functions (`serving/deploy`, `serving/stop`, `serving/load`, `serving/unload`, `serving/delete`) and hosts the reconciler. It is the sole holder of the Popen handles and the **sole `os.replace` writer of `deployments.json`**. Leader by construction — no leader-election, no filelock, no generation counter (all explicitly retired).
- **`worker`** — scales freely (`--scale worker=N`). Registers `doc/extract`, `benchmark/run`, `serving/seed`, and the GC cron. Never spawns a runtime; it calls llama-server endpoints over HTTP and fires lifecycle events at `serving`.

**Routing is load-bearing and must be explicit.** "`serving` owns deploys" only holds if lifecycle events land on the `serving` replica and never on a `worker`. Mechanism: **register the lifecycle functions only in the `serving` service's function set and the extraction/benchmark functions only in the `worker` set** (per-service `inngest.serve(...)`/Connect app with disjoint `functions=[...]` lists). A `worker` that receives a `serving/*` trigger has no handler for it; the event is delivered only to the app that registered it. Deploy-time `_guard_deterministic_advertise` becomes redundant belt-and-suspenders once only the single `serving` replica ever deploys.

**P2 — explicit `ALTER TABLE` migration for the new `model_placement` columns.** `catalog.py` runs `Base.metadata.create_all` with **no migrations** and documents this exact hazard for `size_bytes` (`catalog.py:40-41`): `create_all` never adds columns to a table that already exists. `model_placement` already shipped (commits `f39b7d7`/`5579df1`), so on any live DB the new observed columns silently would not be created and the reconciler's first upsert throws `UndefinedColumn`. Fresh DBs are fine; existing deployments break.

Decision: ship an explicit forward migration as a PR-1 precondition, mirroring the `size_bytes` caveat, and stop trusting `create_all`:

```sql
ALTER TABLE model_placement
  ADD COLUMN IF NOT EXISTS phase           text,        -- hot|cold|loading|evicted|failed (§4)
  ADD COLUMN IF NOT EXISTS pid             integer,     -- serving-namespace pid, advisory only
  ADD COLUMN IF NOT EXISTS pid_create_time double precision, -- psutil create_time() at spawn — PID-reuse guard
  ADD COLUMN IF NOT EXISTS rss_bytes       bigint,      -- measured RSS, last cycle
  ADD COLUMN IF NOT EXISTS health_ok       boolean,     -- last /health result
  ADD COLUMN IF NOT EXISTS last_probe_at   timestamptz,
  ADD COLUMN IF NOT EXISTS last_error      text;
```

Note what is **not** here: `activation` and `pinned` are **not** Postgres columns. They are lifecycle-control metadata and live in `deployments.json` (§1, fix #5) so the DB-optional routing contract survives.

---

## 1. Truth model

### States, homes, one writer per home

| Concern | Source of truth | Home | Writer | Readers |
|---|---|---|---|---|
| **Desired** — what should be running, plus lifecycle controls (`activation`, `pinned`) | `DeploymentSpec` + `desired_state` + lifecycle block | `deployments.json` (shared `serving-state` volume, `DOCIE_SERVING_HOME=/app/.serving`) | **`serving` service only** (deploy/stop/load/unload/reconcile) | `serving`, resolver |
| **Recency** — per-deployment last-served timestamp (LRU/idle input) | monotonic max-timestamp sidecar | `serving-state/recency/<name>` (same volume) | **`worker` extract path** (originator) → folded by reconciler | reconciler |
| **Observed** — what *is* running | extended `model_placement` row | Postgres (Studio owns) | **reconciler only** | api, Studio, resolver |
| **Resources** — node RAM & per-proc RSS | new `serving_node` snapshot | Postgres (one row) | **reconciler only** | api, Studio, sizing |

`deployments.json` keeps its role as *desired* state and gains a small lifecycle block (`activation`, `pinned`). The **observed** surface moves entirely into Postgres so the api/Studio never depend on cross-container process introspection.

**Why lifecycle metadata is in the file, not Postgres (fix #5).** Routing is deliberately DB-optional: `profile_resolver` reads the on-disk store index first and the catalog only as fallback, with repeated "no `DATABASE_URL` must never break routing" guards (`profile_resolver.py:138-210`). If `activation`/`pinned` lived only in `model_placement`, the extract path's "cold+`managed` → auto-load" decision would force a Postgres cross-lookup for a bare-`deployment` request that otherwise routes purely through `deployments.json` — regressing a property the codebase intentionally preserves. Putting `activation`/`pinned` in `deployments.json` (worker-owned, DB-optional, the natural home of desired state) keeps Postgres for display + RSS only, and removes the awkwardness of the reconciler reading its own eviction inputs out of the surface it writes.

**Why recency is a sidecar, not a field in `deployments.json` (fix #5, made safe under P1).** `last_served` must be written by the code path that actually serves traffic — the **`worker` extract path** — because the gateway is a stateless passthrough with no recency signal (`gateway.py`) and hot-model extraction calls `llama-server` directly, so only the workers see a request. But `deployments.json` has exactly one `os.replace` writer by P1 (the `serving` service); letting N scaled workers rewrite that file reintroduces the P1 lost-update race. Resolution: `last_served` is a **monotonic max-timestamp**, so last-write-wins is the correct semantics and a lost update is harmless (worst case: an idle unload slips one ~10s cycle). Each extract writes a per-deployment sidecar `serving-state/recency/<name>` (single-key atomic write, never touches `deployments.json`), and the reconciler folds it into the LRU/idle view each cycle. Same file domain, DB-optional, honors "extract path writes last_served" literally, and preserves the single desired-state writer.

`model_placement` already carries `state`, `endpoint`, `negotiated_style` (`catalog.py:49-79`); §0.1/P2 extends it with the observed columns.

### The reconciler (in `serving`, single writer, singleton by construction)

A background asyncio task started at the `serving` service's Connect seam, alongside `connection.start()` (`worker.py:46-58`). It **must** run against the shared `_serving_control_plane()` (`functions.py:310`), not a fresh `PersistentSupervisor` — only the shared instance holds the `_processes` Popen handles that make liveness reliable. Singleton is now real, not asserted: P1 guarantees exactly one `serving` replica, so exactly one reconciler exists and it owns every process it probes.

**Intra-process serialization — the lock the original design wrongly waived (fix #4).** Even at one replica, the reconciler and concurrent Inngest handlers (`deploy`/`stop`/`load`/`unload`) all share the one `@lru_cache`d `ControlPlane` → one `PersistentSupervisor` → one `_records` dict and one `deployments.json`, mutated with **no internal synchronization**. Interleaved read-modify-`_save()` cycles inside a single process lose writes. So:

- **One thread-safe lock** (`threading.Lock`, not `asyncio.Lock`) funnels *all* supervisor mutations — reconciler cycles and every lifecycle handler.
- A reconcile cycle does blocking work: health probes (up to `health_timeout_seconds` each) + `fsync` on `_save`. Running that on the event loop would stall the Connect heartbeats that the `up()` comment (`control_plane.py:352-356`) exists to protect. **The cycle runs via `asyncio.to_thread`**, and because it blocks a worker thread while holding the lock, the primitive has to be a thread lock. The handlers acquire the same lock around their mutations.

Each cycle (default ~10s), for every record in `deployments.json`:

1. **Liveness by health, not by pid.** pid presence is a hint; the authority is a successful `/health` (`runtime.py:304-312`). Guard PID reuse: compare `psutil.Process(pid).create_time()` against the stored `pid_create_time`; a mismatch means "not our process → treat as dead." The `create_time()` call **must catch `psutil.NoSuchProcess`** (raised when the pid is already gone) and map it to "dead" rather than letting it bubble (fix #8). Only publish `phase=hot`/`health_ok=true` on a real health pass. At one replica the advertised `serving:port` endpoint resolves to this same container, so self-probing is well-defined; the stale `await_ready` comment claiming "127.0.0.1 IS reachable" predates `reachable_launch` and should be corrected to name the advertise host (fix #8).
2. **Measure RSS** of the live process (§2) → `rss_bytes`.
3. **Repair vs. garbage-collect — where naive `reconcile_all()` is dangerous.** The existing repair path respawns any `RUNNING` deployment whose process died, up to `max_restarts=5` under the default `ON_FAILURE` policy (`supervisor.py:150-205,302-312`). Blindly looping `reconcile_all()` would **auto-restart everything that crashes** — the opposite of "unload to free memory," and a direct route to eviction storms. The reconciler splits by intent:
   - `desired=RUNNING`, healthy → publish `hot`.
   - `desired=RUNNING`, dead, **gated restart** → respawn only if (a) restart budget remains **and** (b) a live fit-check passes (§2). A model that OOMs on load must not crash→respawn→OOM 5× in seconds. This is where the planner formula finally earns its keep on the *start* path (today it is plan-only, never consulted at launch).
   - `desired=STOPPED` → ensure no process; publish `cold`/`evicted` per the lifecycle block (§4).
   - record marked deleted → actually delete (below).
4. **Fast death-declaration, separate from slow-load tolerance.** Deploys set `health_failure_threshold=60` so `await_ready` won't degrade-and-kill a still-loading GGUF (`control_plane.py:737,792`). If the reconciler reused 60, a crashed `READY` deployment would take ~10 min (60 misses × 10s) to be declared dead. Give the reconciler its own low threshold for the `READY → unreachable` transition (2–3 misses) while keeping the high tolerance for `STARTING` cold-load.
5. **Publish** the observed row + resources snapshot to Postgres via `UPDATE` (see row-lifecycle rule below). Structured per-cycle logging doubles as the "trace" the owner inspects.

**Observed-row lifecycle — reconciler owns it, only DELETE clears it (coherent with fix #3).** The original "cleared on stop" line is removed: `stop`/`unload`/`evict` all leave the row present and **UPDATE** it (`phase=cold`/`evicted`), so display and auto-reload metadata survive. The row is deleted in exactly one place — real teardown. `endpoint` is `NOT NULL` (`catalog.py:70`), so cold/evicted rows write `endpoint=""` (empty string, not `NULL`) and every reader treats `""` as "no live endpoint" (fix #8).

### Deletions that delete

Add a real teardown reachable from the api: api fires a `serving/delete.requested` event → handled on `serving` → `_DefaultSupervisor.remove` (`control_plane.py:823`, already kills the process and drops the record) → **DELETEs** the `model_placement` row → frees the port. Expose it on `ControlPlane` (the missing `remove` method) and register the delete function in the `serving` set.

### Multi-writer safety — now genuinely closed

- **Cross-container:** P1 makes the `serving` replica the single `os.replace` writer of `deployments.json` and the single reconciler/publisher of Postgres. Scaled `worker`s only write per-deployment recency sidecars (last-write-wins, harmless).
- **Intra-process:** the thread lock above serializes the reconciler against the handlers.
- The one apparent external competitor, host-native `docie up`, writes a **different** `DOCIE_SERVING_HOME` (code default `~/.local/share/docie-bench/serving`, `control_plane.py:252-257`), not the shared `serving-state` volume, so it never races the worker's file — declared **out of scope** (§7). No filelock/generation-counter is engineered for a race that does not occur in the target topology.

**lru_cache staleness is a feature, not a bug:** the cached supervisor is *desired*; the reconciler keeps the observed view fresh and republishes.

**Honest scope of the fix (fix #8):** Postgres is **not required** to kill *liveness* staleness. The api mounts `serving-state` and reads `deployments.json` fresh per request (`serving_api.py:28-36`); once the reconciler `_save()`s each cycle, the file already de-stales the Board's deployment list. Postgres is still needed for RSS/node snapshots and for the `web` container (which mounts no volume). So the Deploy/Board tab reads the observed Postgres surface when the DB is up, and degrades to the fresh-but-lean `deployments.json` when it is not (§7).

---

## 2. Resource tracker

**Measured on `serving`, published to Postgres, read by the api** — or the sizing number is simply wrong. `from_defaults()` measures whatever process calls it (`control_plane.py:245`); a Sizing endpoint running in the api would report the api container's RAM, not the serving node's.

### Node total/free — and how Docker Desktop lies

Two honest failure modes to design around:

- `psutil.virtual_memory()` inside the container reads the **WSL2 VM's** `/proc/meminfo`, not the container cgroup, and the VM total is elastic. Prefer cgroup v2 (`/sys/fs/cgroup/memory.max`, `memory.current`) when meaningful.
- But compose sets **no memory limit** on the serving service, so `memory.max` reads `max` (unlimited) and you fall back to the VM total anyway. **Set an explicit `mem_limit` on the `serving` service** so cgroup numbers become authoritative and sizing has a real denominator. Publish `source: cgroup | vm` in the snapshot so the UI can flag a soft number.

### Per-deployment RSS

`psutil.Process(pid).memory_info().rss` for each live runtime, keyed by the reconciler's PID-reuse-guarded pid. Caveat that matters: **llama.cpp mmaps the GGUF**, so RSS is low right after load and climbs as pages fault in. Calibrating from fresh RSS over-provisions. Calibrate from **steady-state** RSS (sampled after warm-up), not first-probe RSS.

### Predicted footprint (per instance) — reuse the FORMULA, not the plumbing

Reuse the planner's *formula* (`planner.py:274-282`) as the *floor*, but **do not** reuse its input path. `_DefaultPlanner.plan → registry.get(model)` (`control_plane.py:500-510`) reads the **registry** `ModelManifest`; store GGUFs live in `ModelStore`/catalog, not the registry, so `plan()` on a store model raises (fix #6). Weights come from `ModelStoreEntry.size_bytes` (or an on-disk `stat` of the GGUF), never the registry:

```
predicted_bytes ≈ weights + kv_cache + overhead
  weights   = ModelStoreEntry.size_bytes (or on-disk stat) × quant_factor   (≈1.0 for a GGUF already at target quant)
  kv_cache  = per_token_kv × context_length × n_parallel                    (llama.cpp: --ctx-size, --parallel)
  overhead  = fixed runtime slab (llama-server ≈ 0.3–0.5 GB; +mmproj for vision families)
```

Calibrate against reality:
```
footprint(model) = max( steady_state_rss_if_running,  predicted_bytes )
```
Trust the measurement once we have one; fall back to the formula for models never yet run. Persist observed footprints per model so sizing improves over time.

---

## 3. Sizing engine + tab

### Engine (serving-published, api-served)

Inputs: the store models (`catalog.list`), the live observed placements + RSS, and the node snapshot — **all from Postgres**, so sizing is synced with live deployments *by construction* (it reads observed state, never records).

**Avoid the double-count trap:** `psutil` "available" *already* nets out running `llama-server` processes. If you compute `free = available` **and** also subtract predicted footprints of running deployments, you subtract them twice. Pick one model and state it in the UI:

- **(a)** `free = measured_available`; price prospective instances by predicted-per-instance. Simple; inherits the mmap under-count.
- **(b)** `free = total − Σ predicted(running)`. Consistent, formula-driven.
- **Recommended hybrid:** `free = total − Σ max(observed_rss, predicted)` — honest about what's resident, conservative about the mmap ramp.

Fit for a candidate model X:
```
fits = floor( (free − safety_margin) / footprint(X) )
```
`safety_margin` is explicit (e.g. 10–15% of total or a fixed 1 GB), surfaced in the UI — not hidden.

### API
```
GET /v1/serving/resources   → { total, free, source: cgroup|vm, safety_margin,
                                deployments: [{name, rss_bytes, phase}], sum_rss }
GET /v1/serving/sizing       → { free_effective,
                                per_model: [{name, footprint_bytes, fits_now}],
                                assumptions: {context_length, n_parallel, margin} }
POST /v1/serving/sizing/whatif  { plan: [{model, instances, context_length}] }
                             → { total_predicted, remaining, ok, per_item: [...] }
```

### Tab
- **Capacity bar:** total / used (Σ RSS) / free / margin, with a soft-number badge when `source=vm`.
- **Per-model fit table:** model · family · footprint (measured or predicted, labeled) · **fits now = N** · [Deploy 1].
- **What-if selector:** stage a mix ("2× lfm2 @ 4k ctx + 1× nuextract3") → live remaining-RAM readout, red when negative. Backed by the what-if endpoint so the math matches the deploy path exactly.

---

## 4. Dynamic load/unload lifecycle

### State model — mapped onto the existing enum, not a parallel one

Map onto `DesiredState` (`supervisor.py:21-24`):

- **`DesiredState.RUNNING` = hot** (process up, healthy).
- **`DesiredState.STOPPED` = cold/evicted** (record + port reservation retained, no process).

Publish a finer `phase` for the UI (`hot | loading | cold | evicted | failed`), but authority stays with `DesiredState` + health.

**Unload is a DISTINCT path from `stop()` (fix #3) — this is the sharpest correction.** The original design said "unload = reuse `stop()`," but the *facade* `control_plane.stop()` (`:818-821`) calls `_clear_placement`, which **deletes** the `model_placement` row (`catalog.py:233-241`). An evicted deployment must **retain** its row so the extract path knows to auto-reload it. Therefore:

- **`unload(name)`** — a new path that: acquires the supervisor lock, flips `DesiredState → STOPPED`, kills the process, retains the port reservation, and **UPDATEs** the row to `phase=evicted, activation=managed, endpoint="", rss_bytes=0` (never `_clear_placement`).
- **`stop(name)`** (user Stop) — flips `DesiredState → STOPPED`, kills the process, UPDATEs the row to `phase=cold, activation=manual, endpoint=""`. It, too, stops calling `_clear_placement` — deletion is delete's job only.
- **`remove(name)`** (delete) — the *only* path that DELETEs the row and frees the port (§1).

**User-stop vs evicted-stop.** Both land in `STOPPED` but behave differently, driven by the `deployments.json` lifecycle block:
- `activation=manual` (user pressed Stop) → `phase=cold`; a request does **not** auto-load it; stays cold until explicit Start.
- `activation=managed` (autoloader unloaded it for memory) → `phase=evicted`; a request **may** auto-reload it.
- `pinned=true` → never chosen for eviction.

### Transitions

| From → To | Trigger | Executor |
|---|---|---|
| cold/evicted → loading → hot | first request to a `managed` evicted deployment; or manual Start | `serving` (spawn + `await_ready`) |
| hot → evicted (idle unload) | no request for `idle_ttl` (e.g. 15 min), per recency sidecar | reconciler |
| hot → evicted (pressure) | memory pressure, victim selected | reconciler |
| any → deleted | explicit delete | `serving` delete handler |

### Cold-start on demand — an event to `serving`, symmetric for every caller

Under P1, **no `worker` and no `api` can spawn** — only the single `serving` replica holds the Popen handles. So load-on-demand is uniformly an event, and the resolver stays pure. `resolve_extraction_profile` is shared and deliberately side-effect-free (`profile_resolver.py:95-107,293-353`); **do not** bury a load side-effect in it.

- **`worker` extract path** (`extract_document`, `functions.py:132`): before resolving, if the target deployment is cold+`managed`, fire `serving/load.requested` and await ready (realtime channel or bounded poll), then resolve and extract. The request **waits and loads**, never 502s. It also writes the recency sidecar (§1) on success. TTFT for that first request = model load time (seconds to tens of seconds for a large CPU GGUF) — **explicitly accepted**, per the owner's priority.
- **Sync api extract path**: identical shape — fires the load event and either waits on the realtime channel or returns `202 Accepted` / `503 Retry-After`. State this in the UI.
- **`serving` load handler**: spawns + `await_ready`, publishes readiness, UPDATEs the row `phase=loading → hot`.

**The cold-start pileup lock is now global by construction (resolves what was a caveat).** Because loads execute only in the single `serving` replica, a per-deployment in-process load lock/queue there makes N concurrent requests (from any number of scaled workers, from the api, or from the reconciler) funnel to **one** spawn. There is no cross-container `0.0.0.0:port` double-bind, because there is only one container that binds — this is a direct dividend of P1, not a separate mechanism.

**Cold-load timeout + step budget (fix #7).** `await_ready` defaults to `timeout_s=60`; a large CPU GGUF can exceed that, and a timeout returns `STARTING`/`_is_live=False`, failing an extraction that would have been ready at 90s. Two decisions:
1. Give load-on-demand a **generous, size-aware `await_ready` timeout** — scale the budget off `footprint(X)`/on-disk size rather than a flat 60s.
2. **Make the load idempotent and confirm the step budget.** `extract_document` (`functions.py:128-131`) declares **no** step-timeout or retry override, so its `ctx.step.run("extract", …)` inherits the Inngest default; a cold load that outruns that default would let the step retry and re-fire the load. The per-deployment load lock + an "already hot? no-op" guard make a retry harmless (it cannot double-spawn) regardless of the exact budget. Precondition for PR-4: either **run the load as its own durable step** (`ctx.step.run("load", …)`) with an explicit size-aware timeout so it is checkpointed separately from `extract`, **or** confirm the extract step timeout exceeds max cold-load. Prefer the separate durable step — it is the robust answer and does not depend on reading an undocumented default.

### Eviction policy

When a load would exceed `free − margin`: pick victims among `hot`, non-`pinned` deployments by LRU (least-recently-served, from the recency sidecars), unload (the distinct `unload` path above) until the candidate fits, marking victims `evicted`+`managed`. Guard against **eviction storms**: never evict to load something that itself won't fit (fit-check first), rate-limit evictions per cycle, and respect a minimum hot-time so a just-loaded model isn't immediately evicted.

### Composition with the extract path
The resolver's `_is_live` gate (`READY` + endpoint, `profile_resolver.py:110-112`) is unchanged; `serving` simply ensures the deployment *is* live before the resolver runs. `store:<name>` refs already resolve through the placement row (`placement_resolver.py`), which the reconciler now keeps fresh — so a stale `ready` placement (empty `endpoint`) stops routing traffic into a dead endpoint.

---

## 5. API + UI surface

### Endpoints
Reads (api, from Postgres observed state, degrading to `deployments.json`):
```
GET /v1/serving/deployments        → observed rows (phase, rss, health, endpoint, last_error)
GET /v1/serving/resources          → node snapshot (§3)
GET /v1/serving/sizing             → fit table (§3)
```
Actions (api fires `serving/*` events at the `serving` service, returns the event/run id):
```
POST /v1/serving/deployments/{name}/load
POST /v1/serving/deployments/{name}/unload    (distinct path, §4 — UPDATEs row to evicted)
POST /v1/serving/deployments/{name}/stop      (manual → activation=manual, phase=cold)
POST /v1/serving/deployments/{name}/pin        {pinned: bool}
DELETE /v1/serving/deployments/{name}          (real teardown, §1 — the only DELETE)
POST /v1/serving/sizing/whatif
```
The existing `/deployments`, `/ports`, `/store`, `/families` (`serving_api.py`) stay; `/deployments` is repointed at the observed surface.

### Board UX (fits the existing LiteLLM-style shell)
- **Deployments table:** name · model · family · **phase** (hot/cold/loading/evicted/failed with a live dot) · RSS · port · endpoint reachability · actions [Load][Unload][Pin][Delete]. Delete that actually removes the row.
- **Capacity header:** the §3 capacity bar, always visible.
- **Sizing tab:** §3.
- **Honesty affordances:** a "measurements are soft (VM, no cgroup limit)" badge when applicable; `last_error` surfaced inline for failed/degraded rows (the reconciler captures the runtime's stderr tail, `supervisor.py:172-174`); an "observed state unavailable — showing worker-local desired state" banner when Postgres is down.

---

## 6. Phased PR plan

Preconditions P1 (dedicated single-replica `serving` service) and P2 (`ALTER TABLE`) land **before** PR-1 code. Order thereafter: **staleness dies first**, then measurement, then sizing, then dynamic lifecycle. Each PR is shippable and testable.

**PR-0 (preconditions) — topology + migration.**
- Scope: split `worker` into `serving` (`deploy.replicas: 1`, lifecycle functions + reconciler seam) and freely-scaled `worker` (extraction/benchmark); disjoint per-service Inngest function registration; the `ALTER TABLE model_placement` migration; `mem_limit` on `serving` (pulled forward from PR-2 so the reconciler's node snapshot is authoritative from cycle one).
- Files: `docker-compose.yml`, `worker.py` (two app registrations / seam), `catalog.py` (migration + stop trusting `create_all` for these columns).
- Live-verify: `--scale worker=3` and confirm only the `serving` replica handles `serving/deploy.requested` and only it holds Popen handles; existing DB gains the columns.

**PR-1 — Reconciler + observed-state publish + real delete.**
- Scope: in-`serving` reconcile loop under `asyncio.to_thread` + one thread lock shared with handlers (health-authoritative liveness, `NoSuchProcess`-guarded PID-reuse check, fast death-declaration, gated restart); publish observed rows via `UPDATE` each cycle (`endpoint=""` for cold/evicted); wire a real `remove`/`DELETE` (the only path that clears a row).
- Files: `worker.py` (task seam + lock), `supervisor.py` (fast threshold, lock, delete flag), `control_plane.py` (expose `remove`, share lru_cache, lock the facade mutators), `catalog.py` (observed upsert = UPDATE), `functions.py` (`serving/delete.requested` on the `serving` set), `serving_api.py` (`/deployments` → observed, DB-down degrade).
- Stub tests: scripted-adapter (crash mid-life) → READY→FAILED→GC without respawn-storm; `create_time` mismatch and `NoSuchProcess` → treated dead; delete removes row + frees port; **concurrent-invocation test** (reconciler cycle interleaved with a `stop` handler) → no lost write, asserting the lock. Note plainly: the cross-container `deployments.json` race is *designed out* by P1, so there is nothing left to stub for it; the intra-process race is covered by the concurrency test.
- Live-verify: deploy a GGUF; `kill` the `llama-server` inside `serving`; Board flips to failed within ~2 cycles; Delete makes the row and port vanish.
- Risks: respawn storms if the restart gate is wrong; fast/slow threshold split must not kill slow cold-loads — mitigated by the STARTING-vs-READY threshold separation.

**PR-2 — Resource tracker + node snapshot publish.**
- Scope: RSS per process; cgroup-v2-first node total/free with VM fallback + `source` flag; steady-state footprint calibration persisted per model, weights from `ModelStoreEntry.size_bytes` (not the registry).
- Files: new `serving/resources.py`, `catalog.py` (`serving_node` snapshot), reconciler publishes snapshot. (`mem_limit` already landed in PR-0.)
- Stub tests: injected `/proc`/cgroup readers → cgroup preferred when limited, VM fallback flagged; footprint = `max(rss, predicted)`; `ModelStoreEntry` weights path exercised (no registry lookup).
- Live-verify: compare published free/RSS against `docker stats` and `htop` in the WSL2 VM; confirm the mmap ramp.
- Risks: Docker Desktop measurement lies — mitigated by `mem_limit` + soft-number badge.

**PR-3 — Sizing engine + tab.** *(hard dependency on PR-2's publish.)*
- Scope: `/resources`, `/sizing`, `/sizing/whatif`; the tab (capacity bar, fit table, what-if).
- Files: `serving_api.py`, new `serving/sizing.py` (reuses the planner formula with `ModelStoreEntry` weights), frontend Sizing tab.
- Stub tests: fixed snapshot + store list → deterministic `fits_now`; double-count guard asserted (hybrid formula).
- Live-verify: deploy N instances until the bar predicts 0; confirm the (N+1)th deploy fails the fit-check.
- Risks: double-counting; misleading margins — mitigated by the stated free-RAM model + visible margin.

**PR-4 — Dynamic load/unload lifecycle.**
- Scope: `activation`/`pinned` in `deployments.json`; recency sidecars written by the extract path + folded by the reconciler; idle-TTL unload; cold-start-on-demand as a `serving/load.requested` event (separate durable load step, size-aware `await_ready`, idempotent under the global load lock); eviction (LRU, pin, fit-gated); the distinct `unload` path (UPDATE→evicted, never clear); load/unload/pin endpoints.
- Files: `functions.py` (load-on-demand event + recency write on `worker`; load handler on `serving`), reconciler (idle + eviction + recency fold), `supervisor.py`/`control_plane.py` (`unload` path, activation/pinned in the file), `serving_api.py`, frontend actions.
- Stub tests: scripted clock + sidecars → idle unload after TTL; request to evicted `managed` → reload; eviction never picks pinned; storm guard (no evict-to-not-fit); global load lock → N concurrent requests trigger one load; `unload` retains the row (asserts fix #3).
- Live-verify: deploy two models that don't co-fit; request the cold one; watch the hot one evict and the cold one load; measure TTFT; confirm a >60s cold load does not double-spawn under a step retry.
- Risks: eviction storms, cold-start pileup — mitigated by the min-hot-time/rate-limit/fit-gate and the global in-`serving` load lock.

**PR-5 (optional) — Board polish + observability.** Reachability column, `last_error` surfacing, per-cycle reconcile trace view, soft-measurement badges, DB-down banner.

---

## 7. Non-goals + sharpest failure modes

### Explicit non-goals (now)
- **GPU.** CPU-first llama-server only; the GPU branches in the planner stay dormant.
- **Multi-node / cluster scheduling.** One `serving` node (P1); `worker` scale-out is for extraction/benchmark only.
- **True autoscaling / replica load-balancing.** One instance per deployment (the supervisor refuses `replicas>1`, `control_plane.py:695-696`). Sizing *advises* how many fit; it does not auto-provision.
- **Host-native `docie up` in the compose control plane** — different `DOCIE_SERVING_HOME`; out of scope (§1).
- **Preemptive/predictive loading.** Load is reactive (on request) or manual; no traffic forecasting.
- **Leader election / distributed lock over `deployments.json`.** Solved structurally by the single-replica `serving` service; explicitly *not* built.

### Sharpest failure modes & mitigations
| Failure | Why it bites here | Mitigation |
|---|---|---|
| **Reconciler under scale>1** | `--scale worker=1` is a comment; N reconcilers clobber `deployments.json` and mis-probe the round-robin advertise host. | Dedicated single-replica `serving` service owns deploys + reconciler; disjoint Inngest function registration (P1). |
| **New columns silently absent** | `create_all` never alters an existing table (`catalog.py:40-41`). | Explicit `ALTER TABLE` migration as a PR-0 precondition (P2). |
| **Unload erases eviction metadata** | `stop()` facade calls `_clear_placement` → deletes the row (`catalog.py:233-241`). | Distinct `unload` path UPDATEs row to `evicted`/`managed`; only `remove` DELETEs (fix #3). |
| **Lost writes inside one process** | Reconciler + handlers share one unsynchronized `PersistentSupervisor`. | One thread lock funnels all mutations; cycle runs via `asyncio.to_thread` (fix #4). |
| **Routing re-coupled to Postgres** | `activation`/`pinned` in the DB would force a cross-lookup on a DB-optional path. | Lifecycle metadata in `deployments.json`; recency in per-deployment sidecars; Postgres for display/RSS only (fix #5). |
| **No LRU/idle data source** | `gateway.py` is a stateless passthrough; no last-served signal exists. | Extract path writes a monotonic recency sidecar; reconciler folds it (fix #6). |
| **Flapping reconciler** | Repair-by-respawn (`max_restarts=5`) + a tight loop = crash storms. | Gate every (re)start on a live fit-check; respect restart budget; separate fast-death (READY) from slow-load (STARTING) thresholds. |
| **RSS lies on Docker Desktop** | psutil reads the WSL2 VM; llama.cpp mmap under-counts fresh RSS. | cgroup-v2-first + explicit `mem_limit`; steady-state RSS; `max(rss, predicted)`; soft-number badge. |
| **Eviction storm** | Pressure evicts to load, the loaded one triggers the next eviction. | LRU + min-hot-time + fit-before-evict + per-cycle rate limit; pinned deployments. |
| **Sizing plan() raises on store models** | `_DefaultPlanner.plan` reads the registry, not `ModelStore`. | Reuse the formula; take weights from `ModelStoreEntry.size_bytes`/on-disk stat (fix #6). |
| **Cold load outruns await_ready / step budget** | Default `timeout_s=60`; `extract_document` sets no step timeout. | Size-aware `await_ready`; load as a separate durable idempotent step under the global load lock; confirm the extract step budget (fix #7). |
| **Cold-start pileup** | N concurrent requests to one cold model. | Global in-`serving` load lock — one replica binds the port, so N requests → one spawn (dividend of P1). |
| **DB down** | Observed surface is Postgres. | Degrade to fresh `deployments.json` (staleness already dies via reconciler `_save`); visible banner; RSS/sizing unavailable until DB returns. Postgres is *not required* for staleness-death (fix #8). |
| **Stale routing into a dead endpoint** | Extraction routes on `READY`+endpoint. | Reconciler UPDATEs the placement off `ready` (`endpoint=""`) within a couple of cycles; traffic stops fast. |
