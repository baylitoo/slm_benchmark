# Fleet runbook: adding, checking and removing a serving node

Operating a multi-node fleet (#538). For *how* the pieces work, read
{doc}`../architecture/serving`; for the order to do things in the first time,
including turning an existing single machine into the first node, read
{doc}`fleet-cookbook`. This page is the per-task reference.

The single most useful thing on this page is
[`docie-serving doctor`](#check-the-node-before-it-takes-traffic). Most of what
can be wrong with a serving node **fails as a wrong answer rather than as an
error** — the node starts, publishes capacity, accepts deploys, and then serves
requests for its models from a different machine. Run the doctor before a node
takes traffic and after every `.env` change.

## Before the first node

The central stack publishes Postgres and Inngest on loopback, so no other
machine can reach them. In the **central** `.env`:

```
POSTGRES_BIND_HOST=<central host's private IP>
INNGEST_BIND_HOST=<central host's private IP>
```

That one address, never `0.0.0.0`: Postgres answers to real credentials, and
the Inngest dashboard has no authentication at all. Put the fleet on a private
network (WireGuard is what this was designed against) and use those addresses
throughout.

## Add a node

1. **Copy the central `.env`** to the new machine, so every application setting
   matches, then append `.env.serving-node.example` and fill in what it marks
   REQUIRED. Four values have no default and the stack refuses to start
   without them, because each one silently misroutes or over-exposes if
   guessed: `DOCIE_DB_HOST`, `DOCIE_SERVING_ADVERTISE_HOST`,
   `DOCIE_SERVING_PUBLISH_IP`, `POSTGRES_PASSWORD`.

2. **Check it before starting:**

   ```
   docker compose -f docker-compose.serving-node.yml run --rm serving \
     docie-serving doctor --probe-ports
   ```

   `--probe-ports` is worth passing here and only here: binding the
   deploy window means something in the serving container's own network
   namespace, and nothing on the host.

3. **Start it:**

   ```
   docker compose -f docker-compose.serving-node.yml up -d
   ```

4. **Confirm it joined.** The Studio's **Fleet** page lists it within one
   reconcile interval (10s by default), or:

   ```
   curl -s -H "Authorization: Bearer $KEY" https://<central>/v1/serving/nodes
   ```

   A node that appears with `never_reported: true` has a row but has never
   published a measurement — desired state was staged for it and it has not
   started, or it cannot reach Postgres.

5. **Seed the models it should serve.** The catalog is global; the blob store
   is **node-local**. A model listed in the Studio says nothing about which
   nodes hold its weights.

   Seeding is **not node-addressable**: the Studio's "Add model" path is an
   Inngest event on the shared serving app, so it lands on whichever node
   Inngest picks. To put a model on a *chosen* node, seed it on that node:

   ```
   # from a model Ollama already pulled on this machine
   docker compose -f docker-compose.serving-node.yml exec serving \
     docie-serving store seed-ollama lfm2.5:350m --name lfm2.5-350m --family openai_chat

   # or from a GGUF already on this node's disk
   docker compose -f docker-compose.serving-node.yml exec serving \
     docie-serving store add-gguf /models/model.gguf --name lfm2.5-350m --family openai_chat
   ```

   There is no node-local "pull from Hugging Face" command. For a model that
   is not already on the machine, download the GGUF onto the node by whatever
   means (`huggingface-cli download`, `curl`) and then `store add-gguf` it.

   Use the **same `--name` on every node**. That name is what the catalog and
   every deploy refer to, so the same weights registered under two names are
   two different models as far as routing is concerned.

   `docie-serving doctor` lists which catalogued models this node is missing.

   Store writes on a node serialise on a lock in the store root
   (`.index.lock`), so a `store` command run by hand and a seed job landing
   at the same time cannot lose each other's registration. A `store`
   command that reports a lock timeout is waiting on one of those.

### Add a serving node with WireGuard and Coolify

Use one Coolify application per serving host. Once the central stack's private
ports are reachable, clone a known-good serving application for each additional
host. Replace uppercase placeholders in the commands and configuration below
with the values assigned to your fleet.

1. **Join the private network.** Obtain a distinct WireGuard client profile
   from the hub administrator. Keep the assigned address, endpoint and keys.
   Set client `AllowedIPs = FLEET_VPN_CIDR` for this example so the default
   Internet route stays on the public interface. Omit the profile's `DNS` line
   if the host should keep its current resolver. On a peer whose inbound
   connectivity expires while idle, set `PersistentKeepalive = 25` in
   `[Peer]` before installing the profile. Transfer the profile securely.

   Install `wireguard-tools` with `apt` on Ubuntu or `dnf` on Rocky Linux.
   With the profile copied to the SSH user's home on the new node, run:

   ```bash
   sudo install -d -m 700 /etc/wireguard
   sudo install -o root -g root -m 600 ~/wg-fleet.conf /etc/wireguard/wg-fleet.conf &&
     rm ~/wg-fleet.conf
   sudo wg-quick up wg-fleet
   ping -c 3 -W 2 VPN_HUB_IP
   sudo wg show wg-fleet
   ip -4 route get CENTRAL_VPN_IP
   ip -4 route get 1.1.1.1
   ```

   Expect a recent handshake, a VPN route to central and the existing Internet
   route. Keep the original SSH session open while testing. If routes are
   wrong, use `sudo wg-quick down wg-fleet` and correct the profile. Once
   verified, use `sudo systemctl enable wg-quick@wg-fleet` to enable it on boot.

2. **Verify both directions before deploying.** On the new node, test the
   actual host-published ports of central Postgres and Inngest. This example
   uses Postgres `CENTRAL_DB_PORT`, Inngest HTTP `CENTRAL_INNGEST_HTTP_PORT` and Connect `CENTRAL_INNGEST_CONNECT_PORT`.

   ```bash
   for port in CENTRAL_DB_PORT CENTRAL_INNGEST_HTTP_PORT CENTRAL_INNGEST_CONNECT_PORT; do
     if timeout 3 bash -c 'true < "/dev/tcp/CENTRAL_VPN_IP/$1"' _ "$port"; then
       echo "$port OK"
     else
       echo "$port FAILED"
     fi
   done
   ```

   If these fail, check the hub's peer forwarding, firewall policy, and central
   Docker bindings. A loopback binding is unreachable from the new node.
   Changing central bindings requires a separate central redeploy, which can
   unload its running models. Plan that change once, before onboarding nodes.
   Limit access to the intended peers on a shared VPN.

   From the Coolify host, check the new node's SSH endpoint (example port
   `NODE_SSH_PORT`):

   ```bash
   ssh-keyscan -T 5 -p NODE_SSH_PORT NEW_NODE_VPN_IP
   ```

3. **Register the host in Coolify.** Use its VPN IP, actual SSH port and
   login user. Follow [Coolify's SSH key setup](https://coolify.io/docs/core/infrastructure/servers/add-server)
   for that account. For a non-root account,
   verify passwordless sudo with `sudo -n id` as that user. Select
   **Validate Connection** and complete Docker setup if prompted. The serving
   ports are reached directly over the VPN and need no Coolify proxy or domain.

   | Failure | Check |
   |---|---|
   | SSH connection times out | Check the listener, handshake and keepalive. Compare the host test with `sudo docker exec coolify ssh-keyscan -T 5 -p NODE_SSH_PORT NEW_NODE_VPN_IP`. |
   | SSH reports `Permission denied (publickey)` | Compare the active Coolify public key's fingerprint with `ssh-keygen -lf ~/.ssh/authorized_keys` on the target. Check the selected user, ownership and permissions. |
   | Validation fails after SSH | Inspect the failing checkpoint and run `sudo docker version` and `sudo docker compose version` on the target. |

4. **Clone and configure the serving application.** Select the new host as
   the destination. Keep the approved serving revision. In this repository,
   the base directory is `/small-doc-ie-bench` and the Compose location
   relative to it is `/docker-compose.serving-node.yml`. Verify the resolved
   file is `/small-doc-ie-bench/docker-compose.serving-node.yml` and only
   `serving` appears under `services:`. Leave domains empty.

   Update the per-node settings:

   ```text
   DOCIE_SERVING_NODE_ID=new-node-id
   DOCIE_SERVING_ADVERTISE_HOST=NEW_NODE_VPN_IP
   DOCIE_SERVING_PUBLISH_IP=NEW_NODE_VPN_IP
   DOCIE_SERVING_MEM_LIMIT=NODE_MEMORY_LIMIT
   ```

   Use a unique node ID and size the example memory ceiling for the actual
   host. Check that the clone has its own persistent `serving-state` volume.
   Verify these shared settings in the clone (substitute your central
   endpoints):

   ```text
   DOCIE_DB_HOST=CENTRAL_VPN_IP
   DOCIE_DB_PORT=CENTRAL_DB_PORT
   INNGEST_BASE_URL=http://CENTRAL_VPN_IP:CENTRAL_INNGEST_HTTP_PORT
   INNGEST_CONNECT_GATEWAY_URL=ws://CENTRAL_VPN_IP:CENTRAL_INNGEST_CONNECT_PORT/v0/connect
   INNGEST_SERVING_APP_ID=docie-serving
   INNGEST_DEV=0
   DOCIE_SERVING_PORT_RANGE_START=8088
   DOCIE_SERVING_PORT_RANGE_END=8107
   DOCIE_SERVING_DESIRED_APPLY=1
   ```

   Verify `POSTGRES_PASSWORD`, `INNGEST_EVENT_KEY`, and `INNGEST_SIGNING_KEY`
   match central, and keep `LLAMA_REF` pinned to the same revision. Compose
   constructs the node's `DATABASE_URL`. The host-side port range must bind
   only to the node's private IP. The same port range can be reused on each
   host. Coolify may regenerate empty Compose-managed environment fields.
   Check the actual service and required values before deploying.

5. **Deploy and verify.** Check the deployment logs if the container never
   starts. On the new host, locate the running serving container:

   ```bash
   sudo docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}'
   ```

   Set `CID` to the ID in that row, then run:

   ```bash
   sudo docker port "$CID" 8088/tcp
   sudo docker port "$CID" 8107/tcp
   sudo docker logs --since 5m --tail 100 "$CID"
   sudo docker exec "$CID" docie-serving doctor
   ```

   Expect both ports on the node's VPN IP, a successful Inngest Connect
   startup, and a fresh heartbeat for the new ID in Studio **Fleet**. On older
   images, a fresh `doctor` process may report a missing database URL before
   initializing its database engine. Check the worker logs and Fleet heartbeat
   before treating that message as a missing setting. Docker bridge mode also
   puts the host VPN IP outside the container's interfaces. Verify the published
   bindings and test the runtime from the central API container after loading it.

6. **Test one model.** Copy a small GGUF onto the new host and into its
   serving container. Use the same catalog name and identical model bytes if
   it will be a replica of an existing deployment.

   ```bash
   sudo docker cp ./model.gguf "$CID:/tmp/docie-seed.gguf"
   sudo docker exec "$CID" docie-serving store add-gguf \
     /tmp/docie-seed.gguf --name lfm2.5-350m --family openai_chat
   ```

   Confirm the new node holds the model in Fleet. Deploy with the new node
   selected as **Target node**, then run an extraction through the central
   API and check `served_by.node`. If the deployment fails, check the node's
   local blob inventory, memory admission and worker logs. If it starts but
   inference fails, test its advertised address and assigned runtime port from
   the central API container. This path uses a node-local model store.

### What a node actually needs

Three things, and nothing else:

- the central **Postgres**, for the catalog it publishes into and reads
  desired state from;
- the central **Inngest**, for lifecycle events (outbound only — a node
  needs no inbound reachability for job delivery);
- its own **`DOCIE_SERVING_HOME`** volume, which must never be shared
  between machines.

Of what that volume holds, only `node_id` cannot be rebuilt. Everything else
reconverges: `deployments.json` from the desired state in the catalog, the
weights from a seed. So when the volume's lifecycle belongs to something other
than you — a PaaS, an orchestrator, an infrastructure-as-code run — set
`DOCIE_SERVING_NODE_ID` in the node's `.env` and the identity stops depending
on the volume surviving.

Those three are durable. The **model weights are not** — they can be seeded
or pulled again.

By default the weights live at `$DOCIE_SERVING_HOME/models`, a **child** of
the state directory. Removing that directory does *not* remove `node_id`
beside it. What the default costs is **placement**: the weights share one
volume with the durable state, so they cannot be put on a different disk,
their growth consumes that volume, and destroying or resizing it takes both.

### Moving the model cache to its own volume

Optional. Worth doing when the weights outgrow the state volume, or when you
want to reclaim that space without touching anything durable.

**The weights do not move by themselves.** Setting the variable without
copying them makes the node report an empty blob store and re-seed
everything, because the old files are still sitting in the state volume
where nothing is looking for them.

1. Stop the node:

   ```
   docker compose -f docker-compose.serving-node.yml stop serving
   ```

2. Uncomment the `model-cache` mount in `docker-compose.serving-node.yml`
   and set, in the node `.env`:

   ```
   DOCIE_MODEL_STORE_DIR=/app/.cache/models
   ```

3. Copy the existing weights across, with both volumes mounted:

   ```
   docker compose -f docker-compose.serving-node.yml run --rm \
     --entrypoint sh serving -c \
     'mkdir -p /app/.cache/models && cp -a /app/.serving/models/. /app/.cache/models/'
   ```

4. Start it and confirm the move before deleting anything:

   ```
   docker compose -f docker-compose.serving-node.yml up -d serving
   docker compose -f docker-compose.serving-node.yml exec serving \
     docie-serving store list
   ```

   The same models should be listed as before. `docie-serving doctor` also
   reports what this node holds against the catalog.

5. Only then reclaim the old copy:

   ```
   docker compose -f docker-compose.serving-node.yml exec serving \
     rm -rf /app/.serving/models
   ```

A cache root that **is, or contains, `$DOCIE_SERVING_HOME`** is refused at
startup — a directory you intend to clear must not hold `node_id`,
`deployments.json` or the applied-state file.

Leaving the variable unset keeps the current layout exactly. Nothing has to
move to upgrade.

In particular it does **not** need a synchronised copy of `configs/`.
That file is the routing table, read by the central api and the worker
role; the serving role takes its launch arguments from the family contract
in code plus the store entry. The node compose mounts it only so
`docie-serving gateway`/`plan` work if you exec them in the container, and a
stale copy there cannot change what the node serves (#593).

## Check the node before it takes traffic

`docie-serving doctor` exits non-zero when a check FAILS, so a bringup script
can gate on it. Each check maps to a failure that is otherwise invisible:

| Check | What a bad result means |
|---|---|
| **worker role** | `DOCIE_WORKER_ROLE` without `serving`: the process registers no serving functions and will never deploy, reconcile or publish. |
| **serving home** | Not writable: the node cannot remember what it is running, or what it is called. |
| **node id** | Duplicated across nodes. `ModelPlacement.name` is a primary key, so two nodes sharing an id overwrite each other's rows — one node's replica goes unroutable while still holding its RAM. |
| **advertise host** | Resolves to loopback or to *another machine*. Deploys record `http://{this}:{port}/v1`, and the central API resolves that in **its** network. This is the failure that returns a 200 with the wrong model. |
| **model catalog** | Unreachable Postgres. The node keeps serving what is loaded, but it is invisible to the fleet and converges on nothing. |
| **inngest** | Unreachable: no lifecycle events arrive. Desired state still converges, since that is read from the catalog. |
| **desired-state apply** | Off: the node joins, reports healthy, and ignores every deploy aimed at it, every auto-placed deploy that picks it, and the unload half of a drain. Routing exclusion and capacity publishing work regardless. |
| **port window** | Inverted, or ports held by something else. A runtime on an unpublished port records an endpoint nothing can reach. |
| **blob store** | Lists the catalogued models this node lacks. Normal on a fleet; a deploy of one of them aimed here is refused until it is seeded. |
| **clocks** | The database clock against this node's. Liveness and the drain grace period compare timestamps written here against the **central API's** clock, so a large number is a reason to check NTP on both. |

`--json` gives the same verdicts as a list of objects.

## Deploy to a specific node

Deploys are addressable; seeding is not. In the Studio's Deploy form pick a
**Target node**, or let the platform rank the nodes by what actually fits.

A targeted deploy is recorded as **desired state owned by that node**, not fired
as an event, so the response carries no progress stream — the node converges on
its own cycle. If nothing happens, check `desired-state apply` in the doctor
output: that is the setting which makes the difference between a recorded
intent and a running model.

## Take a node out

To stop a node receiving traffic, use the **drain** control on the Fleet page
(or `POST /v1/serving/nodes/{id}/drain`), which excludes it from routing
immediately and lets it stop its runtimes once in-flight requests have
finished. See *Taking a node out* in {doc}`../architecture/serving` for the two
phases and why a drain excludes even the last replica.

### Unattended, from a deployment hook

Replacing a node's container kills every `llama-server` it holds, because they
are its children. A pre-stop hook that drains first and blocks is one command:

```bash
export DOCIE_API_URL=https://<central>
export DOCIE_API_KEY=<key>
docie-serving fleet drain <node-id> --wait --timeout 900
```

Non-zero means **do not replace the container**: a timeout, a node that stopped
reporting while still holding runtimes, or a node that was not draining at all.
The command never adds `--acknowledge-outage` by itself, so a refusal naming a
model about to lose its last replica stops the deploy rather than being worked
around. See {doc}`../reference/serving-cli` for exactly what exit 0 claims —
the guarantee is bounded by the deployments' declared request timeouts, not a
proof that every request finished.

Afterwards:

```bash
docie-serving fleet resume <node-id>
```

Resume is explicit, and comes **after** the replacement container is up and
healthy — a resume puts the node back in the routing rotation, so doing it
first routes requests at a machine that is restarting. The node restores the
deployments the drain stopped on its next cycle; `docie-serving fleet status`
shows that happening.

:::{warning}
Both halves of a drain need the node's reconciler to be converging. A node with
`DOCIE_SERVING_DESIRED_APPLY` off leaves the routing rotation and keeps its
memory, and a resume on it restores nothing. `docie-serving doctor` reports that
setting for exactly this reason.

Note that `docker-compose.serving-node.yml` defaults it ON while the central
`docker-compose.yml` does not set it at all — so on a machine that is central
*and* serving, drain is routing-exclusion only until you set it there.
:::

If drain is not available on your build, stopping the container is the blunt
version:

```
docker compose -f docker-compose.serving-node.yml stop serving
```

Its placements stay in the catalog and are **demoted** rather than removed —
routing prefers replicas on live nodes but will still use a stale node's row
rather than refuse the only one, because a reconciler that stopped publishing
does not prove the runtime died. So stopping the only node serving a model
leaves that model's requests going to a machine that is off.

## Remove a node permanently

1. Move its models elsewhere first: deploy them on another node, and confirm
   from the Fleet page that each has a live placement that is not on this node.
2. Stop and remove the container.
3. Delete its deployments so no placement rows point at it. Rows owned by a
   node that no longer exists are the one case nothing cleans up
   automatically — they are indistinguishable from a node that is temporarily
   off, which is exactly why they are not reaped.
4. Its `serving_node` row stays as a tombstone showing when it last reported.

## Diagnose

| Symptom | Likely cause |
|---|---|
| Requests for a model return a 200 with the **wrong** model's output | `DOCIE_SERVING_ADVERTISE_HOST` resolves to another machine. Run the doctor. |
| A node appears in the Fleet page as `staged` / `never reported` | It has a row but has never published: not started, or cannot reach Postgres. |
| A targeted or auto-placed deploy records and nothing starts | `DOCIE_SERVING_DESIRED_APPLY` is off on that node. |
| A model deploys fine centrally but fails on one node | Its blob is not seeded there. The doctor lists what is missing. |
| A node's replica holds RAM but takes no traffic | Duplicate node id: another node's publish overwrote its placement row. |
| The doctor reports `no node id could be resolved` | The `node_id` file in the serving home is there but unreadable (failing volume, wrong ownership). The node refuses to mint a second identity rather than orphan the rows the old one owns, so repair the volume -- deleting the file is what loses them. |
| A node rejoins as new after a platform redeploy, leaving its old row behind | Its state volume was replaced, so a generated `node_id` was regenerated. Set `DOCIE_SERVING_NODE_ID` explicitly, then delete the orphaned deployments. |
| Every deploy on a node fails to allocate a port | Something outside the container holds the published window. |
| A node flaps between reporting and not | Clock skew between it and the central API, or a reconcile interval longer than the staleness window. |
| A `store` command exits on a store index lock timeout | A seed job or another `store` command on that node still holds `.index.lock`. Wait for it, or look for a stuck seed. |

Logs are node-local. `docker compose -f docker-compose.serving-node.yml logs -f
serving` on the machine itself; there is no central log aggregation, and a
lifecycle action aimed at another node's deployment is refused rather than
proxied.
